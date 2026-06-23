import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

# reproducibility / paths
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = os.getcwd()
REF_PATH = os.path.join(OUT_DIR, 'ref2d.npz')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load reference full-wave 2-D data: Lx (9,) width, Ly (11,) length,
# phase (11,9) deg [rows=Ly, cols=Lx], mag (11,9) |Gamma|
ref = np.load(REF_PATH)
Lx_axis = ref['Lx'].astype(np.float64)          # (9,)
Ly_axis = ref['Ly'].astype(np.float64)          # (11,)
phase_grid = ref['phase'].astype(np.float64)    # (11,9) wrapped deg
mag_grid = ref['mag'].astype(np.float64)        # (11,9)

NY, NX = phase_grid.shape                        # 11, 9
LXMIN, LXMAX = float(Lx_axis.min()), float(Lx_axis.max())   # 2,6
LYMIN, LYMAX = float(Ly_axis.min()), float(Ly_axis.max())   # 2,6
RES_LY = 2.8                                      # resonance region: Ly <= 2.8

# Flattened grid of all 99 points: rows index (iy, ix)
IY, IX = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
IY = IY.ravel(); IX = IX.ravel()
N_ALL = IY.size                                   # 99
LX_all = Lx_axis[IX]                              # (99,)
LY_all = Ly_axis[IY]                              # (99,)
PH_all = phase_grid[IY, IX]                       # (99,) deg
MG_all = mag_grid[IY, IX]                         # (99,)
G_all = MG_all * np.exp(1j * np.deg2rad(PH_all))
RE_all = G_all.real
IM_all = G_all.imag


def norm_xy(Lx, Ly):
    """Map physical (Lx,Ly) mm to normalized [-1,1]^2."""
    nx = 2.0 * (Lx - LXMIN) / (LXMAX - LXMIN) - 1.0
    ny = 2.0 * (Ly - LYMIN) / (LYMAX - LYMIN) - 1.0
    return np.stack([nx, ny], axis=-1)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


# surrogate model: normalized (Lx,Ly) -> (Re Gamma, Im Gamma)
class Surrogate2D(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


def train_surrogate(Lx_a, Ly_a, Re_a, Im_a, width=64, depth=4,
                    iters=20000, lr=2e-3, target_loss=1e-7, seed=SEED):
    torch.manual_seed(seed)
    model = Surrogate2D(width, depth).to(DEVICE)
    xy = torch.tensor(norm_xy(Lx_a, Ly_a), dtype=torch.float32, device=DEVICE)
    tgt = torch.tensor(np.stack([Re_a, Im_a], axis=1),
                       dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.5)
    loss_fn = nn.MSELoss()
    final = None
    for it in range(iters):
        opt.zero_grad()
        loss = loss_fn(model(xy), tgt)
        loss.backward()
        opt.step()
        sched.step()
        final = loss.item()
        if final < target_loss:
            break
    return model, final


def predict(model, Lx, Ly):
    model.eval()
    xy = torch.tensor(norm_xy(Lx, Ly), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(xy).cpu().numpy()
    G = out[:, 0] + 1j * out[:, 1]
    return np.rad2deg(np.angle(G)), np.abs(G)


# metrics: circular phase error on held-out + resonance region
def metrics(model, anchor_set):
    ph_pred, mg_pred = predict(model, LX_all, LY_all)
    err = np.abs(wrap180(ph_pred - PH_all))
    mag_err = np.abs(mg_pred - MG_all)
    held = np.array([i for i in range(N_ALL) if i not in anchor_set])
    res_mask = LY_all <= RES_LY
    res_held = res_mask & np.array([i not in anchor_set for i in range(N_ALL)])
    return dict(
        overall=float(err.mean()),
        heldout=float(err[held].mean()) if held.size else float('nan'),
        resonance=float(err[res_mask].mean()),
        resonance_heldout=float(err[res_held].mean()) if res_held.any() else float('nan'),
        mag_err=float(mag_err.mean()),
        heldout_mag=float(mag_err[held].mean()) if held.size else float('nan'),
    )


# anchor-placement strategies (return flat indices into the 99-point grid)
def _flat(iy, ix):
    return iy * NX + ix


def uniform_anchors(N):
    """N anchors on a near-square sub-grid, evenly spaced in (Lx,Ly), snapped."""
    ny = int(round(np.sqrt(N * NY / NX)))
    ny = max(2, min(NY, ny))
    nx = int(np.ceil(N / ny))
    nx = max(2, min(NX, nx))
    iy_t = np.unique(np.round(np.linspace(0, NY - 1, ny)).astype(int))
    ix_t = np.unique(np.round(np.linspace(0, NX - 1, nx)).astype(int))
    idx = sorted({_flat(a, b) for a in iy_t for b in ix_t})
    # adjust to exactly N: trim extras (drop interior) or add nearest unused
    allidx = list(range(N_ALL))
    if len(idx) > N:
        # keep a spread-out subset by greedy farthest-point on normalized coords
        idx = _farthest_trim(idx, N)
    elif len(idx) < N:
        idx = _fill_to_N(idx, N)
    return sorted(idx)


def random_anchors(N, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(N_ALL, size=N, replace=False).tolist())


def resonance_aware_anchors(N):
    """Greedy curvature-weighted active sampling on the 2-D grid.

    Curvature weight = |2-D Laplacian| of the unwrapped phase (large across the
    sharp small-Ly resonance, tiny on the smooth plateau). Start from the 4
    corners, then repeatedly add the point where a current
    (RBF/inverse-distance) interpolant of unwrapped phase has the largest
    curvature-weighted error -> anchors crowd into the resonance band.
    """
    # unwrap phase along Ly (axis 0) for a smooth field to interpolate
    ph_uw = np.unwrap(np.deg2rad(phase_grid), axis=0)
    ph_uw = np.rad2deg(ph_uw)
    # 2-D discrete Laplacian magnitude as curvature
    curv = np.zeros_like(ph_uw)
    curv[1:-1, :] += np.abs(ph_uw[:-2, :] - 2 * ph_uw[1:-1, :] + ph_uw[2:, :])
    curv[:, 1:-1] += np.abs(ph_uw[:, :-2] - 2 * ph_uw[:, 1:-1] + ph_uw[:, 2:])
    curv[0, :] = curv[1, :]; curv[-1, :] = curv[-2, :]
    curv[:, 0] = curv[:, 1]; curv[:, -1] = curv[:, -2]
    w = 1.0 + curv / (curv.max() + 1e-12)
    w_flat = w.ravel()
    field = ph_uw.ravel()

    # normalized coordinates for distance-based interpolation
    coords = norm_xy(LX_all, LY_all)             # (99,2)

    corners = [_flat(0, 0), _flat(0, NX - 1),
               _flat(NY - 1, 0), _flat(NY - 1, NX - 1)]
    chosen = list(dict.fromkeys(corners))[:N]
    while len(chosen) < N:
        cs = np.array(chosen)
        # inverse-distance interpolant from chosen anchors
        d2 = ((coords[:, None, :] - coords[None, cs, :]) ** 2).sum(-1)  # (99,k)
        d2 = np.maximum(d2, 1e-9)
        wts = 1.0 / d2
        interp = (wts * field[cs][None, :]).sum(1) / wts.sum(1)
        wel = w_flat * np.abs(field - interp)
        wel[cs] = -1.0
        chosen.append(int(np.argmax(wel)))
    return sorted(chosen)


def _farthest_trim(idx, N):
    coords = norm_xy(LX_all, LY_all)
    idx = list(idx)
    # greedy farthest-point selection seeded by corner-most point
    start = int(np.argmax(np.abs(coords[idx]).sum(1)))
    keep = [idx[start]]
    pool = [i for i in idx if i != idx[start]]
    while len(keep) < N and pool:
        kc = coords[keep]
        d = np.array([min(((coords[p] - kc) ** 2).sum(1)) for p in pool])
        nxt = pool[int(np.argmax(d))]
        keep.append(nxt); pool.remove(nxt)
    return keep


def _fill_to_N(idx, N):
    coords = norm_xy(LX_all, LY_all)
    idx = list(idx)
    pool = [i for i in range(N_ALL) if i not in idx]
    while len(idx) < N and pool:
        ic = coords[idx]
        d = np.array([min(((coords[p] - ic) ** 2).sum(1)) for p in pool])
        nxt = pool[int(np.argmax(d))]
        idx.append(nxt); pool.remove(nxt)
    return idx


# basic multi-parameter surrogate demo
def basic_demo():
    print('=== 1) MULTI-PARAMETER DIFFERENTIABLE SURROGATE ===')
    idx = resonance_aware_anchors(25)
    model, floss = train_surrogate(LX_all[idx], LY_all[idx],
                                   RE_all[idx], IM_all[idx])
    m = metrics(model, set(idx))
    print(f'anchors N=25 (resonance-aware), train MSE={floss:.2e}')
    print(f'  overall phase err   : {m["overall"]:.2f} deg')
    print(f'  held-out phase err  : {m["heldout"]:.2f} deg')
    print(f'  resonance (Ly<=2.8) : {m["resonance"]:.2f} deg')
    print(f'  |Gamma| err (mean)  : {m["mag_err"]:.4f}')
    return model


# active-sampling scaling law
def fit_powerlaw(N, err):
    """Fit err ~ C * N^(-k) via log-log least squares. Returns (k, C, r2)."""
    N = np.asarray(N, float); err = np.asarray(err, float)
    good = np.isfinite(err) & (err > 0)
    if good.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    x = np.log(N[good]); y = np.log(err[good])
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = sol
    yhat = A @ sol
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(-slope), float(np.exp(intercept)), float(r2)


def scaling_study(budgets=(9, 12, 16, 20, 25, 36), n_seeds=6):
    print('\n=== 2) ACTIVE-SAMPLING SCALING LAW ===')
    rng_seeds = list(range(n_seeds))

    # storage
    uni_held = {}; uni_res = {}
    res_held = {}; res_res = {}
    rnd_held = {N: [] for N in budgets}
    rnd_res = {N: [] for N in budgets}
    rnd_held_seeds = []  # (N, seed, err)
    rnd_res_seeds = []

    hdr = f'{"N":>3} {"strategy":>16} {"held-out":>9} {"res-region":>10} {"|G|err":>8}'
    print(hdr)
    for N in budgets:
        # uniform
        idx = uniform_anchors(N)
        model, _ = train_surrogate(LX_all[idx], LY_all[idx],
                                   RE_all[idx], IM_all[idx])
        m = metrics(model, set(idx))
        uni_held[N] = m['heldout']; uni_res[N] = m['resonance_heldout']
        print(f'{N:>3} {"uniform":>16} {m["heldout"]:>9.2f} '
              f'{m["resonance_heldout"]:>10.2f} {m["heldout_mag"]:>8.4f}')

        # resonance-aware
        idx = resonance_aware_anchors(N)
        model, _ = train_surrogate(LX_all[idx], LY_all[idx],
                                   RE_all[idx], IM_all[idx])
        m = metrics(model, set(idx))
        res_held[N] = m['heldout']; res_res[N] = m['resonance_heldout']
        print(f'{N:>3} {"resonance-aware":>16} {m["heldout"]:>9.2f} '
              f'{m["resonance_heldout"]:>10.2f} {m["heldout_mag"]:>8.4f}')

        # random multi-seed
        for s in rng_seeds:
            idx = random_anchors(N, seed=1000 + s)
            model, _ = train_surrogate(LX_all[idx], LY_all[idx],
                                       RE_all[idx], IM_all[idx], seed=s)
            m = metrics(model, set(idx))
            rnd_held[N].append(m['heldout'])
            rnd_res[N].append(m['resonance_heldout'])
            rnd_held_seeds.append((N, s, m['heldout']))
            rnd_res_seeds.append((N, s, m['resonance_heldout']))
        rh = np.array(rnd_held[N]); rr = np.array(rnd_res[N])
        print(f'{N:>3} {"random(mean+-std)":>16} '
              f'{rh.mean():>9.2f} {rr.mean():>10.2f} '
              f'  (+-{rh.std():.2f}/{rr.std():.2f})')

    Narr = np.array(budgets, float)
    uni_h = np.array([uni_held[N] for N in budgets])
    uni_r = np.array([uni_res[N] for N in budgets])
    res_h = np.array([res_held[N] for N in budgets])
    res_r = np.array([res_res[N] for N in budgets])
    rnd_h_mean = np.array([np.mean(rnd_held[N]) for N in budgets])
    rnd_h_std = np.array([np.std(rnd_held[N]) for N in budgets])
    rnd_r_mean = np.array([np.mean(rnd_res[N]) for N in budgets])
    rnd_r_std = np.array([np.std(rnd_res[N]) for N in budgets])

    # power-law fits (on held-out error)
    fits = {}
    for label, arr in (('uniform_held', uni_h), ('resaware_held', res_h),
                       ('random_held', rnd_h_mean),
                       ('uniform_res', uni_r), ('resaware_res', res_r),
                       ('random_res', rnd_r_mean)):
        k, C, r2 = fit_powerlaw(Narr, arr)
        fits[label] = (k, C, r2)

    print('\n--- power-law fits  err ~ C * N^(-k)  (held-out phase err) ---')
    for label in ('uniform_held', 'random_held', 'resaware_held'):
        k, C, r2 = fits[label]
        print(f'  {label:>16}: k={k:.3f}  C={C:.2f}  r2={r2:.3f}')
    print('--- power-law fits (resonance-region held-out err) ---')
    for label in ('uniform_res', 'random_res', 'resaware_res'):
        k, C, r2 = fits[label]
        print(f'  {label:>16}: k={k:.3f}  C={C:.2f}  r2={r2:.3f}')

    out = os.path.join(OUT_DIR, 'surr2d_scaling.npz')
    np.savez(
        out,
        N=Narr,
        uniform_heldout=uni_h, uniform_resonance=uni_r,
        resaware_heldout=res_h, resaware_resonance=res_r,
        random_heldout_mean=rnd_h_mean, random_heldout_std=rnd_h_std,
        random_resonance_mean=rnd_r_mean, random_resonance_std=rnd_r_std,
        random_heldout_seeds=np.array(rnd_held_seeds),   # (N,seed,err)
        random_resonance_seeds=np.array(rnd_res_seeds),
        fit_uniform_held=np.array(fits['uniform_held']),
        fit_random_held=np.array(fits['random_held']),
        fit_resaware_held=np.array(fits['resaware_held']),
        fit_uniform_res=np.array(fits['uniform_res']),
        fit_random_res=np.array(fits['random_res']),
        fit_resaware_res=np.array(fits['resaware_res']),
        n_seeds=n_seeds,
    )
    print(f'saved -> {out}')
    return fits


# ML baselines (SVR + small ANN) at equal N
def baseline_study(budgets=(9, 12, 16, 20, 25, 36), n_seeds=6):
    print('\n=== 3) ML BASELINE HEAD-TO-HEAD (SVR / ANN vs MLP surrogate) ===')
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def phase_mae_from_complex(re_pred, im_pred, mask):
        G = re_pred + 1j * im_pred
        ph = np.rad2deg(np.angle(G))
        return float(np.abs(wrap180(ph - PH_all[mask])).mean())

    rows = []
    print(f'{"N":>3} {"anchors":>16} {"MLP":>7} {"SVR":>7} {"ANN":>7}  (held-out phase MAE deg)')
    store = dict(N=[], strategy=[], mlp=[], svr=[], ann=[])
    for N in budgets:
        for sname, sfn in (('uniform', lambda n: uniform_anchors(n)),
                           ('resonance-aware', lambda n: resonance_aware_anchors(n))):
            idx = sfn(N)
            held = np.array([i for i in range(N_ALL) if i not in set(idx)])
            Xtr = norm_xy(LX_all[idx], LY_all[idx])
            Ytr = np.stack([RE_all[idx], IM_all[idx]], axis=1)
            Xall = norm_xy(LX_all, LY_all)

            # MLP surrogate (our torch model)
            model, _ = train_surrogate(LX_all[idx], LY_all[idx],
                                       RE_all[idx], IM_all[idx])
            ph_pred, _ = predict(model, LX_all, LY_all)
            mlp_mae = float(np.abs(wrap180(ph_pred[held] - PH_all[held])).mean())

            # SVR (RBF), one per output channel
            svr_re = make_pipeline(StandardScaler(),
                                   SVR(kernel='rbf', C=100.0, gamma='scale',
                                       epsilon=1e-3)).fit(Xtr, Ytr[:, 0])
            svr_im = make_pipeline(StandardScaler(),
                                   SVR(kernel='rbf', C=100.0, gamma='scale',
                                       epsilon=1e-3)).fit(Xtr, Ytr[:, 1])
            re_p = svr_re.predict(Xall); im_p = svr_im.predict(Xall)
            svr_mae = phase_mae_from_complex(re_p[held], im_p[held], held)

            # small ANN (sklearn MLPRegressor) - "published-style"
            ann = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                               solver='lbfgs', max_iter=5000,
                               random_state=0).fit(Xtr, Ytr)
            yp = ann.predict(Xall)
            ann_mae = phase_mae_from_complex(yp[held, 0], yp[held, 1], held)

            print(f'{N:>3} {sname:>16} {mlp_mae:>7.2f} {svr_mae:>7.2f} '
                  f'{ann_mae:>7.2f}')
            store['N'].append(N); store['strategy'].append(sname)
            store['mlp'].append(mlp_mae); store['svr'].append(svr_mae)
            store['ann'].append(ann_mae)
            rows.append((N, sname, mlp_mae, svr_mae, ann_mae))

    out = os.path.join(OUT_DIR, 'surr2d_baseline.npz')
    np.savez(out,
             N=np.array(store['N']),
             strategy=np.array(store['strategy']),
             mlp_phase_mae=np.array(store['mlp']),
             svr_phase_mae=np.array(store['svr']),
             ann_phase_mae=np.array(store['ann']))
    print(f'saved -> {out}')
    return rows


# differentiable gradient validation
def dphase_dxy(model, Lx, Ly):
    """Autodiff d(phase deg)/dLx and /dLy at physical points. Returns (gx,gy)."""
    model.eval()
    Lxt = torch.tensor(Lx, dtype=torch.float32, device=DEVICE,
                       requires_grad=True).reshape(-1)
    Lyt = torch.tensor(Ly, dtype=torch.float32, device=DEVICE,
                       requires_grad=True).reshape(-1)
    nx = 2.0 * (Lxt - LXMIN) / (LXMAX - LXMIN) - 1.0
    ny = 2.0 * (Lyt - LYMIN) / (LYMAX - LYMIN) - 1.0
    xy = torch.stack([nx, ny], dim=1)
    out = model(xy)
    ph = torch.atan2(out[:, 1], out[:, 0])       # radians
    gx = np.zeros(Lxt.shape[0]); gy = np.zeros(Lyt.shape[0])
    for i in range(Lxt.shape[0]):
        g = torch.autograd.grad(ph[i], [Lxt, Lyt], retain_graph=True)
        gx[i] = g[0][i].item(); gy[i] = g[1][i].item()
    return np.rad2deg(gx), np.rad2deg(gy)        # deg/mm


def gradient_validation(N=25):
    print('\n=== 4) DIFFERENTIABLE GRADIENT VALIDATION ===')
    idx = resonance_aware_anchors(N)
    model, _ = train_surrogate(LX_all[idx], LY_all[idx],
                               RE_all[idx], IM_all[idx])

    # surrogate autodiff on full grid
    gx_s, gy_s = dphase_dxy(model, LX_all, LY_all)

    # full-wave central finite differences of unwrapped phase
    ph_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_grid), axis=0))   # (11,9)
    # also unwrap along Lx for the dLx FD to be safe
    ph_uw_x = np.rad2deg(np.unwrap(np.deg2rad(phase_grid), axis=1))
    gy_fd_grid = np.gradient(ph_uw, Ly_axis, axis=0)               # d/dLy
    gx_fd_grid = np.gradient(ph_uw_x, Lx_axis, axis=1)             # d/dLx
    gx_fd = gx_fd_grid[IY, IX]
    gy_fd = gy_fd_grid[IY, IX]

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def mre(a, b, floor=1.0):
        denom = np.maximum(np.abs(b), floor)
        return float(np.mean(np.abs(a - b) / denom))

    cos_x = cos_sim(gx_s, gx_fd); cos_y = cos_sim(gy_s, gy_fd)
    mre_x = mre(gx_s, gx_fd); mre_y = mre(gy_s, gy_fd)

    print(f'anchors N={N} (resonance-aware)')
    print(f'  dphase/dLx : cosine={cos_x:.4f}  mean-rel-err={mre_x:.4f}')
    print(f'  dphase/dLy : cosine={cos_y:.4f}  mean-rel-err={mre_y:.4f}')

    out = os.path.join(OUT_DIR, 'surr2d_grad.npz')
    np.savez(out,
             Lx=LX_all, Ly=LY_all,
             gx_surrogate=gx_s, gy_surrogate=gy_s,
             gx_finitediff=gx_fd, gy_finitediff=gy_fd,
             cos_x=cos_x, cos_y=cos_y, mre_x=mre_x, mre_y=mre_y,
             anchor_idx=np.array(idx))
    print(f'saved -> {out}')
    return model, (cos_x, cos_y, mre_x, mre_y)


# 2-D inverse design demo
def bilinear_ref(Lx, Ly):
    """Bilinear interpolate openEMS reference (unwrapped phase + mag) at (Lx,Ly)."""
    ph_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_grid), axis=0))   # (11,9)
    # clamp to grid
    Lx = np.clip(Lx, LXMIN, LXMAX); Ly = np.clip(Ly, LYMIN, LYMAX)
    ix = np.searchsorted(Lx_axis, Lx) - 1
    iy = np.searchsorted(Ly_axis, Ly) - 1
    ix = np.clip(ix, 0, NX - 2); iy = np.clip(iy, 0, NY - 2)
    x0, x1 = Lx_axis[ix], Lx_axis[ix + 1]
    y0, y1 = Ly_axis[iy], Ly_axis[iy + 1]
    tx = (Lx - x0) / (x1 - x0); ty = (Ly - y0) / (y1 - y0)

    def bilin(F):
        f00 = F[iy, ix]; f01 = F[iy, ix + 1]
        f10 = F[iy + 1, ix]; f11 = F[iy + 1, ix + 1]
        return (f00 * (1 - tx) * (1 - ty) + f01 * tx * (1 - ty) +
                f10 * (1 - tx) * ty + f11 * tx * ty)

    ph = wrap180(bilin(ph_uw))
    mg = bilin(mag_grid)
    return ph, mg


def inverse_design(model, n_targets=8):
    print('\n=== 5) 2-D INVERSE DESIGN DEMO ===')
    # Pick targets spanning the achievable complex surface. Targets must be
    # genuinely achievable, so for each target phase we read the corresponding
    # achievable |Gamma| directly off the full-wave surface (nearest grid point
    # in phase), rather than imposing a single fixed |Gamma| that the surface
    # cannot deliver across the whole phase swing.
    pmin, pmax = PH_all.min(), PH_all.max()
    margin = 0.05 * (pmax - pmin)
    ph_targets = np.linspace(pmin + margin, pmax - margin, n_targets)
    mg_targets = np.empty(n_targets)
    for i, tp in enumerate(ph_targets):
        j = int(np.argmin(np.abs(wrap180(PH_all - tp))))
        mg_targets[i] = MG_all[j]

    print(f'{"tgt_ph":>8} {"tgt_|G|":>8} {"Lx*":>6} {"Ly*":>6} '
          f'{"ph(ref)":>8} {"|G|(ref)":>9} {"dPh":>7} {"d|G|":>7}')
    rows = []
    for tph, tmg in zip(ph_targets, mg_targets):
        tgt_rad = np.deg2rad(tph)
        tre = tmg * np.cos(tgt_rad); tim = tmg * np.sin(tgt_rad)
        tgt_vec = torch.tensor([tre, tim], dtype=torch.float32, device=DEVICE)
        best = None
        # multi-start over a coarse (Lx,Ly) grid
        for lx0 in np.linspace(LXMIN + 0.2, LXMAX - 0.2, 4):
            for ly0 in np.linspace(LYMIN + 0.2, LYMAX - 0.2, 5):
                p = torch.tensor([[lx0, ly0]], dtype=torch.float32,
                                 device=DEVICE, requires_grad=True)
                opt = torch.optim.Adam([p], lr=0.05)
                for _ in range(600):
                    opt.zero_grad()
                    nx = 2.0 * (p[0, 0] - LXMIN) / (LXMAX - LXMIN) - 1.0
                    ny = 2.0 * (p[0, 1] - LYMIN) / (LYMAX - LYMIN) - 1.0
                    out = model(torch.stack([nx, ny]).reshape(1, 2))
                    # complex match loss (phase + mag together)
                    loss = ((out[0] - tgt_vec) ** 2).sum()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        p[0, 0].clamp_(LXMIN, LXMAX)
                        p[0, 1].clamp_(LYMIN, LYMAX)
                lx = float(p[0, 0].detach()); ly = float(p[0, 1].detach())
                ph_s, mg_s = predict(model, np.array([lx]), np.array([ly]))
                # rank candidates by surrogate complex error
                gs = mg_s[0] * np.exp(1j * np.deg2rad(ph_s[0]))
                e = abs(gs - (tre + 1j * tim))
                if best is None or e < best[-1]:
                    best = (lx, ly, ph_s[0], mg_s[0], e)
        lx, ly, ph_s, mg_s, _ = best
        ph_ref, mg_ref = bilinear_ref(np.array([lx]), np.array([ly]))
        ph_ref = float(ph_ref[0]); mg_ref = float(mg_ref[0])
        dph = abs(wrap180(ph_ref - tph)); dmg = abs(mg_ref - tmg)
        print(f'{tph:>8.1f} {tmg:>8.3f} {lx:>6.3f} {ly:>6.3f} '
              f'{ph_ref:>8.1f} {mg_ref:>9.3f} {dph:>7.2f} {dmg:>7.3f}')
        rows.append((tph, tmg, lx, ly, ph_ref, mg_ref, dph, dmg))

    rows = np.array(rows)
    out = os.path.join(OUT_DIR, 'surr2d_inverse.npz')
    np.savez(out,
             target_phase=rows[:, 0], target_mag=rows[:, 1],
             solved_Lx=rows[:, 2], solved_Ly=rows[:, 3],
             achieved_phase_ref=rows[:, 4], achieved_mag_ref=rows[:, 5],
             phase_err_deg=rows[:, 6], mag_err=rows[:, 7])
    print(f'mean phase err vs ref = {rows[:,6].mean():.2f} deg '
          f'(max {rows[:,6].max():.2f}); mean |G| err = {rows[:,7].mean():.4f}')
    print(f'saved -> {out}')
    return rows


if __name__ == '__main__':
    print(f'device = {DEVICE}')
    print(f'grid: {N_ALL} points  Lx {LXMIN}-{LXMAX} ({NX})  '
          f'Ly {LYMIN}-{LYMAX} ({NY})  phase swing '
          f'{phase_grid.max()-phase_grid.min():.1f} deg')

    basic_demo()
    fits = scaling_study()
    baseline_study()
    gmodel, gstats = gradient_validation(N=25)
    inverse_design(gmodel, n_targets=8)
    print('\nDONE.')
