"""Three-parameter (Lx, Ly, h) differentiable surrogate for the unit cell,
extending surrogate2d.py to three normalized inputs. Reads the openEMS 3-D
reference ref3d.npz (Lx (9,), Ly (9,), h (4,); phase/mag (4,9,9), 324 points;
a fixed (Lx,Ly) gives different phase across h). The network maps normalized
(Lx,Ly,h) to (Re Gamma, Im Gamma) with a 3->64x4->2 tanh MLP; reports circular
phase error (overall and resonance region Ly<=2.8) and magnitude error.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(OUT_DIR, 'ref3d.npz')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the full-wave 3-D reference
ref = np.load(REF_PATH)
Lx_axis = ref['Lx'].astype(np.float64)          # (9,)
Ly_axis = ref['Ly'].astype(np.float64)          # (9,)
h_axis = ref['h'].astype(np.float64)            # (4,)
phase_grid = ref['phase'].astype(np.float64)    # (4,9,9) [h,Ly,Lx] wrapped deg
mag_grid = ref['mag'].astype(np.float64)        # (4,9,9)

NH, NY, NX = phase_grid.shape                    # 4, 9, 9
LXMIN, LXMAX = float(Lx_axis.min()), float(Lx_axis.max())
LYMIN, LYMAX = float(Ly_axis.min()), float(Ly_axis.max())
HMIN, HMAX = float(h_axis.min()), float(h_axis.max())
RES_LY = 2.8                                      # resonance region: Ly <= 2.8

# Flattened grid of all 324 points: index order (ih, iy, ix)
IH, IY, IX = np.meshgrid(np.arange(NH), np.arange(NY), np.arange(NX),
                         indexing='ij')
IH = IH.ravel(); IY = IY.ravel(); IX = IX.ravel()
N_ALL = IH.size                                   # 324
LX_all = Lx_axis[IX]
LY_all = Ly_axis[IY]
H_all = h_axis[IH]
PH_all = phase_grid[IH, IY, IX]                   # deg
MG_all = mag_grid[IH, IY, IX]
G_all = MG_all * np.exp(1j * np.deg2rad(PH_all))
RE_all = G_all.real
IM_all = G_all.imag


def norm_xyh(Lx, Ly, h):
    """Map physical (Lx,Ly,h) to normalized [-1,1]^3."""
    nx = 2.0 * (Lx - LXMIN) / (LXMAX - LXMIN) - 1.0
    ny = 2.0 * (Ly - LYMIN) / (LYMAX - LYMIN) - 1.0
    nh = 2.0 * (h - HMIN) / (HMAX - HMIN) - 1.0
    return np.stack([nx, ny, nh], axis=-1)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Surrogate model: normalized (Lx,Ly,h) -> (Re, Im)
# ---------------------------------------------------------------------------
class Surrogate3D(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyh):
        return self.net(xyh)


def train_surrogate(Lx_a, Ly_a, h_a, Re_a, Im_a, width=64, depth=4,
                    iters=20000, lr=2e-3, target_loss=1e-7, seed=SEED):
    torch.manual_seed(seed)
    model = Surrogate3D(width, depth).to(DEVICE)
    x = torch.tensor(norm_xyh(Lx_a, Ly_a, h_a), dtype=torch.float32,
                     device=DEVICE)
    tgt = torch.tensor(np.stack([Re_a, Im_a], axis=1),
                       dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.5)
    loss_fn = nn.MSELoss()
    final = None
    for it in range(iters):
        opt.zero_grad()
        loss = loss_fn(model(x), tgt)
        loss.backward()
        opt.step()
        sched.step()
        final = loss.item()
        if final < target_loss:
            break
    return model, final


def predict(model, Lx, Ly, h):
    model.eval()
    x = torch.tensor(norm_xyh(Lx, Ly, h), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(x).cpu().numpy()
    G = out[:, 0] + 1j * out[:, 1]
    return np.rad2deg(np.angle(G)), np.abs(G)


# ---------------------------------------------------------------------------
# Metrics: circular phase error on held-out + resonance region
# ---------------------------------------------------------------------------
def metrics(model, anchor_set):
    ph_pred, mg_pred = predict(model, LX_all, LY_all, H_all)
    err = np.abs(wrap180(ph_pred - PH_all))
    mag_err = np.abs(mg_pred - MG_all)
    is_anchor = np.array([i in anchor_set for i in range(N_ALL)])
    held = ~is_anchor
    res_mask = LY_all <= RES_LY
    res_held = res_mask & held
    return dict(
        overall=float(err.mean()),
        heldout=float(err[held].mean()) if held.any() else float('nan'),
        resonance=float(err[res_mask].mean()),
        resonance_heldout=float(err[res_held].mean()) if res_held.any()
        else float('nan'),
        mag_err=float(mag_err.mean()),
        heldout_mag=float(mag_err[held].mean()) if held.any() else float('nan'),
    )


# ---------------------------------------------------------------------------
# Anchor-placement strategies (return flat indices into the 324-point grid)
# ---------------------------------------------------------------------------
def _flat(ih, iy, ix):
    return (ih * NY + iy) * NX + ix


def uniform_anchors(N):
    """N anchors on a near-regular sub-grid in (Lx,Ly,h), snapped to grid.
    Always sample all h-levels (only 4), spread Lx,Ly evenly; trim/fill to N
    by greedy farthest-point in normalized coordinates."""
    # distribute N across h levels and an (ny x nx) tile per level
    per_h = max(2, int(round((N / NH) ** 0.5)))
    ny = max(2, min(NY, per_h))
    nx = max(2, min(NX, int(np.ceil((N / NH) / ny))))
    iy_t = np.unique(np.round(np.linspace(0, NY - 1, ny)).astype(int))
    ix_t = np.unique(np.round(np.linspace(0, NX - 1, nx)).astype(int))
    idx = sorted({_flat(ih, a, b)
                  for ih in range(NH) for a in iy_t for b in ix_t})
    if len(idx) > N:
        idx = _farthest_trim(idx, N)
    elif len(idx) < N:
        idx = _fill_to_N(idx, N)
    return sorted(idx)


def random_anchors(N, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(N_ALL, size=N, replace=False).tolist())


def resonance_aware_anchors(N):
    """Greedy curvature-weighted active sampling on the 3-D grid.

    Curvature weight = |3-D discrete Laplacian| of the unwrapped phase (large
    across the sharp small-Ly resonance and at h-values where it is sharp, tiny
    on the smooth plateau). Start from the 8 corners of the (Lx,Ly,h) box, then
    repeatedly add the grid point where an inverse-distance interpolant of the
    unwrapped phase has the largest curvature-weighted error -> anchors crowd
    into the resonance band and oversample the sharp-h slices.
    """
    # unwrap phase along Ly (axis 1, the resonant length) per h-slice
    ph_uw = np.rad2deg(np.unwrap(np.deg2rad(phase_grid), axis=1))   # (4,9,9)
    # 3-D discrete Laplacian magnitude as curvature
    curv = np.zeros_like(ph_uw)
    curv[:, 1:-1, :] += np.abs(ph_uw[:, :-2, :] - 2 * ph_uw[:, 1:-1, :]
                               + ph_uw[:, 2:, :])
    curv[:, :, 1:-1] += np.abs(ph_uw[:, :, :-2] - 2 * ph_uw[:, :, 1:-1]
                               + ph_uw[:, :, 2:])
    curv[1:-1, :, :] += np.abs(ph_uw[:-2, :, :] - 2 * ph_uw[1:-1, :, :]
                               + ph_uw[2:, :, :])
    # edge replicate
    curv[:, 0, :] = curv[:, 1, :]; curv[:, -1, :] = curv[:, -2, :]
    curv[:, :, 0] = curv[:, :, 1]; curv[:, :, -1] = curv[:, :, -2]
    curv[0, :, :] = curv[1, :, :]; curv[-1, :, :] = curv[-2, :, :]
    w = 1.0 + curv / (curv.max() + 1e-12)
    w_flat = w.ravel()
    field = ph_uw.ravel()

    coords = norm_xyh(LX_all, LY_all, H_all)         # (324,3)

    corners = [_flat(ih, iy, ix)
               for ih in (0, NH - 1) for iy in (0, NY - 1)
               for ix in (0, NX - 1)]
    chosen = list(dict.fromkeys(corners))[:N]
    while len(chosen) < N:
        cs = np.array(chosen)
        d2 = ((coords[:, None, :] - coords[None, cs, :]) ** 2).sum(-1)
        d2 = np.maximum(d2, 1e-9)
        wts = 1.0 / d2
        interp = (wts * field[cs][None, :]).sum(1) / wts.sum(1)
        wel = w_flat * np.abs(field - interp)
        wel[cs] = -1.0
        chosen.append(int(np.argmax(wel)))
    return sorted(chosen)


def _farthest_trim(idx, N):
    coords = norm_xyh(LX_all, LY_all, H_all)
    idx = list(idx)
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
    coords = norm_xyh(LX_all, LY_all, H_all)
    idx = list(idx)
    inset = set(idx)
    pool = [i for i in range(N_ALL) if i not in inset]
    while len(idx) < N and pool:
        ic = coords[idx]
        d = np.array([min(((coords[p] - ic) ** 2).sum(1)) for p in pool])
        nxt = pool[int(np.argmax(d))]
        idx.append(nxt); pool.remove(nxt)
    return idx


# ---------------------------------------------------------------------------
# Power-law fit  err ~ C * N^(-k)
# ---------------------------------------------------------------------------
def fit_powerlaw(N, err):
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


if __name__ == '__main__':
    print(f'device = {DEVICE}')
    print(f'grid: {N_ALL} pts  Lx {LXMIN}-{LXMAX}({NX})  Ly {LYMIN}-{LYMAX}'
          f'({NY})  h {HMIN}-{HMAX}({NH})  phase swing '
          f'{phase_grid.max()-phase_grid.min():.1f} deg')
    idx = resonance_aware_anchors(45)
    model, fl = train_surrogate(LX_all[idx], LY_all[idx], H_all[idx],
                                RE_all[idx], IM_all[idx])
    m = metrics(model, set(idx))
    print(f'N=45 resonance-aware  trainMSE={fl:.2e}')
    print(f'  held-out phase err  : {m["heldout"]:.2f} deg')
    print(f'  resonance held-out  : {m["resonance_heldout"]:.2f} deg')
    print(f'  |Gamma| err         : {m["heldout_mag"]:.4f}')
