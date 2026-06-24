# -*- coding: utf-8 -*-
# Dual-resonance cell: two separated y-dipoles whose lengths are the two sweep params, giving two
# resonance ridges; tests whether the curvature criterion samples both bands. Same protocol as the
# main 2-D study: resonance-aware (curvature) vs uniform vs random, multi-seed, with a bootstrap
# power-law exponent on the resonance-region held-out phase error. Output: dual_scaling.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, numpy as np
import surrogate2d as S      # reuse Surrogate2D, train_surrogate, norm_xy, fit_powerlaw, wrap180

REF = np.load(os.path.join(S.OUT_DIR, 'ref2d_dual.npz'))
Lx_ax = REF['Lx'].astype(float); Ly_ax = REF['Ly'].astype(float)
ph = REF['phase'].astype(float); mg = REF['mag'].astype(float)        # (NY,NX)
NY, NX = ph.shape
IY, IX = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij'); IY = IY.ravel(); IX = IX.ravel()
LXf = Lx_ax[IX]; LYf = Ly_ax[IY]; PHf = ph[IY, IX]
G = mg[IY, IX]*np.exp(1j*np.deg2rad(PHf)); REf, IMf = G.real, G.imag
NALL = PHf.size; RES_LY = 2.8; res_mask = (LXf <= RES_LY) | (LYf <= RES_LY)   # near either resonant strip
COORDS = S.norm_xy(LXf, LYf)
BUD = (9, 12, 16, 20, 25, 36); N_SEEDS = int(os.environ.get('N_SEEDS', '6')); N_BOOT = 2000; RNG = np.random.default_rng(12345)
print(f"cross grid {NALL} pts; phase span {PHf.max()-PHf.min():.0f} deg", flush=True)

def _flat(iy, ix): return iy*NX+ix
def uniform_anchors(N):
    ny = max(2, min(NY, int(round(np.sqrt(N*NY/NX))))); nx = max(2, min(NX, int(np.ceil(N/ny))))
    iy = np.unique(np.round(np.linspace(0, NY-1, ny)).astype(int)); ix = np.unique(np.round(np.linspace(0, NX-1, nx)).astype(int))
    idx = sorted({_flat(a, b) for a in iy for b in ix})
    pool = [i for i in range(NALL) if i not in idx]
    while len(idx) > N: idx.pop()
    k = 0
    while len(idx) < N and k < len(pool):
        if pool[k] not in idx: idx.append(pool[k])
        k += 1
    return sorted(idx[:N])
def random_anchors(N, seed): return sorted(np.random.default_rng(seed).choice(NALL, N, replace=False).tolist())
def curv_anchors(N):
    uw = np.rad2deg(np.unwrap(np.deg2rad(ph), axis=0)); c = np.zeros_like(uw)
    c[1:-1, :] += np.abs(uw[:-2, :]-2*uw[1:-1, :]+uw[2:, :]); c[:, 1:-1] += np.abs(uw[:, :-2]-2*uw[:, 1:-1]+uw[:, 2:])
    c[0, :] = c[1, :]; c[-1, :] = c[-2, :]; c[:, 0] = c[:, 1]; c[:, -1] = c[:, -2]
    w = 1.0+c/(c.max()+1e-12); wf = w.ravel(); field = uw.ravel()
    corners = [_flat(0, 0), _flat(0, NX-1), _flat(NY-1, 0), _flat(NY-1, NX-1)]; chosen = list(dict.fromkeys(corners))[:N]
    while len(chosen) < N:
        cs = np.array(chosen); d2 = np.maximum(((COORDS[:, None]-COORDS[None, cs])**2).sum(-1), 1e-9)
        wts = 1.0/d2; interp = (wts*field[cs][None]).sum(1)/wts.sum(1); sc = wf*np.abs(field-interp); sc[cs] = -1.0
        chosen.append(int(np.argmax(sc)))
    return sorted(chosen)
def evalm(model, idx):
    ph_pred, _ = S.predict(model, LXf, LYf)
    err = np.abs(S.wrap180(ph_pred-PHf)); held = res_mask & np.array([i not in set(idx) for i in range(NALL)])
    return float(err[held].mean()) if held.any() else float('nan')
def train_eval(idx, seed):
    m, _ = S.train_surrogate(LXf[idx], LYf[idx], REf[idx], IMf[idx], seed=seed); return evalm(m, idx)

t0 = time.time(); strategies = ('uniform', 'random', 'resonance-aware')
res = {s: {N: [] for N in BUD} for s in strategies}
for N in BUD:
    uni = uniform_anchors(N); cur = curv_anchors(N)
    for s in range(N_SEEDS):
        res['uniform'][N].append(train_eval(uni, s)); res['resonance-aware'][N].append(train_eval(cur, s))
        res['random'][N].append(train_eval(random_anchors(N, 1000+s), s))
    print(f"  N={N:>3}  rea {np.mean(res['resonance-aware'][N]):.2f}  uni {np.mean(res['uniform'][N]):.2f}  "
          f"rnd {np.mean(res['random'][N]):.2f}  ({time.time()-t0:.0f}s)", flush=True)
Narr = np.array(BUD, float)
def boot(per):
    mat = np.array([per[N] for N in BUD]); ks = []
    for _ in range(N_BOOT):
        sel = RNG.integers(0, mat.shape[1], mat.shape[1]); k, _, _ = S.fit_powerlaw(Narr, mat[:, sel].mean(1))
        if np.isfinite(k): ks.append(k)
    return np.percentile(ks, [2.5, 97.5])
save = dict(N=Narr, n_seeds=N_SEEDS, phase_span=float(PHf.max()-PHf.min())); km = {'uniform':'uni','random':'rnd','resonance-aware':'rea'}
print("\n--- DUAL-RESONANCE: resonance-region err (N=36) and k [95% CI] ---")
for s in strategies:
    rm = np.array([np.mean(res[s][N]) for N in BUD]); rs = np.array([np.std(res[s][N]) for N in BUD])
    k, _, r2 = S.fit_powerlaw(Narr, rm); ci = boot(res[s]); p = km[s]
    save[f'{p}_res_mean'] = rm; save[f'{p}_res_std'] = rs; save[f'{p}_k'] = k; save[f'{p}_k_CI'] = ci
    save[f'{p}_res_raw'] = np.array([res[s][N] for N in BUD])
    print(f"  {s:>15}: N=36 {rm[-1]:.2f}  k={k:.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
np.savez(os.path.join(S.OUT_DIR, 'dual_scaling.npz'), **save)
print(f"\nsaved cross_scaling.npz ({time.time()-t0:.0f}s)", flush=True)
