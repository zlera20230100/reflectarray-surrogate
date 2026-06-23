# -*- coding: utf-8 -*-
# LOLA-Voronoi adaptive-sampling baseline (Crombecq et al., SIAM J. Sci. Comput. 33(4):1948-1974, 2011):
# Voronoi cell size for exploration, local-nonlinearity for exploitation. Runs greedy and offline on the
# 99-point 2-D patch grid with the same surrogate, budgets, seeds and bootstrap as the curvature study.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, numpy as np
import surrogate2d as S

# design-relevant field = unwrapped phase (same observable the curvature criterion uses)
PH_UW = np.rad2deg(np.unwrap(np.deg2rad(S.phase_grid), axis=0)).ravel()  # flat in (iy*NX+ix) order
COORDS = S.norm_xy(S.LX_all, S.LY_all)                                   # (99,2) normalised
NALL = S.N_ALL; NX = S.NX; NY = S.NY
BUD = (9, 12, 16, 20, 25, 36); N_SEEDS = 6; N_BOOT = 2000; RNG = np.random.default_rng(12345)


def _flat(iy, ix): return iy * NX + ix
CORNERS = [_flat(0, 0), _flat(0, NX - 1), _flat(NY - 1, 0), _flat(NY - 1, NX - 1)]


def lola_voronoi_anchors(N, kln=5):
    """Greedy LOLA-Voronoi on the discrete grid. Returns sorted flat indices."""
    chosen = list(dict.fromkeys(CORNERS))[:N]
    D2 = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)        # (99,99) pairwise sq-dist
    while len(chosen) < N:
        cs = np.array(chosen)
        # voronoi assignment of all grid points to nearest chosen anchor
        nn = cs[np.argmin(D2[:, cs], axis=1)]                            # nearest anchor index per grid pt
        # exploration: relative Voronoi cell size V_i = fraction of grid points in anchor i's cell
        V = np.array([np.mean(nn == c) for c in cs])
        # exploitation: LOLA local nonlinearity at each anchor (residual of a local LS plane through
        # the anchor + its kln nearest OTHER anchors); large where the response bends sharply.
        E = np.zeros(len(cs))
        for j, c in enumerate(cs):
            others = cs[cs != c]
            if others.size < 3:
                E[j] = 0.0; continue
            knn = others[np.argsort(D2[c, others])[:max(3, min(kln, others.size))]]
            pts = np.concatenate([[c], knn])
            A = np.column_stack([COORDS[pts], np.ones(pts.size)])        # [x,y,1]
            f = PH_UW[pts]
            coef, *_ = np.linalg.lstsq(A, f, rcond=None)
            E[j] = float(np.sqrt(np.mean((f - A @ coef) ** 2)))          # plane-fit residual = nonlinearity
        H = V + (E / (E.sum() + 1e-12))                                  # Crombecq hybrid score
        star = cs[int(np.argmax(H))]
        # place new sample in the highest-H anchor's Voronoi cell, at the candidate farthest from all chosen
        cell = np.where((nn == star))[0]
        cell = np.array([g for g in cell if g not in set(chosen)])
        if cell.size == 0:                                              # cell full -> global farthest point
            cell = np.array([g for g in range(NALL) if g not in set(chosen)])
        mind = D2[cell][:, cs].min(axis=1)
        chosen.append(int(cell[int(np.argmax(mind))]))
    return sorted(chosen)


def evalm(model, idx):
    ph_pred, _ = S.predict(model, S.LX_all, S.LY_all)
    err = np.abs(S.wrap180(ph_pred - S.PH_all))
    res_mask = S.LY_all <= S.RES_LY
    held = res_mask & np.array([i not in set(idx) for i in range(NALL)])
    return float(err[held].mean()) if held.any() else float('nan')


t0 = time.time()
res = {N: [] for N in BUD}
print("LOLA-Voronoi on the 2-D patch (resonance-region held-out phase error):", flush=True)
for N in BUD:
    idx = lola_voronoi_anchors(N)
    for s in range(N_SEEDS):
        m, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx], S.RE_all[idx], S.IM_all[idx], seed=s)
        res[N].append(evalm(m, idx))
    print(f"  N={N:>3}  lola {np.mean(res[N]):.2f} +- {np.std(res[N]):.2f}  ({time.time()-t0:.0f}s)", flush=True)

Narr = np.array(BUD, float)
mat = np.array([res[N] for N in BUD])
rm = mat.mean(1); rs = mat.std(1)
k, _, r2 = S.fit_powerlaw(Narr, rm)
ks = []
for _ in range(N_BOOT):
    sel = RNG.integers(0, mat.shape[1], mat.shape[1]); kk, _, _ = S.fit_powerlaw(Narr, mat[:, sel].mean(1))
    if np.isfinite(kk): ks.append(kk)
ci = np.percentile(ks, [2.5, 97.5])
print(f"\nLOLA-Voronoi: N=36 {rm[-1]:.2f} deg ; k={k:.2f} [{ci[0]:.2f},{ci[1]:.2f}] (r2={r2:.2f})")
np.savez(os.path.join(S.OUT_DIR, 'lola.npz'), N=Narr, lola_res_mean=rm, lola_res_std=rs,
         lola_k=k, lola_k_CI=ci, lola_raw=mat, n_seeds=N_SEEDS)
print(f"saved lola.npz ({time.time()-t0:.0f}s)")
