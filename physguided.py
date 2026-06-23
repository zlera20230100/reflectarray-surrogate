# -*- coding: utf-8 -*-
# Physics-guided active sampling: a closed-form patch cavity/transmission-line model gives the resonant
# frequency f_res(Lx,Ly), and anchors are placed in the band where f_res ~ f0 (physics-weighted maximin),
# with no oracle and no pilot grid. The model is used only to locate the resonance for sampling, not to
# regularise the fit. Same surrogate, budgets, seeds, bootstrap and metric as the curvature study, on the
# 2-D patch.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, numpy as np
import surrogate2d as S

C0 = 299792458.0
F0 = 24e9; EPSR = 4.4; H_MM = 1.0
COORDS = S.norm_xy(S.LX_all, S.LY_all)
NALL = S.N_ALL; NX = S.NX; NY = S.NY
BUD = (9, 12, 16, 20, 25, 36); N_SEEDS = 6; N_BOOT = 2000; RNG = np.random.default_rng(12345)


def cavity_fres(Lx_mm, Ly_mm):
    """Hammerstad cavity/TL model: TM010 resonant frequency of a patch, resonant length = Ly, width = Lx.
    Closed form, no full-wave solve."""
    W = np.maximum(Lx_mm, 1e-3); L = Ly_mm
    eps_eff = (EPSR + 1) / 2 + (EPSR - 1) / 2 * (1 + 12 * H_MM / W) ** -0.5
    dL = 0.412 * H_MM * (eps_eff + 0.3) * (W / H_MM + 0.264) / ((eps_eff - 0.258) * (W / H_MM + 0.8))
    L_eff_m = (L + 2 * dL) * 1e-3
    return C0 / (2 * L_eff_m * np.sqrt(eps_eff))


# physics prior over the grid: weight high where the analytic resonance is near the operating frequency
FRES = cavity_fres(S.LX_all, S.LY_all)
SIGMA = 6e9                                                     # band half-width in Hz (a priori, not tuned)
WPHYS = np.exp(-((FRES - F0) / SIGMA) ** 2)                     # in [0,1], peaks on the physical resonance


def _flat(iy, ix): return iy * NX + ix
CORNERS = [_flat(0, 0), _flat(0, NX - 1), _flat(NY - 1, 0), _flat(NY - 1, NX - 1)]


def physguided_anchors(N):
    """Physics-weighted maximin: crowd the analytic resonance band while still covering the box."""
    D2 = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)
    chosen = list(dict.fromkeys(CORNERS))[:N]
    while len(chosen) < N:
        cs = np.array(chosen)
        mind = D2[:, cs].min(axis=1)                           # distance to nearest chosen (space-filling)
        score = (0.3 + WPHYS) * mind                           # physics weight x coverage; 0.3 floor = explore
        score[cs] = -1.0
        chosen.append(int(np.argmax(score)))
    return sorted(chosen)


def evalm(model, idx):
    ph_pred, _ = S.predict(model, S.LX_all, S.LY_all)
    err = np.abs(S.wrap180(ph_pred - S.PH_all))
    res_mask = S.LY_all <= S.RES_LY
    held = res_mask & np.array([i not in set(idx) for i in range(NALL)])
    return float(err[held].mean()) if held.any() else float('nan')


print(f"cavity-model f_res range {FRES.min()/1e9:.1f}-{FRES.max()/1e9:.1f} GHz; "
      f"physics band (f_res~24GHz) covers {int((WPHYS>0.5).sum())} of {NALL} grid pts", flush=True)
t0 = time.time(); res = {N: [] for N in BUD}
for N in BUD:
    idx = physguided_anchors(N)
    for s in range(N_SEEDS):
        m, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx], S.RE_all[idx], S.IM_all[idx], seed=s)
        res[N].append(evalm(m, idx))
    print(f"  N={N:>3}  physguided {np.mean(res[N]):.2f} +- {np.std(res[N]):.2f}  ({time.time()-t0:.0f}s)", flush=True)

Narr = np.array(BUD, float); mat = np.array([res[N] for N in BUD]); rm = mat.mean(1); rs = mat.std(1)
k, _, r2 = S.fit_powerlaw(Narr, rm); ks = []
for _ in range(N_BOOT):
    sel = RNG.integers(0, mat.shape[1], mat.shape[1]); kk, _, _ = S.fit_powerlaw(Narr, mat[:, sel].mean(1))
    if np.isfinite(kk): ks.append(kk)
ci = np.percentile(ks, [2.5, 97.5])
print(f"\nPHYSICS-GUIDED: N=36 {rm[-1]:.2f} deg ; k={k:.2f} [{ci[0]:.2f},{ci[1]:.2f}] (r2={r2:.2f})")
np.savez(os.path.join(S.OUT_DIR, 'physguided.npz'), N=Narr, pg_res_mean=rm, pg_res_std=rs,
         pg_k=k, pg_k_CI=ci, pg_raw=mat, n_seeds=N_SEEDS, fres=FRES, wphys=WPHYS, sigma=SIGMA)
print(f"saved physguided.npz ({time.time()-t0:.0f}s)")
