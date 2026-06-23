# -*- coding: utf-8 -*-
# Active-sampling acquisition based on Gaussian-process posterior variance (uncertainty sampling) instead
# of the curvature criterion. It fits the GP only on already-acquired anchors (no oracle). Compared
# against the oracle curvature sampler and uniform/random on the 2-D patch.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, numpy as np
import surrogate2d as S
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

OUT = os.path.join(S.OUT_DIR, 'principled_acq.npz')
BUDGETS = (9, 12, 16, 20, 25, 36)
N_SEEDS = 6; N_BOOT = 2000
RNG = np.random.default_rng(12345)
COORDS = S.norm_xy(S.LX_all, S.LY_all)                 # (99,2)
RE, IM = S.RE_all, S.IM_all

def gp_uncertainty_anchors(N):
    """Greedy uncertainty sampling: start at corners; add the candidate of max GP posterior std of the
    complex reflection, fitting the GP only on acquired anchors (no oracle)."""
    corners = [S._flat(0, 0), S._flat(0, S.NX-1), S._flat(S.NY-1, 0), S._flat(S.NY-1, S.NX-1)]
    chosen = list(dict.fromkeys(corners))[:N]
    def kern():
        return C(1.0, (1e-3, 1e3)) * RBF(0.5, (0.05, 5.0)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    while len(chosen) < N:
        cs = np.array(chosen)
        gre = GaussianProcessRegressor(kernel=kern(), normalize_y=True, alpha=1e-8, n_restarts_optimizer=0)
        gim = GaussianProcessRegressor(kernel=kern(), normalize_y=True, alpha=1e-8, n_restarts_optimizer=0)
        gre.fit(COORDS[cs], RE[cs]); gim.fit(COORDS[cs], IM[cs])
        _, sre = gre.predict(COORDS, return_std=True)
        _, sim = gim.predict(COORDS, return_std=True)
        score = np.sqrt(sre**2 + sim**2); score[cs] = -1.0
        chosen.append(int(np.argmax(score)))
    return sorted(chosen)

def train_eval(idx, seed):
    m, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx], S.RE_all[idx], S.IM_all[idx], seed=seed)
    mt = S.metrics(m, set(idx)); return mt['heldout'], mt['resonance_heldout']

def run():
    t0 = time.time()
    strategies = ('uniform', 'random', 'oracle-curv', 'gp-uncert')
    res = {s: {N: [] for N in BUDGETS} for s in strategies}
    held = {s: {N: [] for N in BUDGETS} for s in strategies}
    for N in BUDGETS:
        uni = S.uniform_anchors(N); ora = S.resonance_aware_anchors(N); gp = gp_uncertainty_anchors(N)
        for s in range(N_SEEDS):
            for name, idx in (('uniform', uni), ('oracle-curv', ora), ('gp-uncert', gp),
                              ('random', S.random_anchors(N, 1000+s))):
                h, r = train_eval(idx, s); held[name][N].append(h); res[name][N].append(r)
        print(f"  N={N:>3} done ({time.time()-t0:5.1f}s)  gp={np.mean(res['gp-uncert'][N]):.2f} "
              f"oracle={np.mean(res['oracle-curv'][N]):.2f} uni={np.mean(res['uniform'][N]):.2f}", flush=True)
    Narr = np.array(BUDGETS, float)
    def boot(per):
        mat = np.array([per[N] for N in BUDGETS]); ks = []
        for _ in range(N_BOOT):
            sel = RNG.integers(0, mat.shape[1], mat.shape[1]); k, _, _ = S.fit_powerlaw(Narr, mat[:, sel].mean(1))
            if np.isfinite(k): ks.append(k)
        return np.percentile(ks, [2.5, 97.5])
    save = dict(N=Narr, n_seeds=N_SEEDS)
    km = {'uniform': 'uni', 'random': 'rnd', 'oracle-curv': 'ora', 'gp-uncert': 'gp'}
    print("\n--- resonance-region error (mean+-std) and k [95% CI] ---")
    for s in strategies:
        rm = np.array([np.mean(res[s][N]) for N in BUDGETS]); rs = np.array([np.std(res[s][N]) for N in BUDGETS])
        k, _, r2 = S.fit_powerlaw(Narr, rm); ci = boot(res[s]); p = km[s]
        save[f'{p}_res_mean'] = rm; save[f'{p}_res_std'] = rs; save[f'{p}_k'] = k; save[f'{p}_k_CI'] = ci; save[f'{p}_r2'] = r2
        save[f'{p}_res_raw'] = np.array([res[s][N] for N in BUDGETS])
        print(f"  {s:>12}: N=36 {rm[-1]:.2f}  k={k:.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
    np.savez(OUT, **save); print(f"\nsaved {OUT} ({time.time()-t0:.1f}s)", flush=True)

if __name__ == '__main__':
    run()
