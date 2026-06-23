"""Multi-seed scaling-law experiment for the 2-D surrogate.

Reuses anchor selection and training from surrogate2d.py. For each strategy
(uniform / random / resonance-aware) and budget N, trains the MLP over >=6 seeds
(seed varies MLP init/training; for 'random' the anchor draw also varies). Reports
mean+-std for each strategy and fits err = C*N^-k with a bootstrap 95% CI on k.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np

import surrogate2d as S   # reuse anchors + training + metrics + grid

OUT_DIR = os.getcwd()
BUDGETS = (9, 12, 16, 20, 25, 36)
N_SEEDS = 6
N_BOOT = 2000
RNG_BOOT = np.random.default_rng(12345)


def train_eval(idx, seed):
    """Train one surrogate on anchor indices idx with the given seed, return
    (heldout circular phase err, resonance-region held-out phase err)."""
    model, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx],
                                 S.RE_all[idx], S.IM_all[idx], seed=seed)
    m = S.metrics(model, set(idx))
    return m['heldout'], m['resonance_heldout']


def run():
    t0 = time.time()
    print(f'device = {S.DEVICE}  grid={S.N_ALL} pts  '
          f'budgets={BUDGETS}  seeds={N_SEEDS}')

    # per-strategy, per-N: arrays of length N_SEEDS for heldout & res errors
    strategies = ('uniform', 'random', 'resonance-aware')
    held = {s: {N: [] for N in BUDGETS} for s in strategies}
    res = {s: {N: [] for N in BUDGETS} for s in strategies}

    for N in BUDGETS:
        # uniform & resonance-aware: anchors are deterministic in N; seed varies
        # only the MLP init/training. random: anchor draw varies by seed too.
        uni_idx = S.uniform_anchors(N)
        rea_idx = S.resonance_aware_anchors(N)
        for s in range(N_SEEDS):
            h, r = train_eval(uni_idx, seed=s)
            held['uniform'][N].append(h); res['uniform'][N].append(r)

            h, r = train_eval(rea_idx, seed=s)
            held['resonance-aware'][N].append(h); res['resonance-aware'][N].append(r)

            rnd_idx = S.random_anchors(N, seed=1000 + s)
            h, r = train_eval(rnd_idx, seed=s)
            held['random'][N].append(h); res['random'][N].append(r)
        print(f'  N={N:>3} done  ({time.time()-t0:5.1f}s)')

    # aggregate mean/std
    Narr = np.array(BUDGETS, float)
    agg = {}
    for s in strategies:
        hm = np.array([np.mean(held[s][N]) for N in BUDGETS])
        hs = np.array([np.std(held[s][N]) for N in BUDGETS])
        rm = np.array([np.mean(res[s][N]) for N in BUDGETS])
        rs = np.array([np.std(res[s][N]) for N in BUDGETS])
        agg[s] = dict(heldout_mean=hm, heldout_std=hs,
                      res_mean=rm, res_std=rs)

    # power-law fit on the per-seed mean curve + bootstrap CI on k
    def fit_k(err_curve):
        k, C, r2 = S.fit_powerlaw(Narr, err_curve)
        return k, C, r2

    def bootstrap_k(per_seed_dict):
        """per_seed_dict[N] = list of seed errors. Resample seeds with
        replacement, recompute mean curve, refit k. Returns array of k."""
        ks = []
        mat = np.array([per_seed_dict[N] for N in BUDGETS])  # (nN, nSeeds)
        nseed = mat.shape[1]
        for _ in range(N_BOOT):
            sel = RNG_BOOT.integers(0, nseed, size=nseed)
            curve = mat[:, sel].mean(axis=1)
            k, _, _ = S.fit_powerlaw(Narr, curve)
            if np.isfinite(k):
                ks.append(k)
        return np.array(ks)

    fits = {}
    print('\n--- power-law fits  err ~ C*N^-k  (bootstrap 95% CI on k) ---')
    for s in strategies:
        # resonance-region
        k_res, _, r2_res = fit_k(agg[s]['res_mean'])
        kb_res = bootstrap_k(res[s])
        ci_res = np.percentile(kb_res, [2.5, 97.5])
        # overall held-out
        k_all, _, r2_all = fit_k(agg[s]['heldout_mean'])
        kb_all = bootstrap_k(held[s])
        ci_all = np.percentile(kb_all, [2.5, 97.5])
        fits[s] = dict(k_res=k_res, k_res_CI=ci_res, r2_res=r2_res,
                       k_all=k_all, k_all_CI=ci_all, r2_all=r2_all)
        print(f'  {s:>15}  res-region: k={k_res:5.2f} '
              f'CI[{ci_res[0]:.2f},{ci_res[1]:.2f}] r2={r2_res:.2f}   '
              f'overall: k={k_all:5.2f} CI[{ci_all[0]:.2f},{ci_all[1]:.2f}] '
              f'r2={r2_all:.2f}')

    # table: resonance-region error mean+-std
    print('\n--- TABLE: resonance-region held-out phase error (deg), mean+-std ---')
    print(f'{"N":>4} | ' + ' | '.join(f'{s:^20}' for s in strategies))
    for j, N in enumerate(BUDGETS):
        cells = []
        for s in strategies:
            m = agg[s]['res_mean'][j]; sd = agg[s]['res_std'][j]
            cells.append(f'{m:6.2f} +- {sd:5.2f}')
        print(f'{N:>4} | ' + ' | '.join(f'{c:^20}' for c in cells))

    print('\n--- TABLE: overall held-out phase error (deg), mean+-std ---')
    print(f'{"N":>4} | ' + ' | '.join(f'{s:^20}' for s in strategies))
    for j, N in enumerate(BUDGETS):
        cells = []
        for s in strategies:
            m = agg[s]['heldout_mean'][j]; sd = agg[s]['heldout_std'][j]
            cells.append(f'{m:6.2f} +- {sd:5.2f}')
        print(f'{N:>4} | ' + ' | '.join(f'{c:^20}' for c in cells))

    # at each N, is res-aware advantage outside +-std bands?
    print('\n--- VERDICT: resonance-region, res-aware vs uniform & random ---')
    print(f'{"N":>4} {"REA mean":>9} {"UNI mean":>9} {"RND mean":>9} '
          f'{"sep>std?(vsUNI/vsRND)":>24}')
    rea = agg['resonance-aware']; uni = agg['uniform']; rnd = agg['random']
    sep_flags = []
    for j, N in enumerate(BUDGETS):
        # separated if REA's upper band < competitor's lower band
        def separated(comp):
            rea_hi = rea['res_mean'][j] + rea['res_std'][j]
            comp_lo = comp['res_mean'][j] - comp['res_std'][j]
            return rea_hi < comp_lo
        s_uni = separated(uni); s_rnd = separated(rnd)
        sep_flags.append((s_uni, s_rnd))
        print(f'{N:>4} {rea["res_mean"][j]:>9.2f} {uni["res_mean"][j]:>9.2f} '
              f'{rnd["res_mean"][j]:>9.2f} '
              f'{("YES" if s_uni else "no"):>11}/{("YES" if s_rnd else "no"):>11}')

    # k distinguishability
    kr = fits['resonance-aware']['k_res_CI']
    ku = fits['uniform']['k_res_CI']
    krn = fits['random']['k_res_CI']
    print('\n--- k distinguishability (res-region, CI overlap test) ---')
    def overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])
    print(f'  REA k_res CI = [{kr[0]:.2f},{kr[1]:.2f}]')
    print(f'  UNI k_res CI = [{ku[0]:.2f},{ku[1]:.2f}]  '
          f'overlap with REA: {overlap(kr, ku)}')
    print(f'  RND k_res CI = [{krn[0]:.2f},{krn[1]:.2f}]  '
          f'overlap with REA: {overlap(kr, krn)}')

    # save
    out = os.path.join(OUT_DIR, 'surr2d_scaling_ms.npz')
    save = dict(N=Narr, n_seeds=N_SEEDS, n_boot=N_BOOT)
    keymap = {'uniform': 'uni', 'random': 'rnd', 'resonance-aware': 'rea'}
    for s in strategies:
        p = keymap[s]
        save[f'{p}_heldout_mean'] = agg[s]['heldout_mean']
        save[f'{p}_heldout_std'] = agg[s]['heldout_std']
        save[f'{p}_res_mean'] = agg[s]['res_mean']
        save[f'{p}_res_std'] = agg[s]['res_std']
        save[f'{p}_k_res'] = fits[s]['k_res']
        save[f'{p}_k_res_CI'] = fits[s]['k_res_CI']
        save[f'{p}_k_all'] = fits[s]['k_all']
        save[f'{p}_k_all_CI'] = fits[s]['k_all_CI']
        save[f'{p}_r2_res'] = fits[s]['r2_res']
        save[f'{p}_r2_all'] = fits[s]['r2_all']
        # raw per-seed matrices for full transparency
        save[f'{p}_res_raw'] = np.array([res[s][N] for N in BUDGETS])
        save[f'{p}_held_raw'] = np.array([held[s][N] for N in BUDGETS])
    np.savez(out, **save)
    print(f'\nsaved -> {out}   (total {time.time()-t0:.1f}s)')
    return agg, fits


if __name__ == '__main__':
    run()
