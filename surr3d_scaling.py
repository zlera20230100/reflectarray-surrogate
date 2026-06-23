"""Active-sampling scaling in 3-D and the 2-D vs 3-D sample-savings comparison.

For each strategy (uniform / random / resonance-aware) and budget
N in {20,30,45,60,90,120} (of 324 grid pts), trains the 3->64x4->2 MLP over
>=5 seeds. Reports mean+-std (overall held-out and resonance-region Ly<=2.8) and
fits err ~ C*N^-k with bootstrap 95% CI on k.

Then computes the samples-to-reach a target held-out error for uniform vs
resonance-aware in 3-D, the savings factor, and compares it to the 2-D case
(loaded from surr2d_scaling_ms.npz).
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np

import surrogate3d as S
from surr3d_strategies import resonance_aware_balanced

# use the balanced resonance-aware sampler (space-filling base + curvature-
# biased refinement). the naive curvature-greedy (S.resonance_aware_anchors)
# collapses onto unwrap artifacts in 3-D and is kept only as an ablation in
# surr3d_strategy_compare.npz.
S.resonance_aware_anchors = resonance_aware_balanced

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUDGETS = (20, 30, 45, 60, 90, 120)
N_SEEDS = 5
N_BOOT = 2000
ITERS = 8000
RNG_BOOT = np.random.default_rng(12345)


def train_eval(idx, seed):
    model, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx], S.H_all[idx],
                                 S.RE_all[idx], S.IM_all[idx],
                                 iters=ITERS, seed=seed)
    m = S.metrics(model, set(idx))
    return m['heldout'], m['resonance_heldout'], m['heldout_mag']


def samples_to_reach(Narr, err_curve, target):
    """Smallest N at which (monotonised) err <= target, via log-log interp.
    Returns np.inf if never reached within the budget range, with linear
    extrapolation in log-log beyond the last point only if trend is downward."""
    Narr = np.asarray(Narr, float)
    e = np.asarray(err_curve, float)
    # use cumulative-min so a single noisy bump doesn't break monotonicity
    em = np.minimum.accumulate(e)
    if em[0] <= target:
        return float(Narr[0])
    if em[-1] > target:
        # extrapolate in log-log using last-two-point slope if decreasing
        if em[-1] < em[-2]:
            lx = np.log(Narr[-2:]); ly = np.log(em[-2:])
            slope = (ly[1] - ly[0]) / (lx[1] - lx[0])
            if slope < 0:
                lN = lx[1] + (np.log(target) - ly[1]) / slope
                return float(np.exp(lN))
        return float('inf')
    # find bracket where it crosses
    for i in range(1, len(em)):
        if em[i] <= target < em[i - 1]:
            lx = np.log(Narr[i - 1:i + 1]); ly = np.log(em[i - 1:i + 1])
            lN = lx[0] + (np.log(target) - ly[0]) * (lx[1] - lx[0]) / (ly[1] - ly[0])
            return float(np.exp(lN))
    return float('inf')


def run():
    t0 = time.time()
    print(f'device={S.DEVICE}  grid={S.N_ALL} pts  budgets={BUDGETS}  '
          f'seeds={N_SEEDS}  iters={ITERS}')
    strategies = ('uniform', 'random', 'resonance-aware')
    held = {s: {N: [] for N in BUDGETS} for s in strategies}
    res = {s: {N: [] for N in BUDGETS} for s in strategies}
    magh = {s: {N: [] for N in BUDGETS} for s in strategies}

    for N in BUDGETS:
        uni_idx = S.uniform_anchors(N)
        rea_idx = S.resonance_aware_anchors(N)
        for s in range(N_SEEDS):
            h, r, mg = train_eval(uni_idx, seed=s)
            held['uniform'][N].append(h); res['uniform'][N].append(r)
            magh['uniform'][N].append(mg)

            h, r, mg = train_eval(rea_idx, seed=s)
            held['resonance-aware'][N].append(h)
            res['resonance-aware'][N].append(r)
            magh['resonance-aware'][N].append(mg)

            rnd_idx = S.random_anchors(N, seed=1000 + s)
            h, r, mg = train_eval(rnd_idx, seed=s)
            held['random'][N].append(h); res['random'][N].append(r)
            magh['random'][N].append(mg)
        print(f'  N={N:>3} done  ({time.time()-t0:5.1f}s)')
        # incremental save of raw matrices so far
        _save_partial(held, res, magh)

    Narr = np.array(BUDGETS, float)
    agg = {}
    for s in strategies:
        agg[s] = dict(
            heldout_mean=np.array([np.mean(held[s][N]) for N in BUDGETS]),
            heldout_std=np.array([np.std(held[s][N]) for N in BUDGETS]),
            res_mean=np.array([np.mean(res[s][N]) for N in BUDGETS]),
            res_std=np.array([np.std(res[s][N]) for N in BUDGETS]),
            mag_mean=np.array([np.mean(magh[s][N]) for N in BUDGETS]),
        )

    def bootstrap_k(per_seed_dict):
        mat = np.array([per_seed_dict[N] for N in BUDGETS])  # (nN,nSeeds)
        nseed = mat.shape[1]; ks = []
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
        k_res, _, r2_res = S.fit_powerlaw(Narr, agg[s]['res_mean'])
        ci_res = np.percentile(bootstrap_k(res[s]), [2.5, 97.5])
        k_all, _, r2_all = S.fit_powerlaw(Narr, agg[s]['heldout_mean'])
        ci_all = np.percentile(bootstrap_k(held[s]), [2.5, 97.5])
        fits[s] = dict(k_res=k_res, k_res_CI=ci_res, r2_res=r2_res,
                       k_all=k_all, k_all_CI=ci_all, r2_all=r2_all)
        print(f'  {s:>15} res: k={k_res:5.2f} CI[{ci_res[0]:.2f},'
              f'{ci_res[1]:.2f}] r2={r2_res:.2f}   overall: k={k_all:5.2f} '
              f'CI[{ci_all[0]:.2f},{ci_all[1]:.2f}] r2={r2_all:.2f}')

    print('\n--- TABLE: overall held-out phase error (deg), mean+-std ---')
    print(f'{"N":>4} | ' + ' | '.join(f'{s:^20}' for s in strategies))
    for j, N in enumerate(BUDGETS):
        cells = [f'{agg[s]["heldout_mean"][j]:6.2f} +-{agg[s]["heldout_std"][j]:5.2f}'
                 for s in strategies]
        print(f'{N:>4} | ' + ' | '.join(f'{c:^20}' for c in cells))

    print('\n--- TABLE: resonance-region (Ly<=2.8) held-out err (deg), mean+-std ---')
    print(f'{"N":>4} | ' + ' | '.join(f'{s:^20}' for s in strategies))
    for j, N in enumerate(BUDGETS):
        cells = [f'{agg[s]["res_mean"][j]:6.2f} +-{agg[s]["res_std"][j]:5.2f}'
                 for s in strategies]
        print(f'{N:>4} | ' + ' | '.join(f'{c:^20}' for c in cells))

    # 2-D vs 3-D sample-savings ratio
    print('\n=== TASK 3: 2-D vs 3-D sample-savings "bite" ===')
    d2 = np.load(os.path.join(OUT_DIR, 'surr2d_scaling_ms.npz'))
    N2 = d2['N']
    uni2 = d2['uniform_held'].mean(axis=1)   # (nN,) overall held-out mean
    rea2 = d2['resaware_held'].mean(axis=1)
    uni2_res = d2['uniform_res'].mean(axis=1)
    rea2_res = d2['resaware_res'].mean(axis=1)

    uni3 = agg['uniform']['heldout_mean']
    rea3 = agg['resonance-aware']['heldout_mean']
    uni3_res = agg['uniform']['res_mean']
    rea3_res = agg['resonance-aware']['res_mean']

    # choose targets reachable in both dims for the overall metric
    targets_all = [10.0, 7.0, 5.0]
    targets_res = [15.0, 10.0, 6.0]

    def ratio_table(label, N2, uni2, rea2, N3, uni3, rea3, targets):
        print(f'\n  [{label}]  samples-to-reach (uniform / resaware / savings factor)')
        print(f'  {"target":>7} | {"2D uni":>7} {"2D rea":>7} {"2D x":>6} '
              f'| {"3D uni":>7} {"3D rea":>7} {"3D x":>6} | {"3D/2D bite":>10}')
        rows = []
        for tg in targets:
            u2 = samples_to_reach(N2, uni2, tg); r2 = samples_to_reach(N2, rea2, tg)
            u3 = samples_to_reach(N3, uni3, tg); r3 = samples_to_reach(N3, rea3, tg)
            x2 = u2 / r2 if np.isfinite(u2) and r2 > 0 else float('inf')
            x3 = u3 / r3 if np.isfinite(u3) and r3 > 0 else float('inf')
            bite = x3 / x2 if np.isfinite(x2) and x2 > 0 else float('inf')
            print(f'  {tg:>7.1f} | {u2:>7.1f} {r2:>7.1f} {x2:>6.2f} '
                  f'| {u3:>7.1f} {r3:>7.1f} {x3:>6.2f} | {bite:>10.2f}')
            rows.append((tg, u2, r2, x2, u3, r3, x3, bite))
        return np.array(rows)

    overall_rows = ratio_table('OVERALL held-out', N2, uni2, rea2,
                               Narr, uni3, rea3, targets_all)
    res_rows = ratio_table('RESONANCE region', N2, uni2_res, rea2_res,
                           Narr, uni3_res, rea3_res, targets_res)

    # verdict
    def med_finite(col):
        v = col[np.isfinite(col)]
        return float(np.median(v)) if v.size else float('inf')
    bite_all = med_finite(overall_rows[:, 7])
    bite_res = med_finite(res_rows[:, 7])
    # also report the raw 2-D and 3-D savings factors (median over finite tgts)
    x2_all = med_finite(overall_rows[:, 3]); x3_all = med_finite(overall_rows[:, 6])
    x2_res = med_finite(res_rows[:, 3]); x3_res = med_finite(res_rows[:, 6])
    print('\n--- VERDICT (bite = 3D savings-factor / 2D savings-factor) ---')
    print(f'  OVERALL  : 2D savings x={x2_all:.2f}  3D savings x={x3_all:.2f}'
          f'  -> bite={bite_all:.2f}')
    print(f'  RESONANCE: 2D savings x={x2_res:.2f}  3D savings x={x3_res:.2f}'
          f'  -> bite={bite_res:.2f}')

    def verdict(x2, x3, name):
        if not (np.isfinite(x2) and np.isfinite(x3)):
            print(f'  [{name}] inconclusive (a savings factor is non-finite).')
            return
        b = x3 / x2
        if b > 1.10:
            print(f'  [{name}] => resonance-aware BITES HARDER in 3-D '
                  f'(savings {x2:.1f}x -> {x3:.1f}x, bite {b:.2f}).')
        elif b < 0.90:
            print(f'  [{name}] => HONEST: savings factor SHRINKS in 3-D '
                  f'({x2:.1f}x -> {x3:.1f}x, bite {b:.2f}).')
        else:
            print(f'  [{name}] => savings factor roughly UNCHANGED 2D->3D '
                  f'({x2:.1f}x -> {x3:.1f}x, bite {b:.2f}).')
    verdict(x2_all, x3_all, 'OVERALL')
    verdict(x2_res, x3_res, 'RESONANCE')

    # save
    out = os.path.join(OUT_DIR, 'surr3d_scaling.npz')
    save = dict(N=Narr, n_seeds=N_SEEDS, n_boot=N_BOOT, iters=ITERS,
                targets_overall=np.array(targets_all),
                targets_res=np.array(targets_res),
                bite_overall_rows=overall_rows, bite_res_rows=res_rows,
                bite_overall_median=bite_all, bite_res_median=bite_res,
                savings2d_overall=x2_all, savings3d_overall=x3_all,
                savings2d_res=x2_res, savings3d_res=x3_res,
                N2d=N2, uni2d_held=uni2, rea2d_held=rea2,
                uni2d_res=uni2_res, rea2d_res=rea2_res)
    keymap = {'uniform': 'uni', 'random': 'rnd', 'resonance-aware': 'rea'}
    for s in strategies:
        p = keymap[s]
        save[f'{p}_heldout_mean'] = agg[s]['heldout_mean']
        save[f'{p}_heldout_std'] = agg[s]['heldout_std']
        save[f'{p}_res_mean'] = agg[s]['res_mean']
        save[f'{p}_res_std'] = agg[s]['res_std']
        save[f'{p}_mag_mean'] = agg[s]['mag_mean']
        save[f'{p}_k_res'] = fits[s]['k_res']
        save[f'{p}_k_res_CI'] = fits[s]['k_res_CI']
        save[f'{p}_k_all'] = fits[s]['k_all']
        save[f'{p}_k_all_CI'] = fits[s]['k_all_CI']
        save[f'{p}_r2_res'] = fits[s]['r2_res']
        save[f'{p}_r2_all'] = fits[s]['r2_all']
        save[f'{p}_res_raw'] = np.array([res[s][N] for N in BUDGETS])
        save[f'{p}_held_raw'] = np.array([held[s][N] for N in BUDGETS])
    np.savez(out, **save)
    print(f'\nsaved -> {out}  (total {time.time()-t0:.1f}s)')
    return agg, fits


def _save_partial(held, res, magh):
    out = os.path.join(OUT_DIR, 'surr3d_scaling_partial.npz')
    save = {}
    keymap = {'uniform': 'uni', 'random': 'rnd', 'resonance-aware': 'rea'}
    Ns = sorted({N for s in held for N in held[s] if held[s][N]})
    save['N_done'] = np.array(Ns, float)
    for s in held:
        p = keymap[s]
        for N in held[s]:
            if held[s][N]:
                save[f'{p}_held_{int(N)}'] = np.array(held[s][N])
                save[f'{p}_res_{int(N)}'] = np.array(res[s][N])
    np.savez(out, **save)


if __name__ == '__main__':
    run()
