# -*- coding: utf-8 -*-
# Online resonance-aware active sampling for the 2-D surrogate. Reads full-wave phase/mag only at
# already-acquired anchors (the surrogate2d.py sampler scores curvature on the full 99-point grid).
# Query-by-committee: seed the 4 corners, then at each step build two inverse-distance interpolants
# of the complex reflection (power p=2 and p=8) and acquire the candidate of maximum disagreement
# |G_sharp - G_smooth|, to budget N. MLP trained on the acquired anchors, same metrics. Multi-seed,
# bootstrap 95% CI on k. Compared to uniform / random / oracle resonance-aware.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import numpy as np
import surrogate2d as S   # grid, train, metrics, oracle samplers, powerlaw fit

OUT = os.path.join(S.OUT_DIR, 'surr2d_online.npz')
BUDGETS = (9, 12, 16, 20, 25, 36)
N_SEEDS = 6
N_BOOT = 2000
RNG_BOOT = np.random.default_rng(12345)

COORDS = S.norm_xy(S.LX_all, S.LY_all)           # (99,2) normalized
RE = S.RE_all; IM = S.IM_all                      # full-wave complex (queried only at anchors)


def online_resaware_anchors(N):
    """Online curvature-seeking acquisition using only acquired-anchor data."""
    corners = [S._flat(0, 0), S._flat(0, S.NX - 1),
               S._flat(S.NY - 1, 0), S._flat(S.NY - 1, S.NX - 1)]
    chosen = list(dict.fromkeys(corners))[:N]

    def idw(cs, power):
        # interpolate complex G from acquired anchors cs to all candidates
        d2 = ((COORDS[:, None, :] - COORDS[None, cs, :]) ** 2).sum(-1)   # (99,k)
        d2 = np.maximum(d2, 1e-9)
        w = 1.0 / d2 ** (power / 2.0)
        gre = (w * RE[cs][None, :]).sum(1) / w.sum(1)
        gim = (w * IM[cs][None, :]).sum(1) / w.sum(1)
        return gre + 1j * gim

    while len(chosen) < N:
        cs = np.array(chosen)
        g_smooth = idw(cs, power=2)
        g_sharp = idw(cs, power=8)
        disagree = np.abs(g_sharp - g_smooth)    # data-driven uncertainty, no oracle
        disagree[cs] = -1.0
        chosen.append(int(np.argmax(disagree)))
    return sorted(chosen)


def train_eval(idx, seed):
    model, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx],
                                 S.RE_all[idx], S.IM_all[idx], seed=seed)
    m = S.metrics(model, set(idx))
    return m['heldout'], m['resonance_heldout']


def run():
    t0 = time.time()
    print(f'device={S.DEVICE} grid={S.N_ALL} budgets={BUDGETS} seeds={N_SEEDS}', flush=True)
    strategies = ('uniform', 'random', 'oracle-resaware', 'online-resaware')
    held = {s: {N: [] for N in BUDGETS} for s in strategies}
    res = {s: {N: [] for N in BUDGETS} for s in strategies}

    for N in BUDGETS:
        uni_idx = S.uniform_anchors(N)
        ora_idx = S.resonance_aware_anchors(N)      # oracle (full-grid curvature)
        onl_idx = online_resaware_anchors(N)        # online (acquired-only)
        for s in range(N_SEEDS):
            h, r = train_eval(uni_idx, s); held['uniform'][N].append(h); res['uniform'][N].append(r)
            h, r = train_eval(ora_idx, s); held['oracle-resaware'][N].append(h); res['oracle-resaware'][N].append(r)
            h, r = train_eval(onl_idx, s); held['online-resaware'][N].append(h); res['online-resaware'][N].append(r)
            rnd_idx = S.random_anchors(N, seed=1000 + s)
            h, r = train_eval(rnd_idx, s); held['random'][N].append(h); res['random'][N].append(r)
        print(f'  N={N:>3} done ({time.time()-t0:5.1f}s)  '
              f'online_res={np.mean(res["online-resaware"][N]):.2f} '
              f'oracle_res={np.mean(res["oracle-resaware"][N]):.2f} '
              f'uni_res={np.mean(res["uniform"][N]):.2f}', flush=True)

    Narr = np.array(BUDGETS, float)
    agg = {}
    for s in strategies:
        agg[s] = dict(
            res_mean=np.array([np.mean(res[s][N]) for N in BUDGETS]),
            res_std=np.array([np.std(res[s][N]) for N in BUDGETS]),
            held_mean=np.array([np.mean(held[s][N]) for N in BUDGETS]),
            held_std=np.array([np.std(held[s][N]) for N in BUDGETS]),
        )

    def boot_k(per):
        mat = np.array([per[N] for N in BUDGETS])
        ks = []
        for _ in range(N_BOOT):
            sel = RNG_BOOT.integers(0, mat.shape[1], size=mat.shape[1])
            k, _, _ = S.fit_powerlaw(Narr, mat[:, sel].mean(1))
            if np.isfinite(k):
                ks.append(k)
        return np.percentile(ks, [2.5, 97.5])

    print('\n--- power-law k (resonance region, bootstrap 95% CI) ---')
    fits = {}
    for s in strategies:
        k, _, r2 = S.fit_powerlaw(Narr, agg[s]['res_mean'])
        ci = boot_k(res[s])
        fits[s] = dict(k=k, ci=ci, r2=r2)
        print(f'  {s:>16}: k={k:5.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] r2={r2:.2f}', flush=True)

    print('\n--- resonance-region error (deg), mean+-std ---')
    print(f'{"N":>4} ' + ' '.join(f'{s:>18}' for s in strategies))
    for j, N in enumerate(BUDGETS):
        print(f'{N:>4} ' + ' '.join(f'{agg[s]["res_mean"][j]:7.2f}+-{agg[s]["res_std"][j]:5.2f}   ' for s in strategies))

    save = dict(N=Narr, n_seeds=N_SEEDS, n_boot=N_BOOT)
    km = {'uniform': 'uni', 'random': 'rnd', 'oracle-resaware': 'ora', 'online-resaware': 'onl'}
    for s in strategies:
        p = km[s]
        save[f'{p}_res_mean'] = agg[s]['res_mean']; save[f'{p}_res_std'] = agg[s]['res_std']
        save[f'{p}_held_mean'] = agg[s]['held_mean']; save[f'{p}_held_std'] = agg[s]['held_std']
        save[f'{p}_k'] = fits[s]['k']; save[f'{p}_k_CI'] = fits[s]['ci']; save[f'{p}_r2'] = fits[s]['r2']
        save[f'{p}_res_raw'] = np.array([res[s][N] for N in BUDGETS])
    np.savez(OUT, **save)
    print(f'\nsaved -> {OUT}  ({time.time()-t0:.1f}s)', flush=True)


if __name__ == '__main__':
    run()
