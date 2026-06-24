# -*- coding: utf-8 -*-
# Applies the resonance-aware curvature acquisition criterion (Eq. 3) to two non-EM two-parameter
# resonant systems, using the same pipeline (Surrogate2D + train_surrogate + curv/uniform/random anchors
# + bootstrap exponent) as the patch study:
#   (A) a driven damped oscillator / series-RLC transfer function H(p1,p2) at a fixed probe;
#   (B) a Fano resonance complex line shape t(p1,p2).
# Only the reference response changes; the criterion, surrogate, metric and statistics are unchanged, and
# the parameters live in the same [2,6] box so S.norm_xy is reused.
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, numpy as np
import surrogate2d as S      # reuse Surrogate2D, train_surrogate, norm_xy, fit_powerlaw, wrap180, predict

NY, NX = 11, 9                                   # rows = p1 (crossing dir, sharp), cols = p2 (sharpness)
Ly_ax = np.linspace(2.0, 6.0, NY)                # abstract design coord 1 (rows)
Lx_ax = np.linspace(2.0, 6.0, NX)                # abstract design coord 2 (cols)
IY, IX = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij'); IY = IY.ravel(); IX = IX.ravel()
LXf = Lx_ax[IX]; LYf = Ly_ax[IY]
COORDS = S.norm_xy(LXf, LYf)
NALL = LXf.size
# fast-diagnostic switch: `python outdomain_scaling.py diag` uses cheap settings
DIAG = (len(sys.argv) > 1 and sys.argv[1] == 'diag')
BUD = (12, 36) if DIAG else (9, 12, 16, 20, 25, 36)
N_SEEDS = int(os.environ.get('N_SEEDS', str(3 if DIAG else 6)))
ITERS = 3000 if DIAG else 20000
N_BOOT = 500 if DIAG else 2000
RNG = np.random.default_rng(12345)


def build_ref(kind):
    """Return (phase_grid(NY,NX) deg, mag_grid(NY,NX) in [0,1], res_mask(NALL,)) for a non-EM resonance.

    Both systems sit in the regime the method targets: a sharp, high-Q resonance localised in a narrow
    band near the low edge of coord 1, leaving a large smooth plateau (like the patch's phase rotation at
    small Ly). The reduced detuning is delta = (t1 - t1c)/hw with hw shrinking along coord 2 (sharper
    toward small p2), so the band sharpens across the surface as the patch resonance does when Lx narrows.
    """
    t1 = (Ly_ax - 2.0) / 4.0                      # rows  in [0,1] -> resonance-crossing coordinate
    t2 = (Lx_ax - 2.0) / 4.0                      # cols  in [0,1] -> sharpness (high-Q at small p2)
    T1, T2 = np.meshgrid(t1, t2, indexing='ij')   # (NY,NX)
    t1c = 0.16                                     # resonance near the low edge, leaving a plateau above
    if kind == 'rlc':
        # Driven damped oscillator / series-RLC response (Lorentzian): H = 1/(delta + j), |H| peaks at
        # resonance, phase swings 180 deg through it. delta = reduced detuning (drive vs natural freq).
        hw = 0.06 + 0.08 * T2                      # high-Q (hw small) at small p2; broader at large p2
        delta = (T1 - t1c) / hw
        H = 1.0 / (delta + 1j)
        res = np.abs(delta) <= 4.0                 # resonance band: within +-4 reduced linewidths
    elif kind == 'fano':
        # Fano complex line shape t = (delta + q)/(delta + j), asymmetry q (optics / condensed matter).
        hw = 0.025 + 0.060 * T2
        q = 1.5
        delta = (T1 - t1c) / hw
        H = (delta + q) / (delta + 1j)
        res = np.abs(delta) <= 4.0
    else:
        raise ValueError(kind)
    H = H / (np.abs(H).max() + 1e-12)             # normalise |H| to [0,1] so targets are O(1) like |Gamma|
    phase = np.rad2deg(np.angle(H))
    mag = np.abs(H)
    return phase, mag, res.ravel()


def _flat(iy, ix): return iy * NX + ix


def uniform_anchors(N):
    ny = max(2, min(NY, int(round(np.sqrt(N * NY / NX))))); nx = max(2, min(NX, int(np.ceil(N / ny))))
    iy = np.unique(np.round(np.linspace(0, NY - 1, ny)).astype(int)); ix = np.unique(np.round(np.linspace(0, NX - 1, nx)).astype(int))
    idx = sorted({_flat(a, b) for a in iy for b in ix}); pool = [i for i in range(NALL) if i not in idx]
    while len(idx) > N: idx.pop()
    k = 0
    while len(idx) < N and k < len(pool):
        if pool[k] not in idx: idx.append(pool[k])
        k += 1
    return sorted(idx[:N])


def random_anchors(N, seed): return sorted(np.random.default_rng(seed).choice(NALL, N, replace=False).tolist())


def curv_anchors(N, ph):
    # same curvature acquisition used for the patch/cross/dual cells
    uw = np.rad2deg(np.unwrap(np.deg2rad(ph), axis=0)); c = np.zeros_like(uw)
    c[1:-1, :] += np.abs(uw[:-2, :] - 2 * uw[1:-1, :] + uw[2:, :]); c[:, 1:-1] += np.abs(uw[:, :-2] - 2 * uw[:, 1:-1] + uw[:, 2:])
    c[0, :] = c[1, :]; c[-1, :] = c[-2, :]; c[:, 0] = c[:, 1]; c[:, -1] = c[:, -2]
    w = 1.0 + c / (c.max() + 1e-12); wf = w.ravel(); field = uw.ravel()
    corners = [_flat(0, 0), _flat(0, NX - 1), _flat(NY - 1, 0), _flat(NY - 1, NX - 1)]; chosen = list(dict.fromkeys(corners))[:N]
    while len(chosen) < N:
        cs = np.array(chosen); d2 = np.maximum(((COORDS[:, None] - COORDS[None, cs]) ** 2).sum(-1), 1e-9)
        wts = 1.0 / d2; interp = (wts * field[cs][None]).sum(1) / wts.sum(1); sc = wf * np.abs(field - interp); sc[cs] = -1.0
        chosen.append(int(np.argmax(sc)))
    return sorted(chosen)


def run(kind):
    phase, mag, res_mask = build_ref(kind)
    PHf = phase[IY, IX]; G = mag[IY, IX] * np.exp(1j * np.deg2rad(PHf)); REf, IMf = G.real, G.imag
    span = float(PHf.max() - PHf.min())
    print(f"\n[{kind}] grid {NALL} pts; phase span {span:.0f} deg; resonance pts {int(res_mask.sum())}", flush=True)

    def evalm(model, idx):
        ph_pred, _ = S.predict(model, LXf, LYf)
        err = np.abs(S.wrap180(ph_pred - PHf)); held = res_mask & np.array([i not in set(idx) for i in range(NALL)])
        return float(err[held].mean()) if held.any() else float('nan')

    def train_eval(idx, seed):
        m, _ = S.train_surrogate(LXf[idx], LYf[idx], REf[idx], IMf[idx], seed=seed, iters=ITERS); return evalm(m, idx)

    t0 = time.time(); strategies = ('uniform', 'random', 'resonance-aware')
    resd = {s: {N: [] for N in BUD} for s in strategies}
    for N in BUD:
        uni = uniform_anchors(N); cur = curv_anchors(N, phase)
        for s in range(N_SEEDS):
            resd['uniform'][N].append(train_eval(uni, s)); resd['resonance-aware'][N].append(train_eval(cur, s))
            resd['random'][N].append(train_eval(random_anchors(N, 1000 + s), s))
        print(f"  N={N:>3}  rea {np.mean(resd['resonance-aware'][N]):.2f}  uni {np.mean(resd['uniform'][N]):.2f}  "
              f"rnd {np.mean(resd['random'][N]):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    Narr = np.array(BUD, float)

    def boot(per):
        mat = np.array([per[N] for N in BUD]); ks = []
        for _ in range(N_BOOT):
            sel = RNG.integers(0, mat.shape[1], mat.shape[1]); k, _, _ = S.fit_powerlaw(Narr, mat[:, sel].mean(1))
            if np.isfinite(k): ks.append(k)
        return np.percentile(ks, [2.5, 97.5])

    save = dict(N=Narr, n_seeds=N_SEEDS, phase_span=span); km = {'uniform': 'uni', 'random': 'rnd', 'resonance-aware': 'rea'}
    print(f"  --- {kind}: resonance-region err (N=36) and k [95% CI] ---")
    for s in strategies:
        rm = np.array([np.mean(resd[s][N]) for N in BUD]); rs = np.array([np.std(resd[s][N]) for N in BUD])
        k, _, r2 = S.fit_powerlaw(Narr, rm); ci = boot(resd[s]); p = km[s]
        save[f'{p}_res_mean'] = rm; save[f'{p}_res_std'] = rs; save[f'{p}_k'] = k; save[f'{p}_k_CI'] = ci
        save[f'{p}_res_raw'] = np.array([resd[s][N] for N in BUD])
        print(f"    {s:>15}: N=36 {rm[-1]:.2f}  k={k:.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
    np.savez(os.path.join(S.OUT_DIR, f'outdomain_{kind}.npz'), **save)
    print(f"  saved outdomain_{kind}.npz", flush=True)


if __name__ == '__main__':
    print(f"device = {S.DEVICE}")
    run('rlc')
    run('fano')
    print("\nDONE.")
