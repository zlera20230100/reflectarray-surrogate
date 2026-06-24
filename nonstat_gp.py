# -*- coding: utf-8 -*-
"""
Non-stationary GP baseline (reviewer rebuttal).

A reviewer noted that the GP uncertainty-sampling baseline tested in the paper is
STATIONARY (single global RBF length-scale), so its failure against resonance-aware
sampling might be an artifact of stationarity rather than of uncertainty sampling
per se. This script adds a FAIR non-stationary GP baseline and re-runs the exact
same comparison.

Choice of non-stationary model: INPUT WARPING of the Ly axis (option (a)).
Rationale:
  * It is the most apples-to-apples change: it reuses the *identical* stationary
    sklearn GaussianProcessRegressor and the *identical* kernel/optimizer settings
    as the existing `principled_acq.gp_uncertainty_anchors` baseline. The ONLY
    difference is a monotone reparameterization of the Ly coordinate that stretches
    the resonance band (small Ly), so a single global length-scale becomes an
    effectively SHORT length-scale inside the resonance band and a long one on the
    smooth plateau -- i.e. genuine non-stationarity in physical space.
  * The warp is data-agnostic w.r.t. the held-out surface: it is a fixed
    geometric prior ("the sharp physics lives at small Ly"), not an oracle that
    peeks at the answer. This is the same prior knowledge the resonance-aware
    sampler is allowed to use, so the comparison is fair to the GP.
  * It cannot leak labels: the warp only reshapes the input coordinate the GP
    sees; acquisition is still pure GP posterior variance on acquired anchors.

Everything else (budgets, seeds, resonance-region metric, held-out evaluation,
surrogate training, corner initialisation, greedy max-variance acquisition,
kernel, alpha, n_restarts) is reused EXACTLY from surrogate2d.py /
principled_acq.py so the three curves are directly comparable.

CPU-only. Run with D:/ANACONDA/python.exe nonstat_gp.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np

import surrogate2d as S
# Force CPU regardless of environment (the project torch build is +cpu, but be safe).
import torch
S.DEVICE = torch.device('cpu')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

OUT = os.path.join(S.OUT_DIR, 'nonstat_gp.npz')
BUDGETS = (9, 12, 16, 20, 25, 36)
N_SEEDS = 6

# Geometry / fields, taken EXACTLY from surrogate2d (same flat ordering).
COORDS = S.norm_xy(S.LX_all, S.LY_all)        # (99,2) normalized [-1,1]^2
RE, IM = S.RE_all, S.IM_all
LY_all = S.LY_all                             # physical Ly (99,)
RES_LY = S.RES_LY                             # resonance region: Ly <= 2.8


# ----------------------------------------------------------------------------
# Identical kernel factory to the existing stationary GP baseline
# (principled_acq.gp_uncertainty_anchors). Reused verbatim for both GPs so the
# ONLY difference between stationary and non-stationary is the input coordinate.
# ----------------------------------------------------------------------------
def _kern():
    return C(1.0, (1e-3, 1e3)) * RBF(0.5, (0.05, 5.0)) + WhiteKernel(1e-4, (1e-8, 1e-1))


# ----------------------------------------------------------------------------
# Non-stationary warp of the Ly axis.
# Map normalized ny in [-1,1] through a monotone function that DENSIFIES the
# resonance band (small Ly, i.e. ny near -1) by stretching it. We use a smooth
# CDF-style warp concentrated at the resonance edge. The warp is applied only to
# the Ly column of the (normalized) coordinates; Lx is left untouched.
#
# ny -> w(ny): monotone increasing, w(-1)=-1, w(1)=1, with large local slope
# (=> short effective length-scale) near the resonance band.
# ----------------------------------------------------------------------------
def _warp_ny(ny):
    # Resonance band edge in normalized coords: Ly=RES_LY -> ny0
    ny0 = 2.0 * (RES_LY - S.LYMIN) / (S.LYMAX - S.LYMIN) - 1.0
    # Stretch factor inside the band: a sigmoid bump in slope centred at ny0.
    # Integrate a slope profile s(t) = 1 + A*exp(-((t-ny0)/sigma)^2) to get a
    # smooth monotone warp, then rescale to [-1,1].
    A = 4.0        # extra stretch (=> ~5x shorter effective length-scale in band)
    sigma = 0.45   # width of the stretched region (covers the resonance band)
    t = np.linspace(-1.0, 1.0, 2001)
    s = 1.0 + A * np.exp(-((t - ny0) / sigma) ** 2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(t))])
    cdf = -1.0 + 2.0 * cdf / cdf[-1]           # rescale to [-1,1]
    return np.interp(ny, t, cdf)


def warp_coords(coords):
    out = coords.copy()
    out[:, 1] = _warp_ny(coords[:, 1])
    return out


WCOORDS = warp_coords(COORDS)                  # (99,2) warped coords for nonstat GP


# ----------------------------------------------------------------------------
# Greedy GP uncertainty acquisition.
# `feat` selects which coordinate representation the GP sees:
#   COORDS  -> stationary GP   (reproduces principled_acq baseline)
#   WCOORDS -> non-stationary GP (input-warped)
# Identical to principled_acq.gp_uncertainty_anchors otherwise.
# ----------------------------------------------------------------------------
def gp_uncertainty_anchors(N, feat):
    corners = [S._flat(0, 0), S._flat(0, S.NX - 1),
               S._flat(S.NY - 1, 0), S._flat(S.NY - 1, S.NX - 1)]
    chosen = list(dict.fromkeys(corners))[:N]
    while len(chosen) < N:
        cs = np.array(chosen)
        gre = GaussianProcessRegressor(kernel=_kern(), normalize_y=True,
                                       alpha=1e-8, n_restarts_optimizer=0)
        gim = GaussianProcessRegressor(kernel=_kern(), normalize_y=True,
                                       alpha=1e-8, n_restarts_optimizer=0)
        gre.fit(feat[cs], RE[cs])
        gim.fit(feat[cs], IM[cs])
        _, sre = gre.predict(feat, return_std=True)
        _, sim = gim.predict(feat, return_std=True)
        score = np.sqrt(sre ** 2 + sim ** 2)
        score[cs] = -1.0
        chosen.append(int(np.argmax(score)))
    return sorted(chosen)


def stationary_gp_anchors(N):
    return gp_uncertainty_anchors(N, COORDS)


def nonstationary_gp_anchors(N):
    return gp_uncertainty_anchors(N, WCOORDS)


# ----------------------------------------------------------------------------
# Train surrogate on the acquired anchors and read the resonance-region held-out
# phase error -- EXACT same path as principled_acq.train_eval / surrogate2d.metrics.
# ----------------------------------------------------------------------------
def train_eval(idx, seed):
    m, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx],
                             S.RE_all[idx], S.IM_all[idx], seed=seed)
    mt = S.metrics(m, set(idx))
    return mt['heldout'], mt['resonance_heldout']


def run():
    t0 = time.time()
    # Pre-compute anchor sets (deterministic; do not depend on surrogate seed).
    anchors = {
        'stationary-GP': {N: stationary_gp_anchors(N) for N in BUDGETS},
        'nonstat-GP':    {N: nonstationary_gp_anchors(N) for N in BUDGETS},
        'resonance-aware': {N: S.resonance_aware_anchors(N) for N in BUDGETS},
    }
    strategies = ('stationary-GP', 'nonstat-GP', 'resonance-aware')

    res = {s: {N: [] for N in BUDGETS} for s in strategies}
    held = {s: {N: [] for N in BUDGETS} for s in strategies}

    for N in BUDGETS:
        for s in range(N_SEEDS):
            for name in strategies:
                idx = anchors[name][N]
                h, r = train_eval(idx, s)
                held[name][N].append(h)
                res[name][N].append(r)
        msg = (f"  N={N:>3} done ({time.time()-t0:5.1f}s)  "
               f"stat={np.mean(res['stationary-GP'][N]):.2f}  "
               f"nonstat={np.mean(res['nonstat-GP'][N]):.2f}  "
               f"resaware={np.mean(res['resonance-aware'][N]):.2f}")
        print(msg, flush=True)

    Narr = np.array(BUDGETS, float)
    save = dict(N=Narr, n_seeds=N_SEEDS, res_ly=RES_LY)
    km = {'stationary-GP': 'stat', 'nonstat-GP': 'nonstat',
          'resonance-aware': 'resaware'}
    for s in strategies:
        p = km[s]
        rm = np.array([np.mean(res[s][N]) for N in BUDGETS])
        rs = np.array([np.std(res[s][N]) for N in BUDGETS])
        hm = np.array([np.mean(held[s][N]) for N in BUDGETS])
        k, _, r2 = S.fit_powerlaw(Narr, rm)
        save[f'{p}_res_mean'] = rm
        save[f'{p}_res_std'] = rs
        save[f'{p}_held_mean'] = hm
        save[f'{p}_res_raw'] = np.array([res[s][N] for N in BUDGETS])  # (len(N), seeds)
        save[f'{p}_k'] = k
        save[f'{p}_r2'] = r2
        # record the anchor sets for inspection
        save[f'{p}_anchors_N36'] = np.array(anchors[s][36])

    np.savez(OUT, **save)

    # ---- pretty table -------------------------------------------------------
    print("\n" + "=" * 74)
    print("Resonance-region (Ly <= %.1f) held-out phase error [deg], mean +- std "
          "over %d seeds" % (RES_LY, N_SEEDS))
    print("=" * 74)
    header = f"{'N':>4} | " + " | ".join(f"{s:>22}" for s in strategies)
    print(header)
    print("-" * len(header))
    for i, N in enumerate(BUDGETS):
        cells = []
        for s in strategies:
            rm = np.mean(res[s][N]); rs = np.std(res[s][N])
            cells.append(f"{rm:7.2f} +- {rs:5.2f}".rjust(22))
        print(f"{N:>4} | " + " | ".join(cells))
    print("-" * len(header))
    # power-law slopes
    krow = []
    for s in strategies:
        rm = np.array([np.mean(res[s][N]) for N in BUDGETS])
        k, _, r2 = S.fit_powerlaw(Narr, rm)
        krow.append(f"k={k:5.2f} (r2={r2:.2f})".rjust(22))
    print(f"{'fit':>4} | " + " | ".join(krow))
    print("=" * 74)

    # ---- N=36 verdict -------------------------------------------------------
    n36 = {s: np.mean(res[s][36]) for s in strategies}
    print("\nN=36 resonance-region error:")
    for s in strategies:
        print(f"  {s:>16}: {n36[s]:.3f} deg")
    ra = n36['resonance-aware']
    best_gp = min(n36['stationary-GP'], n36['nonstat-GP'])
    best_gp_name = ('nonstat-GP' if n36['nonstat-GP'] <= n36['stationary-GP']
                    else 'stationary-GP')
    if ra < best_gp:
        print(f"\nVERDICT: resonance-aware still wins at N=36 "
              f"({ra:.2f} vs best GP {best_gp:.2f} = {best_gp_name}); "
              f"factor {best_gp/ra:.2f}x lower error.")
    else:
        print(f"\nVERDICT: the {best_gp_name} GP now BEATS/MATCHES resonance-aware "
              f"at N=36 ({best_gp:.2f} vs {ra:.2f}).")
    print(f"\nsaved {OUT} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == '__main__':
    run()
