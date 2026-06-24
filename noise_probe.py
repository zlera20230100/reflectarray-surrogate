# -*- coding: utf-8 -*-
# Noise-robustness probe for the resonance-aware curvature acquisition (addresses a reviewer concern
# that the discrete second-difference / curvature criterion may break on NOISY oracles, where the
# curvature is dominated by observation noise).
#
# We take ONE analytic oracle (the cleanest: the driven-oscillator / series-RLC Lorentzian line shape
# from outdomain_scaling.build_ref('rlc')), add zero-mean Gaussian noise to the OBSERVED phase at
# several relative levels (std = 0,1,2,5,10% of the phase swing), and at each level:
#   (a) run the curvature acquisition whose placement uses the NOISY phase, and measure the fraction
#       of selected anchors that land in the resonance band;
#   (b) measure the resulting resonance-region phase error vs UNIFORM sampling at a fixed budget N=20.
# Averaged over several seeds. Saves noise_probe.npz and prints a table.
#
# CPU / numpy only. We reuse the EXACT analytic oracle and the EXACT curvature acquisition score from
# outdomain_scaling.py. To keep the study lightweight and to isolate the *placement criterion* (which
# is what the reviewer questions) from the heavy NN surrogate, the resonance-region error is measured
# with a lightweight, deterministic numpy reconstruction (inverse-distance interpolation of the complex
# response from the chosen anchors to the full grid) -- the same IDW machinery the curvature acquisition
# itself uses for its interpolated residual. This makes the error a clean function of WHERE anchors are
# placed, so any degradation reported here is attributable to the noisy placement, not to NN training noise.

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

# ---- grid + oracle, identical setup to outdomain_scaling.py --------------------------------------
NY, NX = 11, 9
Ly_ax = np.linspace(2.0, 6.0, NY)
Lx_ax = np.linspace(2.0, 6.0, NX)
IY, IX = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij'); IY = IY.ravel(); IX = IX.ravel()
LXf = Lx_ax[IX]; LYf = Ly_ax[IY]
NALL = LXf.size

LXMIN, LXMAX, LYMIN, LYMAX = 2.0, 6.0, 2.0, 6.0


def norm_xy(Lx, Ly):
    nx = 2.0 * (Lx - LXMIN) / (LXMAX - LXMIN) - 1.0
    ny = 2.0 * (Ly - LYMIN) / (LYMAX - LYMIN) - 1.0
    return np.stack([nx, ny], axis=-1)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


COORDS = norm_xy(LXf, LYf)


def build_ref_rlc():
    """EXACT RLC Lorentzian oracle from outdomain_scaling.build_ref('rlc')."""
    t1 = (Ly_ax - 2.0) / 4.0
    t2 = (Lx_ax - 2.0) / 4.0
    T1, T2 = np.meshgrid(t1, t2, indexing='ij')
    t1c = 0.16
    hw = 0.06 + 0.08 * T2
    delta = (T1 - t1c) / hw
    H = 1.0 / (delta + 1j)
    res = np.abs(delta) <= 4.0                                             # band used in outdomain_scaling
    core = np.abs(delta) <= 1.5                                            # sharp core (where curvature peaks)
    H = H / (np.abs(H).max() + 1e-12)
    phase = np.rad2deg(np.angle(H))
    mag = np.abs(H)
    return phase, mag, res.ravel(), core.ravel()


def _flat(iy, ix):
    return iy * NX + ix


# ---- anchor selectors (uniform identical to outdomain_scaling; curv reused verbatim) -------------
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


def curv_anchors(N, ph):
    """EXACT curvature acquisition from outdomain_scaling.curv_anchors. `ph` is the (NY,NX) phase grid
    the criterion sees -- here it will be the NOISY phase."""
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


# ---- lightweight numpy reconstruction (IDW on complex response from anchors -> full grid) --------
def idw_reconstruct(idx, REf, IMf, power=2.0):
    """Inverse-distance-weighted interpolation of the complex response from anchor set -> full grid.
    Returns predicted phase (deg) on every grid point. Anchors reproduce their own (noisy) value."""
    idx = np.asarray(idx)
    d2 = ((COORDS[:, None, :] - COORDS[None, idx, :]) ** 2).sum(-1)        # (NALL, Na)
    G = REf[idx] + 1j * IMf[idx]
    pred = np.empty(NALL, dtype=complex)
    zero = d2 <= 1e-12                                                     # grid point coincides with an anchor
    has_zero = zero.any(axis=1)
    wts = 1.0 / np.maximum(d2, 1e-12) ** (power / 2.0)
    pred_interp = (wts * G[None, :]).sum(1) / wts.sum(1)
    # for exact-anchor points, take the anchor value directly (avoids 1/0)
    for gp in np.where(has_zero)[0]:
        pred[gp] = G[np.argmax(zero[gp])]
    pred[~has_zero] = pred_interp[~has_zero]
    return np.rad2deg(np.angle(pred))


# ---- main probe ---------------------------------------------------------------------------------
def run():
    phase, mag, res_mask, core_mask = build_ref_rlc()
    PHf_clean = phase[IY, IX]                                              # clean truth phase, flat
    G_clean = mag[IY, IX] * np.exp(1j * np.deg2rad(PHf_clean))
    span = float(PHf_clean.max() - PHf_clean.min())                       # phase swing for noise scaling
    n_res = int(res_mask.sum()); n_core = int(core_mask.sum())
    print(f"[rlc] grid {NALL} pts; phase swing {span:.1f} deg; resonance-band pts {n_res} "
          f"({100*n_res/NALL:.0f}% of grid); sharp-core pts {n_core} ({100*n_core/NALL:.0f}%)", flush=True)

    N = 20                                                                 # fixed budget
    NOISE = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]                     # std as fraction of phase swing
    N_SEEDS = 12

    # uniform anchor set is noise-independent (does not look at the phase); fix it once.
    uni = uniform_anchors(N)

    # truth complex parts (the reconstruction always uses NOISY observed values at anchors, see below)
    REc, IMc = G_clean.real, G_clean.imag

    def res_error(idx, REobs, IMobs):
        """Resonance-region phase error: reconstruct from anchors' OBSERVED (noisy) complex values,
        compare to CLEAN truth on held-out resonance-band points."""
        ph_pred = idw_reconstruct(idx, REobs, IMobs)
        err = np.abs(wrap180(ph_pred - PHf_clean))
        held = res_mask & np.array([i not in set(idx) for i in range(NALL)])
        return float(err[held].mean()) if held.any() else float('nan')

    rows = []  # (noise, fb_mean, fb_std, fc_mean, fc_std, rea_m, rea_s, uni_m, uni_s)
    raw = {}   # per-noise raw arrays
    for nl in NOISE:
        sigma = nl * span
        frac_band, frac_core, rea_err, uni_err = [], [], [], []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 + int(nl * 1e4) * 131 + s)
            # zero-mean Gaussian phase noise on the observation (added to the phase grid)
            noise_grid = rng.normal(0.0, sigma, size=phase.shape) if sigma > 0 else np.zeros_like(phase)
            ph_noisy_grid = phase + noise_grid                            # (NY,NX) noisy phase the criterion sees
            PHf_noisy = ph_noisy_grid[IY, IX]
            # noisy OBSERVED complex response (magnitude kept clean; noise is on phase, per task)
            Gobs = mag[IY, IX] * np.exp(1j * np.deg2rad(PHf_noisy))
            REobs, IMobs = Gobs.real, Gobs.imag

            # (a) curvature acquisition placement uses the NOISY phase grid
            cur = curv_anchors(N, ph_noisy_grid)
            frac_band.append(float(res_mask[np.asarray(cur)].mean()))      # fraction of anchors in resonance band
            frac_core.append(float(core_mask[np.asarray(cur)].mean()))     # fraction in the sharp core

            # (b) resonance-region error: both strategies reconstruct from their anchors' NOISY observations
            rea_err.append(res_error(cur, REobs, IMobs))
            uni_err.append(res_error(uni, REobs, IMobs))

        rows.append((nl, np.mean(frac_band), np.std(frac_band), np.mean(frac_core), np.std(frac_core),
                     np.mean(rea_err), np.std(rea_err), np.mean(uni_err), np.std(uni_err)))
        raw[f'{nl:.2f}_frac'] = np.array(frac_band)
        raw[f'{nl:.2f}_core'] = np.array(frac_core)
        raw[f'{nl:.2f}_rea'] = np.array(rea_err)
        raw[f'{nl:.2f}_uni'] = np.array(uni_err)

    # baselines: fraction a RANDOM/uniform anchor would hit by chance
    band_prior = n_res / NALL; core_prior = n_core / NALL
    uni_frac = res_mask[np.asarray(uni)].mean(); uni_core = core_mask[np.asarray(uni)].mean()

    # ---- print table ----
    print(f"\nFixed budget N={N}; {N_SEEDS} seeds.")
    print(f"  band prior (random hit) = {band_prior:.2f}; uniform-in-band = {uni_frac:.2f}")
    print(f"  core prior (random hit) = {core_prior:.2f}; uniform-in-core = {uni_core:.2f}")
    print("\n noise_std   in-band      in-core      resonance-aware err   uniform err       verdict")
    print("  (% swing)   (frac)       (frac)       (deg)                 (deg)")
    print(" " + "-" * 90)
    for (nl, fbm, fbs, fcm, fcs, rm, rs, um, us) in rows:
        beats = "REA<uni" if rm < um else "REA>=uni"
        conc = "core-conc" if fcm > core_prior + 1e-9 else "NO-core-conc"
        print(f"   {100*nl:5.0f}%    {fbm:.2f}+/-{fbs:.2f}  {fcm:.2f}+/-{fcs:.2f}  {rm:6.2f}+/-{rs:5.2f}        "
              f"{um:6.2f}+/-{us:5.2f}    {beats}/{conc}")

    # ---- save ----
    save = dict(
        noise_levels=np.array(NOISE), budget=N, n_seeds=N_SEEDS, phase_swing=span,
        band_prior=band_prior, core_prior=core_prior,
        uniform_anchors_in_band=uni_frac, uniform_anchors_in_core=uni_core,
        n_res=n_res, n_core=n_core, n_all=NALL,
        frac_in_band_mean=np.array([r[1] for r in rows]),
        frac_in_band_std=np.array([r[2] for r in rows]),
        frac_in_core_mean=np.array([r[3] for r in rows]),
        frac_in_core_std=np.array([r[4] for r in rows]),
        rea_err_mean=np.array([r[5] for r in rows]),
        rea_err_std=np.array([r[6] for r in rows]),
        uni_err_mean=np.array([r[7] for r in rows]),
        uni_err_std=np.array([r[8] for r in rows]),
    )
    save.update(raw)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noise_probe.npz')
    np.savez(out, **save)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    run()
