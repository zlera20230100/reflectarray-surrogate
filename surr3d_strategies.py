"""3-D anchor strategies and a head-to-head comparison.

Balanced resonance-aware sampler: a space-filling farthest-point base in
normalized (Lx,Ly,h), then curvature-weighted greedy refinement on the circular
phase residual (wrap-safe), with a coverage penalty.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import surrogate3d as S


def _circ_curvature():
    """Wrap-safe second-difference magnitude of phase on the 3-D grid (deg)."""
    ph = np.deg2rad(S.phase_grid)                      # (4,9,9)
    cph, sph = np.cos(ph), np.sin(ph)

    def circ_2nd(c, s, axis):
        # angle between neighbours via complex; 2nd diff of unwrapped-locally
        d = np.zeros_like(c)
        sl = [slice(None)] * 3
        # forward/backward circular differences -> local curvature
        lo = c.take(range(0, c.shape[axis] - 2), axis) \
            + 1j * s.take(range(0, s.shape[axis] - 2), axis)
        mid = c.take(range(1, c.shape[axis] - 1), axis) \
            + 1j * s.take(range(1, s.shape[axis] - 1), axis)
        hi = c.take(range(2, c.shape[axis]), axis) \
            + 1j * s.take(range(2, s.shape[axis]), axis)
        # local angle steps
        a1 = np.angle(mid * np.conj(lo))
        a2 = np.angle(hi * np.conj(mid))
        curv = np.abs(a2 - a1)
        idx = [slice(None)] * 3
        idx[axis] = slice(1, c.shape[axis] - 1)
        out = np.zeros_like(c)
        out[tuple(idx)] = curv
        return out

    curv = circ_2nd(cph, sph, 1) + circ_2nd(cph, sph, 2) + circ_2nd(cph, sph, 0)
    curv = np.rad2deg(curv)
    # replicate edges
    curv[:, 0, :] = curv[:, 1, :]; curv[:, -1, :] = curv[:, -2, :]
    curv[:, :, 0] = curv[:, :, 1]; curv[:, :, -1] = curv[:, :, -2]
    curv[0, :, :] = curv[1, :, :]; curv[-1, :, :] = curv[-2, :, :]
    return curv


def resonance_aware_balanced(N, base_frac=0.5):
    """Balanced resonance-aware sampler.

    base = ceil(base_frac*N) space-filling farthest-point anchors from the 8 box
    corners. remaining = curvature-weighted farthest-point: pick the candidate
    maximising curvature_weight * (min distance to chosen); the distance term
    keeps the high-curvature picks spread out.
    """
    coords = S.norm_xyh(S.LX_all, S.LY_all, S.H_all)     # (324,3)
    curv = _circ_curvature().ravel()
    w = 1.0 + 3.0 * curv / (curv.max() + 1e-12)          # curvature emphasis

    n_base = max(8, int(np.ceil(base_frac * N)))
    n_base = min(n_base, N)

    # ---- space-filling base via farthest-point, seeded by 8 corners ----
    corners = [(ih * S.NY + iy) * S.NX + ix
               for ih in (0, S.NH - 1) for iy in (0, S.NY - 1)
               for ix in (0, S.NX - 1)]
    chosen = list(dict.fromkeys(corners))[:n_base]
    # precompute min-dist to chosen
    if chosen:
        dmin = np.min(((coords[:, None, :] - coords[None, chosen, :]) ** 2)
                      .sum(-1), axis=1)
    else:
        dmin = np.full(S.N_ALL, np.inf)
    while len(chosen) < n_base:
        cand = int(np.argmax(dmin))
        chosen.append(cand)
        d = ((coords - coords[cand]) ** 2).sum(-1)
        dmin = np.minimum(dmin, d)
        dmin[cand] = -1.0

    # ---- curvature-weighted refinement ----
    chosen_set = set(chosen)
    dmin = np.full(S.N_ALL, np.inf)
    cs = np.array(chosen)
    dmin = np.min(((coords[:, None, :] - coords[None, cs, :]) ** 2).sum(-1),
                  axis=1)
    for c in chosen:
        dmin[c] = -1.0
    while len(chosen) < N:
        score = w * np.sqrt(np.maximum(dmin, 0))
        score[list(chosen_set)] = -1.0
        cand = int(np.argmax(score))
        chosen.append(cand); chosen_set.add(cand)
        d = ((coords - coords[cand]) ** 2).sum(-1)
        dmin = np.minimum(dmin, d)
        dmin[cand] = -1.0
    return sorted(chosen)


if __name__ == '__main__':
    import time
    BUD = (20, 30, 45, 60, 90, 120)
    print('head-to-head: uniform vs random vs naive-rea vs balanced-rea '
          '(5 seeds, iters=8000)')
    print(f'{"N":>4} | {"uni_held":>8} {"rnd_held":>8} {"naive_h":>8} '
          f'{"bal_held":>8} | {"uni_res":>8} {"rnd_res":>8} {"naive_r":>8} '
          f'{"bal_res":>8}')
    t0 = time.time()
    rows = []
    for N in BUD:
        iu = S.uniform_anchors(N)
        ina = S.resonance_aware_anchors(N)
        ib = resonance_aware_balanced(N)
        def ev(idx, nseed=5):
            hs = []; rs = []
            for sd in range(nseed):
                m, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx],
                                         S.H_all[idx], S.RE_all[idx],
                                         S.IM_all[idx], iters=8000, seed=sd)
                mm = S.metrics(m, set(idx))
                hs.append(mm['heldout']); rs.append(mm['resonance_heldout'])
            return np.mean(hs), np.mean(rs)
        uh, ur = ev(iu)
        # random over 5 seed draws
        rh = []; rr = []
        for sd in range(5):
            ir = S.random_anchors(N, 1000 + sd)
            m, _ = S.train_surrogate(S.LX_all[ir], S.LY_all[ir], S.H_all[ir],
                                     S.RE_all[ir], S.IM_all[ir], iters=8000,
                                     seed=sd)
            mm = S.metrics(m, set(ir))
            rh.append(mm['heldout']); rr.append(mm['resonance_heldout'])
        rh, rr = np.mean(rh), np.mean(rr)
        nh, nr = ev(ina)
        bh, br = ev(ib)
        print(f'{N:>4} | {uh:>8.2f} {rh:>8.2f} {nh:>8.2f} {bh:>8.2f} '
              f'| {ur:>8.2f} {rr:>8.2f} {nr:>8.2f} {br:>8.2f}')
        rows.append((N, uh, rh, nh, bh, ur, rr, nr, br))
    rows = np.array(rows)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'surr3d_strategy_compare.npz'),
             rows=rows,
             cols=np.array(['N', 'uni_held', 'rnd_held', 'naive_held',
                            'bal_held', 'uni_res', 'rnd_res', 'naive_res',
                            'bal_res']))
    print(f'done ({time.time()-t0:.1f}s)')
