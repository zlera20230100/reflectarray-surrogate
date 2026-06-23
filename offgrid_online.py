# -*- coding: utf-8 -*-
# Off-grid online sampling: a committee sampler chooses continuous (Lx,Ly) over a dense 41x41 candidate
# grid and openEMS is called at each chosen point. The surrogate is trained on the off-grid anchors and its
# resonance-region phase error is evaluated on the held-out 99-point reference.
import os, sys, time, numpy as np
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
import h5py
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
import surrogate2d as S          # Surrogate2D, train_surrogate, predict, norm_xy, wrap180; loads ref2d
def P(*a, **k): print(*a, flush=True)

f0, fc = 24e9, 12e9; p, h = 8.0, 1.0; epsR, tand = 4.4, 0.02
z1, z2 = 5.0, 6.5; z_src, z_top = 9.0, 13.0; unit = 1e-3
k0 = 2 * np.pi * f0 / C0 * unit
PAPER = r"D:\实践三号“延安”\论文"

def meanEy(sim, tag):
    with h5py.File(os.path.join(sim, tag + '.h5'), 'r') as f:
        Er = np.array(f['/FieldData/FD/f0_real']); Ei = np.array(f['/FieldData/FD/f0_imag'])
    return (Er[1] + 1j * Ei[1]).mean()
def dedupe(vals, tol=1e-3):
    v = np.sort(np.asarray(vals, float)); out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol: out.append(x)
    return out
def run_cell(sim, Lx, Ly):           # openEMS reflection Gamma for a continuous (Lx,Ly) patch
    FDTD = openEMS(EndCriteria=1e-5, NrTS=60000); FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PMC', 'PMC', 'PEC', 'PEC', 'PEC', 'PML_8'])
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX); g = CSX.GetGrid(); g.SetDeltaUnit(unit)
    fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2 * np.pi * f0 * EPS0 * epsR * tand)
    fr4.AddBox([-p / 2, -p / 2, 0], [p / 2, p / 2, h], priority=1)
    CSX.AddMetal('gnd').AddBox([-p / 2, -p / 2, 0], [p / 2, p / 2, 0], priority=10)
    CSX.AddMetal('patch').AddBox([-Lx / 2, -Ly / 2, h], [Lx / 2, Ly / 2, h], priority=10)
    exc = CSX.AddExcitation('src', 0, [0, 1, 0]); exc.AddBox([-p / 2, -p / 2, z_src], [p / 2, p / 2, z_src], priority=5)
    for zz, tag in [(z1, 'E1'), (z2, 'E2')]:
        d = CSX.AddDump(tag, dump_type=10, dump_mode=2, file_type=1, frequency=[f0]); d.AddBox([-p / 2, -p / 2, zz], [p / 2, p / 2, zz])
    g.AddLine('x', dedupe([-p / 2, -Lx / 2, Lx / 2, p / 2])); g.AddLine('y', dedupe([-p / 2, -Ly / 2, Ly / 2, p / 2]))
    g.AddLine('z', [0, h, z1, z2, z_src, z_top])
    g.SmoothMeshLines('x', 0.35); g.SmoothMeshLines('y', 0.35); g.SmoothMeshLines('z', 0.35)
    FDTD.Run(sim, cleanup=True, verbose=0)
    E1 = meanEy(sim, 'E1'); E2 = meanEy(sim, 'E2')
    M = np.array([[np.exp(-1j * k0 * z1), np.exp(1j * k0 * z1)], [np.exp(-1j * k0 * z2), np.exp(1j * k0 * z2)]])
    A, B = np.linalg.solve(M, np.array([E1, E2]))
    return A / B

# continuous candidate pool (41x41)
NC = 41; cx = np.linspace(2.0, 6.0, NC); cy = np.linspace(2.0, 6.0, NC)
CX, CY = np.meshgrid(cx, cy, indexing='ij'); CAND = np.stack([CX.ravel(), CY.ravel()], 1)   # (1681,2)
CC = S.norm_xy(CAND[:, 0], CAND[:, 1])

def idw(anchors_xy, anchors_G, query_xy, power):
    d2 = np.maximum(((query_xy[:, None, :] - anchors_xy[None, :, :]) ** 2).sum(-1), 1e-9)
    w = d2 ** (-power / 2.0); return (w * anchors_G[None, :]).sum(1) / w.sum(1)

N_TOTAL = 20
ax = [(2.0, 2.0), (2.0, 6.0), (6.0, 2.0), (6.0, 6.0)]      # corners seed
P(f"off-grid online: corners + committee disagreement on continuous 41x41 pool; target N={N_TOTAL}", flush=True)
G_list = []; t0 = time.time(); sidx = 0
for (lx, ly) in ax:
    G_list.append(run_cell(rf"D:\openEMS_pkg\sim_offgrid_{sidx}", lx, ly)); sidx += 1
P(f"  seeded 4 corners ({time.time()-t0:.0f}s)", flush=True)
while len(ax) < N_TOTAL:
    axy = S.norm_xy(np.array([a[0] for a in ax]), np.array([a[1] for a in ax]))
    Gv = np.array(G_list)
    g2 = idw(axy, Gv, CC, 2.0); g8 = idw(axy, Gv, CC, 8.0)
    disagree = np.abs(g2 - g8)
    # avoid re-picking near existing anchors
    dmin = np.sqrt(((CC[:, None, :] - axy[None, :, :]) ** 2).sum(-1)).min(1)
    disagree[dmin < 0.05] = -1.0
    j = int(np.argmax(disagree)); lx, ly = float(CAND[j, 0]), float(CAND[j, 1])
    ax.append((lx, ly)); G_list.append(run_cell(rf"D:\openEMS_pkg\sim_offgrid_{sidx}", lx, ly)); sidx += 1
    P(f"  N={len(ax):>2}  picked ({lx:.2f},{ly:.2f})  ({time.time()-t0:.0f}s)", flush=True)

# train surrogate on off-grid anchors; evaluate on held-out 99-pt reference (resonance region)
LXa = np.array([a[0] for a in ax]); LYa = np.array([a[1] for a in ax])
Gv = np.array(G_list); REa, IMa = Gv.real, Gv.imag
res = {}
for n in (9, 12, 16, 20):
    if n > len(ax): break
    m, _ = S.train_surrogate(LXa[:n], LYa[:n], REa[:n], IMa[:n], seed=0)
    ph_pred, _ = S.predict(m, S.LX_all, S.LY_all)
    err = np.abs(S.wrap180(ph_pred - S.PH_all)); res_mask = S.LY_all <= S.RES_LY
    res[n] = float(err[res_mask].mean())
    P(f"  N={n:>2} off-grid anchors -> resonance-region phase error {res[n]:.2f} deg (held-out 99-pt ref)", flush=True)
np.savez(os.path.join(PAPER, "offgrid_online.npz"),
         anchors=np.array(ax), G=Gv, N=np.array(list(res.keys())), res_err=np.array(list(res.values())))
P(f"\nDONE off-grid online ({time.time()-t0:.0f}s); final resonance error {list(res.values())[-1]:.2f} deg", flush=True)
