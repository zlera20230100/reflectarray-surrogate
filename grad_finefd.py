# -*- coding: utf-8 -*-
# Compares d(phase)/dLy from three sources at resonance-band points:
#   (1) coarse-FD from the 0.4-mm reference grid,
#   (2) fine-FD from dedicated +-0.1-mm openEMS runs,
#   (3) surrogate autodiff.
# Reports how close the surrogate and the coarse grid each are to the fine-FD gradient.
# Output: grad_finefd.npz.
import os, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
import h5py
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
import surrogate2d as S
def P(*a): print(*a, flush=True)

# openEMS unit-cell reflection (same as openems2d.py run())
f0, fc = 24e9, 12e9
p, h = 8.0, 1.0; epsR, tand = 4.4, 0.02
z1, z2 = 5.0, 6.5; z_src, z_top = 9.0, 13.0; unit = 1e-3
k0 = 2*np.pi*f0/C0 * unit
def meanEy(sim, tag):
    with h5py.File(os.path.join(sim, tag+'.h5'), 'r') as f:
        Er = np.array(f['/FieldData/FD/f0_real']); Ei = np.array(f['/FieldData/FD/f0_imag'])
    return (Er[1]+1j*Ei[1]).mean()
def dedupe(vals, tol=1e-3):
    v = np.sort(np.asarray(vals, float)); out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol: out.append(x)
    return out
def run(sim, Lx, Ly):
    FDTD = openEMS(EndCriteria=1e-5, NrTS=60000); FDTD.SetGaussExcite(f0, fc)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX); g = CSX.GetGrid(); g.SetDeltaUnit(unit)
    FDTD.SetBoundaryCond(['PMC','PMC','PEC','PEC','PEC','PML_8'])
    fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2*np.pi*f0*EPS0*epsR*tand)
    fr4.AddBox([-p/2,-p/2,0],[p/2,p/2,h], priority=1)
    CSX.AddMetal('gnd').AddBox([-p/2,-p/2,0],[p/2,p/2,0], priority=10)
    CSX.AddMetal('patch').AddBox([-Lx/2,-Ly/2,h],[Lx/2,Ly/2,h], priority=10)
    exc = CSX.AddExcitation('src',0,[0,1,0]); exc.AddBox([-p/2,-p/2,z_src],[p/2,p/2,z_src], priority=5)
    for zz,tag in [(z1,'E1'),(z2,'E2')]:
        d = CSX.AddDump(tag,dump_type=10,dump_mode=2,file_type=1,frequency=[f0]); d.AddBox([-p/2,-p/2,zz],[p/2,p/2,zz])
    g.AddLine('x', dedupe([-p/2,-Lx/2,Lx/2,p/2]))
    g.AddLine('y', dedupe([-p/2,-Ly/2,Ly/2,p/2]))
    g.AddLine('z', [0,h,z1,z2,z_src,z_top])
    g.SmoothMeshLines('x',0.35); g.SmoothMeshLines('y',0.35); g.SmoothMeshLines('z',0.35)
    FDTD.Run(sim, cleanup=True, verbose=0)
    E1 = meanEy(sim,'E1'); E2 = meanEy(sim,'E2')
    M = np.array([[np.exp(-1j*k0*z1),np.exp(1j*k0*z1)],[np.exp(-1j*k0*z2),np.exp(1j*k0*z2)]])
    A,B = np.linalg.solve(M, np.array([E1,E2])); return A/B           # Gamma

# trained surrogate (N=25 resonance-aware, seed 0)
idx = S.resonance_aware_anchors(25)
model, _ = S.train_surrogate(S.LX_all[idx], S.LY_all[idx], S.RE_all[idx], S.IM_all[idx], seed=0)

# coarse-grid reference (0.4-mm Ly)
ref = np.load('ref2d.npz'); Lx_ax = ref['Lx']; Ly_ax = ref['Ly']
ph_grid = ref['phase']                                   # (11,9) deg
ph_uw_Ly = np.rad2deg(np.unwrap(np.deg2rad(ph_grid), axis=0))
gy_coarse_grid = np.gradient(ph_uw_Ly, Ly_ax, axis=0)    # d(phase)/dLy on the 0.4-mm grid

DELTA = 0.1
pts = [(Lx, Ly0) for Lx in (3.0, 4.0) for Ly0 in (2.4, 2.8, 3.2)]
rows = []; sidx = 0
P(f"{'Lx':>4} {'Ly0':>4} | {'coarse-FD':>9} {'fine-FD':>9} {'surrogate':>9} | "
  f"{'|coarse-fine|':>13} {'|surr-fine|':>11}")
for (Lx, Ly0) in pts:
    # fine FD: phase difference across +-DELTA via complex ratio (robust to wrap)
    Gp = run(rf"D:\openEMS_pkg\sim_ffd_{sidx}", Lx, Ly0+DELTA); sidx += 1
    Gm = run(rf"D:\openEMS_pkg\sim_ffd_{sidx}", Lx, Ly0-DELTA); sidx += 1
    dphi = np.angle(Gp/Gm, deg=True)                     # wrapped diff in (-180,180]
    g_fine = dphi/(2*DELTA)                              # deg/mm
    # coarse FD at this grid point
    iy = int(np.argmin(np.abs(Ly_ax-Ly0))); ix = int(np.argmin(np.abs(Lx_ax-Lx)))
    g_coarse = float(gy_coarse_grid[iy, ix])
    # surrogate autodiff
    _, g_surr = S.dphase_dxy(model, np.array([Lx]), np.array([Ly0])); g_surr = float(g_surr[0])
    ec = abs(g_coarse-g_fine); es = abs(g_surr-g_fine)
    P(f"{Lx:>4.1f} {Ly0:>4.1f} | {g_coarse:>9.1f} {g_fine:>9.1f} {g_surr:>9.1f} | {ec:>13.1f} {es:>11.1f}")
    rows.append((Lx, Ly0, g_coarse, g_fine, g_surr, ec, es))

rows = np.array(rows)
cohen_c = np.abs(rows[:,2]-rows[:,3]); cohen_s = np.abs(rows[:,4]-rows[:,3])
# cosine of (coarse vs fine) and (surrogate vs fine) over the sampled points
def cos(a, b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
cos_cf = cos(rows[:,2], rows[:,3]); cos_sf = cos(rows[:,4], rows[:,3])
P(f"\nmean |coarse-fine| = {cohen_c.mean():.1f} deg/mm ; mean |surrogate-fine| = {cohen_s.mean():.1f} deg/mm")
P(f"cosine(coarse,fine) = {cos_cf:.3f} ; cosine(surrogate,fine) = {cos_sf:.3f}")
P("VERDICT: " + ("surrogate is CLOSER to the fine (true) gradient than the coarse grid is "
                 "-> the 0.76 vs-coarse cosine reflects coarse-FD truncation error, not surrogate error."
                 if cohen_s.mean() < cohen_c.mean() else
                 "surrogate not closer than coarse -> the gap is the surrogate's."))
np.savez('grad_finefd.npz', pts=rows[:,:2], g_coarse=rows[:,2], g_fine=rows[:,3],
         g_surr=rows[:,4], err_coarse=rows[:,5], err_surr=rows[:,6],
         mean_err_coarse=cohen_c.mean(), mean_err_surr=cohen_s.mean(),
         cos_coarse_fine=cos_cf, cos_surr_fine=cos_sf, delta=DELTA)
P("saved grad_finefd.npz")
