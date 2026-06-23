# -*- coding: utf-8 -*-
# openEMS unit-cell reflection of a rectangular patch reflectarray cell, 3D geometry sweep over
# Lx (transverse width), Ly (resonant length, E along y) and h (FR4 thickness). Same setup as
# openems2d.py: waveguide simulator (PMC at +-x, PEC at +-y, normal incidence), grounded FR4,
# 24 GHz, two-plane traveling-wave fit Gamma=A/B at z1=5, z2=6.5 mm. The patch sits on top of the
# substrate at z=h; the extraction planes stay in air above it for all h in the swept range.
# Grid Lx=linspace(2,6,9) x Ly=linspace(2,6,9) x h in {0.6,0.9,1.2,1.5} = 324 sims.
# Saves ref3d.npz after each h-slice so partial results survive.
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, numpy as np
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
import h5py
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
def P(*a): print(*a, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'ref3d.npz')

f0, fc = 24e9, 12e9
p = 8.0; epsR, tand = 4.4, 0.02
z1, z2 = 5.0, 6.5            # two clean reference planes, in air above patch for all h<=1.5
z_src, z_top = 9.0, 13.0; unit = 1e-3
k0 = 2*np.pi*f0/C0 * unit    # per-mm

def meanEy(sim, tag):
    fn = os.path.join(sim, tag+'.h5')
    with h5py.File(fn, 'r') as f:
        Er = np.array(f['/FieldData/FD/f0_real']); Ei = np.array(f['/FieldData/FD/f0_imag'])
    return (Er[1]+1j*Ei[1]).mean()

def dedupe(vals, tol=1e-3):
    # remove near-coincident mesh lines (sliver-cell / tiny-timestep guard)
    v = np.sort(np.asarray(vals, dtype=float))
    out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol:
            out.append(x)
    return out

def run(sim, Lx, Ly, h):
    FDTD = openEMS(EndCriteria=1e-5, NrTS=60000); FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PMC','PMC','PEC','PEC','PEC','PML_8'])   # zmin=PEC (ground), zmax=PML
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX); g = CSX.GetGrid(); g.SetDeltaUnit(unit)
    fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2*np.pi*f0*EPS0*epsR*tand)
    fr4.AddBox([-p/2,-p/2,0], [p/2,p/2,h], priority=1)
    CSX.AddMetal('gnd').AddBox([-p/2,-p/2,0], [p/2,p/2,0], priority=10)
    CSX.AddMetal('patch').AddBox([-Lx/2,-Ly/2,h], [Lx/2,Ly/2,h], priority=10)   # patch on top at z=h
    exc = CSX.AddExcitation('src', 0, [0,1,0]); exc.AddBox([-p/2,-p/2,z_src], [p/2,p/2,z_src], priority=5)
    for zz, tag in [(z1,'E1'), (z2,'E2')]:
        d = CSX.AddDump(tag, dump_type=10, dump_mode=2, file_type=1, frequency=[f0]); d.AddBox([-p/2,-p/2,zz], [p/2,p/2,zz])
    # mesh: explicit lines at patch edges + substrate top, smooth, then dedupe to avoid sliver cells
    g.AddLine('x', dedupe([-p/2,-Lx/2,Lx/2,p/2]))
    g.AddLine('y', dedupe([-p/2,-Ly/2,Ly/2,p/2]))
    g.AddLine('z', dedupe([0,h,z1,z2,z_src,z_top]))
    g.SmoothMeshLines('x', 0.35); g.SmoothMeshLines('y', 0.35); g.SmoothMeshLines('z', 0.35)
    FDTD.Run(sim, cleanup=True, verbose=0)
    E1 = meanEy(sim, 'E1'); E2 = meanEy(sim, 'E2')
    # solve E(z)=A e^{-jk z}+B e^{+jk z} at z1,z2; A=reflected(+z), B=incident(-z); Gamma=A/B
    M = np.array([[np.exp(-1j*k0*z1), np.exp(1j*k0*z1)], [np.exp(-1j*k0*z2), np.exp(1j*k0*z2)]])
    A, B = np.linalg.solve(M, np.array([E1, E2]))
    return A/B

mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
if mode == 'test':
    Lx_s = np.array([2.0,4.0,6.0]); Ly_s = np.array([2.0,4.0,6.0]); h_s = np.array([0.6,1.2])
elif mode == 'small':                       # fallback grid if runtime tight: 7x7x4 = 196
    Lx_s = np.linspace(2.0,6.0,7); Ly_s = np.linspace(2.0,6.0,7); h_s = np.array([0.6,0.9,1.2,1.5])
else:                                        # full grid: 9x9x4 = 324
    Lx_s = np.linspace(2.0,6.0,9); Ly_s = np.linspace(2.0,6.0,9); h_s = np.array([0.6,0.9,1.2,1.5])

nH, nY, nX = len(h_s), len(Ly_s), len(Lx_s)
phase = np.full((nH,nY,nX), np.nan)
mag   = np.full((nH,nY,nX), np.nan)
fails = []
sidx = 0
t_start = time.time()
P(f"3D sweep: Lx={nX} Ly={nY} h={nH} -> {nH*nY*nX} sims. h={list(h_s)}")

for ih, h in enumerate(h_s):
    t_h = time.time()
    P(f"\n=== h-slice {ih+1}/{nH}: h={h:.2f} mm ===")
    for iy, Ly in enumerate(Ly_s):
        row_msgs = []
        for ix, Lx in enumerate(Lx_s):
            try:
                G = run(rf"D:\openEMS_pkg\sim_uc3d_{sidx}", float(Lx), float(Ly), float(h))
                phase[ih,iy,ix] = np.angle(G, deg=True); mag[ih,iy,ix] = abs(G)
                row_msgs.append(f"Lx={Lx:.1f}|G|{abs(G):.2f}ph{np.angle(G,deg=True):+.0f}")
            except Exception as ex:
                fails.append((float(Lx), float(Ly), float(h), str(ex)))
                P(f"  !! FAIL Lx={Lx:.2f} Ly={Ly:.2f} h={h:.2f}: {ex}")
            sidx += 1
        P(f"  Ly={Ly:.2f}: " + " | ".join(row_msgs))
    # incremental save after each h-slice
    np.savez(OUT, Lx=Lx_s, Ly=Ly_s, h=h_s, phase=phase, mag=mag,
             f0=f0, p=p, epsR=epsR, z1=z1, z2=z2)
    P(f"  [saved ref3d.npz through h-slice {ih+1}/{nH}; slice {time.time()-t_h:.0f}s, total {(time.time()-t_start)/60:.1f}min]")

# summary stats
P(f"\n==== DONE. total {(time.time()-t_start)/60:.1f} min, {sidx} sims, {len(fails)} fails ====")
fin = phase[~np.isnan(phase)]
if fin.size:
    P(f"phase range: {fin.min():+.1f} .. {fin.max():+.1f} deg (span {fin.max()-fin.min():.1f})")
mfin = mag[~np.isnan(mag)]
if mfin.size:
    P(f"|Gamma| range: {mfin.min():.3f} .. {mfin.max():.3f}")
# locate resonance (min |Gamma|, i.e. strongest absorption / sharpest phase swing) per h-slice
for ih, h in enumerate(h_s):
    sl = mag[ih]
    if np.all(np.isnan(sl)):
        continue
    j = np.nanargmin(sl); iy, ix = np.unravel_index(j, sl.shape)
    P(f"  h={h:.2f}: min|G|={sl[iy,ix]:.3f} at Lx={Lx_s[ix]:.2f} Ly={Ly_s[iy]:.2f} (phase {phase[ih,iy,ix]:+.0f} deg)")
if fails:
    P("fails: " + ", ".join(f"(Lx={a:.2f},Ly={b:.2f},h={c:.2f})" for a,b,c,_ in fails))
else:
    P("no failed sims")
P("\nsaved " + OUT)
