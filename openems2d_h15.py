# -*- coding: utf-8 -*-
# openEMS unit-cell reflection of a rectangular patch reflectarray cell, 2D geometry sweep over
# Lx (transverse width) and Ly (resonant length, E along y), on a thicker h=1.5 mm substrate.
# Waveguide simulator (PMC at +-x, PEC at +-y, normal incidence), grounded FR4, 24 GHz, two-plane
# traveling-wave fit Gamma=A/B at z1=5, z2=6.5 mm.
# Sweeps Ly in linspace(2,6,11) x Lx in linspace(2,6,9) = 99 sims and saves ref2d_h15.npz.
import os, sys, numpy as np
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
import h5py
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
def P(*a): print(*a, flush=True)

f0, fc = 24e9, 12e9
p, h = 8.0, 1.5; epsR, tand = 4.4, 0.02   # thicker substrate (h was 1.0)
z1, z2 = 5.0, 6.5            # two clean reference planes (>0.3 lambda0 above patch at z=1)
z_src, z_top = 9.0, 13.0; unit = 1e-3
k0 = 2*np.pi*f0/C0 * unit    # per-mm

def meanEy(sim, tag):
    fn = os.path.join(sim, tag+'.h5')
    with h5py.File(fn,'r') as f:
        Er=np.array(f['/FieldData/FD/f0_real']); Ei=np.array(f['/FieldData/FD/f0_imag'])
    return (Er[1]+1j*Ei[1]).mean()

def dedupe(vals, tol=1e-3):
    # remove near-coincident mesh lines (sliver-cell / tiny-timestep guard)
    v = np.sort(np.asarray(vals, dtype=float))
    out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol:
            out.append(x)
    return out

def run(sim, Lx, Ly):
    FDTD=openEMS(EndCriteria=1e-5, NrTS=60000); FDTD.SetGaussExcite(f0,fc)
    FDTD.SetBoundaryCond(['PMC','PMC','PEC','PEC','PEC','PML_8'])   # zmin=PEC (ground plane), zmax=PML
    CSX=ContinuousStructure(); FDTD.SetCSX(CSX); g=CSX.GetGrid(); g.SetDeltaUnit(unit)
    fr4=CSX.AddMaterial('FR4',epsilon=epsR,kappa=2*np.pi*f0*EPS0*epsR*tand)
    fr4.AddBox([-p/2,-p/2,0],[p/2,p/2,h],priority=1)
    CSX.AddMetal('gnd').AddBox([-p/2,-p/2,0],[p/2,p/2,0],priority=10)
    CSX.AddMetal('patch').AddBox([-Lx/2,-Ly/2,h],[Lx/2,Ly/2,h],priority=10)   # rectangular patch
    exc=CSX.AddExcitation('src',0,[0,1,0]); exc.AddBox([-p/2,-p/2,z_src],[p/2,p/2,z_src],priority=5)
    for zz,tag in [(z1,'E1'),(z2,'E2')]:
        d=CSX.AddDump(tag,dump_type=10,dump_mode=2,file_type=1,frequency=[f0]); d.AddBox([-p/2,-p/2,zz],[p/2,p/2,zz])
    # mesh: explicit lines at patch edges, then smooth, then dedupe to avoid sliver cells
    g.AddLine('x', dedupe([-p/2,-Lx/2,Lx/2,p/2]))
    g.AddLine('y', dedupe([-p/2,-Ly/2,Ly/2,p/2]))
    g.AddLine('z',[0,h,z1,z2,z_src,z_top])
    g.SmoothMeshLines('x',0.35); g.SmoothMeshLines('y',0.35); g.SmoothMeshLines('z',0.35)
    FDTD.Run(sim,cleanup=True,verbose=0)
    E1=meanEy(sim,'E1'); E2=meanEy(sim,'E2')
    # solve E(z)=A e^{-jk z}+B e^{+jk z}  at z1,z2
    M=np.array([[np.exp(-1j*k0*z1),np.exp(1j*k0*z1)],[np.exp(-1j*k0*z2),np.exp(1j*k0*z2)]])
    A,B=np.linalg.solve(M,np.array([E1,E2]))   # A=reflected (toward +z), B=incident (toward -z)
    G=A/B
    return G, abs(E1), abs(E2)

mode = sys.argv[1] if len(sys.argv)>1 else 'full'
if mode=='test':
    Ly_s = np.array([2.0,4.0,6.0]); Lx_s = np.array([2.0,4.0,6.0])
else:
    Ly_s = np.linspace(2.0,6.0,11); Lx_s = np.linspace(2.0,6.0,9)

nY, nX = len(Ly_s), len(Lx_s)
phase = np.full((nY,nX), np.nan); mag = np.full((nY,nX), np.nan)
fails = []
sidx = 0
for iy, Ly in enumerate(Ly_s):
    row_msgs = []
    for ix, Lx in enumerate(Lx_s):
        try:
            G,e1,e2 = run(rf"D:\openEMS_pkg\sim_uc2dB_{sidx}", float(Lx), float(Ly))
            phase[iy,ix] = np.angle(G,deg=True); mag[iy,ix] = abs(G)
            row_msgs.append(f"Lx={Lx:.2f} |G|={abs(G):.3f} ph={np.angle(G,deg=True):+.0f}")
        except Exception as ex:
            fails.append((float(Lx),float(Ly),str(ex)))
            P(f"  !! FAIL Lx={Lx:.2f} Ly={Ly:.2f}: {ex}")
        sidx += 1
    P(f"row Ly={Ly:.2f}mm: " + " | ".join(row_msgs))

# unwrap phase along Ly (rows) for reference (column-wise, axis=0)
phase_unwrap = np.full_like(phase, np.nan)
for ix in range(nX):
    col = phase[:,ix]
    good = ~np.isnan(col)
    if good.sum() >= 2:
        uw = np.full(nY, np.nan)
        uw[good] = np.unwrap(np.radians(col[good]))*180/np.pi
        phase_unwrap[:,ix] = uw

np.savez(r"D:\实践三号“延安”\论文\ref2d_h15.npz",
         Lx=Lx_s, Ly=Ly_s, phase=phase, mag=mag, phase_unwrap=phase_unwrap,
         f0=f0, p=p, h=h, epsR=epsR, z1=z1, z2=z2)
P("\nsaved ref2d.npz")

# summary stats
fin = phase[~np.isnan(phase)]
if fin.size:
    P(f"phase surface range: {fin.min():+.1f} .. {fin.max():+.1f} deg (span {fin.max()-fin.min():.1f})")
mfin = mag[~np.isnan(mag)]
if mfin.size:
    P(f"|Gamma| range: {mfin.min():.3f} .. {mfin.max():.3f}")
if fails:
    P(f"{len(fails)} failed sims: " + ", ".join(f"(Lx={a:.2f},Ly={b:.2f})" for a,b,_ in fails))
else:
    P("no failed sims")
