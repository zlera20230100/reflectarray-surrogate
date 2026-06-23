# -*- coding: utf-8 -*-
# Phase-gradient reflectarray metasurface, full-wave openEMS. A linear reflection-phase ramp of
# super-period Lambda = M*p reflects a normally-incident plane wave into the angle
# arcsin(lambda0/Lambda) (m=-1 order). Per-cell patch sizes come from inverting the grounded-FR4
# square-patch size->phase library (ref.npz). A finite N_sup-period strip is simulated, the
# reflected field is FFT'd over x to a wavenumber spectrum, and diffraction-order angles/efficiencies
# are extracted. Output: meta_steer.npz + _fig_meta_steer.png/pdf.
# argv: 'test' = quick single-M check; default = full M-sweep.
import os, sys, glob, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
import h5py
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

def P(*a): print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))

# physical constants / cell parameters (match the openEMS unit-cell setup)
f0, fc = 24e9, 12e9
p      = 8.0
epsR, tand = 4.4, 0.02
unit = 1e-3
lam0 = C0/f0/unit          # free-space wavelength in mm  (= 12.49 mm)
k0   = 2*np.pi/lam0        # per-mm
# (substrate thickness h and the z reference planes are set AFTER the library is selected, below)

# ---- size->phase library (invert to get patch size for a target reflection phase) ----
# library selectable via env var META_LIB:  'h1' (ref, h=1mm, 211 deg)  or
# 'h1.5' (thicker cell ref_thick_h1.5, h=1.5mm, ~259 deg phase range).
LIBSEL = os.environ.get('META_LIB', 'h1')
if LIBSEL == 'h1.5':
    lib = np.load(os.path.join(HERE, "ref_thick_h1.5.npz")); H_SUB = 1.5
else:
    lib = np.load(os.path.join(HERE, "ref.npz")); H_SUB = 1.0
Llib   = lib["L"].astype(float)
ph_lib = lib["phase"].astype(float)          # deg, wrapped [-180,180]
mag_lib= lib["mag"].astype(float)
_unw   = lib["phase_unwrap"].astype(float)   # deg, unwrapped vs L
PHASE_RANGE = float(_unw.max()-_unw.min())

# Build the widest monotonic (descending-phase) design branch from the unwrapped curve, so the
# size->phase inversion returns a monotonic ramp instead of nearest-neighbour picks that scramble
# the order. Scan for the longest run over which phase_unwrap is (mostly) decreasing in L.
def _longest_monotone_branch(L, phi):
    best_i, best_j, best_span = 0, len(L)-1, -1
    n = len(L)
    for i in range(n):
        for j in range(i+2, n+1):
            seg = phi[i:j]
            d = np.diff(seg)
            if np.all(d <= 5.0):                      # allow tiny up-jitter (<=5 deg)
                span = seg[0]-seg[-1]
                if span > best_span:
                    best_span, best_i, best_j = span, i, j
    return best_i, best_j
_bi, _bj = _longest_monotone_branch(Llib, _unw)
Lbr   = Llib[_bi:_bj]
phibr = _unw[_bi:_bj]                                  # descending
magbr = mag_lib[_bi:_bj]
BRANCH_SPAN = float(phibr[0]-phibr[-1])               # usable monotone phase span
# ascending arrays for np.interp
_o = np.argsort(phibr)
phibr_s, Lbr_s, magbr_s = phibr[_o], Lbr[_o], magbr[_o]
BR_MIN, BR_MAX = float(phibr.min()), float(phibr.max())

def size_for_phase(target_deg):
    """Invert size->phase along the widest monotonic branch. The target ramp phase is mapped into the
    branch's usable window by adding the right multiple of 360 (phase is periodic), then clamped to
    the realizable [BR_MIN,BR_MAX]. Returns (L, |Gamma|, realized_phase_deg_wrapped)."""
    # bring target into the branch window using 360-periodicity
    t = float(target_deg)
    while t > BR_MAX: t -= 360.0
    while t < BR_MIN: t += 360.0
    # if it overshoots past BR_MAX after the loop (window <360), clamp to nearest edge
    if t > BR_MAX:
        t = BR_MAX if (t-BR_MAX) < (BR_MIN+360-t) else BR_MIN
    t = np.clip(t, BR_MIN, BR_MAX)
    L = float(np.interp(t, phibr_s, Lbr_s))
    m = float(np.interp(t, phibr_s, magbr_s))
    real = ((t + 180) % 360) - 180
    return L, m, real

# ---- substrate thickness and z reference planes (scale with H_SUB so probes stay in clean air) ----
h    = H_SUB
z1   = h + 4.0             # probe plane 1 (clean air above patch)
z2   = h + 5.5            # probe plane 2
z_src= h + 8.0           # plane-wave current sheet
z_top= h + 12.0          # top of air box (PML above)

def meanEy_plane(simdir, tag):
    """Return complex Ey field array on the dump plane (x,y) at f0, plus the x-grid (mm)."""
    fn = os.path.join(simdir, tag+'.h5')
    with h5py.File(fn,'r') as f:
        Er = np.array(f['/FieldData/FD/f0_real'])   # shape (3, nx, ny, nz)
        Ei = np.array(f['/FieldData/FD/f0_imag'])
        # mesh
        gx = np.array(f['/Mesh/x'])  # in m? openEMS stores in the unit set; usually meters*unit
        gy = np.array(f['/Mesh/y'])
    Ey = Er[1] + 1j*Ei[1]            # (nx, ny, nz) ; nz=1 for a plane dump
    return Ey, gx, gy

def build_supercell_sizes(M):
    """Per-cell patch sizes for a blazed phase-gradient supercell. A blazed grating that sends power
    into the single m=-1 order needs a full 360 deg reflection-phase ramp across the super-period
    (per-cell step = 360/M deg). The patch realizes only ~211 deg (h=1mm) or ~259 deg (h=1.5mm), so
    the nearest-phase inverter returns the closest realizable phase; the range deficit leaks power
    into other orders."""
    step = 360.0/M
    targets = -step*np.arange(M)                      # full 360 deg ramp, decreasing, start at 0
    Ls, mags, realized = [], [], []
    for t in targets:
        L, m, tr = size_for_phase(t)                 # nearest realizable phase over whole library
        Ls.append(L); mags.append(m); realized.append(tr)
    return np.array(Ls), np.array(mags), np.array(realized), np.array(targets)

def run_supercell(simdir, M, N_sup, mode='full'):
    """Finite strip = N_sup copies of an M-cell phase-gradient supercell along x.
       Returns dict with reflected-field x-spectrum and diffraction-order powers."""
    Ls, mags, realized, targets = build_supercell_sizes(M)
    Lambda = M*p                      # super-period (mm)
    Ncells = M*N_sup                  # total cells along x
    Wx = Ncells*p                     # strip width along x (mm)
    x0 = -Wx/2.0                      # left edge

    FDTD = openEMS(EndCriteria=1e-4, NrTS=60000 if mode!='test' else 15000)
    FDTD.SetGaussExcite(f0, fc)
    # x: MUR absorbs the lateral edges of the finite strip. y: PMC mirror (y-symmetric, replicates the
    # cell stack in y, matching the library polarization E along y). zmin: PEC ground backing, zmax: PML.
    # boundary order: [xmin,xmax,ymin,ymax,zmin,zmax]
    FDTD.SetBoundaryCond(['MUR','MUR','PMC','PMC','PEC','PML_8'])

    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
    g = CSX.GetGrid(); g.SetDeltaUnit(unit)
    fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2*np.pi*f0*EPS0*epsR*tand)
    fr4.AddBox([x0, -p/2, 0], [x0+Wx, p/2, h], priority=1)
    CSX.AddMetal('gnd').AddBox([x0, -p/2, 0], [x0+Wx, p/2, 0], priority=10)

    patch = CSX.AddMetal('patch')
    xc_list = []
    for n in range(Ncells):
        cx = x0 + (n+0.5)*p          # cell centre
        xc_list.append(cx)
        L = Ls[n % M]
        patch.AddBox([cx-L/2, -L/2, h], [cx+L/2, L/2, h], priority=10)

    # normally-incident plane wave: electric current sheet (E along y) spanning whole strip at z_src
    exc = CSX.AddExcitation('src', 0, [0,1,0])
    exc.AddBox([x0, -p/2, z_src], [x0+Wx, p/2, z_src], priority=5)

    # two full x-line probe planes (z1,z2) for the per-x traveling-wave split that isolates reflected A(x)
    for zz, tag in [(z1,'E1'), (z2,'E2')]:
        d = CSX.AddDump(tag, dump_type=10, dump_mode=2, file_type=1, frequency=[f0])
        d.AddBox([x0, 0.0, zz], [x0+Wx, 0.0, zz])      # y=0 line (structure y-uniform)

    # mesh
    g.AddLine('x', [x0, x0+Wx])
    # add cell edges & patch edges
    xs = [x0, x0+Wx]
    for n in range(Ncells):
        cx = x0 + (n+0.5)*p; L = Ls[n % M]
        xs += [cx-p/2, cx+p/2, cx-L/2, cx+L/2]
    g.AddLine('x', sorted(set(np.round(xs,4))))
    g.AddLine('y', [-p/2, p/2, 0.0])
    g.AddLine('z', [0, h, z1, z2, z_src, z_top])
    res = lam0/20.0                   # ~0.62 mm target mesh in air
    g.SmoothMeshLines('x', min(res, p/8))
    g.SmoothMeshLines('y', p/4)
    g.SmoothMeshLines('z', res)

    FDTD.Run(simdir, cleanup=True, verbose=0)

    # ---- read full x-line Ey at the two probe planes ----
    def lineEy(tag):
        fn = os.path.join(simdir, tag+'.h5')
        with h5py.File(fn,'r') as f:
            Er=np.array(f['/FieldData/FD/f0_real']); Ei=np.array(f['/FieldData/FD/f0_imag'])
            gx=np.array(f['/Mesh/x'])
        Ey=(Er[1]+1j*Ei[1])
        Ey=np.squeeze(Ey)
        if Ey.ndim>1: Ey=Ey.reshape(Ey.shape[0],-1).mean(axis=1)
        xg=np.squeeze(gx)
        xg=xg/unit if xg.max()<1.0 else xg              # h5 mesh stored in meters -> mm
        return Ey, xg
    E1, xg1 = lineEy('E1')
    E2, xg2 = lineEy('E2')
    # common x-grid
    xc = xg1 if E1.size<=E2.size else xg2
    if E2.size!=xc.size: E2=np.interp(xc,xg2,E2.real)+1j*np.interp(xc,xg2,E2.imag)
    if E1.size!=xc.size: E1=np.interp(xc,xg1,E1.real)+1j*np.interp(xc,xg1,E1.imag)

    # per-x traveling-wave split:  E(z)=A e^{-jk z}+B e^{+jk z}  (A=up=reflected, B=down=incident)
    # openEMS phase convention e^{+jwt}: down-going (toward patch, -z) ~ e^{+jk z}.
    M2 = np.array([[np.exp(-1j*k0*z1), np.exp(1j*k0*z1)],
                   [np.exp(-1j*k0*z2), np.exp(1j*k0*z2)]])
    Minv = np.linalg.inv(M2)
    Ax = Minv[0,0]*E1 + Minv[0,1]*E2      # reflected (up-going) field A(x)
    Bx = Minv[1,0]*E1 + Minv[1,1]*E2      # incident  (down-going) field B(x)
    Binc = np.median(np.abs(Bx))          # incident amplitude (should be ~uniform in x)

    # ---- spatial FFT of the reflected field A(x) -> transverse wavenumber spectrum ----
    # crop the central region to suppress MUR/edge artifacts: drop one super-period each side but
    # never more than ~30% total so a strip always survives.
    edge = min(Lambda, 0.3*(xc.max()-xc.min()))   # mm to crop each side
    central = (xc >= xc.min()+edge) & (xc <= xc.max()-edge)
    if central.sum() < 16:                 # safety: keep central 60%
        lo,hi = np.percentile(xc,[20,80]); central=(xc>=lo)&(xc<=hi)
    xcrop = xc[central]; Acrop = Ax[central]
    # uniform resample
    Nfft = 4096
    xu = np.linspace(xcrop.min(), xcrop.max(), Nfft)
    Au = np.interp(xu, xcrop, Acrop.real)+1j*np.interp(xu, xcrop, Acrop.imag)
    dx = xu[1]-xu[0]
    win = np.hanning(len(xu))
    F = np.fft.fftshift(np.fft.fft(Au*win))
    kx = np.fft.fftshift(np.fft.fftfreq(len(xu), d=dx))*2*np.pi   # per-mm
    Pk = np.abs(F)**2
    prop = np.abs(kx) <= k0                                       # propagating window
    theta = np.degrees(np.arcsin(np.clip(kx/k0, -1, 1)))

    # ---- diffraction-order powers of the reflected field at k_x,m = 2*pi*m/Lambda ----
    span_x = xu[-1]-xu[0]
    bw = (2*np.pi/span_x)*1.5             # ~1.5 FFT-bins half-width (narrow, non-overlapping)
    orders = {}
    for m in range(-5, 6):
        kxm = 2*np.pi*m/Lambda
        if abs(kxm) > k0:
            orders[m] = (np.nan, np.nan); continue
        thm = np.degrees(np.arcsin(kxm/k0))
        sel = np.abs(kx-kxm) <= bw
        orders[m] = (thm, float(Pk[sel].sum()))
    total = sum(v[1] for v in orders.values() if not np.isnan(v[1])) + 1e-30
    eff = {m: (orders[m][1]/total if not np.isnan(orders[m][1]) else np.nan) for m in orders}

    return dict(M=M, N_sup=N_sup, Lambda=Lambda, Ncells=Ncells,
                Ls=Ls, mags=mags, realized=realized, targets=targets,
                xc=xc, Ax=Ax, Bx=Bx, Binc=Binc,
                kx=kx, theta=theta, Pk=Pk, prop=prop,
                orders={m:orders[m] for m in orders}, eff=eff,
                theta_pred=float(np.degrees(np.arcsin(min(lam0/Lambda,1.0)))))

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv)>1 else 'full'
    P(f"lambda0 = {lam0:.3f} mm , k0 = {k0:.4f} /mm , p = {p} mm = {p/lam0:.3f} lambda0")
    P(f"CELL LIBRARY = '{LIBSEL}'  (h={h} mm),  full phase RANGE = {PHASE_RANGE:.0f} deg, "
      f"monotone design branch L=[{Lbr.min():.2f},{Lbr.max():.2f}] span={BRANCH_SPAN:.0f} deg, "
      f"|Gamma| {magbr.min():.2f}-{magbr.max():.2f}")
    P(f"  (a clean blazed ramp needs 360 deg; available {BRANCH_SPAN:.0f} deg -> "
      f"{'near-blazing possible' if BRANCH_SPAN>250 else 'partial blaze, expect +-1 leakage'})")

    if mode == 'test':
        Mlist = [4]; N_sup = 2
    else:
        Mlist = [3,4,5,6,8]; N_sup = 4      # 4 super-periods for clean order resolution

    results = []
    for M in Mlist:
        thp = np.degrees(np.arcsin(min(lam0/(M*p),1.0)))
        P(f"\n==== M={M}  Lambda={M*p}mm  predicted theta_-1 = {thp:.2f} deg  (N_sup={N_sup}, {M*N_sup} cells) ====")
        Ls, mags, realized, targets = build_supercell_sizes(M)
        P("  per-cell  L(mm)=" + " ".join(f"{x:.2f}" for x in Ls))
        P("  target ph =" + " ".join(f"{x:+.0f}" for x in targets))
        P("  realiz ph =" + " ".join(f"{x:+.0f}" for x in realized))
        P("  |Gamma|   =" + " ".join(f"{x:.2f}" for x in mags))
        try:
            r = run_supercell(rf"D:\openEMS_pkg\sim_meta_M{M}", M, N_sup, mode)
        except Exception as e:
            P(f"  !! M={M} FAILED: {e}")
            import traceback; traceback.print_exc()
            continue
        # report orders
        om = r['orders']; ef = r['eff']
        P("  diffraction orders (m: theta_deg, eff%):")
        for m in sorted(om):
            th, _ = om[m]
            if not np.isnan(th):
                P(f"    m={m:+d}: theta={th:+6.2f} deg   eff={100*ef[m]:5.1f}%")
        # dominant non-specular (m!=0) order
        cand = [(m, om[m][0], ef[m]) for m in om if m!=0 and not np.isnan(om[m][0])]
        if cand:
            mdom, thdom, efdom = max(cand, key=lambda t: t[2])
            # full-wave realized angle: spectral peak in a narrow window (+-1/4 order spacing) about
            # the dominant order k_x,m, which excludes the specular skirt
            kx, Pk, theta = r['kx'], r['Pk'], r['theta']
            kxm = 2*np.pi*mdom/r['Lambda']
            half = 0.25*2*np.pi/r['Lambda']
            wsel = np.abs(kx-kxm) <= half
            if wsel.any():
                idxs = np.where(wsel)[0]; ipk = idxs[np.argmax(Pk[idxs])]
                th_fw = float(theta[ipk])
            else:
                th_fw = thdom
            # blaze asymmetry: dominant order power vs its mirror (-m) order power
            ef_mirror = ef.get(-mdom, np.nan)
            ratio = efdom/ef_mirror if (ef_mirror and ef_mirror>1e-6) else np.inf
            r['eff0'] = ef.get(0, np.nan)
            P(f"  >> DOMINANT anomalous order m={mdom:+d}: GSL-pred={thdom:+.2f} deg, "
              f"full-wave-peak={th_fw:+.2f} deg, anomalous-eff={100*efdom:.1f}%, "
              f"blaze asym (m/-m)={ratio:.1f}x, specular(m0)={100*r['eff0']:.0f}%")
            r['dom_m']=mdom; r['dom_pred']=thdom; r['dom_fw']=th_fw; r['dom_eff']=efdom; r['ratio']=ratio
        results.append(r)

    # save
    if results:
        save = dict(
            f0=f0, p=p, h=h, epsR=epsR, lam0=lam0, k0=k0,
            lib=LIBSEL, phase_range=PHASE_RANGE,
            Mlist=np.array([r['M'] for r in results]),
            Lambda=np.array([r['Lambda'] for r in results]),
            theta_pred=np.array([r['theta_pred'] for r in results]),
            dom_pred=np.array([r.get('dom_pred',np.nan) for r in results]),
            dom_fw=np.array([r.get('dom_fw',np.nan) for r in results]),
            dom_eff=np.array([r.get('dom_eff',np.nan) for r in results]),
            dom_eff0=np.array([r.get('eff0',np.nan) for r in results]),
            ratio=np.array([r.get('ratio',np.nan) for r in results]),
        )
        for r in results:
            M=r['M']
            save[f'theta_M{M}']=r['theta']
            save[f'Pk_M{M}']=r['Pk']
            save[f'prop_M{M}']=r['prop']
            save[f'Ls_M{M}']=r['Ls']
            save[f'eff_M{M}']=np.array([[m, r['eff'][m]] for m in sorted(r['eff'])])
        np.savez(os.path.join(HERE, f"meta_steer_{LIBSEL}.npz"), **save)
        np.savez(os.path.join(HERE,"meta_steer.npz"), **save)   # canonical = last run
        P(f"\nsaved meta_steer_{LIBSEL}.npz and meta_steer.npz")

        # summary table
        P("\n================ STEERING SUMMARY (lib="+LIBSEL+") ================")
        P(f"{'M':>3} {'Lambda':>7} {'|GSL|':>7} {'|fullwave|':>11} {'anom-eff':>9} {'asym':>6} {'spec':>6}")
        for r in results:
            P(f"{r['M']:>3} {r['Lambda']:>6.0f}m {abs(r.get('dom_pred',np.nan)):>6.2f}d "
              f"{abs(r.get('dom_fw',np.nan)):>10.2f}d {100*r.get('dom_eff',np.nan):>7.1f}% "
              f"{r.get('ratio',np.nan):>5.1f}x {100*r.get('eff0',np.nan):>4.0f}%")
