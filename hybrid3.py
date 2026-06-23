# -*- coding: utf-8 -*-
# 2-D (Lx,Ly) version of the matched-anchor PINN/regression comparison (port of hybrid2.py).
# Three modes, same eval (resonance-region held-out circular phase error):
#   free   = data-free geometry-conditioned total-field PINN (Maxwell residual only, no data)
#   hybrid = Maxwell residual + scarce (Lx,Ly) data (differentiable two-plane Gamma)
#   reg    = pure data-driven regression on the same scarce anchors
# h=1mm fixed (z-structure unchanged). Output: ablation3.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys, time, copy, numpy as np, torch
import torch.nn as nn
DIR = r"D:\实践三号“延安”\论文"
N_ANCH = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(SEED); np.random.seed(SEED)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE={dev}  N_ANCH={N_ANCH} iters={N_ITERS} seed={SEED}", flush=True)

C0 = 299792458.0; f0 = 24e9; epsR = 4.4; tand = 0.02; h = 1.0; p = 8.0
z1, z2 = 5.0, 6.5; z_src = 9.0; z_top = 13.0
k0 = 2*np.pi*f0/C0*1e-3; LMIN, LMAX = 2.0, 6.0; WDATA = 30.0

# 2-D full-wave reference (99 pts)
ref = np.load(os.path.join(DIR, 'ref2d.npz'))
Lx_ax = ref['Lx'].astype(np.float64); Ly_ax = ref['Ly'].astype(np.float64)
ph = ref['phase'].astype(np.float64); mg = ref['mag'].astype(np.float64)   # (11,9)
NY, NX = ph.shape
IY, IX = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij'); IY = IY.ravel(); IX = IX.ravel()
LXf = Lx_ax[IX]; LYf = Ly_ax[IY]; PHf = ph[IY, IX]; MGf = mg[IY, IX]
Gf = MGf*np.exp(1j*np.deg2rad(PHf)); REf = Gf.real; IMf = Gf.imag
NALL = PHf.size; RES_LY = 2.8

# scarce resonance-aware anchors (curvature criterion)
import surrogate2d as S
anch = np.array(S.resonance_aware_anchors(N_ANCH))
is_anchor = np.zeros(NALL, bool); is_anchor[anch] = True
print(f"anchors N={len(anch)}  (resonance-aware)", flush=True)

def wrap(x): return (x+180.0) % 360.0 - 180.0
def evalG(G_all):
    err = np.abs(wrap(np.rad2deg(np.angle(G_all)) - PHf))
    held = ~is_anchor
    res_held = held & (LYf <= RES_LY)
    return float(err[held].mean()), float(err[res_held].mean()) if res_held.any() else float('nan')

# two-plane Gamma constants
e_m1 = np.exp(-1j*k0*z1); e_p1 = np.exp(1j*k0*z1); e_m2 = np.exp(-1j*k0*z2); e_p2 = np.exp(1j*k0*z2)
det = e_m1*e_p2 - e_p1*e_m2

def run_mode(MODE):
    torch.manual_seed(SEED); np.random.seed(SEED)
    La = torch.tensor(np.stack([LXf[anch], LYf[anch]], 1), dtype=torch.float32, device=dev)
    Ga_re = torch.tensor(REf[anch], dtype=torch.float32, device=dev).view(-1, 1)
    Ga_im = torch.tensor(IMf[anch], dtype=torch.float32, device=dev).view(-1, 1)

    if MODE == 'reg':
        class Reg(nn.Module):
            def __init__(s, hid=64, nl=4):
                super().__init__(); lay = [nn.Linear(2, hid), nn.Tanh()]
                for _ in range(nl-2): lay += [nn.Linear(hid, hid), nn.Tanh()]
                lay += [nn.Linear(hid, 2)]; s.net = nn.Sequential(*lay)
            def forward(s, Lxy): return s.net((Lxy-4.0)/2.0)
        reg = Reg().to(dev); opt = torch.optim.Adam(reg.parameters(), lr=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ITERS, eta_min=1e-5)
        tgt = torch.cat([Ga_re, Ga_im], 1)
        for it in range(1, N_ITERS+1):
            opt.zero_grad(); loss = ((reg(La)-tgt)**2).mean(); loss.backward(); opt.step(); sch.step()
        reg.eval()
        with torch.no_grad():
            out = reg(torch.tensor(np.stack([LXf, LYf], 1), dtype=torch.float32, device=dev)).cpu().numpy()
        return out[:, 0] + 1j*out[:, 1]

    # geometry-conditioned total-field PINN, inputs (x,y,z,Lx,Ly)
    class Net(nn.Module):
        def __init__(s, nfeat=64, hid=128, nl=4):
            super().__init__()
            sxy = np.exp(np.random.uniform(np.log(0.3), np.log(3.0), nfeat))
            sz = np.exp(np.random.uniform(np.log(0.45), np.log(1.1), nfeat))
            sL = np.exp(np.random.uniform(np.log(0.2), np.log(1.5), nfeat))
            B = np.stack([np.random.randn(nfeat)*sxy, np.random.randn(nfeat)*sxy,
                          np.random.randn(nfeat)*sz, np.random.randn(nfeat)*sL,
                          np.random.randn(nfeat)*sL])
            s.register_buffer('B', torch.tensor(B, dtype=torch.float32))
            lay = [nn.Linear(2*nfeat, hid), nn.Tanh()]
            for _ in range(nl-2): lay += [nn.Linear(hid, hid), nn.Tanh()]
            lay += [nn.Linear(hid, 2)]; s.net = nn.Sequential(*lay)
        def forward(s, x, y, z, lx, ly):
            xin = torch.cat([x, y, z, lx, ly], 1) @ s.B
            f = torch.cat([torch.sin(2*np.pi*xin), torch.cos(2*np.pi*xin)], 1)
            o = s.net(f); return o[:, 0:1], o[:, 1:2]
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ITERS, eta_min=1e-5); best = 1e18; best_state = None

    def kk(z):
        a = torch.where(z > h, torch.full_like(z, k0**2), torch.full_like(z, k0**2*epsR))
        b = torch.where(z > h, torch.zeros_like(z), torch.full_like(z, -k0**2*epsR*tand)); return a, b
    def srcf(z): return torch.exp(-((z-z_src)**2)/(2*0.3**2))
    def lap(u, x, y, z):
        ux, uy, uz = torch.autograd.grad(u, [x, y, z], torch.ones_like(u), create_graph=True)
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
        uzz = torch.autograd.grad(uz, z, torch.ones_like(uz), create_graph=True)[0]
        return uxx+uyy+uzz
    def randL(n): return (torch.rand(n, 1, device=dev)*(LMAX-LMIN)+LMIN)
    def col(n, zlo, zhi):
        x = ((torch.rand(n, 1, device=dev)*2-1)*p/2).requires_grad_(True)
        y = ((torch.rand(n, 1, device=dev)*2-1)*p/2).requires_grad_(True)
        z = (torch.rand(n, 1, device=dev)*(zhi-zlo)+zlo).requires_grad_(True)
        return x, y, z, randL(n), randL(n)
    gx, gy = np.meshgrid(np.linspace(-p/2, p/2, 14), np.linspace(-p/2, p/2, 14))
    GX = torch.tensor(gx.reshape(-1, 1), dtype=torch.float32, device=dev)
    GY = torch.tensor(gy.reshape(-1, 1), dtype=torch.float32, device=dev)
    def gamma_of(lx, ly):
        n = GX.shape[0]; LXc = torch.full((n, 1), float(lx), device=dev); LYc = torch.full((n, 1), float(ly), device=dev)
        u1, v1 = net(GX, GY, torch.full((n, 1), z1, device=dev), LXc, LYc); E1 = torch.complex(u1.mean(), v1.mean())
        u2, v2 = net(GX, GY, torch.full((n, 1), z2, device=dev), LXc, LYc); E2 = torch.complex(u2.mean(), v2.mean())
        A = (E1*e_p2-E2*e_p1)/det; B = (E2*e_m1-E1*e_m2)/det
        return A/B
    def data_loss():
        lr = torch.zeros((), device=dev)
        for i in range(La.shape[0]):
            G = gamma_of(float(La[i, 0]), float(La[i, 1]))
            lr = lr + (G.real-Ga_re[i, 0])**2 + (G.imag-Ga_im[i, 0])**2
        return lr/La.shape[0]

    t0 = time.time()
    for it in range(1, N_ITERS+1):
        opt.zero_grad(); loss = torch.zeros((), device=dev)
        x, y, z, lx, ly = col(2200, 0, z_top)
        xu, yu, zu, lxu, lyu = col(900, 0.0, h)
        X = torch.cat([x, xu]); Y = torch.cat([y, yu]); Z = torch.cat([z, zu]); LX = torch.cat([lx, lxu]); LY = torch.cat([ly, lyu])
        u, v = net(X, Y, Z, LX, LY); a, b = kk(Z); Ssrc = srcf(Z)
        ru = lap(u, X, Y, Z)+a*u-b*v-Ssrc; rv = lap(v, X, Y, Z)+a*v+b*u
        l_pde = (ru**2+rv**2).mean()
        # rectangular-patch PEC mask at z~h
        npat=1600; xp = ((torch.rand(npat, 1, device=dev)*2-1)*p/2); yp = ((torch.rand(npat, 1, device=dev)*2-1)*p/2)
        zp = h+(torch.rand(npat, 1, device=dev)*2-1)*0.12; lxp = randL(npat); lyp = randL(npat)
        up, vp = net(xp, yp, zp, lxp, lyp)
        m = torch.sigmoid(40.0*(lxp/2-torch.abs(xp)))*torch.sigmoid(40.0*(lyp/2-torch.abs(yp)))
        l_patch = (m*(up**2+vp**2)).mean()
        xg = ((torch.rand(500, 1, device=dev)*2-1)*p/2); yg = ((torch.rand(500, 1, device=dev)*2-1)*p/2)
        ug, vg = net(xg, yg, torch.zeros(500, 1, device=dev), randL(500), randL(500)); l_gnd = (ug**2+vg**2).mean()
        def wall(c):
            xw = ((torch.rand(250, 1, device=dev)*2-1)*p/2).requires_grad_(True)
            yw = ((torch.rand(250, 1, device=dev)*2-1)*p/2).requires_grad_(True)
            zw = (torch.rand(250, 1, device=dev)*z_top).requires_grad_(True)
            if c == 'x': xw = torch.full_like(xw, p/2).requires_grad_(True)
            else: yw = torch.full_like(yw, p/2).requires_grad_(True)
            uw, vw = net(xw, yw, zw, randL(250), randL(250)); q = xw if c == 'x' else yw
            du = torch.autograd.grad(uw, q, torch.ones_like(uw), create_graph=True)[0]
            dv = torch.autograd.grad(vw, q, torch.ones_like(vw), create_graph=True)[0]
            return (du**2+dv**2).mean()
        l_wall = wall('x')+wall('y')
        xt = ((torch.rand(400, 1, device=dev)*2-1)*p/2); yt = ((torch.rand(400, 1, device=dev)*2-1)*p/2)
        zt = torch.full((400, 1), z_top, device=dev, requires_grad=True)
        ut, vt = net(xt, yt, zt, randL(400), randL(400))
        uz = torch.autograd.grad(ut, zt, torch.ones_like(ut), create_graph=True)[0]
        vz = torch.autograd.grad(vt, zt, torch.ones_like(vt), create_graph=True)[0]
        l_top = ((uz-k0*vt)**2+(vz+k0*ut)**2).mean()
        loss = loss + l_pde+50*l_patch+30*l_gnd+5*l_top+l_wall
        if MODE == 'hybrid':
            ld = data_loss(); loss = loss + WDATA*ld
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if it > N_ITERS//3 and loss.item() < best: best = loss.item(); best_state = copy.deepcopy(net.state_dict())
        if it % max(1, N_ITERS//8) == 0 or it == 1:
            msg = f"  [{MODE}] it {it:5d} | loss {loss.item():.2e} | pde {l_pde.item():.2e}"
            if MODE == 'hybrid': msg += f" | data {ld.item():.2e}"
            msg += f" | {(time.time()-t0)/it*1000:.0f}ms/it"; print(msg, flush=True)
    if best_state: net.load_state_dict(best_state)
    net.eval()
    return np.array([complex(gamma_of(float(LXf[i]), float(LYf[i])).detach()) for i in range(NALL)])

results = {}
for MODE in ('free', 'hybrid', 'reg'):
    print(f"\n===== MODE {MODE} =====", flush=True)
    G = run_mode(MODE); ov, rs = evalG(G); results[MODE] = (ov, rs)
    print(f"  {MODE}: held-out overall {ov:.2f} deg | resonance-region {rs:.2f} deg", flush=True)

print("\n===== 2-PARAMETER (Lx,Ly) ABLATION (resonance-region held-out phase err) =====", flush=True)
for MODE in ('free', 'hybrid', 'reg'):
    print(f"  {MODE:7s}: {results[MODE][1]:.2f} deg  (overall {results[MODE][0]:.2f})", flush=True)
print(f"\n  physics-helps? hybrid {results['hybrid'][1]:.2f} vs pure-data reg {results['reg'][1]:.2f} -> "
      + ("physics HURTS (reg better)" if results['reg'][1] < results['hybrid'][1] else "physics helps"), flush=True)
np.savez(os.path.join(DIR, 'ablation3.npz'),
         N_anch=N_ANCH, free_res=results['free'][1], hybrid_res=results['hybrid'][1], reg_res=results['reg'][1],
         free_ov=results['free'][0], hybrid_ov=results['hybrid'][0], reg_ov=results['reg'][0])
print("saved ablation3.npz", flush=True)
