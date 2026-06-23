# -*- coding: utf-8 -*-
# 1-D (square-patch L) PINN/regression comparison at equal data: data-free PINN (free), Maxwell
# residual + data (hybrid), and pure data-driven regression (reg) on the same anchor set. GPU.
#
# Usage: python hybrid2.py MODE N_ANCHOR N_ITERS [SEED]
# Config via env vars (all optional):
#   FORCE_RES=1     -> include nearest indices to L=2.25 and L=2.5 in the anchor set
#   WDATA=<float>   -> data weight (default 30)
#   ANNEAL=1        -> ramp physics weight 0.1->1.0 over first 60% of training
#   CAV=1           -> stronger cavity/interface coupling (more 0<z<h colloc, denser near z=h, higher PEC wt)
#   TAG=<str>       -> filename suffix tag
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys, time, copy, numpy as np, torch
import torch.nn as nn
DIR=r"D:\实践三号“延安”\论文"
MODE   = sys.argv[1] if len(sys.argv)>1 else 'hybrid'
N_ANCH = int(sys.argv[2]) if len(sys.argv)>2 else 6
N_ITERS= int(sys.argv[3]) if len(sys.argv)>3 else 8000
SEED   = int(sys.argv[4]) if len(sys.argv)>4 else 0
torch.manual_seed(SEED); np.random.seed(SEED)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE={dev} ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'})")

FORCE_RES=os.environ.get('FORCE_RES','0')=='1'
WDATA=float(os.environ.get('WDATA','30'))
ANNEAL=os.environ.get('ANNEAL','0')=='1'
CAV=os.environ.get('CAV','0')=='1'
TAG=os.environ.get('TAG','')
print(f"CFG FORCE_RES={FORCE_RES} WDATA={WDATA} ANNEAL={ANNEAL} CAV={CAV} TAG={TAG}")

# geometry / physics constants (match openEMS reference)
C0=299792458.0; f0=24e9; epsR=4.4; tand=0.02; h=1.0; p=8.0
z1,z2=5.0,6.5; z_src=9.0; z_top=13.0
k0=2*np.pi*f0/C0*1e-3                       # per-mm
LMIN,LMAX=2.0,6.0

# full-wave reference (openEMS)
ref=np.load(os.path.join(DIR,'ref.npz'))
L_ref=ref['L'].astype(np.float64)                       # 17 sizes
G_ref=ref['mag']*np.exp(1j*np.radians(ref['phase']))    # complex Gamma at each size
# anchor (training) sizes: evenly spaced subset of the 17; rest are held-out test sizes
anch_idx=np.unique(np.round(np.linspace(0,len(L_ref)-1,N_ANCH)).astype(int))
if FORCE_RES:
    # force resonance-region anchors (nearest grid points to L=2.25 and 2.5)
    extra=[int(np.argmin(np.abs(L_ref-t))) for t in (2.25,2.5)]
    anch_idx=np.unique(np.concatenate([anch_idx,extra]).astype(int))
test_idx=np.array([i for i in range(len(L_ref)) if i not in anch_idx])
La=torch.tensor(L_ref[anch_idx],dtype=torch.float32,device=dev).view(-1,1)
Ga_re=torch.tensor(G_ref[anch_idx].real,dtype=torch.float32,device=dev).view(-1,1)
Ga_im=torch.tensor(G_ref[anch_idx].imag,dtype=torch.float32,device=dev).view(-1,1)
print(f"MODE={MODE} N_ANCHOR={len(anch_idx)} iters={N_ITERS} seed={SEED}")
print(f"anchor L = {L_ref[anch_idx].round(3)}")
print(f"held-out L = {L_ref[test_idx].round(3)}")

# pure data-driven regression baseline
if MODE=='reg':
    class Reg(nn.Module):
        def __init__(s,hid=64,nl=4):
            super().__init__()
            lay=[nn.Linear(1,hid),nn.Tanh()]
            for _ in range(nl-2): lay+=[nn.Linear(hid,hid),nn.Tanh()]
            lay+=[nn.Linear(hid,2)]; s.net=nn.Sequential(*lay)
        def forward(s,L): return s.net((L-4.0)/2.0)        # normalize L
    reg=Reg().to(dev); opt=torch.optim.Adam(reg.parameters(),lr=3e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,N_ITERS,eta_min=1e-5)
    tgt=torch.cat([Ga_re,Ga_im],1)
    for it in range(1,N_ITERS+1):
        opt.zero_grad(); out=reg(La); loss=((out-tgt)**2).mean(); loss.backward(); opt.step(); sch.step()
        if it%max(1,N_ITERS//8)==0 or it==1: print(f"it {it:5d} | data {loss.item():.2e}",flush=True)
    reg.eval()
    with torch.no_grad():
        out=reg(torch.tensor(L_ref,dtype=torch.float32,device=dev).view(-1,1)).cpu().numpy()
    G_pinn=out[:,0]+1j*out[:,1]
else:
    # geometry-conditioned field PINN
    class Net(nn.Module):
        def __init__(s,nfeat=64,hid=128,nl=4):
            super().__init__()
            sxy=np.exp(np.random.uniform(np.log(0.3),np.log(3.0),nfeat))   # patch-edge x,y freqs
            sz =np.exp(np.random.uniform(np.log(0.45),np.log(1.1),nfeat))  # physical z freqs k0..k1
            sL =np.exp(np.random.uniform(np.log(0.2),np.log(1.5),nfeat))   # L-family freqs
            B=np.stack([np.random.randn(nfeat)*sxy,np.random.randn(nfeat)*sxy,
                        np.random.randn(nfeat)*sz, np.random.randn(nfeat)*sL])
            s.register_buffer('B',torch.tensor(B,dtype=torch.float32))
            lay=[nn.Linear(2*nfeat,hid),nn.Tanh()]
            for _ in range(nl-2): lay+=[nn.Linear(hid,hid),nn.Tanh()]
            lay+=[nn.Linear(hid,2)]; s.net=nn.Sequential(*lay)
        def forward(s,x,y,z,L):
            xin=torch.cat([x,y,z,L],1)@s.B
            f=torch.cat([torch.sin(2*np.pi*xin),torch.cos(2*np.pi*xin)],1)
            o=s.net(f); return o[:,0:1],o[:,1:2]
    net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=2e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,N_ITERS,eta_min=1e-5); best=1e18; best_state=None

    def kk(z):                                   # k^2(z): air above h, lossy FR4 below
        a=torch.where(z>h,torch.full_like(z,k0**2),torch.full_like(z,k0**2*epsR))
        b=torch.where(z>h,torch.zeros_like(z),torch.full_like(z,-k0**2*epsR*tand)); return a,b
    def srcf(z): return torch.exp(-((z-z_src)**2)/(2*0.3**2))
    def lap(u,x,y,z):
        ux,uy,uz=torch.autograd.grad(u,[x,y,z],torch.ones_like(u),create_graph=True)
        uxx=torch.autograd.grad(ux,x,torch.ones_like(ux),create_graph=True)[0]
        uyy=torch.autograd.grad(uy,y,torch.ones_like(uy),create_graph=True)[0]
        uzz=torch.autograd.grad(uz,z,torch.ones_like(uz),create_graph=True)[0]
        return uxx+uyy+uzz
    def col(n,zlo,zhi,xr=None):
        xr=p/2 if xr is None else xr
        x=((torch.rand(n,1,device=dev)*2-1)*xr).requires_grad_(True)
        y=((torch.rand(n,1,device=dev)*2-1)*xr).requires_grad_(True)
        z=(torch.rand(n,1,device=dev)*(zhi-zlo)+zlo).requires_grad_(True)
        L=(torch.rand(n,1,device=dev)*(LMAX-LMIN)+LMIN)
        return x,y,z,L
    # fixed xy grid for the differentiable two-plane Gamma extraction (data + eval)
    gx,gy=np.meshgrid(np.linspace(-p/2,p/2,14),np.linspace(-p/2,p/2,14))
    GX=torch.tensor(gx.reshape(-1,1),dtype=torch.float32,device=dev)
    GY=torch.tensor(gy.reshape(-1,1),dtype=torch.float32,device=dev)
    e_m1=np.exp(-1j*k0*z1); e_p1=np.exp(1j*k0*z1); e_m2=np.exp(-1j*k0*z2); e_p2=np.exp(1j*k0*z2)
    det=e_m1*e_p2-e_p1*e_m2
    def gamma_of(Lval):
        """differentiable complex reflection Gamma(L) via two-plane fit, referenced to z=0 (as openEMS)."""
        n=GX.shape[0]; Lc=torch.full((n,1),float(Lval),device=dev)
        u1,v1=net(GX,GY,torch.full((n,1),z1,device=dev),Lc); E1=torch.complex(u1.mean(),v1.mean())
        u2,v2=net(GX,GY,torch.full((n,1),z2,device=dev),Lc); E2=torch.complex(u2.mean(),v2.mean())
        A=(E1*e_p2-E2*e_p1)/det; B=(E2*e_m1-E1*e_m2)/det                      # E=A e^{-jkz}+B e^{+jkz}
        return A/B
    def data_loss():
        lr=torch.zeros((),device=dev)
        for i in range(La.shape[0]):
            G=gamma_of(float(La[i,0]))
            lr=lr+(G.real-Ga_re[i,0])**2+(G.imag-Ga_im[i,0])**2
        return lr/La.shape[0]

    # cavity-coupling knobs
    n_cav   = 1800 if CAV else 900          # collocation in 0<z<h
    n_iface = 700  if CAV else 0            # extra collocation in a thin slab around z=h
    w_patch = 120  if CAV else 50
    w_gnd   = 60   if CAV else 30

    t0=time.time()
    for it in range(1,N_ITERS+1):
        opt.zero_grad(); loss=torch.zeros((),device=dev)
        # physics-weight annealing factor (0.1 -> 1.0 over first 60%)
        if ANNEAL:
            wphys=0.1+0.9*min(1.0, it/(0.6*N_ITERS))
        else:
            wphys=1.0
        if MODE in ('free','hybrid'):
            x,y,z,L=col(2200,0,z_top)
            xu,yu,zu,Lu=col(n_cav,0.0,h,xr=p/2)              # FR4 cavity under patch
            cat_x=[x,xu];cat_y=[y,yu];cat_z=[z,zu];cat_L=[L,Lu]
            if n_iface>0:                                    # thin slab around z=h (interface)
                xi,yi,_,Li=col(n_iface,0,1); zi=(h+(torch.rand(n_iface,1,device=dev)*2-1)*0.25).requires_grad_(True)
                cat_x.append(xi);cat_y.append(yi);cat_z.append(zi);cat_L.append(Li)
            X=torch.cat(cat_x);Y=torch.cat(cat_y);Z=torch.cat(cat_z);LL=torch.cat(cat_L)
            u,v=net(X,Y,Z,LL); a,b=kk(Z); S=srcf(Z)
            ru=lap(u,X,Y,Z)+a*u-b*v-S; rv=lap(v,X,Y,Z)+a*v+b*u
            l_pde=(ru**2+rv**2).mean()
            # masked-PEC: Ey=0 on the L-sized patch top (z~h) and on the ground (z=0)
            npatch=2400 if CAV else 1600
            xp=((torch.rand(npatch,1,device=dev)*2-1)*p/2); yp=((torch.rand(npatch,1,device=dev)*2-1)*p/2)
            zp=h+(torch.rand(npatch,1,device=dev)*2-1)*0.12; Lp=(torch.rand(npatch,1,device=dev)*(LMAX-LMIN)+LMIN)
            up,vp=net(xp,yp,zp,Lp)
            m=torch.sigmoid(40.0*(Lp/2-torch.abs(xp)))*torch.sigmoid(40.0*(Lp/2-torch.abs(yp)))
            l_patch=(m*(up**2+vp**2)).mean()
            xg=((torch.rand(500,1,device=dev)*2-1)*p/2); yg=((torch.rand(500,1,device=dev)*2-1)*p/2)
            Lg=(torch.rand(500,1,device=dev)*(LMAX-LMIN)+LMIN); ug,vg=net(xg,yg,torch.zeros(500,1,device=dev),Lg)
            l_gnd=(ug**2+vg**2).mean()
            # PMC walls: dEy/dn=0 ; radiation BC at z_top
            def wall(c):
                xw=((torch.rand(250,1,device=dev)*2-1)*p/2).requires_grad_(True)
                yw=((torch.rand(250,1,device=dev)*2-1)*p/2).requires_grad_(True)
                zw=(torch.rand(250,1,device=dev)*z_top).requires_grad_(True)
                Lw=(torch.rand(250,1,device=dev)*(LMAX-LMIN)+LMIN)
                if c=='x': xw=torch.full_like(xw,p/2).requires_grad_(True)
                else: yw=torch.full_like(yw,p/2).requires_grad_(True)
                uw,vw=net(xw,yw,zw,Lw); q=xw if c=='x' else yw
                du=torch.autograd.grad(uw,q,torch.ones_like(uw),create_graph=True)[0]
                dv=torch.autograd.grad(vw,q,torch.ones_like(vw),create_graph=True)[0]
                return (du**2+dv**2).mean()
            l_wall=wall('x')+wall('y')
            xt=((torch.rand(400,1,device=dev)*2-1)*p/2); yt=((torch.rand(400,1,device=dev)*2-1)*p/2)
            Lt=(torch.rand(400,1,device=dev)*(LMAX-LMIN)+LMIN); zt=torch.full((400,1),z_top,device=dev,requires_grad=True)
            ut,vt=net(xt,yt,zt,Lt)
            uz=torch.autograd.grad(ut,zt,torch.ones_like(ut),create_graph=True)[0]
            vz=torch.autograd.grad(vt,zt,torch.ones_like(vt),create_graph=True)[0]
            l_top=((uz-k0*vt)**2+(vz+k0*ut)**2).mean()
            l_phys=l_pde+w_patch*l_patch+w_gnd*l_gnd+5*l_top+l_wall
            loss=loss+wphys*l_phys
        if MODE=='hybrid':
            ld=data_loss(); loss=loss+WDATA*ld
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        if it>N_ITERS//3 and loss.item()<best: best=loss.item(); best_state=copy.deepcopy(net.state_dict())
        if it%max(1,N_ITERS//10)==0 or it==1:
            msg=f"it {it:5d} | loss {loss.item():.2e} | wphys {wphys:.2f}"
            if MODE in ('free','hybrid'): msg+=f" | pde {l_pde.item():.2e} | patch {l_patch.item():.2e}"
            if MODE=='hybrid': msg+=f" | data {ld.item():.2e}"
            msg+=f" | {(time.time()-t0)/it*1000:.0f}ms/it"; print(msg,flush=True)
    if best_state: net.load_state_dict(best_state); print(f"best loss {best:.2e}")
    net.eval()
    G_pinn=np.array([complex(gamma_of(float(L)).detach()) for L in L_ref])

# report
ph_pinn=np.degrees(np.angle(G_pinn)); ph_ref=np.degrees(np.angle(G_ref))
print("\n  L      ref_phase   pinn_phase   |ref|   |pinn|   role")
for i,L in enumerate(L_ref):
    role='ANCHOR' if i in anch_idx else 'test'
    print(f" {L:.2f}   {ph_ref[i]:+8.1f}   {ph_pinn[i]:+8.1f}   {abs(G_ref[i]):.3f}   {abs(G_pinn[i]):.3f}   {role}")
def swing(p): u=np.unwrap(np.radians(p))*180/np.pi; return u.max()-u.min()
cdiff=np.degrees(np.abs(np.angle(np.exp(1j*np.radians(ph_pinn-ph_ref)))))   # circular |phase err|, deg
res_mask=L_ref<=2.75                                                         # resonance region
err_all=cdiff.mean(); err_test=cdiff[test_idx].mean() if len(test_idx) else float('nan')
err_res=cdiff[res_mask].mean(); magerr=np.abs(np.abs(G_pinn)-np.abs(G_ref)).mean()
# held-out restricted to resonance region (the hard interpolation)
res_test=[i for i in test_idx if L_ref[i]<=2.75]
err_res_test=cdiff[res_test].mean() if len(res_test) else float('nan')
print(f"\nphase swing: ref={swing(ph_ref):.1f}  pinn={swing(ph_pinn):.1f} deg")
print(f"mean circular |phase err|: all={err_all:.1f}  held-out={err_test:.1f}  resonance(L<=2.75)={err_res:.1f}  res-heldout={err_res_test:.1f} deg")
print(f"mean ||Gamma| err|: {magerr:.3f}")
suff=f"_{TAG}" if TAG else ""
fn=f"hybrid2_{MODE}_N{len(anch_idx)}_s{SEED}{suff}.npz"
np.savez(os.path.join(DIR,fn),
         L=L_ref,G_pinn=G_pinn,G_ref=G_ref,anch_idx=anch_idx,test_idx=test_idx,
         err_all=err_all,err_test=err_test,err_res=err_res,err_res_test=err_res_test,magerr=magerr,
         swing_pinn=swing(ph_pinn),swing_ref=swing(ph_ref))
print(f"saved {fn}")
