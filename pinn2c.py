# -*- coding: utf-8 -*-
# Unit-cell total-field PINN for the patch reflection. Multi-scale Fourier features (x,y up to ~3/mm),
# denser collocation at the patch edges and in the FR4 under the patch. Gaussian source, lr decay,
# best checkpoint, multi-plane least-squares extraction; reports Gamma vs L.
import os, sys, time, copy, numpy as np, torch
import torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C0=299792458.0; f0=24e9; epsR=4.4; tand=0.02; h=1.0; p=8.0
z1,z2=5.0,6.5; z_src=12.0; z_top=16.0; tp=0.25   # raised source + taller domain so higher Floquet modes decay before extraction
L=float(sys.argv[2]) if len(sys.argv)>2 else 2.5
N_ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 9000
k0=2*np.pi*f0/C0*1e-3; k1=k0*np.sqrt(epsR)
print(f"L={L}mm  multi-scale Fourier + adaptive patch sampling")

class Net(nn.Module):
    def __init__(s,nfeat=128,hid=192,nl=5):
        super().__init__()
        # multi-scale: x,y freqs span 0.3..3.0 /mm; z spans 0.25..1.2 /mm
        sxy=np.exp(np.random.uniform(np.log(0.3),np.log(3.0),nfeat))   # high x,y freq for patch edges
        sz =np.exp(np.random.uniform(np.log(0.45),np.log(1.1),nfeat))  # z only physical: k0(air)..k1(FR4)
        B=np.stack([np.random.randn(nfeat)*sxy, np.random.randn(nfeat)*sxy, np.random.randn(nfeat)*sz])
        s.register_buffer('B',torch.tensor(B,dtype=torch.float32))
        lay=[nn.Linear(2*nfeat,hid),nn.Tanh()]
        for _ in range(nl-2): lay+=[nn.Linear(hid,hid),nn.Tanh()]
        lay+=[nn.Linear(hid,2)]; s.net=nn.Sequential(*lay)
    def forward(s,x,y,z):
        xin=torch.cat([x,y,z],1)@s.B
        f=torch.cat([torch.sin(2*np.pi*xin),torch.cos(2*np.pi*xin)],1)
        o=s.net(f); return o[:,0:1],o[:,1:2]
net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=2e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=N_ITERS,eta_min=1e-5); best=1e9; best_state=None

def kk(z):
    a=torch.where(z>h,torch.full_like(z,k0**2),torch.full_like(z,k0**2*epsR))
    b=torch.where(z>h,torch.zeros_like(z),torch.full_like(z,-k0**2*epsR*tand)); return a,b
def srcf(z): return torch.exp(-((z-z_src)**2)/(2*0.3**2))
def lap(u,x,y,z):
    ux,uy,uz=torch.autograd.grad(u,[x,y,z],torch.ones_like(u),create_graph=True)
    uxx=torch.autograd.grad(ux,x,torch.ones_like(ux),create_graph=True)[0]
    uyy=torch.autograd.grad(uy,y,torch.ones_like(uy),create_graph=True)[0]
    uzz=torch.autograd.grad(uz,z,torch.ones_like(uz),create_graph=True)[0]
    return uxx+uyy+uzz
def rp(n,zlo,zhi,xr=None):
    xr=p/2 if xr is None else xr
    x=(torch.rand(n,1,device=dev)*2-1)*xr; y=(torch.rand(n,1,device=dev)*2-1)*xr
    z=torch.rand(n,1,device=dev)*(zhi-zlo)+zlo
    return x.requires_grad_(True),y.requires_grad_(True),z.requires_grad_(True)
def edge(n):  # dense ring around the patch perimeter at z~h
    a=torch.rand(n,1,device=dev)*2*np.pi
    r=L/2+ (torch.rand(n,1,device=dev)*2-1)*0.4
    x=(r*torch.cos(a)); y=(r*torch.sin(a)); z=h+(torch.rand(n,1,device=dev)*2-1)*0.35
    return x.requires_grad_(True),y.requires_grad_(True),z.requires_grad_(True)

t0=time.time()
for it in range(1,N_ITERS+1):
    opt.zero_grad()
    x,y,z=rp(3500,0,z_top)                              # global
    xu,yu,zu=rp(1500,0.0,h,xr=L/2+0.3)                  # under/around patch in FR4 (cavity)
    xe,ye,ze=edge(1500)                                  # patch edges
    X=torch.cat([x,xu,xe]);Y=torch.cat([y,yu,ye]);Z=torch.cat([z,zu,ze])
    u,v=net(X,Y,Z); a,b=kk(Z); S=srcf(Z)
    ru=lap(u,X,Y,Z)+a*u-b*v-S; rv=lap(v,X,Y,Z)+a*v+b*u
    l_pde=(ru**2+rv**2).mean()
    xg,yg,_=rp(700,0,0); zg=torch.zeros_like(xg,requires_grad=True); ug,vg=net(xg,yg,zg); l_gnd=(ug**2+vg**2).mean()
    xp=(torch.rand(2000,1,device=dev)*2-1)*L/2; yp=(torch.rand(2000,1,device=dev)*2-1)*L/2
    zp=(h+torch.rand(2000,1,device=dev)*tp); xp.requires_grad_(True);yp.requires_grad_(True);zp.requires_grad_(True)
    up,vp=net(xp,yp,zp); l_patch=(up**2+vp**2).mean()
    xt,yt,_=rp(700,0,0); zt=torch.full_like(xt,z_top,requires_grad=True); ut,vt=net(xt,yt,zt)
    uz=torch.autograd.grad(ut,zt,torch.ones_like(ut),create_graph=True)[0]; vz=torch.autograd.grad(vt,zt,torch.ones_like(vt),create_graph=True)[0]
    l_top=((uz-k0*vt)**2+(vz+k0*ut)**2).mean()
    def wall(c):
        xw=(torch.rand(300,1,device=dev)*2-1)*p/2; yw=(torch.rand(300,1,device=dev)*2-1)*p/2; zw=torch.rand(300,1,device=dev)*z_top
        if c=='x': xw=torch.full_like(xw,p/2)
        else: yw=torch.full_like(yw,p/2)
        xw.requires_grad_(True);yw.requires_grad_(True);zw.requires_grad_(True); uw,vw=net(xw,yw,zw); q=xw if c=='x' else yw
        du=torch.autograd.grad(uw,q,torch.ones_like(uw),create_graph=True)[0]; dv=torch.autograd.grad(vw,q,torch.ones_like(vw),create_graph=True)[0]
        return (du**2+dv**2).mean()
    l_wall=wall('x')+wall('y')
    loss=l_pde+50*l_patch+30*l_gnd+5*l_top+l_wall
    loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
    if it>N_ITERS//3 and loss.item()<best: best=loss.item(); best_state=copy.deepcopy(net.state_dict())
    if it%max(1,N_ITERS//10)==0 or it==1:
        print(f"it {it:5d} | loss {loss.item():.2e} | pde {l_pde.item():.2e} | patch {l_patch.item():.2e} | {(time.time()-t0)/it*1000:.0f}ms/it")
if best_state: net.load_state_dict(best_state); print(f"best loss {best:.2e}")

# L-BFGS second-order polish (helps recover high-frequency/resonant modes)
print("L-BFGS polish ...")
with torch.no_grad():
    pass
def fix(t): return t.detach().requires_grad_(True)
# fixed comprehensive batch
Xg_,Yg_,Zg_=rp(6000,0,z_top); Xu_,Yu_,Zu_=rp(3000,0.0,h,xr=L/2+0.3); Xe_,Ye_,Ze_=edge(3000)
Xx_,Yx_,Zx_=rp(5000,7.0,11.0)   # dense in the extraction region to constrain the field there
XI=fix(torch.cat([Xg_,Xu_,Xe_,Xx_])); YI=fix(torch.cat([Yg_,Yu_,Ye_,Yx_])); ZI=fix(torch.cat([Zg_,Zu_,Ze_,Zx_]))
xg0,yg0,_=rp(1200,0,0); ZG=torch.zeros_like(xg0,requires_grad=True); XG=fix(xg0); YG=fix(yg0)
XPp=fix((torch.rand(3000,1,device=dev)*2-1)*L/2); YPp=fix((torch.rand(3000,1,device=dev)*2-1)*L/2); ZPp=fix(h+torch.rand(3000,1,device=dev)*tp)
xt0,yt0,_=rp(1200,0,0); ZT=torch.full_like(xt0,z_top,requires_grad=True); XT=fix(xt0); YT=fix(yt0)
def wallset(c,n=800):
    xw=(torch.rand(n,1,device=dev)*2-1)*p/2; yw=(torch.rand(n,1,device=dev)*2-1)*p/2; zw=torch.rand(n,1,device=dev)*z_top
    if c=='x': xw=torch.full_like(xw,p/2)
    else: yw=torch.full_like(yw,p/2)
    return fix(xw),fix(yw),fix(zw)
WXx,WXy,WXz=wallset('x'); WYx,WYy,WYz=wallset('y')
def total_loss():
    u,v=net(XI,YI,ZI); a,b=kk(ZI); S=srcf(ZI)
    ru=lap(u,XI,YI,ZI)+a*u-b*v-S; rv=lap(v,XI,YI,ZI)+a*v+b*u; lp=(ru**2+rv**2).mean()
    ug,vg=net(XG,YG,ZG); lg=(ug**2+vg**2).mean()
    up,vp=net(XPp,YPp,ZPp); lpa=(up**2+vp**2).mean()
    ut,vt=net(XT,YT,ZT); uz=torch.autograd.grad(ut,ZT,torch.ones_like(ut),create_graph=True)[0]; vz=torch.autograd.grad(vt,ZT,torch.ones_like(vt),create_graph=True)[0]
    lt=((uz-k0*vt)**2+(vz+k0*ut)**2).mean()
    def w(xw,yw,zw,c):
        uw,vw=net(xw,yw,zw); q=xw if c=='x' else yw
        du=torch.autograd.grad(uw,q,torch.ones_like(uw),create_graph=True)[0]; dv=torch.autograd.grad(vw,q,torch.ones_like(vw),create_graph=True)[0]
        return (du**2+dv**2).mean()
    lw=w(WXx,WXy,WXz,'x')+w(WYx,WYy,WYz,'y')
    return lp+50*lpa+30*lg+5*lt+lw
opt2=torch.optim.LBFGS(net.parameters(),max_iter=200,history_size=80,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
def closure():
    opt2.zero_grad(); l=total_loss(); l.backward(); return l
l_before=float(total_loss()); opt2.step(closure); l_after=float(total_loss())
print(f"L-BFGS: loss {l_before:.2e} -> {l_after:.2e}")
net.eval()
def meanEy(zv):
    n=4000; x=(torch.rand(n,1,device=dev)*2-1)*p/2; y=(torch.rand(n,1,device=dev)*2-1)*p/2; z=torch.full_like(x,zv)
    with torch.no_grad(): u,v=net(x,y,z)
    return (u+1j*v).cpu().numpy().mean()
zs=np.linspace(7.5,10.5,16); E=np.array([meanEy(float(z)) for z in zs])   # clean window: higher modes decayed, below raised source
Mt=np.stack([np.exp(-1j*k0*zs),np.exp(1j*k0*zs)],1); sol,*_=np.linalg.lstsq(Mt,E,rcond=None); A,B=sol; G=A/B
misfit=np.linalg.norm(E-Mt@sol)/np.linalg.norm(E)
print(f"\nPINN reflection: |Gamma|={abs(G):.3f}  phase={np.angle(G,deg=True):+.1f} deg  (misfit={misfit:.2f})")
oe={2.0:59.2,2.5:-133.5,3.0:-149.8,3.5:-152.1}
print(f"openEMS ref @L={L}: {oe.get(L,'?')} deg")
torch.save(net.state_dict(), rf"D:\实践三号“延安”\论文\pinn2c_L{L}.pth")
