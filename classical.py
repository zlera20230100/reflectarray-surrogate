# -*- coding: utf-8 -*-
# Classical-interpolant baseline: compares RBF interpolation and GP kriging against the MLP on the
# same anchors, held-out circular phase MAE on the 2-param/99-pt surface.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, importlib.util, sys
DIR=r"D:\实践三号“延安”\论文"
spec=importlib.util.spec_from_file_location("uc2d",os.path.join(DIR,"surrogate2d.py"))
uc=importlib.util.module_from_spec(spec); sys.modules["uc2d"]=uc; spec.loader.exec_module(uc)
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPRBF, ConstantKernel as C, WhiteKernel

def wrap180(x): return (x+180.0)%360.0-180.0
def pmae(re,im,held):
    ph=np.degrees(np.angle(re+1j*im)); return float(np.abs(wrap180(ph-uc.PH_all[held])).mean())

BUD=[9,16,25,36]
print(f"{'N':>3} {'strategy':>16} {'MLP':>7} {'RBF':>7} {'kriging':>8}  (held-out phase MAE deg)")
Ns=[]; strat=[]; mlp_a=[]; rbf_a=[]; kr_a=[]
for N in BUD:
    for sname,sfn in (('uniform',uc.uniform_anchors),('resonance-aware',uc.resonance_aware_anchors)):
        idx=sfn(N); held=np.array([i for i in range(uc.N_ALL) if i not in set(idx)])
        Xtr=uc.norm_xy(uc.LX_all[idx],uc.LY_all[idx]); Xall=uc.norm_xy(uc.LX_all,uc.LY_all)
        Ytr=np.stack([uc.RE_all[idx],uc.IM_all[idx]],1)
        model,_=uc.train_surrogate(uc.LX_all[idx],uc.LY_all[idx],uc.RE_all[idx],uc.IM_all[idx],iters=12000)
        php,_=uc.predict(model,uc.LX_all,uc.LY_all); mlp=float(np.abs(wrap180(php[held]-uc.PH_all[held])).mean())
        try:
            rbf=RBFInterpolator(Xtr,Ytr,kernel='thin_plate_spline',smoothing=0.0); yr=rbf(Xall)
            rbf_mae=pmae(yr[held,0],yr[held,1],held)
        except Exception as e: rbf_mae=float('nan')
        k=C(1.0)*GPRBF(0.5,(0.05,5))+WhiteKernel(1e-4,(1e-8,1e-1))
        gp_re=GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=2).fit(Xtr,Ytr[:,0])
        gp_im=GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=2).fit(Xtr,Ytr[:,1])
        kr_mae=pmae(gp_re.predict(Xall)[held],gp_im.predict(Xall)[held],held)
        print(f"{N:>3} {sname:>16} {mlp:>7.2f} {rbf_mae:>7.2f} {kr_mae:>8.2f}",flush=True)
        Ns.append(N); strat.append(sname); mlp_a.append(mlp); rbf_a.append(rbf_mae); kr_a.append(kr_mae)
np.savez(os.path.join(DIR,"classical.npz"), N=np.array(Ns), strategy=np.array(strat),
         mlp=np.array(mlp_a), rbf=np.array(rbf_a), kriging=np.array(kr_a))
print("saved classical.npz")
