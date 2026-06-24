# -*- coding: utf-8 -*-
# Builds the Paper-3 figures (setup, scaling, eval, ablation, meta-steer, baseline2, ood) with a
# common plot style.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# plot style
mpl.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
    'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,
    'axes.linewidth':0.9,'font.size':11,
})
DIR=os.path.dirname(os.path.abspath(__file__))
NEU='#444444'; SIG='#1f5fa6'; ACC='#c0392b'; GRN='#1e7a45'
ORG='#d98a3d'
def L(f): return np.load(os.path.join(DIR,f),allow_pickle=True)
def save(fig,name):
    fig.savefig(os.path.join(DIR,name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(DIR,name+'.pdf'),bbox_inches='tight')
    plt.close(fig); print('  saved',name+'.png /',name+'.pdf')
def panel(ax,txt,x=-0.12,y=1.04):
    ax.text(x,y,txt,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')

ref=L('ref2d.npz'); bl=L('surr2d_baseline.npz'); gr=L('surr2d_grad.npz')
inv=L('surr2d_inverse.npz'); sc=L('surr2d_scaling_ms.npz')
cl=L('classical.npz'); ood=L('surr2d_inverse_ood.npz')
ms=L('meta_steer.npz'); ms1=L('meta_steer_h1.npz')

print('Regenerating figures...')

# fig_p3_setup
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.9))
ax1.set_xlim(0,10); ax1.set_ylim(2.6,8.2); ax1.axis('off')
# unit-cell cross-section (left)
ax1.add_patch(Rectangle((0.7,3.55),3.4,0.45,fc='#9a9a9a',ec='k',lw=0.9))          # ground plane
ax1.add_patch(Rectangle((0.7,4.00),3.4,1.25,fc='#cfe3b8',ec='k',lw=0.9))          # FR4
ax1.add_patch(Rectangle((1.7,5.25),1.4,0.30,fc=ACC,ec='k',lw=0.9))                # patch (center x=2.4)
ax1.text(0.95,4.62,'FR4',fontsize=8,rotation=90,va='center',ha='center',color=GRN)
ax1.text(2.40,3.30,'ground plane',ha='center',va='top',fontsize=8,color=NEU)
# reflection-coefficient double arrow well above the patch (no overlap)
ax1.annotate('',xy=(2.40,6.95),xytext=(2.40,5.75),arrowprops=dict(arrowstyle='<->',color=NEU,lw=1.1))
ax1.text(2.62,6.45,r'$\Gamma(L_x,L_y)$',fontsize=9.5,va='center',color=NEU)
# patch label via a thin leader into clear space (no overlap with FR4)
ax1.annotate(r'patch $L_x\!\times\!L_y$',xy=(3.05,5.40),xytext=(4.05,5.95),
             fontsize=8,color=ACC,ha='left',va='center',
             arrowprops=dict(arrowstyle='-',color=ACC,lw=0.7))
ax1.text(2.40,7.55,r'waveguide simulator (PMC$\perp x$, PEC$\perp y$)',
         ha='center',fontsize=7.8,style='italic',color=NEU)
# differentiable surrogate box (right)
ax1.add_patch(FancyBboxPatch((5.95,3.95),3.45,2.55,
              boxstyle='round,pad=0.02,rounding_size=0.18',fc='#eef2f7',ec=SIG,lw=1.2))
ax1.text(7.68,6.75,'differentiable surrogate',ha='center',fontsize=8.8,color=SIG)
# forward flow:  (Lx,Ly) -> MLP -> Gamma
ax1.text(6.35,5.62,r'$L_x$',fontsize=9.5,va='center',color=NEU)
ax1.text(6.35,5.08,r'$L_y$',fontsize=9.5,va='center',color=NEU)
ax1.add_patch(Rectangle((6.95,4.95),1.25,0.80,fc='white',ec=SIG,lw=1.1))
ax1.text(7.575,5.35,'MLP',ha='center',va='center',fontsize=9.5,color=SIG)
ax1.annotate('',xy=(6.92,5.35),xytext=(6.62,5.35),arrowprops=dict(arrowstyle='->',color=NEU,lw=1.0))
ax1.annotate('',xy=(8.78,5.35),xytext=(8.25,5.35),arrowprops=dict(arrowstyle='->',color=NEU,lw=1.0))
ax1.text(8.98,5.35,r'$\Gamma$',fontsize=10.5,ha='center',va='center',color=NEU)
# autodiff: arrow ABOVE, label BELOW it (clearly separated, no overlap)
ax1.annotate('',xy=(6.70,4.55),xytext=(8.55,4.55),arrowprops=dict(arrowstyle='->',color=ACC,lw=1.2))
ax1.text(7.625,4.22,r'$\partial\Gamma/\partial(L_x,L_y)$ autodiff',ha='center',va='center',fontsize=7.8,color=ACC)
# anchors arrow from cell to surrogate
ax1.annotate('',xy=(5.90,5.35),xytext=(4.55,5.35),arrowprops=dict(arrowstyle='->',color=NEU,lw=1.4))
ax1.text(5.22,5.62,'anchors',ha='center',va='center',fontsize=7.8,color=NEU)
ax1.set_title('(a) unit cell + differentiable surrogate',loc='left',fontsize=11,fontweight='bold')
Lx=ref['Lx']; Ly=ref['Ly']; ph=ref['phase']
im=ax2.pcolormesh(Lx,Ly,ph,shading='auto',cmap='viridis')
ax2.set_xlabel(r'$L_x$ (mm)'); ax2.set_ylabel(r'$L_y$ (mm)')
cb=fig.colorbar(im,ax=ax2,pad=0.02,fraction=0.046); cb.set_label(r'$\angle\Gamma$ (deg)',fontsize=9); cb.ax.tick_params(labelsize=8)
ax2.set_title(r'(b) full-wave reflection-phase surface',loc='left',fontsize=11,fontweight='bold')
ax2.annotate('resonance',xy=(4,2.25),xytext=(3.0,3.0),color='white',fontsize=9,ha='center',
             arrowprops=dict(arrowstyle='->',color='white',lw=1.0))
save(fig,'fig_p3_setup')

# fig_p3_scaling (multi-seed bands + k CI)
N=sc['N']
def mstat(a): return a.mean(1), a.std(1)
um,us=mstat(sc['uni_res_raw']); rm,rs=mstat(sc['rnd_res_raw']); am,as_=mstat(sc['rea_res_raw'])
def ktriple(p): return (float(sc[p+'_k_res']), float(sc[p+'_k_res_CI'][0]), float(sc[p+'_k_res_CI'][1]))
kr=ktriple('rea'); ku=ktriple('uni'); kk=ktriple('rnd')
fig,ax=plt.subplots(figsize=(5.4,4.3))
ax.plot(N,am,'o-',color=GRN,lw=2,ms=6,label=f'resonance-aware ($k$={kr[0]:.1f} [{kr[1]:.1f},{kr[2]:.1f}])')
ax.fill_between(N,np.maximum(am-as_,1e-3),am+as_,color=GRN,alpha=0.20,lw=0)
ax.plot(N,um,'s--',color=SIG,lw=1.8,ms=5,label=f'uniform ($k$={ku[0]:.1f} [{ku[1]:.1f},{ku[2]:.1f}])')
ax.fill_between(N,np.maximum(um-us,1e-3),um+us,color=SIG,alpha=0.15,lw=0)
ax.plot(N,rm,'^:',color=ACC,lw=1.6,ms=5,label=f'random ($k$={kk[0]:.1f} [{kk[1]:.1f},{kk[2]:.1f}])')
ax.fill_between(N,np.maximum(rm-rs,1e-3),rm+rs,color=ACC,alpha=0.12,lw=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(N); ax.set_xticklabels([int(n) for n in N])
ax.set_xlabel(r'number of full-wave anchors $N$'); ax.set_ylabel('resonance-region phase error (deg)')
ax.legend(frameon=False,fontsize=8.5); ax.grid(alpha=0.25,which='both')
ax.set_title(r'Data efficiency (20 seeds, mean$\pm$std)',fontsize=11,fontweight='bold')
save(fig,'fig_p3_scaling')

# fig_p3_eval (a baselines, b gradient, c inverse)
fig,(a,b,c)=plt.subplots(1,3,figsize=(13,3.9))
Nb=bl['N']; strat=bl['strategy']; mask=strat=='resonance-aware'
a.plot(Nb[mask],bl['mlp_phase_mae'][mask],'o-',color=GRN,lw=2,ms=6,label='our MLP')
a.plot(Nb[mask],bl['svr_phase_mae'][mask],'s--',color=SIG,lw=1.6,ms=5,label='SVR (RBF)')
a.plot(Nb[mask],bl['ann_phase_mae'][mask],'^:',color=ACC,lw=1.6,ms=5,label='ANN')
a.set_yscale('log'); a.set_xlabel(r'anchors $N$ (resonance-aware)'); a.set_ylabel('phase MAE (deg)')
a.legend(frameon=False); a.grid(alpha=0.25,which='both')
a.set_title('(a) surrogate vs ML baselines',loc='left',fontsize=11,fontweight='bold')
b.scatter(gr['gx_finitediff'],gr['gx_surrogate'],s=16,color=SIG,edgecolor='none',
          label=fr"$\partial\phi/\partial L_x$ (cos {float(gr['cos_x']):.2f})")
b.scatter(gr['gy_finitediff'],gr['gy_surrogate'],s=16,color=ACC,marker='^',edgecolor='none',
          label=fr"$\partial\phi/\partial L_y$ (cos {float(gr['cos_y']):.2f})")
allfd=np.concatenate([gr['gx_finitediff'],gr['gy_finitediff']])
lim=[allfd.min(),allfd.max()]
b.plot(lim,lim,'--',color=NEU,lw=1,alpha=0.7,label=r'$y=x$')
b.set_xlabel('full-wave FD gradient'); b.set_ylabel('autodiff gradient')
b.legend(frameon=False,fontsize=8); b.grid(alpha=0.25)
b.set_title('(b) differentiable gradient validation',loc='left',fontsize=11,fontweight='bold')
tp=inv['target_phase']; ap=inv['achieved_phase_ref']
c.plot([-180,90],[-180,90],'--',color=NEU,lw=1,alpha=0.7,label=r'$y=x$')
c.scatter(tp,ap,s=48,color=GRN,edgecolor='k',linewidth=0.4,zorder=5)
c.set_xlabel('target phase (deg)'); c.set_ylabel('full-wave achieved phase (deg)')
c.text(0.05,0.88,fr"mean err {np.mean(inv['phase_err_deg']):.1f}$^\circ$ (max {np.max(inv['phase_err_deg']):.1f}$^\circ$)",
       transform=c.transAxes,fontsize=9)
c.legend(frameon=False,fontsize=8,loc='lower right'); c.grid(alpha=0.25)
c.set_title('(c) 2D inverse design',loc='left',fontsize=11,fontweight='bold')
save(fig,'fig_p3_eval')

# fig_p3_ablation (1-param vs 2-param geometry families)
fig,ax=plt.subplots(figsize=(6.0,4.1))
groups=['data-free\nPINN','physics+data\nhybrid','data-driven\nres-aware']
v1=[91.8,35.0,1.2]      # 1-parameter (square-patch L) family
v2=[99.4,82.2,2.66]     # 2-parameter (Lx,Ly) family, 12 anchors (ablation3.npz)
x=np.arange(3); w=0.38
b1=ax.bar(x-w/2,v1,w,color=SIG,edgecolor='k',lw=0.5,label='1-param ($L$) family')
b2=ax.bar(x+w/2,v2,w,color=ACC,edgecolor='k',lw=0.5,label='2-param ($L_x,L_y$), 12 anchors')
for rects,vv in ((b1,v1),(b2,v2)):
    for r,v in zip(rects,vv): ax.text(r.get_x()+r.get_width()/2,v*1.08,fr'{v:.1f}$^\circ$',ha='center',fontsize=8)
ax.set_yscale('log'); ax.set_ylim(0.8,300)
ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=9)
ax.set_ylabel('resonance-region phase error (deg)')
ax.set_title('Physics residual is unnecessary/harmful in both 1-D and 2-D',fontsize=10.5,fontweight='bold')
ax.legend(frameon=False,fontsize=8.5,loc='upper right')
ax.grid(alpha=0.25,axis='y',which='both')
save(fig,'fig_p3_ablation')

# fig_meta_steer
Ms=list(ms['Mlist']); p=float(ms['p']); lam0=float(ms['lam0'])
cmap=plt.cm.viridis(np.linspace(0.08,0.85,len(Ms)))
fig,(axA,axB)=plt.subplots(1,2,figsize=(9.4,3.8))
OFF=1.05
for k,(c,M) in enumerate(zip(cmap,Ms)):
    th=ms[f'theta_M{M}']; Pk=ms[f'Pk_M{M}']; prop=ms[f'prop_M{M}']; Lambda=M*p
    sgn=np.sign(ms['dom_fw'][list(Ms).index(M)]) or 1.0
    sel=prop & (np.sign(th)==sgn)
    x=np.abs(th[sel]); y=Pk[sel]; o=np.argsort(x); x=x[o]; y=y[o]
    y=y/(y.max()+1e-30); base=k*OFF
    axA.fill_between(x,base,base+y,color=c,alpha=0.30,lw=0)
    axA.plot(x,base+y,color=c,lw=1.5)
    thp=np.degrees(np.arcsin(min(lam0/Lambda,1.0)))
    axA.plot([thp,thp],[base,base+1.0],color='0.25',ls=':',lw=0.9)
    ipk=np.argmax(y); axA.plot(x[ipk],base+y[ipk],marker='*',ms=9,color=c,mec='k',mew=0.4,zorder=5)
    axA.text(76,base+0.30,fr'$M$={M}'+'\n'+fr'$\Lambda$={Lambda:.0f} mm',fontsize=7.6,color=c,va='center',ha='right')
axA.set_xlim(0,80); axA.set_ylim(-0.1,len(Ms)*OFF+0.3)
axA.set_xlabel(r'reflection angle  $|\theta|$ (deg)')
axA.set_ylabel('normalised reflected power (offset per gradient)')
axA.set_yticks([]); axA.set_xticks(range(0,76,15))
axA.spines['left'].set_visible(False)
panel(axA,'(a)',x=-0.06)
axA.text(0.5,1.03,r'$\star$ measured peak    $\cdots$ generalized-Snell  $\theta=\arcsin(\lambda_0/\Lambda)$',
         transform=axA.transAxes,ha='center',va='bottom',fontsize=7.6,color='0.3')
def pts(dd): return np.abs(dd['dom_pred']),np.abs(dd['dom_fw']),dd['dom_eff']*100,dd['Mlist']
gp,fw,eff,MM=pts(ms); gp1,fw1,eff1,_=pts(ms1)
axB.plot([5,40],[5,40],color='0.55',ls='--',lw=1.0,zorder=0,label=r'ideal $\theta_{\rm fw}=\theta_{\rm GSL}$')
scp=axB.scatter(gp,fw,c=eff,s=70,marker='o',cmap='plasma',vmin=0,vmax=45,edgecolor='k',linewidth=0.5,zorder=3,label=r'259$^\circ$ cell ($h$=1.5 mm)')
axB.scatter(gp1,fw1,c=eff1,s=55,marker='^',cmap='plasma',vmin=0,vmax=45,edgecolor='k',linewidth=0.5,zorder=3,label=r'211$^\circ$ cell ($h$=1.0 mm)')
for x,y,M in zip(gp,fw,MM): axB.annotate(f'M={M}',(x,y),textcoords='offset points',xytext=(5,4),fontsize=7.5,color='0.2')
cb=fig.colorbar(scp,ax=axB,pad=0.02,fraction=0.046); cb.set_label('anomalous-order efficiency (%)',fontsize=9); cb.ax.tick_params(labelsize=8)
axB.set_xlim(8,35); axB.set_ylim(8,35)
axB.set_xlabel(r'generalized-Snell angle $|\theta_{\rm GSL}|$ (deg)')
axB.set_ylabel(r'full-wave dominant angle $|\theta_{\rm fw}|$ (deg)')
axB.legend(frameon=False,fontsize=7.6,loc='upper left')
panel(axB,'(b)',x=-0.18)
save(fig,'fig_meta_steer')

# fig_p3_baseline2 (classical interpolant comparison)
Ncl=cl['N']; scl=cl['strategy']
fig,(pa,pb)=plt.subplots(1,2,figsize=(10,4.0),sharey=True)
for ax,strat,tag in [(pa,'uniform','(a) uniform anchors'),(pb,'resonance-aware','(b) resonance-aware anchors')]:
    mcl=scl==strat
    Nc=Ncl[mcl]
    ax.plot(Nc,cl['mlp'][mcl],'o-',color=GRN,lw=2,ms=6,label='MLP (neural)')
    ax.plot(Nc,cl['rbf'][mcl],'D--',color=SIG,lw=1.7,ms=5,label='RBF interpolant')
    ax.plot(Nc,cl['kriging'][mcl],'v:',color=ACC,lw=1.7,ms=5,label='kriging')
    mbl=bl['strategy']==strat
    ax.plot(bl['N'][mbl],bl['svr_phase_mae'][mbl],'s-.',color=NEU,lw=1.3,ms=4,alpha=0.85,label='SVR (RBF)')
    ax.plot(bl['N'][mbl],bl['ann_phase_mae'][mbl],'^-',color=ORG,lw=1.3,ms=4,alpha=0.85,label='ANN')
    ax.set_yscale('log'); ax.set_xscale('log')
    ax.set_xticks([9,12,16,20,25,36]); ax.set_xticklabels([9,12,16,20,25,36])
    ax.set_xlabel(r'number of full-wave anchors $N$')
    ax.grid(alpha=0.25,which='both')
    ax.set_title(tag,loc='left',fontsize=11,fontweight='bold')
pa.set_ylabel('held-out phase MAE (deg)')
pa.legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('Neural surrogate is most accurate; RBF/kriging degrade under clustered anchors',
             fontsize=10.5,fontweight='bold',y=1.02)
save(fig,'fig_p3_baseline2')

# fig_p3_ood (on- vs off-surface inverse design)
on_e=ood['on_phase_err_deg']; off_e=ood['off_phase_err_deg']
on_hit=int(ood['on_boundary_hits']); off_hit=int(ood['off_boundary_hits'])
on_n=len(ood['on_boundary_hit']); off_n=len(ood['off_boundary_hit'])
on_mean=float(ood['on_phase_err_mean']); off_mean=float(ood['off_phase_err_mean'])
fig,ax=plt.subplots(figsize=(5.6,4.3))
bp=ax.boxplot([on_e,off_e],positions=[1,2],widths=0.5,patch_artist=True,
              showmeans=True,meanprops=dict(marker='D',mfc='white',mec='k',ms=6),
              medianprops=dict(color=NEU,lw=1.2),
              flierprops=dict(marker='o',ms=4,mfc=NEU,mec='none',alpha=0.5))
for patch,c in zip(bp['boxes'],[GRN,ACC]): patch.set_facecolor(c); patch.set_alpha(0.30); patch.set_edgecolor(c)
rng=np.random.default_rng(0)
for pos,data,c in [(1,on_e,GRN),(2,off_e,ACC)]:
    jx=pos+rng.uniform(-0.12,0.12,len(data))
    ax.scatter(jx,data,s=26,color=c,edgecolor='k',linewidth=0.3,zorder=5,alpha=0.85)
ax.set_yscale('log')
ax.set_xticks([1,2]); ax.set_xticklabels(['on-surface\n(achievable targets)','off-surface\n(unreachable targets)'],fontsize=9.5)
ax.set_ylabel('inverse-design phase error (deg)')
ax.set_title('Graceful degradation off the achievable manifold',fontsize=10.5,fontweight='bold')
ax.grid(alpha=0.25,axis='y',which='both')
ax.annotate(fr'mean {on_mean:.1f}$^\circ$',xy=(1,on_mean),xytext=(1.18,on_mean*1.8),
            fontsize=9,color=GRN,fontweight='bold')
ax.annotate(fr'mean {off_mean:.1f}$^\circ$',xy=(2,off_mean),xytext=(2.16,off_mean*1.2),
            fontsize=9,color=ACC,fontweight='bold')
ax.text(0.56,0.045,f'boundary hits {on_hit}/{on_n}',ha='left',va='center',fontsize=8.4,color=GRN)
ax.text(1.56,0.16,f'boundary hits {off_hit}/{off_n}',ha='left',va='center',fontsize=8.4,color=ACC)
save(fig,'fig_p3_ood')

print('ALL DONE')
