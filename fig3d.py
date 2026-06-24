# -*- coding: utf-8 -*-
# Builds fig_p3_3d from the 3-D datasets (surr3d_scaling.npz, surr3d_strategy_compare.npz,
# surr3d_bite_finite.npz): data-efficiency curves and 2-D vs 3-D resonance-error floors.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
    'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,
    'axes.linewidth':0.9,'font.size':11,
})
DIR=os.path.dirname(os.path.abspath(__file__))
NEU='#444444'; SIG='#1f5fa6'; ACC='#c0392b'; GRN='#1e7a45'; ORG='#d98a3d'
def L(f): return np.load(os.path.join(DIR,f),allow_pickle=True)
def save(fig,name):
    fig.savefig(os.path.join(DIR,name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(DIR,name+'.pdf'),bbox_inches='tight')
    plt.close(fig); print('  saved',name)
def panel(ax,txt,x=-0.14,y=1.03):
    ax.text(x,y,txt,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')

d=L('surr3d_scaling.npz'); sc=L('surr3d_strategy_compare.npz'); bf=L('surr3d_bite_finite.npz')
N=d['N']
um,us=d['uni_res_mean'],d['uni_res_std']
rm,rs=d['rnd_res_mean'],d['rnd_res_std']
am,as_=d['rea_res_mean'],d['rea_res_std']
ku=(float(d['uni_k_res']),)+tuple(d['uni_k_res_CI'])
kk=(float(d['rnd_k_res']),)+tuple(d['rnd_k_res_CI'])
kr=(float(d['rea_k_res']),)+tuple(d['rea_k_res_CI'])
cols=list(sc['cols']); rows=sc['rows']
naive=rows[:,cols.index('naive_res')]

fig,(axA,axB)=plt.subplots(1,2,figsize=(10.2,4.1))

# (a) 3-D data-efficiency curves
axA.plot(N,am,'o-',color=GRN,lw=2,ms=6,label=fr'resonance-aware ($k$={kr[0]:.2f} [{kr[1]:.2f},{kr[2]:.2f}])')
axA.fill_between(N,np.maximum(am-as_,1e-3),am+as_,color=GRN,alpha=0.20,lw=0)
axA.plot(N,um,'s--',color=SIG,lw=1.8,ms=5,label=fr'uniform ($k$={ku[0]:.2f} [{ku[1]:.2f},{ku[2]:.2f}])')
axA.fill_between(N,np.maximum(um-us,1e-3),um+us,color=SIG,alpha=0.15,lw=0)
axA.plot(N,rm,'^:',color=ACC,lw=1.6,ms=5,label=fr'random ($k$={kk[0]:.2f} [{kk[1]:.2f},{kk[2]:.2f}])')
axA.fill_between(N,np.maximum(rm-rs,1e-3),rm+rs,color=ACC,alpha=0.12,lw=0)
axA.plot(N,naive,'x-',color=NEU,lw=1.1,ms=6,alpha=0.8,label='naive (global criterion)')
axA.set_xscale('log'); axA.set_yscale('log')
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
axA.xaxis.set_major_locator(FixedLocator(list(N))); axA.xaxis.set_minor_locator(NullLocator())
axA.xaxis.set_minor_formatter(NullFormatter()); axA.set_xticklabels([int(n) for n in N])
axA.set_xlabel(r'number of full-wave anchors $N$ (of 324)')
axA.set_ylabel('resonance-region phase error (deg)')
axA.legend(frameon=False,fontsize=8); axA.grid(alpha=0.25,which='both')
panel(axA,'(a)')
axA.set_title(r'three-parameter $(L_x,L_y,h)$ cell, 16 seeds',loc='left',fontsize=10.5,fontweight='bold')

# (b) resonance-error floors, 2D vs 3D.
# Floors taken from the higher-seed re-runs (surr2d_scaling_ms 20 seeds, surr3d_scaling 16 seeds)
# so the bar chart matches the curves and the manuscript text; bf provides only the target.
target=float(bf['common_res_target'])
d2=L('surr2d_scaling_ms.npz')
fu2=float(np.min(d2['uni_res_mean'])); fr2=float(np.min(d2['rea_res_mean']))
fu3=float(np.min(d['uni_res_mean']));  fr3=float(np.min(d['rea_res_mean']))
x=np.arange(2); w=0.36
b1=axB.bar(x-w/2,[fu2,fu3],w,color=SIG,edgecolor='k',lw=0.5,label='uniform floor')
b2=axB.bar(x+w/2,[fr2,fr3],w,color=GRN,edgecolor='k',lw=0.5,label='resonance-aware floor')
axB.set_yscale('log'); axB.set_ylim(0.2,40)
axB.axhline(target,color=ACC,ls='--',lw=1.2)
axB.text(1.46,target*1.05,fr'target {target:.0f}$^\circ$',color=ACC,fontsize=8.5,ha='right',va='bottom')
for rect,v in zip(list(b1)+list(b2),[fu2,fu3,fr2,fr3]):
    axB.text(rect.get_x()+rect.get_width()/2,v*1.08,fr'{v:.2f}$^\circ$' if v<1 else fr'{v:.1f}$^\circ$',
             ha='center',fontsize=8.4)
axB.set_xticks(x); axB.set_xticklabels(['2-D $(L_x,L_y)$','3-D $(L_x,L_y,h)$'],fontsize=9.5)
axB.set_ylabel('best attained resonance error (deg)')
axB.legend(frameon=False,fontsize=8.5,loc='upper left')
panel(axB,'(b)',x=-0.16)
axB.set_title('curse-of-dimensionality bite',loc='left',fontsize=10.5,fontweight='bold')

save(fig,'fig_p3_3d')
print('DONE')
