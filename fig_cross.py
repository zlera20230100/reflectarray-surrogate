# -*- coding: utf-8 -*-
# fig_p3_cross: resonance-region error vs anchor budget for resonance-aware / uniform / random on a
# second cell topology (cross dipole).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,'axes.linewidth':0.9,'font.size':11})
DIR=os.path.dirname(os.path.abspath(__file__)); GRN='#1e7a45'; SIG='#1f5fa6'; ACC='#c0392b'
d=np.load(os.path.join(DIR,'cross_scaling.npz')); N=d['N']
def k(p): return float(d[p+'_k']), tuple(d[p+'_k_CI'])
fig,ax=plt.subplots(figsize=(5.4,4.2))
for p,lab,c,ls,mk in [('rea','resonance-aware',GRN,'o-',6),('uni','uniform',SIG,'s--',5),('rnd','random',ACC,'^:',5)]:
    m=d[p+'_res_mean']; s=d[p+'_res_std']; kk,ci=k(p)
    ax.plot(N,m,ls,color=c,lw=1.9,ms=mk,label=f'{lab} ($k$={kk:.2f} [{ci[0]:.2f},{ci[1]:.2f}])')
    ax.fill_between(N,np.maximum(m-s,1e-3),m+s,color=c,alpha=0.15,lw=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(N); ax.set_xticklabels([int(n) for n in N])
ax.set_xlabel(r'number of full-wave anchors $N$ (of 99)')
ax.set_ylabel('resonance-region phase error (deg)')
ax.legend(frameon=False,fontsize=8.6); ax.grid(alpha=0.25,which='both')
ax.set_title('Advantage recurs on a second cell topology (cross dipole)',fontsize=10.2,fontweight='bold')
fig.savefig(os.path.join(DIR,'fig_p3_cross.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(DIR,'fig_p3_cross.pdf'),bbox_inches='tight')
print('saved fig_p3_cross')
