# -*- coding: utf-8 -*-
# fig_p3_transfer: warm-start (adapt the h=1.0 surrogate) vs train-from-scratch on a new substrate
# (h=1.5), resonance-region error vs number of new full-wave anchors.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,'axes.linewidth':0.9,'font.size':11})
DIR=os.path.dirname(os.path.abspath(__file__)); GRN='#1e7a45'; ACC='#c0392b'
d=np.load(os.path.join(DIR,'transfer.npz')); k=d['budgets']
wm,ws,sm,ss=d['warm_mean'],d['warm_std'],d['scratch_mean'],d['scratch_std']
fig,ax=plt.subplots(figsize=(5.4,4.1))
ax.plot(k,wm,'o-',color=GRN,lw=2,ms=6,label='warm-start (adapt $h{=}1.0$ surrogate)')
ax.fill_between(k,np.maximum(wm-ws,1e-3),wm+ws,color=GRN,alpha=0.18,lw=0)
ax.plot(k,sm,'s--',color=ACC,lw=1.8,ms=5,label='train from scratch on $h{=}1.5$')
ax.fill_between(k,np.maximum(sm-ss,1e-3),sm+ss,color=ACC,alpha=0.15,lw=0)
ax.axhline(sm[-1],color='0.5',ls=':',lw=1.0)
ax.annotate(f'scratch best ({sm[-1]:.1f}$^\\circ$, $k$=16);\nwarm reaches it at $k$=4',
            xy=(9.5,sm[-1]),xytext=(9.5,12.5),ha='center',fontsize=8.2,color=GRN,
            arrowprops=dict(arrowstyle='->',color=GRN,lw=0.9))
ax.set_yscale('log'); ax.set_xticks(k); ax.set_xticklabels([int(x) for x in k])
ax.set_xlabel('number of NEW full-wave anchors on the target substrate ($h{=}1.5$ mm)')
ax.set_ylabel('resonance-region phase error (deg)')
ax.legend(frameon=False,fontsize=8.6); ax.grid(alpha=0.25,which='both')
ax.set_title('Transfer across substrate: train once, adapt cheaply',fontsize=10.5,fontweight='bold')
fig.savefig(os.path.join(DIR,'fig_p3_transfer.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(DIR,'fig_p3_transfer.pdf'),bbox_inches='tight')
print('saved fig_p3_transfer')
