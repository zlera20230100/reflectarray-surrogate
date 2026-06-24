# -*- coding: utf-8 -*-
# Elsevier graphical abstract: 13 x 5 cm landscape banner (>= 1328 x 531 px).
# Three panels: phase surface with anchors, surrogate box, acquisition comparison.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
from matplotlib.patches import FancyBboxPatch
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; SIG = '#1f5fa6'; ACC = '#c0392b'

fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))                      # 13 x 5 cm
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.5,
                      left=0.07, right=0.985, bottom=0.30, top=0.80)

def caption(ax, s):
    ax.text(0.5, -0.34, s, transform=ax.transAxes, ha='center', va='top', fontsize=7.0, color='0.15')

# panel 1: phase surface, anchors crowded into the resonance band
ax0 = fig.add_subplot(gs[0, 0])
ref = np.load(os.path.join(DIR, 'ref2d.npz')); ph = ref['phase']; Lx = ref['Lx']; Ly = ref['Ly']
ax0.pcolormesh(Lx, Ly, ph, cmap='RdBu_r', shading='auto', rasterized=True)
rng = np.random.default_rng(0)
ax0.scatter(rng.uniform(2, 6, 9), rng.uniform(2.0, 2.8, 9), s=11, c='k', marker='o', lw=0)
ax0.scatter(rng.uniform(2, 6, 4), rng.uniform(3.2, 6, 4), s=11, c='k', marker='o', lw=0)
ax0.set_xlabel(r'$L_x$', fontsize=8, labelpad=0); ax0.set_ylabel(r'$L_y$', fontsize=8, labelpad=-2)
ax0.set_xticks([]); ax0.set_yticks([])
caption(ax0, 'sample where it matters:\ncrowd the resonance band')

# panel 2: surrogate maps geometry to reflection and its gradient
ax1 = fig.add_subplot(gs[0, 1]); ax1.axis('off')
ax1.add_patch(FancyBboxPatch((0.16, 0.46), 0.68, 0.34, boxstyle='round,pad=0.02,rounding_size=0.06',
              fc='#eef3f8', ec=SIG, lw=1.2, transform=ax1.transAxes))
ax1.text(0.5, 0.63, 'differentiable\nsurrogate', transform=ax1.transAxes, ha='center', va='center',
         fontsize=8.4, fontweight='bold', color=SIG)
ax1.annotate('', xy=(0.14, 0.63), xytext=(-0.01, 0.63), transform=ax1.transAxes,
             arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.3))
ax1.text(0.06, 0.85, r'$(L_x,L_y)$', transform=ax1.transAxes, ha='center', fontsize=7.2)
ax1.annotate('', xy=(1.01, 0.63), xytext=(0.86, 0.63), transform=ax1.transAxes,
             arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.3))
ax1.text(0.93, 0.85, r'$\Gamma,\ \partial\Gamma/\partial L$', transform=ax1.transAxes, ha='center', fontsize=7.2)
caption(ax1, 'autodiff gradients drive\ninverse design ($0.9^\\circ$)')

# panel 3: phase error vs anchor budget for each acquisition rule.
# resonance-aware + uniform from the 20-seed headline run; GP and LOLA from the joint comparison run.
ax2 = fig.add_subplot(gs[0, 2])
ms = np.load(os.path.join(DIR, 'surr2d_scaling_ms.npz')); lola = np.load(os.path.join(DIR, 'lola.npz'))
gp = np.load(os.path.join(DIR, 'principled_acq.npz')); N = ms['N']
for m, c, lab in [(ms['rea_res_mean'], GRN, 'resonance-aware'), (lola['lola_res_mean'], '#e67e22', 'LOLA-Voronoi'),
                  (gp['gp_res_mean'], SIG, 'GP variance'), (ms['uni_res_mean'], '0.55', 'uniform')]:
    ax2.plot(N, m, '-', color=c, lw=1.6, label=lab)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.xaxis.set_major_locator(FixedLocator([9, 36])); ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xticklabels(['9', '36']); ax2.tick_params(labelsize=7, length=2)
ax2.set_xlabel('full-wave anchors $N$', fontsize=8, labelpad=1)
ax2.set_ylabel('phase error (deg)', fontsize=8, labelpad=1)
ax2.legend(frameon=False, fontsize=5.9, loc='upper right', handlelength=1.1, borderpad=0.1, labelspacing=0.22)
for s in ['top', 'right']: ax2.spines[s].set_visible(False)
# annotate the resonance-aware decay rate (uniform stays the flat ~N^-0.8 baseline)
ax2.text(24, 1.4, r'$N^{-2.3}$ vs $N^{-0.8}$', color='0.15', fontsize=6.6, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
caption(ax2, r'$0.29^\circ$ at $N{=}36$: ~$20\times$ vs GP,' '\n' r'~$9\times$ vs published LOLA-Voronoi')

fig.suptitle('Resonance-aware active sampling for a data-efficient, differentiable resonant-cell surrogate',
             fontsize=8.6, fontweight='bold', y=0.955)
fig.savefig(os.path.join(DIR, 'fig_p3_graphical.png'), dpi=300)
fig.savefig(os.path.join(DIR, 'fig_p3_graphical.pdf'))
from PIL import Image
w, h = Image.open(os.path.join(DIR, 'fig_p3_graphical.png')).size
print(f'saved fig_p3_graphical.png  {w} x {h} px  ok={w>=1328 and h>=531}')
