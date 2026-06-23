# -*- coding: utf-8 -*-
# fig_p3_outdomain (2-panel): the resonance-aware curvature criterion vs uniform/random on two non-EM
# resonance-dominated responses -- (a) a driven damped oscillator / series-RLC transfer function,
# (b) a Fano resonance line shape.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9, 'axes.linewidth': 0.9, 'font.size': 11})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; SIG = '#1f5fa6'; ACC = '#c0392b'


def panel(ax, npz, title):
    d = np.load(os.path.join(DIR, npz)); N = d['N']
    for p, lab, c, ls, mk in [('rea', 'resonance-aware', GRN, 'o-', 6), ('uni', 'uniform', SIG, 's--', 5), ('rnd', 'random', ACC, '^:', 5)]:
        m = d[p + '_res_mean']; s = d[p + '_res_std']; kk = float(d[p + '_k']); ci = tuple(d[p + '_k_CI'])
        ax.plot(N, m, ls, color=c, lw=1.9, ms=mk, label=f'{lab} ($k$={kk:.2f} [{ci[0]:.2f},{ci[1]:.2f}])')
        ax.fill_between(N, np.maximum(m - s, 0.25 * m), m + s, color=c, alpha=0.15, lw=0)  # floor relative to mean (log axis); std can exceed mean for random
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(N); ax.set_xticklabels([int(n) for n in N])
    ax.set_xlabel(r'number of oracle samples $N$ (of 99)'); ax.legend(frameon=False, fontsize=8.0)
    ax.grid(alpha=0.25, which='both'); ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold')


fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2))
panel(a, 'outdomain_rlc.npz', '(a) driven oscillator / series-RLC (non-EM)')
a.set_ylabel('resonance-region phase error (deg)')
panel(b, 'outdomain_fano.npz', '(b) Fano resonance line shape (non-EM)')
fig.savefig(os.path.join(DIR, 'fig_p3_outdomain.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_p3_outdomain.pdf'), bbox_inches='tight')
print('saved fig_p3_outdomain (2-panel: RLC + Fano)')
