# -*- coding: utf-8 -*-
# fig_p3_acq: acquisition-strategy comparison on the 2-D patch (resonance-region phase error vs N).
# Compares resonance-aware (oracle) and its online variant, LOLA-Voronoi, GP max-variance, and
# uniform/random.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8.4, 'axes.linewidth': 0.9, 'font.size': 11})
DIR = os.path.dirname(os.path.abspath(__file__))
onl = np.load(os.path.join(DIR, 'surr2d_online.npz'))
gp = np.load(os.path.join(DIR, 'principled_acq.npz'))
lola = np.load(os.path.join(DIR, 'lola.npz'))
N = onl['N']

# (label, mean, std, k, colour, linestyle, marker)
def k_of(d, key): return float(d[key])
series = [
    ('resonance-aware (oracle)',      onl['ora_res_mean'], onl['ora_res_std'], k_of(onl,'ora_k'), '#1e7a45', 'o-', 6),
    ('resonance-aware (online, deployable)', onl['onl_res_mean'], onl['onl_res_std'], k_of(onl,'onl_k'), '#27ae60', 'o-', 5),
    ('LOLA-Voronoi (published adaptive)', lola['lola_res_mean'],lola['lola_res_std'],k_of(lola,'lola_k'),'#e67e22','v--',5),
    ('GP max-variance (textbook AL)', gp['gp_res_mean'],   gp['gp_res_std'],   k_of(gp,'gp_k'),   '#1f5fa6', 's--', 5),
    ('uniform',                       onl['uni_res_mean'], onl['uni_res_std'], k_of(onl,'uni_k'), '#7f8c8d', 'P:', 5),
    ('random',                        onl['rnd_res_mean'], onl['rnd_res_std'], k_of(onl,'rnd_k'), '#c0392b', '^:', 5),
]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for lab, m, s, k, c, ls, mk in series:
    ax.plot(N, m, ls, color=c, lw=1.8, ms=mk, label=lab)
    ax.fill_between(N, np.maximum(m - s, 1e-3), m + s, color=c, alpha=0.10, lw=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(N); ax.set_xticklabels([int(n) for n in N])
ax.set_ylim(0.03, 45)   # extend the lower decade so the legend fits in the (empty) lower-left
ax.set_xlabel(r'number of full-wave anchors $N$ (of 99)')
ax.set_ylabel('resonance-region phase error (deg)')
ax.grid(alpha=0.25, which='both')
ax.legend(frameon=True, framealpha=0.92, edgecolor='none', fontsize=8.0, loc='lower left', borderaxespad=0.6)
ax.set_title('Acquisition strategies: the resonance-aware family beats every off-the-shelf rule',
             loc='left', fontsize=10.0, fontweight='bold')
fig.savefig(os.path.join(DIR, 'fig_p3_acq.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_p3_acq.pdf'), bbox_inches='tight')
print('saved fig_p3_acq')
print('N=36 (deg):',
      f"oracle {onl['ora_res_mean'][-1]:.2f} | online {onl['onl_res_mean'][-1]:.2f} | "
      f"LOLA {lola['lola_res_mean'][-1]:.2f} | GP {gp['gp_res_mean'][-1]:.2f} | "
      f"uni {onl['uni_res_mean'][-1]:.2f} | rnd {onl['rnd_res_mean'][-1]:.2f}")
