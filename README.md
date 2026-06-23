# Reproducibility repository — Paper 3

**Resonance-Aware Active Sampling for Data-Efficient Differentiable Surrogates of
Resonant Responses: A Reflectarray Unit-Cell Case Study**

Long Zhang\* (corresponding: 20230100@huat.edu.cn), Xuan Shi, Liang Ma, Shengjun Wu, Zijuan Wu
Hubei University of Automotive Technology, Shiyan 442002, China

This repository contains the code, the full-wave reference / result data, and the figure
generators needed to reproduce every figure and table in the paper. All results are
simulation-only (openEMS FDTD reference + PyTorch surrogates).

---

## 1. Environment

- Python 3.11
- PyTorch 2.6.0 (CUDA 12.4 build used; CPU also works, slower)
- NumPy, SciPy, scikit-learn, Matplotlib (Times New Roman / STIX for figures)
- h5py
- **openEMS 0.0.36** (with the Python `CSXCAD` / `openEMS` bindings) — only needed to regenerate
  the full-wave *reference* data from scratch; not needed to reproduce figures from the provided `.npz`.

Set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows if you hit an OpenMP duplicate-runtime error.

Run all scripts **from inside this folder** (scripts load their `.npz` inputs by relative name).

---

## 2. Quick reproduction (figures from provided data — no openEMS, no GPU needed)

```bash
python figs_unified.py     # -> figures/fig_p3_setup, _scaling, _eval, _baseline2, _ood, _ablation, meta_steer
python fig3d.py            # -> figures/fig_p3_3d
python fig_online.py       # -> figures/fig_p3_online
```

These read the frozen `.npz` files in this folder and write the PDFs/PNGs used in the manuscript.

---

## 3. Figure / table → script + data map

| Manuscript item | Generator script | Data file(s) |
|---|---|---|
| Fig. 1 (unit cell + Γ surface) | `figs_unified.py` | `ref2d.npz` |
| Fig. 2 (online deployable sampling) | `fig_online.py` | `surr2d_online.npz` |
| Fig. 3 (data-efficiency scaling) | `figs_unified.py` | `surr2d_scaling_ms.npz` |
| Fig. 4 (3-D widening + curse of dim.) | `fig3d.py` | `surr3d_scaling.npz`, `surr3d_strategy_compare.npz`, `surr3d_bite_finite.npz` |
| Fig. 5 (baselines / gradient / inverse) | `figs_unified.py` | `surr2d_baseline.npz`, `surr2d_grad.npz`, `surr2d_inverse.npz` |
| Fig. 6 (classical interpolant baselines) | `figs_unified.py` | `classical.npz`, `surr2d_baseline.npz` |
| Fig. 7 (inverse-design OOD robustness) | `figs_unified.py` | `surr2d_inverse_ood.npz` |
| Fig. 8 (physics-unnecessary ablation) | `figs_unified.py` | `ablation3.npz` (+ 1-D values in script) |
| Fig. 9 (anomalous-reflection closure) | `figs_unified.py` | `meta_steer.npz`, `meta_steer_h1.npz` |
| Table: data-efficiency / exponents | (values) | `surr2d_scaling_ms.npz` |
| Table: 3-D | (values) | `surr3d_scaling.npz`, `surr3d_strategy_compare.npz` |
| Table: baselines | (values) | `surr2d_baseline.npz` |
| Table: fine-FD gradient | (values) | `grad_finefd.npz` |
| Table: inverse design | (values) | `surr2d_inverse.npz` |
| Table: metagrating orders | (values) | `meta_steer.npz` |

---

## 4. Regenerating results from scratch

**Surrogate experiments (PyTorch; minutes):**
```bash
python surrogate2d.py        # 2-D surrogate: scaling/baseline/gradient/inverse .npz
python surr2d_scaling_ms.py  # multi-seed scaling + bootstrap exponents -> surr2d_scaling_ms.npz
python surr2d_online.py      # online (no-oracle) acquisition           -> surr2d_online.npz
python classical.py          # RBF / kriging / SVR / ANN baselines       -> classical.npz
python surr3d_scaling.py     # 3-D (Lx,Ly,h) scaling + bootstrap         -> surr3d_scaling.npz, ...
python hybrid3.py 12 8000 0  # 2-parameter physics-unnecessary ablation  -> ablation3.npz
python hybrid2.py            # 1-parameter ablation reference
```

**Full-wave finite-difference gradient check (openEMS; ~12 short runs):**
```bash
python grad_finefd.py        # fine +-0.1 mm full-wave FD -> grad_finefd.npz
```

**Full-wave reference / closure (openEMS; longer):**
```bash
python openems2d.py          # 99-point 2-D (Lx,Ly) reflection reference -> ref2d.npz
python openems3d.py          # 324-point 3-D (Lx,Ly,h) reference         -> ref3d.npz
python meta_steer.py            # anomalous-reflection metagrating closure  -> meta_steer*.npz
```

Note: openEMS may change the process working directory mid-run; scripts that save with a relative
path are launched from this folder, and a couple save via an absolute path — check the printed
"saved ..." line for the output location.

---

## 5. Notes

- Seeds, anchor budgets, network sizes, and bootstrap settings are fixed inside each script and
  listed in the manuscript's reproducibility table.
- The full-wave reference uses a waveguide-simulator unit cell (PMC⊥x, PEC⊥y) on grounded FR4 at 24 GHz.
- License: MIT (see `LICENSE`). Citation metadata: `CITATION.cff`; archival metadata: `.zenodo.json`.

- `hybrid3_adv.py` -> `ablation3_adv.npz`: advanced (NTK/gradient-norm-balanced) hybrid ablation (still ~26x worse than pure data).
- `principled_acq.py` -> `principled_acq.npz`: principled GP-uncertainty active-learning baseline (curvature beats it ~20x).
- `openems2d_h15.py` -> `ref2d_h15.npz`: condition-B reference (h=1.5mm) for transfer.
- `transfer.py` -> `transfer.npz` + `fig_transfer.py`: cross-substrate transfer (warm-start ~4x fewer anchors).
- `openems2d_cross.py` -> `ref2d_cross.npz`; `cross_scaling.py` -> `cross_scaling.npz` + `fig_cross.py`: second cell topology (cross dipole).
- `openems2d_dual.py` -> `ref2d_dual.npz`; `dual_scaling.py` -> `dual_scaling.npz`: third cell (dual-resonance, two y-dipoles).
