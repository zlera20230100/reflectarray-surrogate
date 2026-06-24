# -*- coding: utf-8 -*-
# Transfer of the surrogate across substrates: pretrain on condition A (h=1.0 mm), then adapt to
# condition B (h=1.5 mm, which shifts the resonance) by warm-starting from the A-trained weights.
# Warm-start vs training from scratch on the same B-anchors and the same optimiser budget; only the
# initialisation differs.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time, copy, numpy as np, torch
import surrogate2d as S            # Surrogate2D, norm_xy, train_surrogate, wrap180

DIR = S.OUT_DIR
A = np.load(os.path.join(DIR, 'ref2d.npz'))          # condition A (h=1.0)
B = np.load(os.path.join(DIR, 'ref2d_h15.npz'))      # condition B (h=1.5)
dev = S.DEVICE
Lx_ax = A['Lx'].astype(float); Ly_ax = A['Ly'].astype(float)   # same grid for A and B
NY, NX = B['phase'].shape
IY, IX = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij'); IY = IY.ravel(); IX = IX.ravel()
LXf = Lx_ax[IX]; LYf = Ly_ax[IY]
def grids(ref):
    ph = ref['phase'].astype(float); mg = ref['mag'].astype(float)
    G = mg[IY, IX]*np.exp(1j*np.deg2rad(ph[IY, IX])); return ph, G.real, G.imag, ph[IY, IX]
phA, REA, IMA, PHA = grids(A)
phB, REB, IMB, PHB = grids(B)
NALL = PHB.size; RES_LY = 2.8
res_mask = LYf <= RES_LY

def curv_anchors(ph_grid, N):
    """resonance-aware (curvature) anchors on a given grid (same criterion as the paper)."""
    uw = np.rad2deg(np.unwrap(np.deg2rad(ph_grid), axis=0))
    c = np.zeros_like(uw)
    c[1:-1, :] += np.abs(uw[:-2, :]-2*uw[1:-1, :]+uw[2:, :]); c[:, 1:-1] += np.abs(uw[:, :-2]-2*uw[:, 1:-1]+uw[:, 2:])
    c[0, :] = c[1, :]; c[-1, :] = c[-2, :]; c[:, 0] = c[:, 1]; c[:, -1] = c[:, -2]
    w = 1.0+c/(c.max()+1e-12); wf = w.ravel(); field = uw.ravel()
    coords = S.norm_xy(LXf, LYf)
    corners = [0, NX-1, (NY-1)*NX, (NY-1)*NX+NX-1]; chosen = list(dict.fromkeys(corners))[:N]
    while len(chosen) < N:
        cs = np.array(chosen); d2 = np.maximum(((coords[:, None]-coords[None, cs])**2).sum(-1), 1e-9)
        wts = 1.0/d2; interp = (wts*field[cs][None]).sum(1)/wts.sum(1)
        sc = wf*np.abs(field-interp); sc[cs] = -1.0; chosen.append(int(np.argmax(sc)))
    return sorted(chosen)

def train(idx, re, im, seed, init_state=None, iters=4000, lr=2e-3):
    torch.manual_seed(seed); net = S.Surrogate2D().to(dev)
    if init_state is not None: net.load_state_dict(init_state)
    xy = torch.tensor(S.norm_xy(LXf[idx], LYf[idx]), dtype=torch.float32, device=dev)
    tg = torch.tensor(np.stack([re[idx], im[idx]], 1), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 1e-5)
    for _ in range(iters):
        opt.zero_grad(); loss = ((net(xy)-tg)**2).mean(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(S.norm_xy(LXf, LYf), dtype=torch.float32, device=dev)).cpu().numpy()
    return net.state_dict(), out[:, 0]+1j*out[:, 1]

def res_err(G, idx):
    err = np.abs(S.wrap180(np.rad2deg(np.angle(G))-PHB))
    held = res_mask & np.array([i not in set(idx) for i in range(NALL)])
    return float(err[held].mean()) if held.any() else float('nan')

# 1) pretrain on condition A (rich anchor set), once
A_idx = curv_anchors(phA, 36)
A_state, _ = train(A_idx, REA, IMA, seed=0, iters=8000, lr=2e-3)
print(f"pretrained on A (h=1.0), {len(A_idx)} anchors", flush=True)

# 2) transfer to B at small budgets: warm-start vs scratch, same anchors, same budget
BUD = [4, 6, 9, 12, 16]; SEEDS = list(range(6)); t0 = time.time()
warm = {k: [] for k in BUD}; scratch = {k: [] for k in BUD}
for k in BUD:
    bidx = curv_anchors(phB, k)
    for s in SEEDS:
        _, Gw = train(bidx, REB, IMB, seed=s, init_state=A_state, iters=4000, lr=2e-3)
        _, Gs = train(bidx, REB, IMB, seed=s, init_state=None,   iters=4000, lr=2e-3)
        warm[k].append(res_err(Gw, bidx)); scratch[k].append(res_err(Gs, bidx))
    print(f"  k={k:2d}  warm {np.mean(warm[k]):6.2f}  scratch {np.mean(scratch[k]):6.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

wm = np.array([np.mean(warm[k]) for k in BUD]); ws = np.array([np.std(warm[k]) for k in BUD])
sm = np.array([np.mean(scratch[k]) for k in BUD]); ss = np.array([np.std(scratch[k]) for k in BUD])
print("\n===== TRANSFER (h=1.0 -> h=1.5): warm-start vs scratch, resonance-region err =====")
for i, k in enumerate(BUD):
    print(f"  k={k:2d}:  warm {wm[i]:6.2f}+-{ws[i]:4.2f}   scratch {sm[i]:6.2f}+-{ss[i]:4.2f}   "
          f"(warm {sm[i]/max(wm[i],1e-9):.1f}x better)")
# transfer saving: smallest k where warm <= scratch's best (largest-k) error
target = sm[-1]
kw = next((BUD[i] for i in range(len(BUD)) if wm[i] <= target), None)
print(f"\n  scratch needs k={BUD[-1]} to reach {target:.2f} deg; warm-start reaches it at k={kw} "
      f"-> ~{BUD[-1]/kw:.1f}x fewer B-anchors" if kw else "  (warm did not reach scratch-best target)")
np.savez(os.path.join(DIR, 'transfer.npz'), budgets=np.array(BUD),
         warm_mean=wm, warm_std=ws, scratch_mean=sm, scratch_std=ss, hA=1.0, hB=1.5)
print("saved transfer.npz")
