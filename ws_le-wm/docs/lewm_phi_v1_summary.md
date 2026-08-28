# lewm-phi v1 ablation summary

**Date:** 2026-08-28  
**Branch:** `feat/lewm-phi` (parent + `ws_le-wm/le-wm` submodule)  
**Train:** live WeakPolicy bank (60 eps / 6000 steps), frozen HF trunk, GPU `cuda:0`  
**φ checkpoint:** `stablewm/checkpoints/pusht/lewm_phi/reach.pt` (best val corr(d,k)=**0.569**)  
**Eval:** `online_offset` + `pair_mode=short_horizon`, 8 episodes (E1/E2/E4), seed 0

## Metrics

| ID | Cost | Interface | n | Success % | mean min pose dist ↓ | mean plan time (s) |
|----|------|-----------|---|-----------|----------------------|--------------------|
| E0 | L2 `z` | archived `paper_cem_kin10` | 10 | 0.0 | **68.6** | 1.76 |
| E1 | L2 `z` | C1 cache | 8 | 12.5 | 93.7 | 1.66 |
| E2 | `d_φ` trained | C1 cache | 8 | 12.5 | 107.2 | 1.66 |
| E4 | `d_φ` random | C1 cache | 8 | 12.5 | 96.3 | 1.78 |

Artifacts: `le-wm/eval_results/pusht/lewm_phi_v1/{E0_baseline,E1_l2_c1,E2_phi_c1,E4_random}/`

## Gate (design spec §11)

- **Go** requires E2 > E1 under C1 (φ cost beats L2).
- Here E2 **ties** E1 on success (12.5%) and is **worse** on mean min distance (107 vs 94).
- E2 also does not beat random φ (E4) on these 8 episodes.

### Verdict: **pivot**

`φ` learns a moderate temporal signal (val corr ≈ 0.57) but does **not** yet improve PushT short-horizon CEM vs L2-in-`z` under this protocol.

## Likely next probes (in order)

1. Larger / matched eval (n≥50, same pairs across E1/E2) — 8 eps is noisy.
2. Inspect cost scale: does CEM see usable dynamic range for `d_φ` vs L2?
3. Stronger live bank (more steps, mix kinematic + weak) and longer `train_phi`.
4. Only then: quasimetric / IQL upgrade (deferred D3) — not hierarchy yet.

## Training visibility

Per-epoch logs live at:

- `stablewm/checkpoints/pusht/lewm_phi/metrics.csv`
- `stablewm/checkpoints/pusht/lewm_phi/training_curves.png`

Regenerate plot: `python scripts/plot_train_phi.py`
