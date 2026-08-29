# lewm-phi v2 campaign — train + matched re-eval

**Date:** 2026-08-28  
**Note:** Does **not** overwrite `lewm_phi_v1` / `lewm_phi` / `lewm_phi_fixed` artifacts.

## Training runs (preserved side-by-side)

| Run | Path | Data | Epochs | Split | Best held-out val corr |
|-----|------|------|--------|-------|------------------------|
| v1 leaky | `stablewm/checkpoints/pusht/lewm_phi/` | 6k weak steps | 5 | broken (same-bank resample) | 0.569 (overstated) |
| fixed | `stablewm/checkpoints/pusht/lewm_phi_fixed/` | 6k weak steps | 5 | 54/6 episode holdout | **0.487** @ ep3 |
| **v2** | `stablewm/checkpoints/pusht/lewm_phi_v2/` | **24k** weak steps (241 eps) | **15** | 217/24 holdout | **0.536** @ ep13 |

v2 also logs: `metrics.csv`, `training_curves.png`, `train_phi_meta.json`, `train_console.log`.

## Eval protocol (this campaign)

- `online_offset` + `pair_mode=short_horizon`
- **n=20** (short_horizon only yields ~21 valid pairs even with 320 kin eps; n=32 not feasible without widening the band)
- seed 0, kinematic collector, `--collect-episodes 320`
- C1 goal-emb cache on
- Artifacts: `le-wm/eval_results/pusht/lewm_phi_v2_eval/{E1_l2_c1,E2_phi_fixed,E2_phi_v2,E4_random}/pusht/pusht_seed0/`

## Matched results (n=20)

| ID | Cost | Weights | Success % | mean min pose ↓ | mean min state ↓ | plan t (s) |
|----|------|---------|-----------|-----------------|------------------|------------|
| E1 | L2 `z` | — | 25.0 (5/20) | 27.34 | 107.1 | 1.57 |
| E2_fixed | `d_φ` | `lewm_phi_fixed/reach.pt` | 20.0 (4/20) | 28.52 | 104.3 | 1.58 |
| **E2_v2** | `d_φ` | `lewm_phi_v2/reach.pt` | **35.0 (7/20)** | **23.86** | **85.3** | 1.45 |
| E4 | `d_φ` random | — | 20.0 (4/20) | 32.43 | 144.6 | 1.58 |

## Reference: old v1 (n=8, leaky φ, not matched to this table)

| ID | Success % | mean min pose | mean min state |
|----|-----------|---------------|----------------|
| E1 | 12.5 | 33.5 | 93.7 |
| E2 | 12.5 | 29.7 | 107.2 |
| E4 | 12.5 | 31.4 | 96.3 |

## Gate (design spec §11)

- **Go** requires E2 > E1 under C1.
- **E2_fixed ≯ E1** (20% vs 25%) — small fixed run still insufficient.
- **E2_v2 > E1** on success (35% vs 25%), mean min pose (23.9 vs 27.3), and mean min state (85 vs 107).
- E2_v2 also beats E4 (random φ).

### Verdict: **weak go** (v2 weights only)

Larger live bank + longer train + held-out checkpointing yields a planning cost that **beats L2-in-z on this matched n=20 short_horizon protocol**. Still a small-n result; next step is a wider-band or `offset` protocol at n≥50 for a firmer claim.

## Commands (repro)

```bash
# train v2 (already run)
python train_phi.py --collector weak --collect-steps 24000 --epochs 15 \
  --samples-per-epoch 4096 --seed 0 --device cuda \
  --out-dir ../stablewm/checkpoints/pusht/lewm_phi_v2

# eval (example E2_v2)
python eval_live.py --env pusht --protocol online_offset --pair-mode short_horizon \
  --episodes 20 --seed 0 --collector kinematic --collect-episodes 320 \
  --plan-cost phi_d --phi-weights ../stablewm/checkpoints/pusht/lewm_phi_v2/reach.pt \
  --no-video --log-dir eval_results/pusht/lewm_phi_v2_eval/E2_phi_v2
```
