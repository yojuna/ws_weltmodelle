# lewm-phi v3 campaign — large trains + offset n=50 eval

**Date:** 2026-08-28  
**Note:** Does **not** overwrite prior checkpoints or `lewm_phi_v1` / `v2` summaries.

## New training runs

| Run | Path | Collector / data | Epochs | Holdout | Best val corr(d,k) |
|-----|------|------------------|--------|---------|---------------------|
| **v3_kin** | `stablewm/checkpoints/pusht/lewm_phi_v3_kin/` | kinematic 256 eps / 20.5k steps | 20 | 230/26 | **0.731** @ ep1 |
| **v3_weak** | `stablewm/checkpoints/pusht/lewm_phi_v3_weak/` | weak 481 eps / **48k** steps | 20 | 433/48 | **0.535** @ ep19 |

Prior (unchanged): `lewm_phi` (leaky), `lewm_phi_fixed`, `lewm_phi_v2`.

### Training notes

- **v3_kin** learns temporal structure easily on kinematic rollouts (train corr → 0.91) but **overfits**: held-out peaks at epoch 1 (0.73) and does not improve. Checkpoint is early-stop by val corr.
- **v3_weak** matches v2-scale corr (~0.54) despite 2× data — weak-policy hindsight saturates around this range.

## Eval protocol (this campaign)

- `online_offset` + **`pair_mode=offset`** (paper-style t→t+25; harder / more pairs than short_horizon)
- **n=50**, seed 0, kinematic bank `--collect-episodes 200`
- C1 goal-emb cache on
- Artifacts: `le-wm/eval_results/pusht/lewm_phi_v3_eval/{E1_l2_c1,E2_phi_v2,E2_phi_v3_kin,E2_phi_v3_weak,E4_random}/`
- Full console: `.../lewm_phi_v3_eval/campaign_console.log`

## Matched results (offset, n=50)

| ID | Cost | Weights | Success % | mean min pose ↓ | mean min state ↓ |
|----|------|---------|-----------|-----------------|------------------|
| **E1** | L2 `z` | — | **8.0 (4/50)** | **45.4** | 139.4 |
| E2_v2 | `d_φ` | `lewm_phi_v2` | 6.0 (3/50) | 45.0 | **130.6** |
| E2_v3_kin | `d_φ` | `lewm_phi_v3_kin` | 8.0 (4/50) | 50.3 | 185.6 |
| E2_v3_weak | `d_φ` | `lewm_phi_v3_weak` | 6.0 (3/50) | 45.8 | 144.5 |
| E4 | random `d_φ` | — | 4.0 (2/50) | 45.9 | 166.4 |

## Cross-campaign context

| Campaign | Protocol | n | Best φ vs E1 |
|----------|----------|---|--------------|
| v1 | short_horizon | 8 | tie / worse (pivot) |
| v2 | short_horizon | 20 | **E2_v2 35% > E1 25%** (weak go) |
| **v3** | **offset** | **50** | **no φ beats E1 on success** |

## Gate (design spec §11)

- **Go** requires E2 > E1 under C1 on the claim protocol.
- On **offset n=50**, no trained φ improves success over L2; v3_kin ties (8%) with **worse** pose/state error; v2/v3_weak are slightly worse on success.
- High kinematic `(d,k)` corr does **not** transfer to better CEM costs on this harder offset set.

### Verdict: **pivot on offset / large-n**

The short_horizon weak-go (v2) does **not** hold under the firmer offset n=50 protocol. Next probes (in order):

1. Cost-to-go mismatch: hindsight `k` ≠ optimal steps — try quasimetric / IQL (deferred D3).
2. Re-check v3_kin / v2 on **short_horizon n=20** with these new weights (optional, for continuity).
3. Do **not** scale weak/kinematic regression further expecting a planning win — signal saturates.

## Commands (repro)

```bash
# trains (already run)
python train_phi.py --collector kinematic --collect-episodes 256 --epochs 20 \
  --samples-per-epoch 4096 --out-dir ../stablewm/checkpoints/pusht/lewm_phi_v3_kin
python train_phi.py --collector weak --collect-steps 48000 --epochs 20 \
  --samples-per-epoch 8192 --out-dir ../stablewm/checkpoints/pusht/lewm_phi_v3_weak

# eval example
python eval_live.py --env pusht --protocol online_offset --pair-mode offset \
  --episodes 50 --seed 0 --collect-episodes 200 --plan-cost phi_d \
  --phi-weights ../stablewm/checkpoints/pusht/lewm_phi_v3_kin/reach.pt \
  --no-video --log-dir eval_results/pusht/lewm_phi_v3_eval/E2_phi_v3_kin
```
