# lewm-phi IQL v1 (Protocol T3) — train + matched eval

**Date:** 2026-08-28  
**Normative plan:** [`03_quasimetric_iql_t3.md`](03_quasimetric_iql_t3.md)  
**Note:** Does **not** overwrite prior Euclidean φ runs (`lewm_phi*`).

## Training

| Field | Value |
|-------|--------|
| Path | `stablewm/checkpoints/pusht/lewm_phi_iql_v1/` |
| Trunk | frozen `hf_pusht` |
| Distance | IQE-sum (`k=8,l=8`, `φ` dim 64) |
| Loss | Destrade Eq. (1) expectile, `γ=0.93`, `τ=0.60` |
| Bank | kinematic 256 eps / 20.5k steps; holdout 230/26 |
| Best ckpt | epoch **15**, val `L_VF=0.0353` |
| Artifacts | `reach.pt`, `metrics.csv`, `training_curves.png`, `train_phi_iql_meta.json` |

### Training diagnostics

- Loss finite and gently decreasing (train ~0.055 → ~0.033; best val 0.035 @ ep15).
- Mean IQE-sum `d` stays ~11–12 (nontrivial, not collapsed to 0).
- `frac_s==g` ≈ **0** on the sampled batches (exact index match is rare when `t ∈ [0,T-2]` and goals are terminal/random). Reward term is almost always `-1`; learning is almost pure discounted value consistency.
- Smoke: `attach_reach_head` loads `distance_mode=iqe_sum`; self-distance 0; CEM cost finite.

## Eval protocols (matched)

Artifacts: `le-wm/eval_results/pusht/lewm_phi_iql_v1/{short,offset}/…`  
Console: `…/lewm_phi_iql_v1/campaign_console.log`

1. **short_horizon n=20** — continuity with v2 weak-go (seed 0, kin bank 320 eps).
2. **offset n=50** — firm gate (seed 0, kin bank 200 eps).

## Short-horizon n=20

| ID | Cost | Weights | Success % | mean min pose ↓ | mean min state ↓ | plan t (s) |
|----|------|---------|-----------|-----------------|------------------|------------|
| **E1** | L2 `z` | — | **25.0 (5/20)** | **27.34** | **107.1** | 1.57 |
| E2_iql | IQE-sum `d_φ` | `lewm_phi_iql_v1` | 10.0 (2/20) | 27.73 | 120.4 | 1.68 |
| E4 | random IQE `d_φ` | — | 30.0 (6/20) | 24.83 | 96.8 | 1.67 |

E2_iql **loses** to E1 and even to random φ (E4) on success.

## Offset n=50

| ID | Cost | Weights | Success % | mean min pose ↓ | mean min state ↓ | plan t (s) |
|----|------|---------|-----------|-----------------|------------------|------------|
| **E1** | L2 `z` | — | **8.0 (4/50)** | **45.43** | **139.4** | 1.70 |
| E2_iql | IQE-sum `d_φ` | `lewm_phi_iql_v1` | 4.0 (2/50) | 46.83 | 144.2 | 1.72 |
| E4 | random IQE `d_φ` | — | 6.0 (3/50) | 51.27 | 156.8 | 1.72 |

E2_iql **regresses** vs E1 (half the success rate).

## Gate (design §11 / `03_quasimetric_iql_t3.md` §6)

| Criterion | Result |
|-----------|--------|
| E2_iql > E1 on short_horizon n=20 | **Fail** (10% < 25%) |
| E2_iql does not lose to L2 on offset n=50 | **Fail** (4% < 8%) |

**Verdict: No-go.** Paper-faithful VF_quasi on frozen-LeWM `φ` does not beat L2 CEM under our locked D6 adaptation.

## Cross-campaign context

| Campaign | Protocol | n | Best φ vs E1 |
|----------|----------|---|--------------|
| v2 Euclidean | short_horizon | 20 | E2_v2 35% > E1 25% |
| v3 Euclidean | offset | 50 | no φ beats E1 |
| **IQL T3** | short | 20 | **worse than E1 and E4** |
| **IQL T3** | offset | 50 | **worse than E1** |

## Likely failure modes (for next pivot)

1. **D6 vs Destrade Sep:** paper’s winning VF_quasi trains the *state encoder* with `L_VF`; we only train thin `φ` on detached `z`. Geometry may not be rich enough.
2. **Near-zero `s==g` hits:** with indexed equality and `t < T-1`, the immediate reward is almost always `-1` — sparse goal-identity signal.
3. **Cost scale / landscape:** IQE distances ~11–12 may be flatter for CEM than L2-`z`; random IQE sometimes beats trained φ on short_horizon.
4. **Kinematic bank:** good for IQL theory vs weak policy, but may not stress goal diversity the way paper expert/mixed data does.

## Next options (not executed here)

- Destrade-style **separate pixel encoder** for value (true Sep), keep LeWM for rollouts.
- Increase intentional `s==g` sampling / terminal-as-`s_t` mixes without inventing pose rewards.
- Cost normalization / temperature for CEM with IQE.
- Revisit Euclidean v2 short_horizon signal with different offset pairing — already known not to transfer to offset n=50.
