# Euclidean φ multi-seed replicate

**Date opened:** 2026-08-28  
**Purpose:** Get enough planning evidence to tell whether the v2 short_horizon win was real, and whether Euclidean `φ` ever helps on the firm offset gate — **before** opening Sep / more IQL.

**Frozen weights (no retrain):** `stablewm/checkpoints/pusht/lewm_phi_v2/reach.pt`  
(Euclidean hindsight-`k`, weak 24k-step bank; see `lewm_phi_v2_summary.md`)

---

## Question

Under locked D6 (frozen LeWM + thin Euclidean `φ`):

1. Does **E2 > E1** on **short_horizon** replicate across seeds?
2. Does **E2** beat or match **E1** on **offset** across seeds?
3. Does **E2 > E4** (trained beats random)?

If (1) fails → the only positive signal was likely noise.  
If (1) holds and (2) fails → protocol/transfer problem, not “φ never works.”  
If both fail → Euclidean-on-frozen-`z` is weak evidence against the Phase A claim.

---

## Protocol (matched)

| Knob | Value |
|------|--------|
| Trunk | `hf_pusht` frozen |
| Seeds | **0, 1, 2** |
| Conditions | **E1** `l2_z` · **E2** `phi_d` + v2 weights · **E4** `phi_d` random |
| Short | `pair_mode=short_horizon`, **n=20**, kin bank `--collect-episodes 320` |
| Offset | `pair_mode=offset`, **n=50**, kin bank `--collect-episodes 200` |
| Other | C1 goal cache on, `--no-video`, `device=cuda` |

Each `(seed, protocol, condition)` is a separate `eval_live.py` run. Bank collection uses that seed.

**Artifacts root:** `le-wm/eval_results/pusht/lewm_phi_euclid_multiseed/`

```text
lewm_phi_euclid_multiseed/
  campaign_console.log
  aggregate.json          # filled by scripts/aggregate_euclid_multiseed.py
  short/seed{0,1,2}/{E1_l2_c1,E2_phi_v2,E4_random}/...
  offset/seed{0,1,2}/{E1_l2_c1,E2_phi_v2,E4_random}/...
```

**Runner:** `le-wm/scripts/run_euclid_multiseed.sh`  
**Aggregator:** `le-wm/scripts/aggregate_euclid_multiseed.py`

---

## Go / no-go readout (after aggregate)

| Check | Pass if |
|-------|---------|
| Short replicate | mean_seed success(E2) > mean_seed success(E1) |
| Short vs random | mean_seed success(E2) ≥ mean_seed success(E4) |
| Offset transfer | mean_seed success(E2) ≥ mean_seed success(E1) |

Report **per-seed** tables plus **mean ±** (sample std over 3 seeds). With n=20/50, treat gaps &lt;~10pp as fragile.

---

## Results

See [`lewm_phi_euclid_multiseed_summary.md`](lewm_phi_euclid_multiseed_summary.md) (seeds 0–2 complete).

| Check | Outcome |
|-------|---------|
| Short E2 > E1 | **PASS** (36.7±7.6% vs 16.7±7.6%) |
| Short E2 ≥ E4 | **PASS** (36.7% vs 25.0%) |
| Offset E2 ≥ E1 | **PASS but weak** (6.7±1.2% vs 4.7±3.1%; random 7.3% still highest) |