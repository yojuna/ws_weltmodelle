# Findings — Encoder-floor bracket and pre-retrain confirmation

**Date:** 2026-08-31  
**Question:** Is frozen LeWM’s predictor `P` already wrong at one step (retrain), or only when unrolled (protocol / re-encode)?  
**Verdict:** **CONFIRMED_INFIDELITY** on the live-bank, then **BLOCK_INFIDELITY** on a bank selected for real block motion. Re-derived on a **correct ruler** after the old fork was found to be keyed on the wrong column. One-step `P` is a **wrong-direction map** (not frozen, not deaf) on **pusher and block**. **Part B / CA-train / C1 are not started from this file.**  
**Cuts:** frozen in `le-wm/thresholds.yaml` *before* the GPU runs (`frac ≤ 0.25` ACCUMULATION, `≥ 0.5` INFIDELITY). Not retuned.  
**Artifacts:** `eval_results/pusht/encoder_floor/seed0/encoder_floor.json`, `eval_results/pusht/infidelity_confirm/summary.json`, `eval_results/pusht/block_motion_eval/seed0/summary.json`. Chronicle: [`experiment_log.md`](experiment_log.md). Plan: [`16_fidelity_retrain_plan.md`](16_fidelity_retrain_plan.md) (Part B now names a **block-moving eval bank** and a **small-step tercile** success criterion).

---

## 0. The catch that makes the fork trustworthy

The old CA0 fork rested on **“1.43 vs a round 1.0.”** That 1.43 was never the per-step fidelity quantity. It is mean ‖ẑ_end − z*‖ at `m=1` — distance to the **goal** encoding after teacher-forcing. On this live-bank the goal image *is* the last path frame, so 1.43 equals mean **last-frame** one-step error, not the median over all steps.

The fork should have been keyed on **median one-step error over all t ≥ history**, bracketed between two real reference predictors (encoder floor, identity). Replacing a near-miss against an arbitrary constant with that ratio is not a refinement: it is noticing the original verdict was computed on a subtly wrong column and rebuilding it. Same-state re-encode **0** and fresh-`P` matching the dump mean the new ruler is sound *before* you read `frac`. This is the same “verify the instrument” discipline that caught the packing bug and the weak oracle, applied to the number a retrain depended on.

---

## 1. What we were measuring

The world model maps a camera frame to a 192-d vector **z** (encoder **E**) and guesses the next **z** from recent **z** plus actions (predictor **P**). Distances below are L2 in that raw embedding (`encode()['emb'][:,0]`, no SIGReg reshape).

**`frac`** is the one number the fork uses:

```
frac = (one-step error − encoder floor) / (adjacent true-z − encoder floor)
```

- **One-step error:** ‖P(true 3-frame history, true action) − encode(next frame)‖, teacher-forced (`m=1`).
- **Encoder floor:** encode the *same* pixels twice. Here **0**.
- **Adjacent true-z:** ‖encode(o_t) − encode(o_{t+1})‖ — error of the trivial predictor “nothing moved.”

`frac → 0` means per-step faithful (do not retrain). `frac → 1` means `P` barely beats identity. `frac > 1` means `P` is *worse* than identity.

CA0’s old **1.43** is a different column (see §0). The fork uses **`frac`**, not 1.43 vs 1.0.

---

## 2. The ruler is working

| Check | Result |
|-------|--------|
| Same-state re-encode (32 frames, two forwards, batch-position) | **0.0** |
| Dump z vs a fresh encode of the same frames | rel **0.0** |
| Dump `d_end` m=1 vs CA0 `summary.json` | **1.426** matches |
| Fresh `P` m=1 vs dump ẑ (50 pairs) | rel **0.0** |

Calibration passed. Later distances are not inflated by a stochastic encoder or a units mismatch.

---

## 3. Seed-0 live-bank bracket (the original fork, re-decided)

50 GoalPush+Weak short-horizon windows, 1150 teacher-forced steps.

| Quantity | Value |
|----------|-------|
| Encoder floor | 0.0 |
| One-step median | **1.21** |
| Adjacent true-z median (identity) | **1.23** |
| Random-pair spread (latent diameter) | 18.7 |
| CA0 mean `d_end` m=1 (not used in `frac`) | 1.43 |
| **frac** | **0.983** |

Frozen cut `≥ 0.5` → **INFIDELITY**. Per-step `frac` median 1.02 (p10 0.51, p90 2.66): on a typical step `P` is identity-level, sometimes worse. The latent has scale (spread 18.7); `P` is not using it for one-step motion.

---

## 4. What we tried to overturn — and did not

Pre-registered: a later seed `frac ≤ 0.25`, pose-Δ vs ‖Δz‖ Spearman `< 0.30`, or random-action `frac ≤ 0.25` would have stopped or narrowed a retrain. None of those fired.

| Arm | n | frac | Reading |
|-----|---|------|---------|
| Live-bank seed 0 | 50 pairs / 1150 steps | **0.99** | INFIDELITY |
| Live-bank seed 1 | 50 / 1150 | **1.09** | INFIDELITY |
| Live-bank seed 2 | 50 / 1150 | **0.99** | INFIDELITY |
| Random `env.step` | 80 segs / 1840 steps | **1.40** | INFIDELITY (worse than identity) |

**Not a seed-0 fluke. Not GoalPush-window-only.** Random-action is *more* broken than the live-bank (one-step 1.33 vs adjacent 0.95).

---

## 5. Mechanism: a wrong-direction map (the load-bearing diagnosis)

The boring explanations are false. `P` is neither stuck nor deaf. The fault is a **structured wrong map**: motion of roughly the right magnitude, action-sensitive, pointing Δz about **halfway to orthogonal**. That is more hopeful than “the predictor is broken” — it is the kind of error more training signal can correct — and it is what A5 was groping at (pair-0 PCA-plane 73.9° was a shadow; **full-space bank ~41–46°** is the citable scalar). The two instruments agreeing once A5 is honest is corroboration, not a new claim.

On the seed-0 live-bank:

| Probe | Result | Meaning |
|-------|--------|---------|
| Predicted-move / adjacent | **1.40** | `P` *does* emit motion (not frozen at z_t) |
| Mean / median full-space angle | **46° / 41°** | Direction systematically off (isotropic would be ~90°) |
| Shuffle actions vs true | gap / adjacent **0.58** (one-step 1.92 vs 1.21) | `P` hears actions |
| Zero actions vs true | gap / adjacent **0.22** (one-step 1.48 vs 1.21) | True actions help a little; still infidelity |
| Pose Spearman (‖Δz‖ vs agent+block xy) | **0.67** (cut 0.30) | Adjacent z tracks **agent** motion, not encoder jitter |
| Block xy step median | **0** | These short windows barely move the block; the latent step is mostly the **pusher** |

Random-action bank: same pattern — hears shuffles (gap 0.56), still moves (ratio 1.69), angle ~53°, large-step tercile **0.70**.

### 5.1 Confound — what fidelity did we actually measure?

`frac ≈ 1` on these windows is **`P` mis-predicting the pusher’s own motion** (the thing most directly determined by the action). The block barely moves (per-step block-xy median **0**); Spearman 0.67 is z-tracks-*agent*. That is arguably *worse* than a block-only fault (if it cannot predict the actuated pusher, block dynamics are hopeless), but it is a **different claim** than “`P` cannot predict the task.” PushT *is* the block.

**As-run (2026-08-31):** block-moving bank n=50, median per-step ‖Δblock_xy‖ cut **2.0** (frozen), mean pair median **7.34 px**, per-step block-xy median **5.26**. Bank `frac` **0.887**, angle **~51°**, Spearman vs block **0.43**. Decision **BLOCK_INFIDELITY** — the claim strengthens to pusher **and** block. Spec: [`16`](16_fidelity_retrain_plan.md) B.4a / B.eval-block.

### 5.2 Terciles — infidelity throughout; worst on small steps

| True-step tercile | frac (seed-0 live-bank) |
|-------------------|-------------------------|
| Small | **2.17** |
| Mid | **0.96** |
| Large | **0.66** |

Even large steps clear the 0.5 cut, so it is infidelity throughout. The *shape* is the B-relevant fact: `P` is relatively **least** bad on large motions and **most** bad on small ones. Small steps are where identity is a strong baseline and where directional error swamps a tiny true Δz. That is exactly the regime that **accumulates** over a rollout. An average-only `frac` drop could hide it. **Named B target:** small-step tercile, not just the bank median. Spec: [`16`](16_fidelity_retrain_plan.md) B.4a / B.eval-tercile.

**As-run on the block-moving bank (frozen live-bank edges, not re-fit):** small **1.70** (n=100) / mid **1.09** (n=453) / large **0.72** (n=597). Same shape; small-step is still worse than the bank median (0.89). Mass shifts into the large tercile because these windows actually move.

Strata (still infidelity where defined): free **0.97** (n=1000), contact **0.82** (n=104). Wall has adjacent median 0 (block parked) so `frac` is undefined there.

---

## 6. How this sits on the earlier campaign

- **C0 Outcome B** (replay works, imagination does not) is the parent. This investigation localizes that failure to **one-step `P`**, not only to long unrolls.
- **CA0 m=25** end-dist 8.23 / 2% toward-goal is the open-loop symptom. Teacher-forcing does not rescue per-step fidelity (`frac ≈ 1`).
- **CA2** hard elbow (~22 / 192-d, 58% dead) still says **do not scale the token.**
- **D6** (no reachability grads into the encoder) is unchanged. A later retrain is prediction/fidelity, not a cost-head or `φ` flip.
- **C1** (actor) stays gated: scoring imagined futures through this `P` is still scoring a lie.

---

## 7. What this does *not* say

- It does not say the encoder is useless: same-state noise is 0, pose is linearly readable (B1/C0.1), adjacent z tracks agent motion.
- It does not say `P` ignores actions. Shuffle and zero both hurt.
- It does not say only walls or only contact are broken.
- It does not start a retrain. **CONFIRMED_INFIDELITY + BLOCK_INFIDELITY gate Part B; they do not execute it.** B still needs a matched 1-step baseline vs multi-step objective, stop-grad targets, frozen `φ` (D6), no scaling — and must still **pass** on the block-moving bank and the small-step tercile, not pusher-only bank-median `frac`.

**Compounding (descriptive, corroboration not a gate):** naive linear ~4.9 and √T ~2.2 over CEM T=5 both exceed one-step ~1.2 and **bracket** the observed 25-step open-loop **8.23**. That is a consistency check that the one-step wrong map is the **parent** of the long-unroll symptom, not a separate problem — so fixing one-step fidelity is the right lever, not a local patch.
