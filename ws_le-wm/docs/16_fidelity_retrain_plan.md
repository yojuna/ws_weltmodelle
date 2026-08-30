# Plan 16 — Validate the Encoder Floor, then Multi-Step Fidelity Retrain (prove on PushT)

**Date:** 2026-08-31
**Spec lineage:** gates and executes the **CA-train** branch that [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md) said CA0-INFIDELITY would motivate; uses the instrument in [`15_viz_toolkit_spec.md`](15_viz_toolkit_spec.md).
**Inherits:** posture + stability rules in `00_decisions.md`; verdicts in `13`/CA-summary.
**Stack:** `le-wm/` on stable-worldmodel; live eval; reuse `phase_b_dump/`, `c0_oracle_livebank/`, `ca0_closed_loop/`.
**Change control:** unchanged. Part B is a **retrain** — a large move — so every design choice is recorded, and Part B does not start until Part A's gate reports. Pure-JEPA discipline (D6, no reward latents, no reconstruction, no scaling) holds throughout.

---

## 0. Orientation — why two parts and a hard gate

CA0 returned **INFIDELITY**, but the verdict rests on one near-miss: **m=1 mean ‖ẑ_end − z*‖ ≈ 1.43** against a **round-number** guard of **1.0**. That 1.43 is **distance-to-goal** after teacher-forcing, not the one-step residual ‖P(true history, a) − z_{t+1}‖ (A2 median ≈ 1.21). Before spending a retrain, we must (A) prove the *ruler* is correct and re-decide the fork on a **bracket of the one-step residual**; then, only if it still reads PARTIAL or INFIDELITY, (B) retrain for multi-step fidelity and prove it on PushT.

The rigor upgrade over "1.43 vs 1.0": **bracket the one-step error between two reference predictors** —
- a **perfect** predictor (lower bound = the encoder's own noise floor), and
- a **trivial identity** predictor (upper bound = how far one real step moves).

Where the **one-step residual** sits in that bracket is what decides the fork, not a hand-picked constant and not `d_end`. Keep the two columns: fork metric = mean `d_end` at `m=1`; bracket metric = one-step.

---

# PART A — Encoder-floor validation (GATE ZERO)

Two sub-goals, kept separate: **A-calibrate** (is the ruler correct?) and **A-decide** (where does the **one-step residual** fall on the correct ruler?). Do not put 1.43 in the bracket.

## A-calibrate — is the measurement correct?

Before trusting any distance, confirm the instrument:

**A.c1 — Encoder determinism at eval.** Encode the *same* observation twice (eval mode: dropout off, augmentation off, deterministic ops, same device/dtype) → distance should be ≈0. Repeat across batch positions and two forward passes. *If nonzero,* the encoder is stochastic at eval and every distance in the program is inflated by this — fix before proceeding. This is also the **perfect-predictor lower bound** (A.d1).

**A.c2 — Units / space consistency.** Confirm the single-step residual, the CA0 `m=1` number, and all floor scales are measured in the **same latent space** with the **same normalization** (raw projector output vs any L2/SIGReg-shaped space). A units mismatch makes the bracket meaningless. Record exactly which tensor (`info["emb"]`) and which norm.

**A.c3 — Instrument cross-check (two columns, not one).**
1. **Fork metric.** Recompute mean `d_end` at `m=1` from `ca0.npz` and confirm it matches `summary.json` (~1.43). Also record mean ‖z_true[-1] − z*‖ (true last frame vs goal). This is what CA0 forked on.
2. **Bracket metric.** ‖P(true history 3, true actions) − z_{t+1}‖ for t ≥ HISTORY. From the dump: `z_hat[:, m=1]` vs `z_true`. Fresh GPU: `imagine_closed_loop(..., m=1)` with `oracle_actions` from the **live-bank** (same padding as `closed_loop_imagine._pad_frames_actions`). `ca0.npz` has no actions; do **not** use kinematic `--dump phase_b_dump`.

These two numbers are **not** required to match. `frac` uses the bracket metric only. *If dump `d_end` disagrees with `summary.json`, or fresh P disagrees with dump `ẑ`,* stop (CALIB_FAIL). *If one-step ≠ 1.43,* that is expected — do not treat it as a failed xcheck.

**A-calibrate passes** only when: determinism floor ≈ 0 (or its value is recorded and subtracted), units are consistent, dump `d_end` reproduces the reported 1.43, and fresh P reproduces dump one-step.

## A-decide — where does the one-step residual fall on the correct ruler?

Compute, as **distributions over the bank** (not a single pair — the report's A1 single-pair issue applies here too):

| Reference | Quantity | Role |
|---|---|---|
| **Perfect-predictor floor** | ‖encode(o) − encode(o)‖ (A.c1) | lower bound: best any predictor can do |
| **Single-step error** | ‖P(true history, a_t) − encode(o_{t+1})‖ | the number under test (**not** 1.43) |
| **Adjacent-step size** | ‖encode(o_t) − encode(o_{t+1})‖ | how far one real step moves |
| **Null (identity) predictor** | error if P outputs its input = adjacent-step size | upper bound: trivial predictor |
| **Random-pair spread** | ‖encode(o_i) − encode(o_j)‖ random | overall latent scale (context) |

**The decisive scalar is the bracket position:**
```
frac = (single_step_error − perfect_floor) / (null_predictor − perfect_floor)
```
- `frac → 0` (error near the encoder floor) → per-step prediction is as faithful as the representation allows → **ACCUMULATION**.
- `frac → 1` (error near identity) → predictor barely beats "predict no movement" → **INFIDELITY** (and of a specific, damning kind: near-trivial).
- `frac` in between → **partially faithful**: predictor captures real dynamics but with residual larger than noise; retrain is justified as *tightening*, not *fixing-broken* — still gate B, but framed honestly.

**A.d-extra — is the residual big enough to matter for planning?** Even a "faithful-per-step" model can fail 25-step planning if per-step error compounds. So also report `single_step_error / adjacent_step_size` (per-step relative error) and project its compounding over the planning horizon. This distinguishes "per-step fine, protocol fix suffices" from "per-step small but compounds past the planning budget."

## A — recalibrated guard (pre-register BEFORE re-reading the fork)

Replace the 1.0 round number with a **bracket-based** threshold, fixed in `thresholds.yaml` before looking at the result:
- `frac ≤ 0.25` → ACCUMULATION (do **not** retrain; re-open C1 with a closed-loop scorer, per `14`).
- `frac ≥ 0.5` → INFIDELITY (proceed to Part B).
- `0.25 < frac < 0.5` → partial: proceed to Part B as *tightening*, with the honest framing recorded.
- Any A-calibrate failure → stop; fix the instrument; re-run.

## A — outcomes & what they gate

| Outcome | Meaning | Gate |
|---|---|---|
| A-calibrate fails | ruler wrong (stochastic encoder / units / metric mismatch) | Fix instrument; re-run. **Nothing downstream trusted.** |
| ACCUMULATION (`frac ≤ 0.25`) | per-step faithful; drift accumulates | **Do NOT retrain.** Protocol fix + re-open C1 (`14` CA0-ACCUMULATION branch). Major save. |
| PARTIAL (`0.25 < frac < 0.5`) or INFIDELITY (`frac ≥ 0.5`) | per-step residual real | **Proceed to Part B.** |

## A — commands

Cuts are frozen in `le-wm/thresholds.yaml` (`encoder_floor`) **before** the GPU run. Do not retune `0.25` / `0.5` after seeing `frac`.

```bash
# A-calibrate + A-decide. Live-bank + CA0 dump — not kinematic phase_b_dump.
python scripts/encoder_floor.py \
  --ca0 eval_results/pusht/ca0_closed_loop/seed0 \
  --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0 \
  --out eval_results/pusht/encoder_floor/seed0 \
  --device cuda
# emits: perfect_floor, one-step {mean,median,dist}, adjacent, null, random-pair,
#        frac, xcheck_reproduces_d_end, xcheck_onestep_vs_d_end, dump_vs_fresh_P,
#        guard_decision {ACCUMULATION|PARTIAL|INFIDELITY|CALIB_FAIL}
```

Record the outcome in `experiment_log.md`. **Part B starts only on PARTIAL or INFIDELITY.**

## A-confirm — thorough checks before spending the retrain (same cuts)

Seed-0 live-bank `frac ≈ 0.98` can still be a slow GoalPush window, a seed, or an action-mix. Frozen in `thresholds.yaml` `infidelity_confirm` **before** this suite (same 0.25 / 0.5; do not retune):

| Finding | Decision |
|---|---|
| Any extra seed `frac ≤ 0.25` | **SEED_FLUKE** — do not start Part B |
| Pose-Δ vs ‖Δz‖ Spearman `< 0.30` | **ENCODER_JITTER** — do not start Part B |
| Live-bank INFIDELITY and random-action `frac ≤ 0.25` | **BANK_SPECIFIC** — Part B data must include on-policy windows |
| Live-bank + extra seeds + random-action all `frac ≥ 0.5` | **CONFIRMED_INFIDELITY** |

Also report (not new gates): predicted-move / adjacent (frozen identity if `< 0.25`); live-bank action shuffle and zero-action; contact/wall/large-step terciles.

```bash
python scripts/infidelity_confirm.py \
  --ca0 eval_results/pusht/ca0_closed_loop/seed0 \
  --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0 \
  --out eval_results/pusht/infidelity_confirm \
  --seeds 1 2 --device cuda
```

**Part B still does not start from this file** until A-confirm reports and someone explicitly scopes the retrain.

**A-confirm as-run (2026-08-31):** **CONFIRMED_INFIDELITY** (live-bank seeds 0–2 `frac` 0.99 / 1.09 / 0.99; random-action `frac` 1.40). Mechanism: wrong-direction map (~41–46°), not frozen/deaf; measured mostly on **pusher** motion (block-xy step median 0). Findings: [`16a_infidelity_investigation.md`](16a_infidelity_investigation.md). Two eval rules below were folded into B from that mechanism **before any retrain**.

**B.eval-block / B.eval-tercile as a pre-retrain run** (not launching B): freeze `b_eval_block.median_step_block_xy_min` and `b_eval_tercile.adj_q33/q67` in `thresholds.yaml`, then

```bash
python scripts/block_motion_eval.py \
  --ca0 eval_results/pusht/ca0_closed_loop/seed0 \
  --out eval_results/pusht/block_motion_eval/seed0 \
  --device cuda
```

**B.eval as-run (2026-08-31):** **BLOCK_INFIDELITY**. GoalPush 80 eps → 237 windows ≥ 2 px/step median; n=50 used; mean pair block-step median **7.34**. Bank `frac` **0.887**, small-step tercile **1.70** (frozen edges), angle **~51°**, block-xy step median **5.26**. Cuts were not retuned. Artifact: `eval_results/pusht/block_motion_eval/seed0/summary.json`. **Still does not launch Part B.**

---

# PART B — Multi-step fidelity retrain (CONDITIONAL on Part A)

**Do not start unless A = PARTIAL or INFIDELITY, and A-confirm is not SEED_FLUKE / ENCODER_JITTER.** As-run: CONFIRMED_INFIDELITY. This is Step 1 of the endgame path (fidelity → PushT → multi-task → few-shot); scope here is **PushT only**. **This section records the retrain design; it does not launch it.**

A-confirm localized the fault: `P` emits motion and hears actions, but Δz is **~41–46° off** in full space (wrong map), **worst on small steps** (`frac` 2.17 / 0.96 / 0.66), and the live-bank `frac` is **mostly pusher motion**. B’s evaluation must therefore include a **block-moving bank** and a **small-step tercile** named target — or “fidelity fixed” could pass on the wrong object and the wrong part of the step-size distribution.

## B.0 — Why this specific move

Multi-step rollout loss is the **one lever that directly targets multi-step fidelity**, it is **impossible on frozen weights**, it **de-confounds** ("this checkpoint drifts" → "1-step-trained JEPA drifts; multi-step loss does/doesn't fix it"), and it is the **on-ramp** to the representation-first / multi-task arc. It changes the *training objective*, not the architecture or the purity.

The compounding aside (descriptive in A) is corroboration that this is the right lever: naive linear ~4.9 and √T ~2.2 over CEM T=5 both exceed one-step ~1.2 and **bracket** CA0’s 25-step **8.23** — the long-unroll symptom is the child of the one-step wrong map, not a separate disease.

## B.1 — Objective (pure JEPA, asymmetry-aware)

```
L = L_pred(1-step)  +  λ_R · L_rollout(n-step)  +  λ_S · L_SIGReg
```
- **L_rollout(n-step):** unroll `P` for `n` steps from a true-encoded start under true actions; match each predicted latent to the **stop-gradient** encoding of the true future frame. Targets detached — gradients flow through the predictor and the *rolling* encoder path, never into the target encoder. This is the LeVLJEPA/asymmetry lesson: an asymmetric predict-with-stop-grad avoids the symmetric-collapse trap that a naive symmetric multi-step MSE would invite.
- **SIGReg unchanged** on the encoder marginal (anti-collapse). It is structurally separate from L_rollout (different locus), so this is *not* the forbidden "stack SIGReg on a value loss" — confirm no rank collapse via effective-rank monitoring (`viz` D1) during training.
- **D6 held:** no reachability/reward gradients into the trunk. Pure prediction/fidelity training.
- **No scaling:** CA2 showed effective rank ≈22/192 with a hard elbow and 58% dead dims — capacity is not the constraint on PushT. Keep `embed_dim=192`, `history=3`; change only the objective. (Token utilization is a *multi-task* goal, not a PushT one — deferred to the next phase.)

## B.2 — Design knobs (record choices)

- **Rollout horizon `n`:** sweep `n ∈ {3, 5, 8}`. `n=5` matches the CEM planning horizon; longer tests whether training-time horizon must exceed planning horizon to be faithful at it.
- **Per-step weighting:** compare uniform vs later-step-weighted (drift shows late, so up-weighting late steps may target the fault; but may destabilize early prediction — ablate).
- **Stop-grad on targets:** on by default (B.1). Ablate off to *confirm* asymmetry matters (expect off → collapse/worse, per LeVLJEPA).
- **Data:** live PushT rollouts (no HDF5 in tree; `data_source.md`). Small high-quality set is acceptable (FF-JEPA showed 40× data cuts hold with a tight success filter). **Must include windows with real block motion**, not only the current short_horizon live-bank (per-step block-xy median 0). Mixing on-policy GoalPush/Weak with random-action is allowed; pusher-only hops are not sufficient as the *sole* train or eval set.

## B.3 — De-confounding (the publishable comparison)

Train, on **identical data / architecture / budget**, two models:
- **B-baseline:** 1-step objective only (reproduces the frozen checkpoint's training regime, but as *our* run).
- **B-multistep:** the n-step objective above.

This turns any result into an architecture-level claim: the difference between them isolates *the objective*, not the checkpoint. Without B-baseline, "multi-step is better" is confounded by data/seed/pipeline.

## B.4 — Success = fidelity fixed, THEN PushT solved

Two gates, in order, both measured with the `viz`/report instrument (pre/post overlays). Cuts go in `thresholds.yaml` **before** training, not after.

**B.4a — Fidelity fixed (primary):**
- CA0 drift curve **flattens**: single-step error `frac` (Part A metric) drops toward the encoder floor; multi-step ‖ẑ_end−z*‖ and toward-goal recover **without** teacher-forcing.
- A5 **full-space** action-direction angle **shrinks** (the wrong-map scalar; ~41–46° pre).
- **Wall-band drift** (the localized weak spot, ~1.5× free) improves — named success criterion, since a fix that leaves walls broken isn't a fix for hard PushT.
- Effective rank does **not** collapse (D1 monitor) — fidelity gained without killing the representation.

**B.eval-block — Block-moving fidelity bank (named, not optional).** The A-confirm live-bank has per-step block-xy median **0**; `frac` there is pusher motion. PushT is the block. Collect (or window) a bank whose windows are selected for **non-trivial block displacement** over the horizon. Freeze the selection cut in `thresholds.yaml` *before* collecting (e.g. window-sum or median per-step ‖Δblock_xy‖ above a pre-registered floor — not retuned after seeing post-train `frac`). Report `frac` and full-space angle on that bank **separately** from the pusher-heavy live-bank. **Fidelity-fixed does not pass on pusher-only `frac`.** If block-moving windows stay infidelity while pusher `frac` drops, the retrain did not fix the task.

**B.eval-tercile — Step-size stratification (small-step named target).** Pre `frac` is worst on small true steps (seed-0: small 2.17, mid 0.96, large 0.66). Small steps are where identity is a strong baseline and where directional error dominates; they are what **accumulates** over a rollout. Report `frac` in the same adjacent-‖Δz‖ terciles as A-confirm (tercile edges frozen from the **pre** bank, not re-fit post). **Fidelity-fixed requires the small-step tercile to improve**, not only the bank-median. An average-only win that leaves small-step `frac` near 2 would not be expected to flatten drift.

**B.4b — PushT solved (the proof):** on a **faithful** B-multistep model (B.4a **and** B.eval-block **and** B.eval-tercile), run the **light asymmetric propose-and-score actor** (the re-opened C1 — now justified because scoring runs through a model that no longer lies) on the **hard-offset** band that has been the standing claim. Pre-register the success bar. If a faithful model + light actor still fails hard offset, *then* hard geometry is the genuine residual — but we can only ask that cleanly once fidelity holds on pusher, block, and small steps.

## B.5 — Ablations (defend the claim)

- `n` sweep (3/5/8): does training horizon need to meet/exceed planning horizon?
- multistep vs 1-step baseline (B.3): the core claim.
- stop-grad on/off: confirms the asymmetry is load-bearing.
- uniform vs late-weighted rollout loss.

## B — pre-registered criteria (in `thresholds.yaml` **before** training)

Numeric bars are **not** filled in from A-confirm after the fact. Freeze them in yaml immediately before the B run. Structure (required):

- **Fidelity-fixed:**
  - bank-median `frac` post ≤ agreed fraction of `frac` pre, on the **live-bank** *and* on the **block-moving bank**;
  - **small-step tercile** `frac` post ≤ agreed fraction of that tercile’s `frac` pre (tercile edges frozen from pre);
  - full-space A5 angle shrinks to an agreed bar;
  - untethered multi-step toward-goal ≥ agreed bar;
  - wall-band drift ≤ agreed bar.
- **PushT-solved:** hard-offset success ≥ agreed bar with the light actor.

A run that only quotes live-bank median `frac` is an incomplete B.4a.

---

## Decision table & triggers

| ID | State | Status | Trigger |
|---|---|---|---|
| **A-calibrate** | pass (floor 0; dump = fresh P) | **Done.** | Fail → fix instrument, halt. |
| **A-decide + A-confirm** | **CONFIRMED_INFIDELITY** | **Done.** Fork re-derived on `frac`, not 1.43 vs 1.0. | ACCUMULATION would have stopped retrain. |
| **B.eval-block / B.eval-tercile** | **BLOCK_INFIDELITY** | **Done (pre-retrain).** Bank `frac` 0.887; small-step 1.70. Claim is pusher **and** block. | Do not retune cuts. |
| **CA-train (retrain)** | Not started | **Gated on CONFIRMED_INFIDELITY + BLOCK_INFIDELITY; eval must include both banks + small-step tercile.** | Explicit B launch, not this file. |
| **C1 actor** | Gated off | **Re-opens** either via A-ACCUMULATION (closed-loop scorer) or B.4b (faithful-model scorer). | Whichever fires. |
| **Scaling (token/model)** | Off | **Stays off.** | CA2 verdict; only a multi-task rank climb reopens it (future). |
| **D6 / purity** | Keep | **Unchanged.** | n/a |

---

## Risks

| Tier | Risk | Cheap guard |
|---|---|---|
| 1 | Retrain launched on a fork that was really ACCUMULATION | **Part A bracket before Part B** (the whole point) |
| 1 | Ruler wrong (stochastic encoder / units / metric) inflates distances | **A-calibrate** determinism + units + two-column xcheck (d_end vs one-step) |
| 2 | Multi-step loss collapses the encoder (symmetric-collapse) | Stop-grad targets + effective-rank monitor (D1) |
| 2 | "Better" is confounded by data/seed/pipeline | **B-baseline** matched 1-step run |
| 2 | Fidelity fixed globally but wall-band still broken | Wall-band as a **named** success criterion |
| 2 | Fidelity “fixed” on pusher-only short hops; block dynamics still lie | **B.eval-block** — block-moving bank required for B.4a |
| 2 | Bank-median `frac` drops; small-step tercile stays ~2 and rollouts still drift | **B.eval-tercile** — small-step named target; tercile edges frozen pre |
| 3 | Fidelity fixed but hard-offset still fails | That is then the *clean* geometry result; acceptable, recorded |
| 3 | Token still ~22-rank after retrain | Expected on PushT; token fills only multi-task (next phase) |

---

## Success criteria (overall)

- **Part A done** = calibration checks pass (or instrument fixed), bracket `frac` computed as a distribution, guard recalibrated and pre-registered, **fork re-decided with reason recorded**.
- **Part B done (if reached)** = B-baseline vs B-multistep trained matched; fidelity-fixed evaluated on live-bank **and** block-moving bank **and** small-step tercile; if passed, hard-offset PushT attempted with the light actor and result recorded.
- **Phase pass** = either (a) A flips to ACCUMULATION and we save the retrain (re-open C1 closed-loop instead), or (b) A confirms infidelity, B-multistep demonstrably improves rollout fidelity over the matched baseline, and we can state whether a faithful pure-JEPA model solves hard PushT.
- **Non-goals:** multi-task, token utilization, few-shot task config, real robots — all deferred to the next phase, which this fidelity result gates.

---

## Immediate actions

1. **Part A first** — done: ruler calibrated, `frac` on the correct column, A-confirm **CONFIRMED_INFIDELITY**. Log + [`16a`](16a_infidelity_investigation.md).
2. **Before launching B** — freeze B.4 numeric bars in `thresholds.yaml`; collect/specify the **block-moving** eval bank; wire small-step tercile reporting with **pre**-frozen edges.
3. **Only then** — matched B-baseline + B-multistep, `n`-sweep, stop-grad ablation; `viz` pre/post drift **and** A5 full-space angle **and** tercile/`block` `frac`.
4. Keep purity invariants asserted in code: no reach/reward grads into trunk (D6), token size fixed, SIGReg retained, targets stop-grad.

**Part A did its job.** The fork is INFIDELITY on a validated ruler, with a wrong-direction map as the mechanism. Part B is **earned, not launched**. The two eval rules (block motion, small-step tercile) exist so the retrain is judged on what PushT needs, not on pusher-only average `frac`.
