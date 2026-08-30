# Phase C Design Spec & Plan — Confirm the Bottleneck, then Build the Actor

**Date:** 2026-08-30  
**Status (2026-08-30):** C0 **ran**. Gate resolved to **Outcome B** (model fidelity) after the live-bank redo — see [`13_phase_c0_report.md`](13_phase_c0_report.md) and [`12a_c03_redo.md`](12a_c03_redo.md). **Do not build C1.** Current plan: [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). This file remains the C0 spec and the (gated-off) C1/C2/C3 design.

**Supersedes for planning (historical):** the B3/B4 forward-look in `09_phase_b_plan.md §5` and `11_phase_b_report.md`'s "B3 one row."  
**Inherits:** posture and stability rules in `00_decisions.md`; diagnostics + verdicts in `09`–`11`.  
**Stack:** `le-wm/` on stable-worldmodel + stable-pretraining; live eval for PushT and Reacher.  
**Change control:** unchanged from `00_decisions.md` — deferred items flip only on *observed* failure, recorded with date + reason. D7 (asymmetry) was proposed here; it stays proposed, not locked in `00`.  
**C0.3-redo:** [`12a_c03_redo.md`](12a_c03_redo.md) (2026-08-30) — live-bank oracle; C0.3-as-run was underpowered.

---

## 0. One-paragraph orientation

Three phases have now **falsified, in order**, three bottleneck hypotheses: the cost head (Phase A), representational entanglement (Phase B, B1: frozen `z` linearly exposes block pose and `P` responds to it), and predictor-rollout horizon (Phase B, B2: shortening CEM `T` does not lift offset). What remains standing, by elimination, is the **actor/search** problem — producing actions that reach hard PushT configurations. That convergent narrowing is the strongest result in the program and is the spine of the eventual writeup. **But** the Phase-B verdict rests on (i) a single seed and (ii) an action-liveness number measured only on the *kinematic* bank, where `P` barely responds to the action (shuffle−true gap 0.002). Because "action-insensitive predictor" would masquerade as "hard geometry," Phase C **does not build the actor yet.** It opens with a cheap **confirmation gate (C0)** that either clears the actor build or redirects to predictor work. LeVLJEPA folds in as an effective-rank test for the flat-`φ` landscape and an **asymmetry principle** for anything we build.

---

## 1. Where we stand — the convergent narrowing

| Hypothesis | Phase | Status | Load-bearing evidence |
|---|---|---|---|
| Cost head is the bottleneck | A | **Falsified** | Random `φ` ≈ trained; offset flat ~2–10% for all heads |
| Entanglement (info not linearly usable) | B1 | **Falsified (seed 0)** | Block pose linear R² 0.90/0.78; intervention on `block_x` moves `P`, hit 1.0 |
| Rollout-horizon OOD | B2 | **Falsified** | CEM `T`∈{2,3,5,8} flat at 4–6% offset; shorter imagination does not help |
| **Actor/search binds** | — | **Standing (by elimination)** | Nothing else survives; needs *positive* confirmation (C0) |

D6 recommendation from `11`: **keep / extract** — control-relevant factors are linearly readable and causally live, so the trunk need not be reshaped. Phase C holds this as the working decision and records the confirmations that would still overturn it.

---

## 2. What LeVLJEPA adds (mechanistic, not directional)

LeVLJEPA is vision-language pretraining — no world model, planning, or actions — so most of it does not transfer. One result does, and two tools come with it.

**The lesson — symmetric embedding-regression collapses onto the common-information subspace.** They found a symmetric MSE between image and text embeddings (even with per-modality SIGReg) collapses: effective rank drops, both distributions meet on a shared low-dim subspace capturing only *common* (coarse) information. The fix is **asymmetry** — a predictor with a stop-gradient target, so each encoder is shaped by its own branch and SIGReg is not fighting a symmetric term.

Two consequences for us:

- **H2 (flat `φ` CEM landscape) may be symmetric-collapse, not just conditioning.** Our `φ` is trained symmetrically: `d(φ(z_t), φ(z_{t+k}))` regressed to scalar `k`, same `φ` both arms. The LeVLJEPA mechanism predicts exactly the low-dynamic-range surface we saw (cand. std φ≈0.48 vs L2≈6.3). **Testable:** effective rank of `u=φ(z)` vs `z` (C0.4). Cheap, reuses dumps.
- **Scalar-`k` is an informationally thin target.** Captions under-specify fine visual structure; a single scalar `k` likewise under-specifies the fine *pose geometry* hard offset goals need. This is a mechanistic account of why scalar-`k` reachability has an intrinsic low ceiling — independent of frozen-vs-reshape.

**The asymmetry theme is now triple-confirmed:** Destrade (quasimetric asymmetric value > Euclidean), PiJEPA (asymmetric policy-prior warm-start > symmetric search), LeVLJEPA (asymmetric predictor+stop-grad fixes symmetric collapse). We elevate this to a design principle (§7). Note: our D6 stop-grad-into-trunk is *not* the same as asymmetric-predictor-structure; we only had the first.

---

## 3. Why Phase C opens with a gate, not a build

Two gaps in the Phase-B evidence must close before committing to an actor:

1. **Single seed.** B1 is seed-0 only; `09 §11` requires seeds 1–2. The strong block-pose numbers must replicate.
2. **Action-liveness measured on the wrong bank.** The 0.002 shuffle−true gap is on the *kinematic* PushT bank, where actions are too smooth to test sensitivity. It sits uneasily against B1's intervention hit 1.0 (latent-nudge ≠ action-input). If `P` is genuinely action-insensitive on **diverse real actions**, then no action sequence changes the imagined outcome, "geometry binds" is wrong, and the true bottleneck is an under-conditioned predictor — a **model-fidelity** problem, not a search problem. Building an actor on an action-deaf model would be the mistake this gate exists to prevent.

"Geometry binds" is currently a verdict *by elimination*; C0 converts it to a verdict *by demonstration* (oracle test) on an *action-live, replicated, non-collapsed* model — or redirects us.

---

## 4. Revised thesis for Phase C

> Phase B located the bottleneck at the **actor/search** layer by elimination. Phase C first *confirms* this positively — the predictor is action-live on diverse actions, the legibility replicates across seeds, and privileged/oracle actions succeed where CEM fails — and only then builds a **light, asymmetric propose-and-score actor** (policy-prior + world-model scoring), keeping D6. If confirmation fails, the redirect is predictor action-conditioning, not an actor.

---

## 5. Research posture (carried forward + one addition)

Unchanged: JEPA-primary; smallest system that falsifies; diagnose before build; flip on measured failure only.

Added: **Asymmetry principle** (proposed D7, §7). Any learned reachability/cost/actor geometry is built asymmetrically — a predictor `φ(z_t) → target` with stop-grad — not a symmetric `d(φ(a), φ(b))`. Rationale: triple-confirmed (§2).

---

## 6. Phase C plan

### C0 — Confirmation gate *(cheap; decides build-actor vs back-to-predictor)*

All four reuse existing dumps/eval; none is a new module.

1. **Seeds 1–2 on B1.** Re-run state probes + intervention. **Pass:** block-pose linear R² and intervention hit replicate within noise. Also fix the Reacher factor bookkeeping (negative mean R² with `qpos_0` at 0.56/0.83 screams mislabeled `factor_*` columns) and re-probe, or formally drop Reacher from the legibility claim.
2. **Diverse-action liveness (the critical one).** Rebuild the PushT bank with **diverse/random actions** (not the kinematic collector) and recompute shuffle−true drift at h=5. **Pass:** meaningful gap (P responds to actions). **Fail:** near-zero → predictor is action-deaf → **redirect to C-alt** (action conditioning), do *not* build the actor.
3. **Oracle-action test (positive confirmation).** Replay a successful demo's action sequence (or a privileged controller) through the planner toward hard offset goals. **Pass (search binds):** oracle actions succeed where CEM fails → the fix is a better proposal distribution. **Fail (model fidelity):** even oracle-ish actions fail in imagination → predictor cannot represent hard-goal outcomes → redirect to predictor work.
4. **Effective-rank check (LeVLJEPA).** Effective rank of `u=φ(z)` vs `z`. **If `u` is collapsed:** H2 is symmetric-collapse; a future learned cost must be asymmetric (§7) — informs C3, not a blocker.

**Gate outcome:**
- **All of {1 pass, 2 pass, 3 = search binds}** → proceed to C1 (build actor).
- **2 fail or 3 = model fidelity** → **C-alt**: strengthen predictor action conditioning (AdaLN is present; test whether the signal is *live*, consider action injection depth / multi-step rollout loss per PiJEPA), then re-enter C0. Keep D6.

### C1 — Actor / search layer *(conditional on C0; the build)*

Smallest actor that tests "better proposals fix hard geometry," built **asymmetric** and **light**:

- **Propose-and-score, not heavy MPPI.** PiJEPA/Q-Planning: world-model *scoring* of policy proposals captured most of the benefit; iterative refinement added little. Start with: policy-prior proposes → `P` rollout scores → Q-weighted select.
- **Policy prior** `π(a | z, z*)` seeded from offline PushT actions (behavior-regularized to stay in-support). Asymmetric by construction (maps state→action).
- **Keep the existing cost** (L2-in-`z` or `φ`) for scoring initially — C0 already established the cost head is not the binder, so do not co-develop a new cost here. Change one thing at a time.
- **Ablations:** prior-warm-started select vs uninformed CEM (does the prior lift offset?); scoring-only vs +refinement (is refinement worth it?).

### C2 — Composition / hierarchy *(conditional; only if C1 underperforms on long/compositional goals)*

Per `11`'s revised trigger: fire hierarchy as a **composition/search** candidate (action-free subgoals or subgoal-conditioned prior), **not** as rollout-shortening (B2 already showed shorter `T` doesn't help). FF-JEPA-style subgoal planner; also drops the goal-image dependence (subsumes part of C1-anchors). Qualitative citation only — its 3.5%→92% is a different protocol.

### C3 — Asymmetric richer-target geometry *(conditional; only if a learned cost re-enters)*

If C1/C2 need a learned cost, do **not** rebuild scalar-`k` `φ`. Use an **asymmetric predictor** toward a **richer-than-scalar** target — relative pose displacement or next-subgoal latent — with stop-grad, per §2/§7. Effective rank of the output is a health check.

### C4 — Build only what the gate selected

Do not implement policy prior, hierarchy, or new cost before C0 clears the relevant branch.

---

## 7. Decision updates & triggers

| ID | Prior state | Phase-C status | Trigger |
|---|---|---|---|
| **D6** | `φ` only, λ≈0 | **Keep / extract** (per `11`). | Overturn only if C0.1 fails to replicate legibility **or** intervention misses across seeds. |
| **D7 (new, proposed)** | — | **Asymmetry principle:** learned cost/actor geometry is predictor+stop-grad, not symmetric distance. | Ratify into `00_decisions.md` with date + rationale (Destrade + PiJEPA + LeVLJEPA). |
| **Hierarchy / `G`** | Candidate (`09`) | **Held for C2**, composition only. | C1 underperforms on long/compositional goals. |
| **Actor / policy-prior** | Out of v1 (`00`) | **Primary C1 build, gated on C0.** | C0 = {legible, action-live, search-binds}. |
| **Predictor conditioning** | AdaLN present | **C-alt fallback.** | C0.2 fail (action-deaf) or C0.3 = model fidelity. |
| **D3 scalar-`k` `φ`** | Frozen | **Retired as a target.** | Only returns as an *asymmetric richer-target* cost (C3). |
| **D5** (no `S` in loss) | None | **Unchanged.** | n/a |

---

## 8. Adopt / Avoid ledger (updated)

**Adopt:**
- Light propose-and-score / policy-prior over heavy MPPI (PiJEPA, Q-Planning).
- **Asymmetric predictor + stop-grad** for any learned geometry (Destrade, PiJEPA, **LeVLJEPA**).
- Richer-than-scalar targets if a cost is learned (relative pose / subgoal latent) — LeVLJEPA's coarse-target lesson.
- Action-free hierarchical subgoals *if* C2 triggers (FF-JEPA), composition not rollout-shortening.
- Effective rank as a collapse diagnostic (LeVLJEPA).

**Avoid:**
- Symmetric embedding-distance objectives as the primary shaper (LeVLJEPA collapse; our flat-`φ`).
- Stacking SIGReg on a value/alignment loss by naive addition — needs structural separation (separate branches, stop-grad, predictor absorbs asymmetry): triple-confirmed (LeVLJEPA, Destrade, our D6).
- Co-training prediction + value in one encoder; two separate encoders (Destrade).
- Building an actor before confirming the predictor is action-live (C0.2/C0.3).
- corr(d,k) or on-path Spearman(`k`) as a proxy.
- Mixing campaign numbers (Euclid E2 36.7 vs imagined-campaign random 26.7).

---

## 9. Metrics & diagnostics

- Per-factor **linear vs MLP R²** across seeds 1–2; intervention hit-rate (C0.1).
- **Diverse-action** shuffle−true drift at h=5 (C0.2) — distinct from the kinematic-bank number.
- **Oracle-action success** vs CEM success on hard offset (C0.3) — the positive search-binds test.
- **Effective rank** of `u` vs `z` (C0.4).
- Actor ablations: prior-warm-start vs uninformed CEM offset; scoring-only vs +refinement (C1).
- Task metrics unchanged (live-eval success, pose/angle error, replan counts).

---

## 10. Longer arc (Phase D — out of Phase C scope)

The actor C1 builds is the substrate the eventual **open-ended self-play / online loop** grows: self-generated diverse goals as anti-FER pressure + self-supervised reward + coverage (honoring D5); mech-interp (the C0/B1 probe suite, now with effective rank) as the measurement layer; the online learn-from-failures loop as the untouched gap on JEPA backbones. The falsifiable Phase-D core experiment is unchanged: does open-ended goal-diverse training measurably reduce FER (linear-decodability, effective rank, MIG/DCI) vs standard prediction? Phase C's job is only to establish that a legible, action-live world model plus a light asymmetric actor closes hard PushT goals — the precondition for growing that actor online.

---

## 11. Risks

| Tier | Risk | Cheap early test |
|---|---|---|
| 1 | "Geometry binds" is wrong; predictor is action-deaf | **C0.2** diverse-action liveness; **C0.3** oracle test |
| 1 | Legibility was a seed-0 fluke | **C0.1** seeds 1–2 |
| 2 | Flat-`φ` is symmetric-collapse, silently caps any future cost | **C0.4** effective rank |
| 2 | Actor overfits offline PushT actions, drifts on hard goals | Behavior-regularize prior; ablate warm-start vs uninformed |
| 2 | Reacher legibility claim rests on broken factor bookkeeping | Fix `factor_*` columns or drop Reacher from the claim |
| 3 | Asymmetric richer-target cost reintroduces SIGReg-composition conflict | Structural separation per §8; effective-rank check |
| 3 | 192-dim token saturates as actor + cost read it | Track probe R² / effective rank as consumers are added |

---

## 12. Success criteria

- **C0 done** = seeds 1–2 legibility + intervention; diverse-action liveness at h=5; oracle-action vs CEM on hard offset; effective rank of `u` — **with a gate decision recorded** (build actor vs C-alt).
- **C1 done (if reached)** = policy-prior propose-and-score built asymmetric; ablation of warm-start vs uninformed CEM on offset; **one clear statement**: does a better proposal distribution lift hard-goal success on a legible, action-live model?
- **Phase C pass** = either (a) a light asymmetric actor measurably lifts hard PushT offset over uninformed CEM on a confirmed-legible, action-live model → the actor/search verdict is demonstrated, not just inferred; or (b) C0 redirects to predictor fidelity with evidence, retiring the "geometry binds" claim honestly.
- **Non-goals:** high absolute PushT success as headline, real robots, open-endedness, the online loop.

---

## 13. Immediate next actions

**Done (2026-08-30).** C0 ran; C0.3-as-run was underpowered; C0.3-redo → Outcome B. Commands below are the C0 reproduction recipe, not current next work. Current next step: CA0 in [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). D7 stays proposed (not ratified into `00` from this file).

```bash
# C0.2 — diverse-action bank + liveness (NEW collector flag, not kinematic)
python scripts/latent_dump.py --env pusht --seed 0 --action-mode diverse --device cuda
python scripts/predictor_drift.py --dump eval_results/pusht/phase_b_dump_diverse/seed0/dump.npz

# C0.3 — oracle-action vs CEM on hard offset
python eval_live.py --env pusht --pair-mode offset --actor oracle_replay --collect-episodes
python eval_live.py --env pusht --pair-mode offset --actor cem_l2 --collect-episodes

# C0.1 — seeds 1-2 legibility + intervention
for s in 1 2; do
  python scripts/latent_dump.py --env pusht --seed $s --device cuda
  python scripts/latent_probe.py --dump eval_results/pusht/phase_b_dump/seed$s/dump.npz --intervene-live
done

# C0.4 — effective rank of u vs z (add --effective-rank to the probe)
python scripts/latent_probe.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz --effective-rank
```

The gate is ordered by risk: C0.2 is the one that could overturn everything, so it runs first.
