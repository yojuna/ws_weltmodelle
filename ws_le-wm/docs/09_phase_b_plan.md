# Phase B Design Spec & Plan — Diagnose two axes, then pick one fix

**Date:** 2026-08-29 (revised same day after code/artifact review)  
**Supersedes for planning:** the “Decision fork” in `07_status_synthesis.md` and `§11 Success criteria` in `01_design_spec.md`.  
**Inherits:** posture and stability rules in `00_decisions.md`; module/stack surfaces in `01_design_spec.md`.  
**Stack:** `le-wm/` on stable-worldmodel + stable-pretraining; **live eval** for PushT and Reacher (no author HDF5 for evaluation).  
**Change control:** unchanged from `00_decisions.md` — deferred items flip only on *observed* failure, recorded with date + reason. This document flips nothing silently.

**Code:** `le-wm/scripts/latent_dump.py`, `latent_probe.py`, `predictor_drift.py`, `run_horizon_sweep.sh`; `eval_live.py --horizon`.

---

## 0. One-paragraph orientation

Phase A tested a thin reachability readout `φ` on a frozen JEPA latent and returned a useful negative: **the cost head is not the bottleneck**. The short-hop Euclidean-φ “win” is a **replicated phenomenon** that is only **partly** learned reachability (random `φ` also beats L2; trained still leads random by ~10–12pp on the matched Euclid campaign). Hard (offset) goals sit in a **~2–10% band for every head**. Phase B therefore **stops iterating the cost head** and diagnoses two **separate** axes the Phase A write-ups had collapsed: **(G) geometric goal hardness** vs **(D) predictor drift as a function of imagination horizon**, plus **(R) what frozen `z` linearly exposes**. Hierarchy, Sep, and residual policies stay gated on those measurements.

---

## 1. Where we stand — Phase A verdict (corrected)

The ledger in `07` is honest on the tables. Several causal readings in the first draft of this file (and parts of `08`) overclaimed. Restating with matched artifacts:

| Phase A finding | Matched evidence | Corrected reading |
|---|---|---|
| Euclid `φ` > L2 on short | **36.7±7.6 vs 16.7±7.6** (seeds 0–2, n=20) | **Replicated.** Random `φ` on the **same** campaign is **25.0±8.7**, not 26.7 (26.7 is imagined-campaign E4). Trained beats random by ~10–12pp on the mean; seed 0 ties random at 35%. Part of beating L2 is CEM-conditioning; part is a mild local signal. |
| Offset capped for all heads | L2 4.7±3.1, φ 6.7±1.2, random 7.3±2.3 (Euclid); imagined random 4.0±2.0 | **Cost head is not the bottleneck.** Band is ~2–10% (1–5 successes on n=50), not a sharp 5% ceiling. Same `--seed` **reuses the same pairs**; random swings are **re-init of φ**, not redrawn banks. |
| Local signal vs range | short trained > random; offset trained ≈ random (noise-wide) | Compatible with a **mild, local** signal that does not transfer to harder **geometry**. Not proof that “distance in time” is the axis — see §1.1. |
| Imagined rollouts OOD | mean ‖ẑ−z‖₂ ≈ **6.3** over a **25-step** true-action path | **25-step open-loop drift**, including 3 teacher-forced frames at 0 error (so 6.3 **underestimates** predicted-only drift). **Not** CEM-horizon OOD. CEM imagines `T≈5`. Candidate L2 **std** ≈ 6.3 is a different quantity. |
| `φ` CEM landscape flat | cand. std φ≈0.48 vs L2≈6.3 (autopsy `horizon=5`) | Dynamic-range / conditioning; orthogonal to “is reachability in `z`.” |
| Real-path Spearman ≈ 0.99 for L2, φ, **and random** | autopsy n=24 | **Nearly tautological:** remaining-steps `k` decreases along a real path; any time-smooth `z` plus a Lipschitz map ranks `k`. **Not** a FER / D6 trigger. |
| IQL-on-frozen-`φ` | seed 0: short 10% vs L2 25% vs **random IQE 30%**; offset 4% vs 8% | Gate failed. Random beating *trained* IQL looks like a **ruined CEM landscape**, not “info present but only encoder-shaping can use it.” |
| Higher corr ≠ better planning | imagined-φ corr 0.75 → offset 2.0% | **corr(d,k) is the wrong diagnostic.** On-path Spearman(`k`) is demoted the same way. |

**Net:** Phase A succeeded by falsifying “thin φ is the planning bottleneck.” Keep the thin-readout *instinct* as the right first bet. Do **not** treat “Euclidean φ > L2 on short: Supported” as “learned reachability unlocked planning.”

### 1.1 Two axes Phase A mixed

In [`pairs.py`](../le-wm/eval_logging/pairs.py), **`short_horizon` and `offset` share `goal_offset=25`**. The difference is the **pose band**:

| Mode | Temporal Δ | Pose band | Angle |
|------|------------|-----------|-------|
| `short_horizon` | 25 steps | **[12, 25]** | ≤ 0.25 |
| `offset` | 25 steps | **[20, 55]** | ≤ 0.6 |

CEM is already `horizon=5`, `receding_horizon=5`, `action_block=5` ([`pusht.yaml`](../le-wm/config/eval/pusht.yaml)) → 25 env steps per replan, `eval_budget=50` → two CEM cycles.

So:

- **Axis G (geometry):** harder PushT pose goals (offset band). Already measured. Not “longer imagination.”
- **Axis D (drift vs horizon):** autopsy measured **25-step** open-loop ‖ẑ−z‖. CEM uses **~5** imagined steps. **Not yet measured** at the horizon CEM actually uses.
- **Axis R (representation):** what frozen `z` linearly exposes. Autopsy **did not persist latents** — only scalars/plots. B1 must re-encode (or dump on the next run).

Predictor action conditioning is **already per-layer AdaLN** (`module.py` `ConditionalBlock`). B2 action-liveness asks whether the signal is *live*, not whether to add AdaLN.

---

## 2. The central fork (held until B1 state probes)

Destrade’s IQL **won** by shaping the **encoder**. Ours **failed** as a thin head on a **frozen** trunk (D6), and lost to random IQE on short. That is **weaker** evidence for “present-but-entangled” than Spearman-on-`k`.

The fork is still real; the **gate is state-factor decodability + intervention**, not on-path `k`:

- **Extract (keep D6):** control-relevant factors are linearly readable and causally live in `P`. Problem is search / geometry / in-distribution scoring.
- **Reshape (break D6):** factors are present only nonlinearly **and** linear directions do not drive `P`. Sep (without stacking SIGReg on value) becomes the documented exception.

We do **not** guess. **B1 decides extract vs reshape. B2 decides rollout vs composition.**

---

## 3. Revised thesis for Phase B

> Phase A mixed **geometric hardness** with **predictor-horizon OOD**. Phase B measures them separately, plus whether frozen `z` linearly exposes sim state. Hierarchy is a **candidate** (composition and/or short hops), not the leading fix a priori. FF-JEPA’s 3.5%→92% at t=75 is a **different protocol** and is not this live eval’s target.

This keeps the representation-first bet and concedes Phase A’s measured point: a thin cost head alone does not unlock hard PushT goals.

---

## 4. Research posture

Unchanged: JEPA-primary; smallest system that falsifies; flip on measured failure only.

Added for Phase B:

- **Diagnose before build.** No new cost head or architecture module ships before B1/B2.
- **Better diagnostics than corr(d,k) or on-path Spearman(k).** State-factor R², causal intervention, per-step drift **at CEM’s h**, CEM `T` sweep on matched pairs.
- **Two envs.** PushT (primary, offset pairs) and **Reacher** (`live_reset` + random-action banks). Do **not** invent a Reacher offset-pair protocol; kinematic banks are PushT-only.

---

## 5. Phase B plan (diagnose-first ordering)

### B0 — Bookkeeping (optional)

`07`’s Option A (hybrid `L2 + α·d_φ`) stays **optional reviewer bookkeeping**. Random-φ already bounds how much a second head can add once L2 carries range. A cheaper H2 test, if needed: **z-score φ costs** inside `criterion` (eval-only). Not a research result.

### B1 — Interpretability gate *(decides extract vs reshape)*

**Question:** Which **ground-truth state factors** are linearly vs nonlinearly decodable from frozen `z`? Does intervening along a probe direction move `P`?

On-path remaining-steps `k` is a **sanity check only** (expected to be easy).

1. Dump `z`, `ẑ` (true-action), state factors, remaining `k`, remaining **pose** error, episode id.  
2. Ridge vs 2-layer MLP per factor; report R² and the nonlinear−linear gap. Day-1 FER metric = **per-factor linear R²**. Full MIG/DCI is optional after R² exists — do not block.  
3. Causal intervention: add ε along the linear direction for `block_x` (PushT) / a joint (Reacher), one `P` step with a **fixed** action, re-read the probe. Hit = predicted factor moves with the right sign.

**D6 trigger (revised):** flip only if state factors are present (high MLP R²) but linear R² is poor **and/or** linear directions do not causally drive `P`. Do **not** flip because `k` is “only nonlinear.”

### B2 — Predictor-rollout fidelity *(parallel with B1)*

1. Per-step ‖ẑ−z‖ vs h = 1…L under true actions; mark **h=5**. Report mean **including vs excluding** teacher-forced history.  
2. Action liveness: same start, **shuffled** future actions vs true; compare Δz. If dead, then consider stronger conditioning — AdaLN is already there.  
3. CEM `horizon` sweep `T ∈ {2, 3, 5, 8}` on **matched** PushT offset n=50 seed 0 (**E1 L2 only**) and Reacher `live_reset` same seeds. This is the real “shorten imagination” test.  
4. Optional H2 z-score of φ costs only if (3) is ambiguous.

**Hierarchy / receding-horizon trigger (revised):** fire if drift at h=5 is already large vs L2 discriminative spread, **or** shortening `T` lifts offset, **or** drift at h=5 is small while hard geometric goals still fail (composition, not OOD).

### B3 — Choose *one* architectural fix (docs only until numbers)

| If B1/B2 says… | Fix |
|---|---|
| Linear state R² high + intervention works + drift@5 small | Keep D6. Search/geometry → hierarchy or policy-prior, not Sep. |
| Linear state R² high + drift@5 large + shorter `T` helps | Rollout wall → shorter execute and/or hierarchy; **still keep D6**. |
| State only nonlinear + intervention fails | Record D6 flip; Sep next **without** stacking SIGReg on value. |
| Linear **and** MLP R² both low | Information may be absent; JEPA-for-control bet is in trouble (TD-MPC2 comparison is Phase C, not now). |

Do **not** implement AdaLN-from-scratch, Sep, FF-JEPA `G`, or a residual policy in this phase.

### B4 — Build the chosen fix (conditional)

Scope after B3. Smallest module that tests the mechanism.

---

## 6. Decision updates & triggers

| ID | Prior state | Phase-B status | Trigger to flip |
|---|---|---|---|
| **D6** | `φ` only, λ≈0 | **Held pending B1 state-factor + intervention.** | High MLP R², poor linear R², **and/or** intervention miss — not on-path `k`. |
| **Hierarchy / FF-JEPA `G`** | Out of v1 | **Re-opened as a candidate**, not pre-selected. | B2: drift@5 large, or shorter `T` helps offset, or drift@5 small + Axis G still binds. |
| **D3** | Euclidean v2 best readout | **Frozen; no more head churn.** | Only inside B3-reshape as encoder loss, not a thin head. |
| **Option A** (hybrid cost) | One of two postures in `07` | **Demoted to optional bookkeeping.** | Reviewer demand only. |
| **Predictor OOD** | Autopsy H1 (25-step) | **First-class B2; measure at CEM h=5.** | n/a |
| **D5** | No `S` in loss | **Unchanged.** | n/a |
| **D4** | PushT claim env | PushT remains the claim; Reacher is a **diagnostic** env in Phase B. | n/a |

---

## 7. Adopt / Avoid ledger

**Adopt (if we get to B3-reshape / B3-search):**

- Quasimetric over Euclidean *if* B3-reshape (Destrade).  
- Light propose-and-score / policy-prior (PiJEPA), not heavy MPPI as the first controller.  
- Action-free hierarchical subgoals *if* the hierarchy trigger fires (FF-JEPA) — citation is qualitative, not a 92% target for this live protocol.  
- Shared frozen encoder across components (PiJEPA).

**Avoid:**

- Co-training prediction + value in one encoder; SIGReg stacked on value (Destrade).  
- Two separate encoders for pred vs cost (Destrade: no benefit).  
- Adding AdaLN before measuring liveness (already present).  
- corr(d,k) or on-path Spearman(`k`) as a planning or FER proxy.  
- IQL as a thin head on a frozen trunk (in-house).  
- Mixing Euclid E2 (36.7) with imagined-campaign random (26.7).

---

## 8. Metrics

- Per-factor linear vs MLP R² (B1); on-path `k` as sanity only.  
- Causal-intervention hit-rate along one probe direction.  
- Per-step ‖ẑ−z‖ vs h; predicted-only vs all-frames mean; marker at h=5.  
- Action-shuffle vs true-action Δz.  
- PushT offset / Reacher live_reset success vs CEM `horizon` (E1 L2).  
- Optional: CEM-selected vs true-best action cost gap (if cheap from dumps).  
- Task metrics unchanged (live-eval success, pose/angle or finger–target error).

---

## 9. Longer arc (out of Phase B)

Unchanged in spirit: representation-first bet; Phase C open-endedness / online loop; FER as measurement. B1’s **state-factor** suite is the instrument, not MIG-on-day-one.

---

## 10. Risks

| Tier | Risk | Cheap early test |
|---|---|---|
| 1 | No config keeps JEPA purity *and* usable control geometry | **B1** state-factor + intervention |
| 1 | “Why not TD-MPC2 from pixels?” | B1: if factors are extractable, JEPA still earns a look; if only reshape works, be honest later |
| 2 | 25-step autopsy OOD overstated vs CEM | **B2** drift@5 + `T` sweep |
| 2 | Axis G (hard geometry) misread as Axis D | Keep pair_mode and `horizon` as **separate** knobs |
| 2 | Stacked components | One module behind a gate |
| 3 | 192-dim CLS saturates | Track probe R² as consumers are added |

---

## 11. Success criteria

- **B1 done** = per-factor linear vs MLP R² for PushT + Reacher, plus one intervention; **D6 keep/flip recorded** using the revised trigger.  
- **B2 done** = per-step drift curve with h=5 marked + CEM `T` sweep; **hierarchy trigger evaluated** (fire or not, with reason).  
- **B3 done** = exactly **one** fix selected from the table in §5, justified by B1+B2 (not preference).  
- **Phase B pass** = we can state which of {geometry, rollout-at-CEM-horizon, entanglement} binds.  
- **Non-goals:** high PushT success as headline, robots, open-endedness, online loop.

---

## 12. Immediate next actions (engineering)

**Code is in the tree** (see [`10_implementation_status.md`](10_implementation_status.md)). GPU dumps / probes / `T` sweep have **not** been run; D6 stays held.

1. Dump latents (do **not** assume autopsy `.pt` files exist).  
2. Run B1 probes + intervention.  
3. Run B2 drift + `--horizon` sweep (L2, matched pairs).  
4. Record outcomes in `experiment_log.md`; flip D6 / hierarchy only per §6.

Commands:

```bash
# dumps
python scripts/latent_dump.py --env pusht --seed 0 --device cuda
python scripts/latent_dump.py --env reacher --seed 0 --device cuda

# B1
python scripts/latent_probe.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz
python scripts/latent_probe.py --dump eval_results/reacher/phase_b_dump/seed0/dump.npz

# B2 plots from dump
python scripts/predictor_drift.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz

# CEM horizon (L2 only)
bash scripts/run_horizon_sweep.sh
```
