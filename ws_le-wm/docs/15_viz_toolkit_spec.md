# Visualization Toolkit Spec (v3) — `viz.py` + scientific report: prioritized, dump-driven, projection-honest

**Date:** 2026-08-30 (v3 adds explicit priority tiers + the report design)
**Status:** implemented in `le-wm/viz.py` + `scripts/report.py`. The v0 gallery (`15_viz_toolkit_spec_v0.md`) remains the record of the first toolkit; do not treat those figures as satisfying this document.
**Purpose:** a reusable visual layer for understanding the world model, rollout, cost, planner, and task — attaching to dumps we already produce, with rigor rules that stop the tooling from hiding the effects we hunt, and an **explicit priority tier** on every figure so we build the decisive ones and treat the rest as rounding.
**Consumes:** `phase_b_dump/seed{N}/dump.npz`, `c0_oracle_livebank/seed{N}/`, `ca0_closed_loop/`, and a new `cem_capture.npz`.
**Feeds:** [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). Every figure names the **scalar it motivates**.
**Change control:** additive tooling.

---

## 0. Principles (non-negotiable)

1. **PCA, not t-SNE/UMAP, for dynamics/distance/drift.** Nonlinear embeddings distort global geometry; allowed only for exploratory clustering, never for "does this path drift."
2. **Fit the projection on REAL latents; APPLY it to imagined ones.** A joint fit swallows the drift and hides it.
3. **Report captured variance on every projection.**
4. **For every lossy view, show the loss.** Pair each projection/aggregation with the honest scalar it could hide: 2D overlay ↔ full-space distance curve (+ in-plane drift fraction); norm-drift ↔ directional decomposition; NN-retrieval ↔ retrieval distance; scree ↔ nonlinear intrinsic dimension. If a view and its scalar disagree, the scalar wins.
5. **A figure motivates a scalar; it does not replace one.**
6. **Decoder-free** (nearest-neighbor retrieval, never reconstruction).
7. **(v3) A plot that does not change what you do next is Tier 3 at best.** Tier 1 is small on purpose. Aesthetics are not a reason to build or foreground a figure.

---

## 1. Priority tiers (build order and reporting prominence follow this)

| Tier | Figures | Role | When to build | Report placement |
|------|---------|------|----------------|------------------|
| **1 — Decision-critical** | **A1, A2, B1** | Resolve the two things the phase turns on: the CA0 fork (**accumulation vs per-step infidelity**) and **search-problem vs model-problem**. | **First. Gate the phase on these.** | Top, full-size, uncollapsed |
| **2 — Mechanism** | **A3, B2, D1** | Explain *why* the Tier-1 verdict and *scope the fix*: contact-driven vs steady drift; CEM converged vs gave up; capacity/structure (is scaling ever warranted). | After Tier 1 reports a verdict. | Middle sections |
| **3 — Rounding / writeup** | A4, A5, B3, B4, C1, C2, D2, E1, F1 | Deepen mechanism, support the objective/geometry narrative, qualitative intuition. | Only as the writeup demands; **never before a fork decision.** | Collapsible appendix |

**Rationale for Tier 1 being exactly three:** A2 states the accumulation-vs-infidelity fork from the *model* side (single-step error vs compounding); B1 states the *same* fork from the *planner* side (does CEM fail to find the good action, or does the model score it as bad); A1 establishes the drift phenomenon they both explain. Those three, cross-checking each other, are the entire decision. Everything else tells you *why* or *rounds the picture* — valuable, but not the gate.

---

## 2. Figure catalog (each tagged: **[Tier] — decision it drives**)

### Family A — Rollout fidelity (the CA0 fork)

**A1 — Oracle-imagine overlay + full-space distance curve** · **[T1] — does the rollout drift, and how**
2D `RealFittedProjector` overlay (real path, open-loop `m=25`, closed-loop `m=5`, start, `z*`) **paired with** the honest full-space curve ‖ẑ_t−z*‖ vs step, and the printed **in-plane drift fraction** (if small, the overlay is decorative and the curve is the evidence). Reads: overshoot → scale/calibration; orthogonal-from-step-1 → action mis-represented; track-then-diverge → accumulation. *Scalar:* ‖ẑ_end−z*‖ and toward-goal fraction vs `m`.

**A2 — Error-compounding decomposition** · **[T1] — accumulation vs per-step infidelity (THE fork, model side)**
(a) histogram of single-step teacher-forced ‖ẑ_{t+1}−z_{t+1}‖; (b) accumulation + CA0 re-encode-every-`m` recovery curve on shared axes. Large single-step → infidelity (→ CA-train). Tiny single-step, steep accumulation → accumulation (→ protocol fix, re-open C1). *Scalar:* median single-step error; accumulation slope; recovery-`m`.

**A3 — Drift vs contact events, directionally decomposed** · **[T2] — why it drifts (contact vs steady; in what factor)**
Per-step drift curve with contact/wall bands, **paired with** `probe_decompose` of the drift onto block/agent/angle directions, + per-episode spread. Reads: jumps at contact → contact-representation fault; block-dominated → the model loses the object. *Scalar:* free-space vs contact drift; per-factor drift share.

**A4 — Drift vs action magnitude/direction** · **[T3] — is fidelity uniform or regime-limited**
One-step error vs |action|, binned by direction, boundary actions flagged. *Scalar:* error–|action| correlation; in-box vs at-bound error.

**A5 — Action-effect vector comparison** · **[T3] — does the model get action *direction* right**
Quiver: real vs imagined one-step displacement. *Scalar:* mean angular error.

### Family B — Planner behavior (search vs model)

**B1 — Terrain-free CEM panels** · **[T1] — search-problem vs model-problem (THE fork, planner side)**
No height map. **B1a** cost vs distance-from-selected (bowl / flat plate / rugged). **B1b** selected→oracle **line-search** — cost along the interpolation; **down toward oracle → search problem; up → model problem** (1-D, loses least, targets the exact decision). **B1c** PCA-of-candidates scatter, cost-colored, oracle+selected marked, **variance printed**. *Scalar (named result):* **signed cost gap = cost(oracle) − cost(selected)** — negative → CEM failed to find (search); positive → model scores good action worse (model).

**B2 — CEM convergence** · **[T2] — gave-up (search) vs converged-confidently-to-bad (model)**
Elite best/mean cost vs iteration; sample-variance shrink. Still improving at last iter → under-budget; early plateau + collapsed variance → confident convergence to a bad optimum. *Scalar:* iters-to-plateau; final elite gap to oracle.

**B3 — Per-horizon-step structure** · **[T3] — which action in the chunk is wrong**
Per-step selected-vs-oracle deviation + per-step cost. First-right/later-wrong → receding-horizon fine, lookahead is the problem. *Scalar:* per-step |selected−oracle|.

**B4 — Cost-head agreement** · **[T3] — do L2 / φ / random-φ rank candidates alike**
Pairwise candidate-cost scatter; rank correlation; disagreement regions. *Scalar:* Kendall-τ between heads.

### Family C — Cost / objective quality

**C1 — Cost-calibration** · **[T3, high] — does the objective track true progress, and where it breaks**
Planning cost (L2, φ) vs **true** remaining steps/pose, binned with a calibration line, **by goal-hardness band** (not one correlation — on-path `k` corr was near-tautological). *Scalar:* calibration slope + dynamic range per band.

**C2 — Probe-faithfulness map + spatial residual** · **[T3] — where block-pose legibility fails physically**
Predicted-vs-true scatter + residual heatmap over the board. *Scalar:* conditional R² near-wall vs center.

### Family D — Representation structure

**D1 — Rank spectrum + nonlinear ID** · **[T2] — is capacity/structure the issue (scaling verdict)**
Scree (participation-ratio marked) + per-dim variance (dead vs quiet) **paired with** a nonlinear intrinsic-dimension estimate (linear scree can't see a curved manifold). Sharp elbow + dead tail + low nonlinear ID → genuine low-rank, don't scale. *Scalar:* participation ratio; nonlinear ID; dead-dim fraction.

**D2 — Intervention: sweep, compound, combine** · **[T3] — is legible geometry durably, independently actionable**
ε-sweep vs one-step movement; multi-step persistence; two-factor combination (independent vs interfering). *Scalar:* slope/linearity; decay rate; cross-factor interference.

### Family E — Task / goal structure

**E1 — Success-by-goal-region map** · **[T3, high] — do failures cluster by pose (unifying story)**
Spatial map of target poses by CEM success/failure; oracle-replay overlay. Overlap with high-residual (C2) / high-drift (A3) regions → one "hard region" story. *Scalar:* success by region; overlap fraction.

### Family F — Qualitative

**F1 — Rollout-vs-reality filmstrip** · **[T3] — where physically imagination diverges (intuition only)**
True pixels beside nearest real frame to `ẑ_t`, **with retrieval distance shown** (large → "imagining a state unlike any real frame"). Exploratory label. *Scalar:* step where retrieval distance crosses threshold.

---

## 3. Shared utilities

`RealFittedProjector` (real-only fit, variance, raises on imagined fit) · `probe_decompose(vec, dirs)` · `contact_events(state)` (validated mask) · `nn_retrieve` (returns idx **and** distance) · `action_effect(state, action)` · `nonlinear_id(Z)` · `cost_calibration_pairs(dump)` · CEM-capture loader (candidates, per-iteration elites, per-horizon-step).

### Figure-function contract (so `viz.py` and `report.py` compose)

Every `fig_*` returns a `FigureResult`:
```
FigureResult:
  figure     : matplotlib Figure
  tier       : 1 | 2 | 3
  question   : str            # the one question it answers
  scalars    : {name: {value, threshold, verdict}}   # the citable numbers
  caption    : {what, how_to_read, reading_here, would_overturn}
```
This lets the report auto-assemble (§4) and lets tests grep for the named scalar.

---

## 4. Anti-footgun checklist (CI where possible)

Projector raises on imagined/joint fit · every projection prints variance · **every norm-aggregate paired with a directional decomposition** · **every 2D overlay paired with a full-space distance curve + in-plane fraction** · **every NN-retrieval shows distance, large flagged** · **every scree paired with nonlinear ID + caveat** · no t-SNE/UMAP in distance/dynamics figures (lint) · filmstrip labeled qualitative · each `fig_*` names its scalar (test greps).

---

## 5. The scientific report (`report.py` → `diagnostic_report.html` + `.md`)

The report is not a gallery. It is a **paper-style results artifact** whose order is the *diagnostic logic*, whose top is the *verdict*, and whose every figure carries a scalar and a falsifier. It regenerates from dumps, so it is never hand-assembled or stale.

### 5.1 Design principles (scientific, anti-aesthetic)

- **BLUF — verdict first.** The reader learns the answer in ten seconds; evidence follows.
- **Order = diagnostic logic, not figure ID.** Sections are *questions*, resolved in the sequence the phase actually decides them.
- **Every figure carries its scalar + a falsifier.** No figure appears without a stated question and a "what would overturn this."
- **Tier drives prominence.** Tier-1 at the top, full-size; Tier-2 in the middle; Tier-3 in a collapsed appendix. A reader who stops after Tier-1 has the decision.
- **Provenance for reproducibility.** Dump paths, seeds, checkpoint hash, git SHA, `DUMP_VERSION`, date in a banner.
- **Self-contained + regenerable.** One HTML with embedded figures for sharing; a `.md` twin for the repo; both emitted by `report.py` from `FigureResult`s.

### 5.2 Structure (the narrative)

```
diagnostic_report.html
├─ 0. Provenance banner            dump/seed/checkpoint-hash/git-SHA/date/DUMP_VERSION
├─ 1. Bottom line (BLUF)           the question + the verdict, filled from pre-registered
│                                  scalar thresholds (e.g. "ACCUMULATION → protocol fix +
│                                  re-open C1", or "INFIDELITY → CA-train")
├─ 2. Scalars table                every named scalar · value · threshold · pass/fail
│                                  (the citable, machine-readable core; figures illustrate it)
├─ 3. Q-A  Does the rollout drift?          → A1                     [T1]
├─ 4. Q-B  Accumulation or per-step?        → A2                     [T1]  ← the fork (model side)
├─ 5. Q-C  Search or model problem?         → B1, B2                 [T1/T2] ← the fork (planner side)
├─ 6. Q-D  Why / where does it drift?       → A3 (+A4, A5)           [T2/T3]
├─ 7. Q-E  Objective & representation       → C1, D1                 [T3/T2]
├─ 8. Q-F  Task geometry                     → E1, C2                 [T3]
└─ 9. Appendix (collapsed)                   → F1, per-figure raw scalars, config dump
```

Sections 3–5 are the spine; a reader can stop after §5 with the verdict in hand. §6–8 are mechanism and support; §9 is intuition and provenance detail.

### 5.3 Per-figure block (identical template everywhere)

Rendered from `FigureResult.caption` so no figure is unexplained to someone who wasn't in the room:

> **[Figure]**
> **What this is** — one sentence.
> **How to read it** — what the axes/regions/shapes mean.
> **Reading here** — the observed scalar and which pre-registered branch it falls in.
> **What would overturn this** — the falsifier.

### 5.4 Section verdict lines

Each question-section ends with a one-line **verdict** computed from its figures' scalars against pre-registered thresholds (e.g. "§B verdict: single-step error median 0.4 ≪ accumulation slope → **ACCUMULATION**"). The BLUF (§1) is just the composition of these section verdicts, so the headline can never drift from the evidence.

### 5.5 Implementation

- `report.py` imports the `fig_*` functions, calls each, renders figures to embedded base64 (HTML) or a `figures/` dir (MD), pulls `scalars`/`caption`/`tier`/`question` from each `FigureResult`, and fills a single template (Jinja2 or f-string).
- **Pre-registered thresholds live in one `thresholds.yaml`**, read by both the section-verdict logic and the tests — so "pass/fail" and the BLUF are reproducible and not editorializable after the fact.
- `--tiers 1` / `--tiers 1,2` / `--all` controls how much renders (fast Tier-1-only report during iteration; full report for the writeup).
- Emits both `diagnostic_report.html` (share) and `diagnostic_report.md` (repo/PR review).

---

## 6. Implementation surfaces

| Concern | File |
|---|---|
| Core figures | `le-wm/viz.py` (returns `FigureResult`) |
| Figure CLI | `scripts/viz.py --dump … --figs …` |
| Report assembly | `scripts/report.py --dump … --tiers … → diagnostic_report.{html,md}` |
| Pre-registered thresholds | `le-wm/thresholds.yaml` |
| CEM capture (B1–B3) | `eval_live.py --capture-cem → cem_capture.npz` |
| Env-specific | `viz.py::contact_events`, `viz.py::action_effect` |
| Tests | `scripts/test_viz.py` (guardrails) + `scripts/test_report.py` (BLUF matches scalars; scalars table matches emitted `FigureResult`s) |

---

## 7. Build order (tier-driven; decisive first, report scaffolded early)

1. **`RealFittedProjector` + `load_dump` + `FigureResult` contract + A1 + A2** (Tier 1, model side).
2. **`--capture-cem` + B1 (+ B2)** (Tier 1/2, planner side). A1+A2+B1 now resolve the phase fork.
3. **`report.py` + `thresholds.yaml` + Tier-1-only report.** Stand this up *as soon as A1/A2/B1 exist* — the report is how the fork decision gets recorded and reviewed, not an afterthought. Run with `--tiers 1`.
4. **A3, D1** (Tier 2 mechanism) → extend report to `--tiers 1,2`.
5. **A4, A5, B3, B4, C1, C2, D2, E1, F1** (Tier 3) as the writeup demands → `--all`.
6. **`test_viz.py` + `test_report.py`** in parallel from step 1.

The report is scaffolded at step 3, not the end: its Tier-1 form *is* the artifact that records whether the phase's fork resolved to protocol-fix-and-re-open-C1 or to retrain — the same decision `14`'s CA0 gate turns on. Everything after enriches a report whose headline is already correct.
