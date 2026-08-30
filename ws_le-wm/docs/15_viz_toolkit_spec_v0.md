# Visualization Toolkit Spec — `viz.py` (reusable, dump-driven, projection-honest)

**Date:** 2026-08-30  
**Status:** implemented 2026-08-30 — `le-wm/viz.py` + Figs 1–7 + `scripts/viz_report.py`. This is the spec that landing followed. The upgrade target is [`15_viz_toolkit_spec.md`](15_viz_toolkit_spec.md) (v3).
**Purpose:** a reusable visual layer for understanding the world model, the rollout, and the planner — attaching to dumps we already produce, with rigor rules that prevent the tooling from hiding the very effects we hunt (esp. drift).
**Consumes:** existing artifacts — `phase_b_dump/seed{N}/dump.npz`, `c0_oracle_livebank/seed{N}/`, `ca0_closed_loop/`. No new data pipeline.
**Feeds:** the checks in [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). Every figure names the **scalar it motivates** — plots are navigation, not evidence.
**Change control:** additive tooling; no `00_decisions.md` impact.

---

## 0. Principles (non-negotiable — this is where rigor usually dies)

1. **PCA, not t-SNE/UMAP, for anything about dynamics, distance, or drift.** Nonlinear embeddings distort global geometry and fabricate/destroy trajectory continuity. They are allowed *only* for exploratory "are there clusters" questions, never for "does this path drift." Any structure seen only under t-SNE/UMAP is a hypothesis to confirm with a metric, never a finding.
2. **Fit the projection on REAL latents; APPLY it to imagined ones.** Never fit jointly on real+imagined — a joint fit rotates the subspace to accommodate the drift and *hides it*. The projector is estimated from `z` (real) and then used, frozen, to transform `ẑ` (imagined). This single rule is what makes drift visible.
3. **Always report captured variance.** Every 2D/3D projection prints the fraction of `z`-variance its components capture. Given `z`'s effective rank ≈ 22, top-2 PCA captures a real fraction here — but say how much, every time.
4. **A figure motivates a scalar; it does not replace one.** Each figure's caption states the numeric test it points to. The scalar is what gets cited.
5. **Decoder-free.** LeWM has no pixel decoder; "what the model imagines" is shown by nearest-neighbor retrieval against an encoded bank, never by reconstruction.

---

## 1. Architecture

Single module `le-wm/viz.py` + thin CLI `scripts/viz.py`. Pure function core; each figure is one function `fig_*(dump, **opts) -> Figure` that takes a loaded dump and returns a matplotlib Figure. No global state, no training deps.

```
viz.py
  load_dump(path) -> Dump            # npz + meta, validates DUMP_VERSION
  RealFittedProjector               # PCA fit on real z, .transform(any)
  contact_events(state) -> mask      # sim-state → contact/wall step mask
  nn_retrieve(z_query, bank) -> idx  # decoder-free "what P thinks"
  fig_oracle_overlay(...)            # Fig-1
  fig_cem_landscape(...)             # Fig-2
  fig_drift_contacts(...)            # Fig-3
  fig_rollout_filmstrip(...)         # Fig-4
  fig_probe_faithfulness(...)        # Fig-5
  fig_intervention_sweep(...)        # Fig-6
  fig_rank_spectrum(...)             # Fig-7
scripts/viz.py --dump ... --figs oracle_overlay,cem_landscape,...
```

**Inputs each figure needs (all already in dumps):** `z` (real latents), `z_hat` (imagined), `z_hat_shuf`, `state` factors, `remaining_k`, pose error, episode ids; oracle bank adds `a_{t:t+25}`, `z*`, true intermediates; CEM captures (Fig-2) need candidate actions + costs (new lightweight hook, §Fig-2).

---

## 2. Figure catalog — ordered by decisiveness

### Fig-1 — Oracle-imagine overlay *(build first; answers the CA0 fork visually)*

**Question:** when fed known-good actions, does the imagined rollout drift, and *how* — overshoot, orthogonal wander, or late divergence?

**Method:** `RealFittedProjector` fit on the real trajectory's `z_0…z_T`. Plot three connected paths in that 2D frame: real `z` trajectory, imagined `ẑ` from oracle actions (open-loop `m=25`), and — overlaid — the closed-loop `m=5` imagined path from CA0. Mark start and `z*`. Arrows for time direction.

**Reads (each implies a different fix):**
- Heads toward `z*` then overshoots → scale/step-size or calibration.
- Wanders orthogonally from step 1 → action effect mis-represented.
- Tracks then diverges late → pure accumulation (supports CA0-ACCUMULATION).
**Motivates:** ‖ẑ_end−z*‖ and toward-goal fraction vs `m` (CA0 curve).

### Fig-2 — CEM cost landscape with oracle overlay *(build second; separates "can't find" from "doesn't look good")*

**Question:** does the good action sit in a low-cost basin CEM should descend (→ search problem) or does imagination score it as *bad* (→ model problem)?

**Method:** at a planning step, capture CEM candidate action sequences + their costs (add a `--capture-cem` hook in `eval_live.py` that dumps candidates/costs for one step). Project candidate actions to 2D (PCA on the candidate set, or cost vs distance-from-selected). Color by cost. Overlay: the **CEM-selected** action, and the **oracle** action sequence's location + its cost.

**Reads:**
- Oracle action in a low-cost region CEM missed → **search/optimizer** problem.
- Oracle action scored *high-cost* (looks bad in imagination) → **model/cost** problem (consistent with C0 Outcome B).
- Flat plate → the H2 flat-landscape, now visible.
**Motivates:** cost(oracle actions) − cost(CEM-selected) scalar (planner regret).

*This is the most decisive single figure for the search-vs-model question; prioritize the `--capture-cem` hook.*

### Fig-3 — Drift vs contact events *(CA1)*

**Question:** steady accumulation or contact-triggered jumps?

**Method:** per-step ‖ẑ_t − z_t‖ curve, x-axis = step, with vertical bands at contact/wall events from `contact_events(state)`. Split-color free-space vs contact steps.

**Reads:** jumps at contacts → contact-representation fault; smooth ramp → generic accumulation.
**Motivates:** free-space vs contact mean-drift scalars.

### Fig-4 — Rollout-vs-reality filmstrip *(decoder-free "what P thinks")*

**Question:** where, physically, does the imagination diverge from reality?

**Method:** for one episode, per step show side-by-side: (a) true pixels, (b) the frame whose encoded `z` is nearest to `ẑ_t` (`nn_retrieve` against the episode/bank). No decoder. A drift becomes visceral — you watch retrieved frames stop matching reality and see if the block is hallucinated off-position.

**Reads:** qualitative localization of divergence (which physical event breaks it).
**Motivates:** the step index where NN-retrieval distance crosses a threshold → ties to Fig-3.

### Fig-5 — Probe-faithfulness map

**Question:** where does the block-pose readout (R²=0.90) actually fail — the 0.10?

**Method:** scatter probe-predicted `block_x` vs true, colored by true block position (or by near-wall / angle bucket). Highlight high-residual regions on the board.

**Reads:** residuals clustering near walls/extreme angles → legibility is state-dependent, and those regions likely coincide with planning failures.
**Motivates:** conditional R² by region (e.g. near-wall vs center).

### Fig-6 — Intervention magnitude sweep *(CA3)*

**Question:** is the legible geometry cleanly actionable?

**Method:** x = ε along a probe direction; y = predicted factor movement after one `P` step; one line per base-state region (free vs near-contact).

**Reads:** monotone/linear → usable for latent-space steering; kink/saturation → bounded usable region.
**Motivates:** slope + linearity (R² of the ε→movement fit) per region.

### Fig-7 — Rank spectrum + per-dim variance *(CA2)*

**Question:** why is effective rank ~22/192 — hard low-rank or soft?

**Method:** (a) scree plot of sorted `z`-covariance eigenvalues (log-y), mark the participation-ratio point; (b) per-dimension variance histogram; (c) decodability of block pose vs number of PCA components included.

**Reads:** sharp elbow + dead tail → genuine low-rank (don't scale); moderate correlated tail → entanglement.
**Motivates:** the CA2 verdict scalar (elbow index; dead-dim fraction).

---

## 3. Shared utilities (the load-bearing correctness bits)

- **`RealFittedProjector`** — fits PCA on a supplied *real*-latent matrix, exposes `.transform()` and `.explained_variance_ratio_`. Refuses to fit on any array flagged `imagined=True` (guardrail against Principle 2 violations). Caches per (episode, seed).
- **`contact_events(state)`** — env-specific; PushT: block–pusher relative distance below contact threshold, block-to-wall distance below margin. Returns per-step boolean masks. Reacher stub later.
- **`nn_retrieve(z_query, bank, metric='l2')`** — nearest encoded frame for decoder-free visualization; returns index + distance.

---

## 4. Anti-footgun checklist (CI-enforced where possible)

- Projector fit only on real `z`; unit test asserts a joint-fit call raises.
- Every figure prints captured-variance in the title/caption.
- No t-SNE/UMAP import inside any `fig_*` used for dynamics/distance (lint rule).
- Filmstrip uses retrieval, never a decoder (none exists).
- Each `fig_*` docstring states the scalar it motivates; a test greps for it.

---

## 5. Reusability going forward

- **Env-agnostic core:** dump schema + projector + figures are env-independent; only `contact_events` is env-specific. Reacher and future envs plug in a mask function.
- **Phase-agnostic:** works on any `dump.npz`, so it serves CA-train (pre/post-retrain drift overlays), a re-opened C1 (closed-loop scorer landscapes), and later real-robot logs (same schema).
- **Writeup-ready:** figures export vector PDF at publication size; Fig-1/Fig-2/Fig-3 are the paper's mechanism figures for the rollout-fidelity result.
- **Cheap to extend:** new figure = one pure function + one CLI entry; no pipeline change.

---

## 6. Implementation surfaces

| Concern | File |
|---|---|
| Core module | `le-wm/viz.py` |
| CLI | `scripts/viz.py --dump … --figs …` |
| CEM candidate capture (Fig-2) | `eval_live.py --capture-cem` → `cem_capture.npz` |
| Contact extraction | `viz.py::contact_events` (PushT now; Reacher later) |
| Tests | `scripts/test_viz.py` (projector guardrail, variance print, no-nonlinear lint, retrieval sanity) |

---

## 7. Build order (decisiveness-first)

1. **`RealFittedProjector` + `load_dump` + Fig-1 (oracle overlay).** Directly serves the CA0 fork; reuses the oracle bank. *Highest value, lowest cost.*
2. **`--capture-cem` hook + Fig-2 (CEM landscape w/ oracle).** The decisive search-vs-model figure; needs the one eval hook.
3. **Fig-3 (drift+contacts)** and **Fig-7 (rank spectrum)** — CA1/CA2 companions, pure-from-dump.
4. **Fig-4 filmstrip, Fig-5 probe-faithfulness, Fig-6 intervention sweep** — mechanism depth, as CA1/CA3 mature.
5. **`test_viz.py`** in parallel from step 1 (the guardrails matter most on the plots you trust most).

Fig-1 and Fig-2 first: together they visually resolve the two live questions — *does the rollout drift* (Fig-1) and *is the failure search or model* (Fig-2) — that the entire C-alt fork turns on.
