# Phase C-alt Plan — Localize the Rollout-Fidelity Fault, Cheap Before Expensive

**Date:** 2026-08-30
**Spec lineage:** resolves the redirect recorded in [`13_phase_c0_report.md`](13_phase_c0_report.md) (Outcome B, model fidelity) and [`12a_c03_redo.md`](12a_c03_redo.md).
**Inherits:** posture + stability rules in `00_decisions.md`; diagnostics in `09`–`13`.
**Stack:** `le-wm/` on stable-worldmodel; live eval; reuse existing dumps (`phase_b_dump/`, `c0_oracle_livebank/`).
**Change control:** D6 keep is recorded in `00` (not a flip). D7 (asymmetry) stays proposed. Nothing gets built past a gate until the gate reports. This plan does **not** start CA0 code.
**Visual artifacts:** several checks below emit figures specified in [`15_viz_toolkit_spec.md`](15_viz_toolkit_spec.md). The scalar is the citable result; the figure motivates the next scalar.

---

## 0. Orientation — what C0 actually established, and the trap to avoid

C0 resolved to **Outcome B**: on reachable on-policy (`short_horizon`) oracle pairs, replay succeeds in env (94%) but `P` imagines the *known-correct* actions ending **8.23** from `z*` when the start was only **2.61** — the rollout lands *further* from the goal than it began. Two facts constrain the redirect:

1. **The redirect is NOT "action-conditioning."** C0.2 already showed `P` is action-**live** on diverse actions (shuffle−true gap 1.77). AdaLN works. So the fault is **rollout fidelity over horizon**, not a deaf predictor. Do not reach for conditioning fixes.
2. **The root is on the *easy* goals.** This fidelity fault appears on reachable `short_horizon` pairs, so Phase B's "hard-offset geometry binds" is downstream of a predictor whose open-loop rollout already drifts on easy goals. **Reframe for the writeup: rollout fidelity is the root; hard-offset geometry was a red herring.**

The trap: "Outcome B → retrain the predictor" is premature. There is a cheaper fork that decides whether retraining is even needed.

---

## 1. The central C-alt fork (this orders everything)

The 8.23 endpoint drift has two very different explanations, with different fixes and different costs:

- **Accumulation (cheap fix):** the per-step model is faithful, but 25 steps of open-loop rollout accumulate drift off the encoder manifold. Fix = **rollout/execution protocol** (shorter open-loop segments; receding-horizon re-encode). No retrain. May re-open the actor (C1) with a closed-loop scorer.
- **Per-step infidelity (expensive fix):** even single/few-step prediction leaves the manifold near the goal. Fix = **retrain** (multi-step rollout loss), which is also the de-confounding, architecture-level experiment.

**CA0 discriminates these for near-zero cost. Run it first. Build nothing until it reports.**

Note vs B2: B2's flat `T`-sweep varied the **CEM planning+execution horizon** on the **hard-offset** band with the real eval loop. CA0 varies **open-loop imagination length inside the scoring rollout** on **reachable** oracle pairs. Different knob, different band — no contradiction. A coherent unified outcome is possible: drift is accumulation (CA0 recovers) *and* hard-offset is a separate search/reachability problem (B2 flat).

---

## 2. CA0 — Closed-loop imagine discriminator *(the hinge; cheapest; first)*

**Question:** does re-encoding the true observation every `m` steps collapse the 8.23 endpoint drift?

**Method.** Reuse the `c0_oracle_livebank/seed0/` pairs (oracle actions `a_{t:t+25}`, start, `z*`, and the *true intermediate observations* along each oracle rollout — these exist because the bank was collected live). For `m ∈ {1, 3, 5, 12, 25}`:

1. Imagine `m` steps open-loop from the current latent using the oracle actions.
2. Re-encode the **true** observation at step `+m` (teacher-force the latent back onto the manifold).
3. Continue until 25 steps consumed.
4. Record ‖ẑ_end − z*‖ and toward-goal fraction, as a function of `m`.

`m=25` is pure open-loop (the C0 result: 8.23 / 2%). `m=1` is fully teacher-forced (near-perfect by construction). **The shape of the curve between them is the finding.**

**Emits:** the CA0 recovery curve, and the **oracle-imagine overlay** figure (`15` §Fig-1) for `m=25` vs `m=5`.

**Pre-registered outcomes:**

- **CA0-ACCUMULATION** — drift collapses by `m=5` (e.g. toward-goal ≥ ~60% and ‖ẑ_end−z*‖ ≤ ~3 at `m=5`).
  → Per-step model is faithful; the fault is open-loop accumulation.
  → **Fix is protocol, not retrain:** (a) scoring rollouts use short segments; (b) execution uses receding-horizon re-encode every ~5 steps. **This re-opens C1** (actor with a closed-loop scorer) — record it as a *conditional un-gating* of C1, to be re-specced.
  → Retrain (CA-train) is **not** motivated by fidelity. Proceed to CA1/CA2 for mechanism, then re-spec C1.

- **CA0-INFIDELITY** — drift persists even at `m=3` (or `m=1` is itself poor).
  → The per-step predictor is genuinely unfaithful near the manifold.
  → **Retrain is motivated (CA-train).** This is the deeper, stronger negative result and the architecture-level experiment.

- **Interpretation guard.** Confirm the `m=1` teacher-forced case is near-perfect; if it is *not*, the fault is single-step prediction (strongest infidelity signal) — go straight to CA-train.

**Deployment caveat (state in the writeup):** CA0 re-encodes *true future* observations, which is a **diagnostic** (you can't see the future while planning). Its positive result maps to a **deployable** receding-horizon *execution* protocol (act `m`, re-encode, replan) — not to omniscient scoring. Be precise about this so a reader doesn't mistake CA0 for a planner.

---

## 3. CA1 — Drift mechanism: steady vs contact-driven *(parallel with CA0)*

**Question:** is the drift smooth accumulation, or does it jump at physical events (block–pusher contact, wall contact)?

**Method.** Per-step ‖ẑ_t − z_t‖ under true actions (reuse `c0_oracle_livebank` and the diverse bank), overlaid with **contact events** extracted from the sim `state` (relative pose crossing contact distance; block near wall). Decompose mean drift into **free-space steps** vs **contact steps**.

**Emits:** drift-with-contacts curve (`15` §Fig-3).

**Reads:**
- Flat drift with jumps at contacts → **contact-representation** fault; the fix (whether protocol or retrain) must target contact dynamics specifically, and PushT's difficulty is concentrated there.
- Smooth monotone accumulation → generic rollout accumulation; consistent with CA0-ACCUMULATION.

This sharpens *which* fix, and it feeds the writeup's mechanism section either way.

---

## 4. CA2 — The rank investigation: why `z` uses ~22 of 192 dims

**Question:** is the low effective rank (22.5/192) a problem (over-compression / under-training) or benign (PushT is simply low-complexity)? This decides whether "utilize current size" is even a meaningful goal before any scaling talk.

**Method.**
1. **Scree/spectrum:** sorted eigenvalues of the `z` covariance (the participation ratio's source). Sharp elbow at ~22 vs a long moderate tail distinguishes hard low-rank from soft.
2. **Per-dimension variance across trajectories/time:** are the ~170 quiet dims dead (near-zero variance) or moderately active but correlated (entanglement eating effective rank)?
3. **Cross-check against legibility:** does adding more PCA components improve linear decodability of block pose beyond the top ~22? If not, the extra dims carry nothing control-relevant.

**Emits:** spectrum + per-dim variance figures (`15` §Fig-7).

**Reads:**
- Hard elbow + dead tail → SIGReg compression or task-simplicity; capacity is genuinely ~22, scaling the token is pointless (confirms the earlier "don't scale" call with mechanism).
- Moderate correlated tail → entanglement; the representation *could* carry more but doesn't factor it — relevant to the longer representation-first arc.

**Non-goal:** this does **not** trigger any change now. It is understanding + writeup, and it informs CA-train's design if reached.

---

## 5. CA3 — Intervention sweeps: is the legible geometry cleanly *actionable*

**Question:** B1 showed block_x is linearly readable and one intervention moved `P`. Is the mapping *clean* (monotone, usable for gradient-style steering) or kinked/saturating?

**Method.** For the `block_x` (and `block_y`) probe direction, sweep magnitude `ε` across a range, apply `z + ε·dir`, roll `P` one step with a fixed action, plot predicted factor movement vs `ε`. Repeat at several base states (free-space vs near-contact).

**Emits:** intervention magnitude-sweep figure (`15` §Fig-6).

**Reads:** monotone clean line → geometry is usable for a search/actor that steers in latent space; kink/saturation, especially near contact → the usable region is bounded, which bounds any latent-space actor and connects back to CA1.

---

## 6. CA-train — Retrain from scratch *(CONDITIONAL: only if CA0-INFIDELITY)*

**Do not start unless CA0 reports INFIDELITY.** If reached, this is the de-confounding, architecture-level experiment — and the gateway to the longer representation-first arc.

**Why it's motivated only here:** everything to date is entangled with a checkpoint you did not train. "Frozen LeWM checkpoint drifts" is a weak claim; "JEPA trained-for-prediction drifts, and a multi-step rollout loss does/doesn't fix it" is an architecture-level claim you can only make by owning the run.

**Design (spec on trigger, not now):**
- Retrain LeWM on PushT with an explicit **multi-step rollout loss** (predict `n>1` steps, match to encoded futures) alongside `L_pred + λ_SIGReg`. The single knob most likely to fix rollout fidelity, and unavailable on frozen weights.
- Keep D6 discipline (reachability stays out of the trunk) unless a *separate* gate flips it.
- Compare rollout drift (CA0 curve) pre/post as the primary readout.
- **This is also the only place the open-endedness / representation-first direction can begin** — goal-diverse or better-structured training requires a run you own. Note as the on-ramp; do not scope it here.

**Explicitly NOT part of CA-train:** scaling the token or model. CA2's rank result (22/192) says capacity is not the binding constraint; scaling stays off until a harder task drives effective rank toward the dimension.

---

## 7. Decision table & triggers

| ID | State | C-alt status | Trigger |
|---|---|---|---|
| **CA0 fork** | Open | **Run first.** | ACCUMULATION → protocol fix + re-open C1. INFIDELITY → CA-train. |
| **C1 actor** | Gated off (C0 Outcome B) | **Conditionally re-opened** by CA0-ACCUMULATION (closed-loop scorer). | CA0-ACCUMULATION recorded. |
| **CA-train (retrain)** | Not started | **Gated on CA0-INFIDELITY.** | CA0 reports per-step infidelity. |
| **D6** | Keep/extract | **Unchanged.** | n/a in this plan. |
| **Scaling (token/model)** | Off | **Stays off.** | CA2 shows rank→dim on a harder task (future). |
| **Sep / new cost head** | Off | **Stays off.** | Not on any C-alt path. |
| **Hierarchy / C2** | Composition-only | **Untriggered.** | Only if a re-specced C1 underperforms on long goals. |

---

## 8. Metrics

- CA0 recovery curve: ‖ẑ_end−z*‖ and toward-goal fraction vs re-encode interval `m`.
- `m=1` teacher-forced sanity (must be near-perfect).
- CA1: free-space vs contact-step mean drift; drift-jump magnitude at contacts.
- CA2: eigenvalue spectrum, per-dim variance histogram, decodability vs #PCA-components.
- CA3: predicted-factor-movement vs `ε` slope/linearity, per base-state region.
- All pre-registered thresholds fixed **before** running (CA0 especially).

---

## 9. Risks

| Tier | Risk | Cheap guard |
|---|---|---|
| 1 | Retrain launched on a fidelity fault that was just open-loop accumulation | **CA0 before CA-train** (the whole point) |
| 1 | CA0 misread because `m=1` isn't actually near-perfect | Interpretation guard in §2 |
| 2 | Contact-drift mistaken for generic accumulation | **CA1** decomposition |
| 2 | "Utilize current size" chased when capacity is genuinely ~22 | **CA2** spectrum before any capacity work |
| 2 | Re-opened C1 built before the closed-loop scorer is validated | C1 re-spec must inherit CA0's `m` |
| 3 | Viz over-reading (pretty picture ≠ result) | `15` discipline rules; every figure names the scalar it motivates |

---

## 10. Success criteria

- **CA0 done** = recovery curve across `m`, with a **fork decision recorded** (ACCUMULATION → protocol + re-open C1; INFIDELITY → CA-train).
- **CA1/CA2/CA3 done** = mechanism (contact vs steady), rank verdict (hard vs soft low-rank), actionability (clean vs kinked) — each a figure + a scalar.
- **C-alt pass** = we can state, with data, *why* the rollout drifts (accumulation vs per-step infidelity; contact vs steady) and therefore *which* fix (protocol/re-open-C1 vs retrain) is warranted — replacing "model fidelity fails" with a mechanism.
- **Non-goals:** scaling; building the actor before CA0; retraining before CA0; real robots; open-endedness.

---

## 11. Immediate actions (ordered by decisiveness)

```bash
# CA0 — the hinge. Reuse the oracle bank; sweep re-encode interval m.
python scripts/closed_loop_imagine.py \
  --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0/ \
  --reencode-every 1 3 5 12 25 --pack tile_block \
  --out eval_results/pusht/ca0_closed_loop/seed0/
#   -> also emits the oracle-imagine overlay (15 Fig-1) for m=25 vs m=5

# CA1 — drift vs contact events (parallel)
python scripts/drift_by_event.py \
  --dump eval_results/pusht/c0_oracle_livebank/seed0/ \
  --out eval_results/pusht/ca1_drift_contacts/seed0/

# CA2 — rank spectrum
python scripts/rank_spectrum.py \
  --dump eval_results/pusht/phase_b_dump/seed0/dump.npz \
  --out eval_results/pusht/ca2_rank/seed0/

# CA3 — intervention magnitude sweep
python scripts/intervene_sweep.py \
  --dump eval_results/pusht/phase_b_dump/seed0/dump.npz \
  --factor block_x block_y --eps-range -3 3 --out eval_results/pusht/ca3_sweep/seed0/

# record CA0 fork in experiment_log.md; only then spec re-opened-C1 or CA-train
```

**CA0 first.** It is the one experiment that decides whether the next big move is a cheap protocol change (and a re-opened actor) or an expensive retrain — and it costs almost nothing because the oracle bank already exists.
