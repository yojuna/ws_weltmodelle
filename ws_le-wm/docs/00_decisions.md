# Locked Design Decisions

**Date locked:** 2026-08-28  
**Scope:** v1 on PushT, reuse `ws_le-wm` / LeWM stack  
**Normative:** If something conflicts with brainstorm notes in `drafts.md`, this file wins.

---

## Research posture

- Fully exploit JEPA: prediction-rich latent world model is primary.
- Planning alignment lives in a **thin** readout, not by reshaping the whole encoder for control.
- No goal-image stream at deployment; task is configured by few-shot latent anchors.
- Prefer the smallest system that can falsify the claim; add complexity only after measured failures.

---

## Decision table

| ID | Topic | Choice | Deferred / fallback |
|----|--------|--------|---------------------|
| D1 | Deploy configurator | **C1** — few-shot latent anchor (encode 1–N example observations once → `z*`) | C3 privileged→latent eval adapter later |
| D2 | Training schedule | **Joint** updates of `E`, `P`, `φ` | Freeze-`E` windows only if goals/`φ` unstable |
| D3 | Reach loss | **Euclidean hindsight-`k` (best planning signal so far = v2)**; IQL T3 failed; imagined-φ H1 failed offset | No more thin-head churn. H2 hybrid is optional bookkeeping. Hierarchy / C2 untriggered; next work is C-alt ([`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md)), not a new cost head |
| D4 | Env | **PushT** is the v1 claim env | Reacher is a Phase B **diagnostic** env (live_reset + probes), not a second claim. Dropped from the PushT legibility claim after C0.1 |
| D5 | Task success `S` in learning | **None** | Eval metrics may still use sim success |
| D6 | Trunk shaping from reachability | **`φ` only (`λ ≈ 0`)** — stop-grad into `E` | **Keep / extract** (B1 2026-08-29; C0.1 seeds 1–2 2026-08-30). Do not start Sep. Do not flip on on-path Spearman(`k`) |

---

## Explicitly out of v1

- Per-step / streaming goal images in the control loop (LeWM/DINO default)
- Family A (in-scene goal as the only task interface)
- FF-JEPA latent planner `G` / hierarchy — re-opened as a Phase B candidate, **not enabled**. C2 is composition-only and untriggered. Do not fire as a rollout-shortening patch (B2 `T` sweep was flat).
- Dense rewards or TD-MPC-style reward latents
- Affordance / arena self-play outer loop as a required component
- Proposer–solver asymmetric self-play
- Behavior-cloning policy replacing CEM
- Cross-env shared weights

---

## Stability / encoding rule (always on)

Store **goal pixels** (or dataset indices), not stale latents, for reachability targets.  
**Re-encode** with the current `E` whenever computing `L_reach` or setting deploy anchors.

This avoids building freeze-`E` until joint training proves unstable.

---

## Ablations required to defend the claim

1. **Cost:** L2(`z`, `z*`) vs `d_φ(φ(z), φ(z*))` with the same trunk and planner.
2. **Head:** trained `φ` vs frozen random `φ` (or identity).
3. **Configurator:** C1 few-shot anchors vs legacy goal-image eval (existing stack) as reference—not as the training target.

---

## Change control

When flipping a deferred item on:

1. Record date + reason (failure observed, not anticipation).
2. Update `01_design_spec.md` and `02_implementation_plan.md`.
3. Keep v1 claim falsifiable; do not silently change the bet.

**2026-08-30 (not a flip):** D6 stay-as-keep recorded after B1 + C0.1. C0.3-redo resolved to Outcome B (model fidelity) → planning follows [`14_phase_c_alt_plan.md`](14_phase_c_alt_plan.md). D7 (asymmetry) stays proposed, not locked here. C1 actor is gated off until CA0 says otherwise.
