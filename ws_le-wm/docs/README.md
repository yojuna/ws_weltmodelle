# JEPA Reachability Project Docs

Working documents for the **LeWM trunk + thin reachability projection `φ` + latent-goal practice + few-shot deploy** line of work.

## Documents

| Doc | Purpose |
|-----|---------|
| [00_decisions.md](00_decisions.md) | Locked design decisions (source of truth for scope) |
| [14_phase_c_alt_plan.md](14_phase_c_alt_plan.md) | **Current plan:** localize rollout fidelity; CA0 reported INFIDELITY; CA-train not started |
| [15_viz_toolkit_spec.md](15_viz_toolkit_spec.md) | Viz toolkit **v3** (upgrade spec: tiers + scientific report) |
| [15_viz_toolkit_spec_v0.md](15_viz_toolkit_spec_v0.md) | Viz toolkit **v0** (implemented: `le-wm/viz.py` Figs 1–7) |
| [13_phase_c0_report.md](13_phase_c0_report.md) | Phase C0 confirmation gate (liveness, oracle, seeds, rank) |
| [12a_c03_redo.md](12a_c03_redo.md) | C0.3-redo: live-bank oracle, two-outcome gate |
| [12_phase_c_plan.md](12_phase_c_plan.md) | Phase C spec: C0 gate then actor / C-alt (C1 gated off) |
| [11_phase_b_report.md](11_phase_b_report.md) | Phase B B1/B2 results, findings, D6/hierarchy gates |
| [10_implementation_status.md](10_implementation_status.md) | What is in the tree vs still to run |
| [09_phase_b_plan.md](09_phase_b_plan.md) | Phase B spec: diagnose geometry vs drift vs representation |
| [08_phase_a_report.md](08_phase_a_report.md) | Full narrative: design, every campaign, results |
| [01_design_spec.md](01_design_spec.md) | Technical design: architecture, losses, data, train/eval protocols |
| [02_implementation_plan.md](02_implementation_plan.md) | Phased implementation plan mapped to this repo (Phase A; then points at 14) |
| [03_quasimetric_iql_t3.md](03_quasimetric_iql_t3.md) | Protocol T3 IQL reference (attempted; gate failed) |
| [04_euclid_multiseed.md](04_euclid_multiseed.md) | Multi-seed Euclidean φ replicate (short + offset) |
| [05_offset_autopsy.md](05_offset_autopsy.md) | Offset failure autopsy protocol (real vs imagined costs) |
| [06_imagined_phi.md](06_imagined_phi.md) | H1 fix: train φ on predictor futures (eval failed) |
| [07_status_synthesis.md](07_status_synthesis.md) | Phase A consolidated results (historical fork) |
| [experiment_log.md](experiment_log.md) | Chronological test log |
| [drafts.md](drafts.md) | Earlier brainstorm / FER notes (historical; not normative) |

## Campaign summaries

| Summary | One-line |
|---------|----------|
| [lewm_phi_v2_summary.md](lewm_phi_v2_summary.md) | First short-horizon φ > L2 signal |
| [lewm_phi_v3_summary.md](lewm_phi_v3_summary.md) | Offset n=50: no Euclidean φ beats L2 |
| [lewm_phi_euclid_multiseed_summary.md](lewm_phi_euclid_multiseed_summary.md) | Short win replicates; offset still weak |
| [lewm_phi_iql_v1_summary.md](lewm_phi_iql_v1_summary.md) | T3 IQL no-go |
| [lewm_phi_offset_autopsy_summary.md](lewm_phi_offset_autopsy_summary.md) | H1+H2 diagnosis |
| [lewm_phi_imagined_v1_summary.md](lewm_phi_imagined_v1_summary.md) | H1 train OK, offset worse |

## One-line claim (v1)

> On PushT, a frozen JEPA world model plus a thin projection `φ` trained by hindsight temporal regression enables planning to few-shot latent anchors without a goal-image stream, outperforming L2-in-`z` planning—without task success rewards or trunk value-shaping.

**Claim status:** partially supported (short-horizon, multi-seed). Not supported for offset / `φ`-alone. Current bottleneck is open-loop rollout fidelity (C0 Outcome B), not the cost head.

## Code anchors

- Model: `le-wm/jepa.py`, `le-wm/module.py`
- Train: `le-wm/train_phi.py`, `train_phi_imagined.py`, `train_phi_iql.py`
- Eval / logging: `le-wm/eval_live.py`, `le-wm/eval_logging/`, `scripts/offset_autopsy.py`
- Phase B diagnostics: `le-wm/phase_b.py`, `scripts/latent_dump.py`, `latent_probe.py`, `predictor_drift.py`, `run_horizon_sweep.sh`
- C0 oracle: `eval_logging/oracle_bank.py`, `scripts/oracle_bank.py`, `scripts/oracle_imagine.py`
- C-alt dumps / viz: `scripts/closed_loop_imagine.py`, `drift_by_event.py`, `le-wm/viz.py`, `scripts/viz.py`, `scripts/viz_report.py`
- PushT eval config: `le-wm/config/eval/pusht.yaml`

## Status (2026-08-30)

- **Short-horizon:** Euclidean `φ` (v2) **beats L2** multi-seed (36.7 vs 16.7); matched random `φ` is **25.0** — phenomenon replicated, only partly learned reachability.
- **Offset / firm gate:** no thin cost head reliably wins; absolute success ~2–10%. Phase B `T` sweep still **4–6%** L2 for T=2,3,5,8.
- **Tried & failed gates:** IQL-on-`φ` (T3); imagined-future `φ` (H1).
- **Phase B B1/B2 (run):** PushT linear state R² ~0.70, live `block_x` intervention hit-rate 1.0 → **D6 keep**. Offset L2 stays **4–6%** for T=2,3,5,8. Writeup: [`11_phase_b_report.md`](11_phase_b_report.md).
- **Phase C0 (run):** `P` is action-live on diverse actions (shuffle−true gap 1.77). Live-bank oracle: replay **94%**, CEM-L2 **50%**, imagine toward-goal **2%** (‖ẑ_end−z\*‖ 8.23 vs start 2.61) → **Outcome B, model fidelity.** Do not build C1. Writeup: [`13_phase_c0_report.md`](13_phase_c0_report.md).
- **C-alt CA0–CA3 (run, seed 0):** **CA0-INFIDELITY** (`m=1` toward 84% / d_end 1.43; `m=25` matches C0 8.23 / 2%). CA1 contact/free drift ratio ~1.2, not a contact spike. CA2 rank 22.5/192, 58% dead dims. CA3 `block_x` ε-sweep linear in free and contact. C1 stays gated; CA-train motivated, not started. Compact: [`experiment_log.md`](experiment_log.md).
- **Next:** viz toolkit **v3** ([`15_viz_toolkit_spec.md`](15_viz_toolkit_spec.md)) then CA-train only when explicitly specced. Not Sep, not a new cost head, not C1.
- **Env:** PushT is the claim env; Reacher is diagnostic only (dropped from the PushT legibility claim).
