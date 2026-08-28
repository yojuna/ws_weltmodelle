# JEPA Reachability Project Docs

Working documents for the **LeWM trunk + thin reachability projection `φ` + latent-goal practice + few-shot deploy** line of work.

## Documents

| Doc | Purpose |
|-----|---------|
| [00_decisions.md](00_decisions.md) | Locked design decisions (source of truth for scope) |
| [01_design_spec.md](01_design_spec.md) | Technical design: architecture, losses, data, train/eval protocols |
| [02_implementation_plan.md](02_implementation_plan.md) | Phased implementation plan mapped to this repo |
| [03_quasimetric_iql_t3.md](03_quasimetric_iql_t3.md) | Protocol T3 IQL reference (attempted; gate failed) |
| [04_euclid_multiseed.md](04_euclid_multiseed.md) | Multi-seed Euclidean φ replicate (short + offset) |
| [05_offset_autopsy.md](05_offset_autopsy.md) | Offset failure autopsy protocol (real vs imagined costs) |
| [06_imagined_phi.md](06_imagined_phi.md) | H1 fix: train φ on predictor futures (eval failed) |
| [07_status_synthesis.md](07_status_synthesis.md) | Consolidated results + next-step fork |
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

## Code anchors

- Model: `le-wm/jepa.py`, `le-wm/module.py`
- Train: `le-wm/train_phi.py`, `train_phi_imagined.py`, `train_phi_iql.py`
- Eval / logging: `le-wm/eval_live.py`, `le-wm/eval_logging/`, `scripts/offset_autopsy.py`
- PushT eval config: `le-wm/config/eval/pusht.yaml`

## Status (2026-08-28 EOD)

- **Short-horizon:** Euclidean `φ` (v2) **beats L2** multi-seed — claim partially supported.
- **Offset / firm gate:** no thin cost head reliably wins; absolute success ~2–8%.
- **Tried & failed gates:** IQL-on-`φ` (T3); imagined-future `φ` (H1).
- **Next:** choose fork in `07_status_synthesis.md` (H2 hybrid bookkeeping vs system change).
- **Env v1:** PushT only · **D6** still in force unless Sep is explicitly flipped.
