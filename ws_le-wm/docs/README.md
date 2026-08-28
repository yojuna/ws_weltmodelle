# JEPA Reachability Project Docs

Working documents for the **LeWM trunk + thin reachability projection `φ` + latent-goal practice + few-shot deploy** line of work.

## Documents

| Doc | Purpose |
|-----|---------|
| [00_decisions.md](00_decisions.md) | Locked design decisions (source of truth for scope) |
| [01_design_spec.md](01_design_spec.md) | Technical design: architecture, losses, data, train/eval protocols |
| [02_implementation_plan.md](02_implementation_plan.md) | Phased implementation plan mapped to this repo |
| [drafts.md](drafts.md) | Earlier brainstorm / FER notes (historical; not normative) |

## One-line claim (v1)

> On PushT, a frozen JEPA world model plus a thin projection `φ` trained by hindsight temporal regression enables planning to few-shot latent anchors without a goal-image stream, outperforming L2-in-`z` planning—without task success rewards or trunk value-shaping.

## Code anchors

- Model: `le-wm/jepa.py`, `le-wm/module.py`
- Train: `le-wm/train.py`, `le-wm/config/train/`
- Eval / logging: `le-wm/eval.py`, `le-wm/eval_live.py`, `le-wm/eval_logging/`
- PushT eval config: `le-wm/config/eval/pusht.yaml`

## Status

- **Decisions locked:** 2026-08-28 (see `00_decisions.md`)
- **Env v1:** PushT only
- **Not in v1:** IQL, freeze-`E` schedule, C3 privileged goals, Maze, FF-JEPA hierarchy, task success `S` in learning, `λ>0` trunk shaping
