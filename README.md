# ws_weltmodelle

Local workspace for JEPA / world-model research.

## Layout

| Path | Role |
|------|------|
| `ws_le-wm/le-wm` | [yojuna/le-wm](https://github.com/yojuna/le-wm) **submodule** (`feat/lewm-phi`); upstream [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) |
| `ws_le-wm/docs` | Design specs and decisions for lewm-phi |
| `ws_le-wm/stablewm` | Local `$STABLEWM_HOME` cache (checkpoints ignored) |
| `docker/` | CUDA 12.6 GPU container (see [`docs/docker_usage.md`](docs/docker_usage.md)) |
| `docs/` | Workspace how-tos (Docker, …) |
| `ws_dino_wm` | Placeholder for DINO-WM experiments |
| `wiki` | Reference papers / notes |

## Submodule

```bash
git clone --recurse-submodules git@github.com:yojuna/ws_weltmodelle.git
# or after clone:
git submodule update --init --recursive
```

Work on LeWM code inside `ws_le-wm/le-wm`. Topic branches use the same name
in both repos: `exp/<name>` (campaigns), `feat/<name>` (lasting features),
`tooling/<name>` (Docker, viz, rules). See `.cursor/rules/31-git-hygiene.mdc`.
Merge PRs into **`main`** on both repos (submodule `main` is the fork, not Lucas).

Research docs: [`ws_le-wm/docs/README.md`](ws_le-wm/docs/README.md). Locked decisions: [`ws_le-wm/docs/00_decisions.md`](ws_le-wm/docs/00_decisions.md). Current plan: [`ws_le-wm/docs/16_fidelity_retrain_plan.md`](ws_le-wm/docs/16_fidelity_retrain_plan.md) (Part B not started).

GPU work: see [`docs/docker_usage.md`](docs/docker_usage.md). Always `cd docker && ./run.sh …` (including CPU tests). Do not use host/system Python or create `ws_le-wm/le-wm/.venv`.
