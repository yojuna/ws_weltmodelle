# ws_weltmodelle

Local workspace for JEPA / world-model research.

## Layout

| Path | Role |
|------|------|
| `ws_le-wm/le-wm` | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) **submodule** (feature work on `feat/lewm-phi`) |
| `ws_le-wm/docs` | Design specs and decisions for lewm-phi |
| `ws_le-wm/stablewm` | Local `$STABLEWM_HOME` cache (checkpoints ignored) |
| `docker/` | CUDA 12.6 GPU container (see [`docs/docker_usage.md`](docs/docker_usage.md)) |
| `docs/` | Workspace how-tos (Docker, …) |
| `ws_dino_wm` | Placeholder for DINO-WM experiments |
| `wiki` | Reference papers / notes |

## Submodule

```bash
git clone --recurse-submodules <this-repo>
# or after clone:
git submodule update --init --recursive
```

Work on LeWM code inside `ws_le-wm/le-wm` on branch `feat/lewm-phi`.

GPU work: see [`docs/docker_usage.md`](docs/docker_usage.md). Short version: `cd docker && ./run.sh up --build` once, then `./run.sh`. The container does not auto-start on reboot; `run.sh` starts it and stops it when the session ends. Do not recreate `ws_le-wm/le-wm/.venv` on the host.
