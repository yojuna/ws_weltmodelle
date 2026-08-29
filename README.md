# ws_weltmodelle

Local workspace for JEPA / world-model research.

## Layout

| Path | Role |
|------|------|
| `ws_le-wm/le-wm` | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) **submodule** (feature work on `feat/lewm-phi`) |
| `ws_le-wm/docs` | Design specs and decisions for lewm-phi |
| `ws_le-wm/stablewm` | Local `$STABLEWM_HOME` cache (checkpoints ignored) |
| `docker/` | CUDA 12.6 GPU container; Python baked at `/opt/venv` (see `docker/README.md`) |
| `ws_dino_wm` | Placeholder for DINO-WM experiments |
| `wiki` | Reference papers / notes |

## Submodule

```bash
git clone --recurse-submodules <this-repo>
# or after clone:
git submodule update --init --recursive
```

Work on LeWM code inside `ws_le-wm/le-wm` on branch `feat/lewm-phi`.

GPU work: `cd docker && ./run.sh up --build` once (bakes torch + `stable-worldmodel` into the image), then `./run.sh` for a shell. The named container `weltmodelle-lewm` uses `restart: unless-stopped`. Code, checkpoints, and `eval_results` live on the host bind (`..:/workspace`). Do not recreate `ws_le-wm/le-wm/.venv` on the host. Prefer `./run.sh stop` / `start` over `docker compose down`.
