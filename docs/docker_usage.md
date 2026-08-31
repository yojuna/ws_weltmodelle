# Docker usage

GPU container for this workspace. Files: [`../docker/`](../docker/). Always use
[`../docker/run.sh`](../docker/run.sh) — never `docker compose run` (it fights
the fixed name `weltmodelle-lewm`).

## Image

`weltmodelle-lewm:local` — CUDA 12.6, Python 3.10, torch (cu126),
`stable-worldmodel[train,env]`, scikit-learn, matplotlib, `datasets>=2`. The
venv is **`/opt/venv` inside the image**. Startup does not run pip.

Torch and CEM use the NVIDIA GPU. MuJoCo / PushT viewers use the previous
path: **EGL** offscreen, **GLFW** (or a pixel window) on the host X display.
Do not force `__GLX_VENDOR_LIBRARY_NAME=nvidia` (this laptop’s X screen is
Intel).

The named container does **not** start on boot. `./run.sh` starts it; leaving
the session (or `./run.sh stop`) sends SIGTERM and waits up to 30s. The repo is
bind-mounted; there are no Docker volumes. Host files survive stop and reboot.

| Host | In container | Role |
|------|----------------|------|
| this `ws_weltmodelle/` tree | `/workspace` | compose bind `..:/workspace` from `docker/`; **no** named volumes |
| `ws_le-wm/le-wm/` | `/workspace/ws_le-wm/le-wm` | cwd |
| `ws_le-wm/docs/` | `/workspace/ws_le-wm/docs` | research docs (parent git) |
| `ws_le-wm/stablewm/` | `/workspace/ws_le-wm/stablewm` | `$STABLEWM_HOME` |
| `docker/home/` | `/workspace/docker/home` | `$HOME` |
| (not a host path) | `/opt/venv` | image Python |

Needs NVIDIA driver + nvidia-container-toolkit. Do not create
`ws_le-wm/le-wm/.venv` on the host.

## Run

From `docker/`:

```bash
./run.sh up --build    # first time, or after Dockerfile changes (leaves it running)
./run.sh               # start if needed, interactive bash, then graceful stop
./run.sh python …      # start if needed, run the command, then graceful stop
./run.sh up            # start and leave running until stop or host shutdown
./run.sh stop          # graceful SIGTERM
```

If the container was already up (`./run.sh up`), a later `./run.sh python …`
does **not** stop it — call `./run.sh stop` when finished.

Examples:

```bash
./run.sh python -c "import torch; print(torch.cuda.get_device_name(0))"
./run.sh python eval_live.py --env reacher --episodes 1 --viewer --no-viewer-hold --no-video
```

**Don't** `docker compose down` unless you mean to delete the container
(image and host files stay). Don't `up --build` every session.

To change Python packages, edit the `Dockerfile` and `./run.sh up --build` once.
Do not install into a host `docker/.venv`.
