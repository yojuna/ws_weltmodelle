# weltmodelle GPU container

Dedicated CUDA 12.6 container for LeWM / lewm-phi. **Python is baked into the
image** at `/opt/venv`. The **workspace (code, checkpoints, eval results, shell
home)** is bind-mounted from the host. No named or anonymous Docker volumes.

The named container `weltmodelle-lewm` uses `restart: unless-stopped`, so it
comes back after a reboot. Prefer `./run.sh stop` / `./run.sh start` over
`docker compose down` (down removes the container; the image and host files stay).

Base image (already on this machine): `nvidia/cuda:12.6.3-base-ubuntu22.04`.
`uv` is copied from the local `ghcr.io/astral-sh/uv:0.8.22` image.

## Layout

| Host path | In container |
|-----------|----------------|
| `ws_weltmodelle/` | `/workspace` |
| image `/opt/venv` | torch cu126 + `stable-worldmodel[train,env]` |
| `docker/home` | `$HOME` (shell history, HF cache) |
| `ws_le-wm/le-wm` | cwd |
| `ws_le-wm/stablewm` | `$STABLEWM_HOME` |

## Start

```bash
cd docker
chmod +x run.sh entrypoint.sh
./run.sh up --build   # once: bake the image (slow pip). later: ./run.sh up
./run.sh              # interactive bash
```

`run.sh` **never** uses `docker compose run`. That would conflict with the
fixed `container_name: weltmodelle-lewm`. Every command is `compose up` (if
needed) then `compose exec`.

```bash
./run.sh python -c "import torch; print(torch.cuda.get_device_name(0))"
./run.sh python scripts/test_phase_b.py
./run.sh python eval_live.py --env reacher --episodes 1 --viewer --no-viewer-hold --no-video
```

After a host reboot the container should already be running (`unless-stopped`).
If Docker itself is stopped, start it then `./run.sh start`.

Rebuild the image only when deps change:

```bash
./run.sh up --build
```

## GPU

Requires NVIDIA driver + nvidia-container-toolkit. Headless MuJoCo/PushT use
`MUJOCO_GL=egl`. With `DISPLAY` set, do not force `SDL_VIDEODRIVER=dummy`.
