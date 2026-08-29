# GPU runtime for le-wm. Source is bind-mounted; do not copy the repo in.
# Python stack is baked at /opt/venv so the container is ready on first start.
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    STABLEWM_HOME=/workspace/ws_le-wm/stablewm \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# Ubuntu 22.04 already ships Python 3.10. GL/EGL + codecs for PushT / MuJoCo Reacher.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        curl \
        ca-certificates \
        pkg-config \
        patchelf \
        ffmpeg \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 \
        libglib2.0-0 \
        libgomp1 \
        libosmesa6 \
        libglfw3 \
        libglew2.2 \
        libx11-6 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libxrandr2 \
        libxi6 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Already on this machine as ghcr.io/astral-sh/uv:0.8.22
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

RUN uv venv --python 3.10 /opt/venv

# gymnasium[all] (via stable-worldmodel[env]) builds box2d-py from source.
# Own layer so the earlier apt install stays cached.
RUN apt-get update && apt-get install -y --no-install-recommends swig \
    && rm -rf /var/lib/apt/lists/*

# CUDA wheels first so the extra does not pull a CPU-only torch.
# BuildKit cache is compile-time only; runtime has no named volumes.
# Serial-ish downloads: parallel CUDA wheels starve extract on a slow link.
# Timeout on this RUN only — do not bump the top-level ENV (that rebuilds apt).
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=2 \
    uv pip install --python /opt/venv/bin/python \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126 \
    && UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=2 \
    uv pip install --python /opt/venv/bin/python \
        'stable-worldmodel[train,env]' \
        scikit-learn \
        matplotlib \
    && /opt/venv/bin/python -c "import torch; print('torch', torch.__version__)"

# Host uid 1000 runs the container; keep the venv world-readable/executable.
RUN chmod -R a+rX /opt/venv

# GL: libOpenGL.so.0 is GLVND dispatch (not pulled by libgl1). NVIDIA EGL ICD
# for offscreen pixels on the CUDA GPU. A GLX ICD is written then removed below
# so GLFW windows stay on Mesa (host X is Intel) — same as before the GLX force.
RUN apt-get update && apt-get install -y --no-install-recommends libopengl0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d /usr/share/glvnd/glx_vendor.d \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
         > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0"}}' \
         > /usr/share/glvnd/glx_vendor.d/10_nvidia.json

# HuggingFace datasets 1.x (pulled transitively) is incompatible with pyarrow 24.
RUN UV_HTTP_TIMEOUT=300 uv pip install --python /opt/venv/bin/python 'datasets>=2.14,<4' \
    && chmod -R a+rX /opt/venv/lib/python3.10/site-packages

COPY entrypoint.sh /opt/lewm/entrypoint.sh
RUN chmod +x /opt/lewm/entrypoint.sh
# GLFW/GLX must follow the host X GPU (Intel Mesa). Keep NVIDIA EGL ICD only.
RUN rm -f /usr/share/glvnd/glx_vendor.d/10_nvidia.json

WORKDIR /workspace/ws_le-wm/le-wm
ENTRYPOINT ["/opt/lewm/entrypoint.sh"]
CMD ["bash"]
