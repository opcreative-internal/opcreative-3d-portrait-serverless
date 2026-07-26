# RunPod Serverless image for opcreative-3d-portrait pipeline_v14.
# Slim base: nvidia/cuda 12.4 runtime (~3GB). GH Actions runner has ~14GB free — must stay under.
# Weights are downloaded at first cold-start (persist on network volume if attached).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# PYTHONPATH: put git-cloned repos on path so their inner packages import cleanly.
# /opt/DAv2 contains a `depth_anything_v2/` package (pipeline_v11 imports it).
# /opt/TripoSG contains scripts/inference_triposg (module_a_retune uses it as subprocess).
ENV HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/DAv2:/opt/TripoSG

# System deps (minimal - just what pipeline_v14 needs at runtime)
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git wget curl ca-certificates \
        libopengl0 libgl1 libgles2 libglu1-mesa libegl1 libglvnd0 \
        libxrender1 libxi6 libxext6 libxrandr2 libx11-xcb1 \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# Torch + CUDA 12.4 wheel (must go first for diso build isolation)
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Pipeline v14 deps
RUN python -m pip install \
        Pillow==11.0.0 \
        mediapipe==0.10.35 \
        torchvision==0.19.0 \
        timm==1.0.9 \
        trimesh==4.4.9 \
        pygltflib==1.16.2 \
        xatlas==0.0.9 \
        pymeshlab \
        opencv-python-headless \
        scipy==1.14.1 \
        scikit-image==0.24.0 \
        numpy \
        huggingface_hub \
        matplotlib \
        fast-simplification \
        runpod \
        diffusers \
        transformers \
        accelerate \
        omegaconf \
        einops

# V21f4: pin diffusers/transformers/accelerate to versions compatible with torch 2.4.0.
# The un-pinned `diffusers transformers accelerate` above resolves to latest which pull
# APIs not yet in torch 2.4 -> "infer_schema(func): Parameter q has unsupported type
# torch.Tensor" on diffusers autoencoder_kl import; peft 0.19+ needs DTensor from newer
# torch. Force-downgrade to a set known to work with torch 2.4 + TripoSG.
RUN python -m pip install --force-reinstall --no-deps \
        transformers==4.44.2 \
        diffusers==0.30.3 \
        accelerate==0.34.2

# TripoSG needs: peft, jaxtyping, typeguard. `diso` is a CUDA extension that requires
# nvcc (not in this cudnn-runtime base). Per Fable v3 verdict 20260725:
# 1. install the Python-only TripoSG deps EXPLICITLY (no silent failures)
# 2. stub diso.py so the `from diso import DiffDMC` import passes; runtime path forced
#    to hierarchical_extract_geometry (skimage marching cubes) — 10-20s slower, no diso
# 3. TripoSG's own requirements.txt is skipped because `diso` in it kills the whole
#    `pip install -r` transaction (single-shot atomic install)
RUN python -m pip install \
        peft==0.11.1 \
        jaxtyping==0.2.34 \
        typeguard==2.13.3

RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSG.git /opt/TripoSG \
    && rm -rf /opt/TripoSG/.git
# Stub diso — /opt/TripoSG is on PYTHONPATH so this shim resolves first.
# Since DiffDMC.__init__ raise means any flash-decoder code path bombs, force
# use_flash_decoder=False at pipeline default level. Non-fatal if grep finds none
# (upstream may already default False in newer commit).
RUN printf 'class DiffDMC:\n    def __init__(self, *a, **kw): raise RuntimeError("diso stub reached at runtime; force use_flash_decoder=False")\n' > /opt/TripoSG/diso.py \
    && sed -i 's/use_flash_decoder\s*:\s*bool\s*=\s*True/use_flash_decoder: bool = False/g' /opt/TripoSG/triposg/pipelines/pipeline_triposg.py 2>/dev/null || true \
    && sed -i 's/use_flash_decoder\s*=\s*True/use_flash_decoder=False/g' /opt/TripoSG/triposg/pipelines/pipeline_triposg.py 2>/dev/null || true \
    && sed -i 's/use_flash_decoder\s*=\s*True/use_flash_decoder=False/g' /opt/TripoSG/scripts/inference_triposg.py 2>/dev/null || true \
    && echo "=== diso stub + use_flash_decoder patches applied ==="

# Bake TripoSG + RMBG weights into the exact local_dirs the CLI reads from.
# `scripts/inference_triposg.py` uses snapshot_download(local_dir='pretrained_weights/...')
# relative to cwd=/opt/TripoSG, so weights MUST live at these paths (HF cache won't be found).
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='VAST-AI/TripoSG', local_dir='/opt/TripoSG/pretrained_weights/TripoSG'); \
print('TripoSG weights baked at /opt/TripoSG/pretrained_weights/TripoSG')"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='briaai/RMBG-1.4', local_dir='/opt/TripoSG/pretrained_weights/RMBG-1.4'); \
print('RMBG weights baked at /opt/TripoSG/pretrained_weights/RMBG-1.4')"

# Depth-Anything-V2 code (imports `depth_anything_v2.dpt.DepthAnythingV2`)
# PYTHONPATH env above adds /opt/DAv2 to sys.path -- proper mechanism (site-packages
# dist-packages layout differs between distros so writing a .pth file is fragile).
# Also install its extra deps if any (typically just torch/torchvision/timm already).
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 /opt/DAv2 \
    && rm -rf /opt/DAv2/.git \
    && (test -f /opt/DAv2/requirements.txt && python -m pip install --no-deps -r /opt/DAv2/requirements.txt || echo "DAv2 has no extra reqs")

# Pre-fetch face_landmarker (~4MB) into /workspace where pipeline_v11 default expects it.
# Also set env var to be explicit; and mirror to /models.
RUN mkdir -p /workspace /models \
    && curl -fL -o /workspace/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
    && cp /workspace/face_landmarker.task /models/face_landmarker.task
ENV FACE_LANDMARKER_MODEL=/workspace/face_landmarker.task

# Pre-fetch Depth-Anything-V2 base weights (~380MB) to HF cache so first job is fast.
# Use hf_hub_download inside python to place in canonical HF_HOME layout.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='depth-anything/Depth-Anything-V2-Base', filename='depth_anything_v2_vitb.pth'); \
print('DAv2 vitb cached at:', p)" || echo "DAv2 preload warning (will fetch at runtime)"

# App code + verifier (goes LAST so edits don't invalidate heavy layers above)
WORKDIR /app
COPY handler.py pipeline_v14.py pipeline_v15.py pipeline_v16.py pipeline_v17.py pipeline_v18.py pipeline_v19.py pipeline_v21.py pipeline_v11_depth_umeyama.py module_a_retune.py module_e_texture.py module_e_engrave.py module_e_ssle.py module_f_graft.py module_wrap_texture.py module_wrap_texture_v19.py module_wrap_texture_v21.py module_autofront_v21h.py module_bgremove_v21h.py verify_imports.py /app/

# Comprehensive build-time verification. Import list is a real .py file (not python -c)
# because multi-line `for/try:` inside python -c is a SyntaxError. This step FAILS the
# build if ANY runtime dependency is missing.
RUN python /app/verify_imports.py

# Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
