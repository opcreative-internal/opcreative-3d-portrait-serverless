# RunPod Serverless image for opcreative-3d-portrait pipeline_v14.
# Slim base: nvidia/cuda 12.4 runtime (~3GB). GH Actions runner has ~14GB free — must stay under.
# Weights are downloaded at first cold-start (persist on network volume if attached).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

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

# diso (CUDA extension — needs torch already installed)
RUN python -m pip install --no-build-isolation diso || echo "diso install warning"

# TripoSG repo (code only — weights fetched at runtime)
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSG.git /opt/TripoSG \
    && rm -rf /opt/TripoSG/.git \
    && python -m pip install -r /opt/TripoSG/requirements.txt || echo "TripoSG reqs partial"

# Depth-Anything-V2 code
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 /opt/DAv2 \
    && rm -rf /opt/DAv2/.git \
    && echo "/opt/DAv2" > /usr/lib/python3.10/site-packages/dav2.pth || true

# App code
WORKDIR /app
COPY handler.py pipeline_v14.py pipeline_v11_depth_umeyama.py module_a_retune.py module_e_texture.py /app/

# Pre-fetch face_landmarker (only 4MB) - weights >100MB download at cold-start
RUN mkdir -p /models \
    && curl -fL -o /models/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Quick sanity import test at build time
RUN python -c "\
import sys; sys.path.insert(0, '/opt/TripoSG'); sys.path.insert(0, '/opt/DAv2'); \
import torch, torchvision, PIL, cv2, mediapipe, trimesh, xatlas, pymeshlab, runpod; \
print('OK torch:', torch.__version__, 'cuda:', torch.version.cuda)"

# Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
