# RunPod Serverless Docker image for opcreative-3d-portrait pipeline_v14.
# Base: RunPod's PyTorch 2.8 + CUDA 12.8 image with Python 3.10 preinstalled.
# Build: docker build -t <dockerhub-user>/cowork-3d-v14:latest .
# Push:  docker push <dockerhub-user>/cowork-3d-v14:latest
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"

# ==== SYSTEM DEPS ====
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        git wget curl ca-certificates \
        libopengl0 libgl1 libgles2 libglu1-mesa libegl1 libglvnd0 \
        libxrender1 libxi6 libxext6 libxrandr2 libx11-xcb1 \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==== MINIFORGE + tripo310 conda env ====
RUN wget -q -O /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/mf.sh -b -p /opt/miniforge \
    && rm /tmp/mf.sh \
    && /opt/miniforge/bin/conda create -y -n tripo310 python=3.10 \
    && /opt/miniforge/bin/conda clean -afy

ENV PATH="/opt/miniforge/envs/tripo310/bin:${PATH}"
ENV PY310=/opt/miniforge/envs/tripo310/bin/python

# ==== TORCH first (for diso build isolation) ====
RUN ${PY310} -m pip install --no-cache-dir --upgrade pip \
    && ${PY310} -m pip install --no-cache-dir \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# ==== TripoSG repo + its requirements ====
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSG.git /opt/TripoSG \
    && ${PY310} -m pip install --no-cache-dir -r /opt/TripoSG/requirements.txt

# ==== diso CUDA extension ====
RUN ${PY310} -m pip install --no-cache-dir --no-build-isolation diso

# ==== Pipeline v14 deps (pinned) ====
RUN ${PY310} -m pip install --no-cache-dir \
        Pillow==11.0.0 \
        mediapipe==0.10.35 \
        torchvision==0.21.0 \
        timm==1.0.9 \
        trimesh==4.12.2 \
        pygltflib==1.16.2 \
        xatlas==0.0.11 \
        pymeshlab \
        opencv-python \
        scipy==1.18.0 \
        scikit-image==0.26.0 \
        huggingface_hub \
        matplotlib \
        fast-simplification \
        runpod

# ==== Depth-Anything-V2 python module ====
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 /opt/DAv2 \
    && echo "/opt/DAv2" > /opt/miniforge/envs/tripo310/lib/python3.10/site-packages/dav2.pth

# ==== Pre-download model weights to volume-independent /models ====
RUN mkdir -p /models/hf \
    && ${PY310} -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Base')" \
    && ${PY310} -c "from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/TripoSG', local_dir='/opt/TripoSG/pretrained_weights/TripoSG')" \
    && curl -fL -o /models/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ==== Copy pipeline code + handler ====
WORKDIR /app
COPY handler.py pipeline_v14.py pipeline_v11_depth_umeyama.py module_a_retune.py module_e_texture.py /app/

# ==== Sanity import test at build time ====
RUN ${PY310} -c "\
import sys; sys.path.insert(0, '/opt/TripoSG'); \
import torch, torchvision, PIL, cv2, mediapipe, trimesh, xatlas, pymeshlab; \
from depth_anything_v2.dpt import DepthAnythingV2; \
from triposg.pipelines.pipeline_triposg import TripoSGPipeline; \
print('OK torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# ==== Runpod Serverless entrypoint ====
CMD ["/opt/miniforge/envs/tripo310/bin/python", "-u", "handler.py"]
