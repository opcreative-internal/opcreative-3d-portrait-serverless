"""Build-time import verifier for RunPod Serverless image.

Fails the Docker build (exit != 0) if any module used at runtime by the
pipeline_v14 chain cannot be imported. Cheaper to fail here than at
cold-start on the first job.
"""
import importlib
import os
import sys

MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "PIL.Image",
    "cv2",
    "mediapipe",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "trimesh",
    "trimesh.visual.texture",
    "trimesh.visual.material",
    "pygltflib",
    "xatlas",
    "pymeshlab",
    "scipy",
    "scipy.spatial",
    "skimage",
    "skimage.transform",
    "huggingface_hub",
    "fast_simplification",
    "runpod",
    "timm",
    "depth_anything_v2",
    "depth_anything_v2.dpt",
    "omegaconf",
    "einops",
    "diffusers",
    "transformers",
    "accelerate",
    # V21f: TripoSG import chain (Fable v3 verdict — catch the diso/peft/jaxtyping cascade at build time)
    "peft",
    "jaxtyping",
    "typeguard",
    "diso",                                 # stubbed at /opt/TripoSG/diso.py
    "triposg",
    "triposg.pipelines",
    "triposg.pipelines.pipeline_triposg",
    "triposg.inference_utils",
]

failed = []
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, type(e).__name__, str(e)[:200]))

print("=== IMPORT VERIFY ===")
for m in MODULES:
    if not any(f[0] == m for f in failed):
        print("  OK  ", m)
for m, cls, msg in failed:
    print("  FAIL", m, "->", cls, msg)

if failed:
    print("\n!! MISSING MODULES:", [f[0] for f in failed])
    sys.exit(1)

print(f"\n=== ALL {len(MODULES)} MODULES OK ===")

import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)

face_task = "/workspace/face_landmarker.task"
if os.path.exists(face_task):
    print("face_landmarker.task:", os.path.getsize(face_task), "bytes")
else:
    print("!! face_landmarker.task NOT AT", face_task)
    sys.exit(2)

# Verify DAv2 checkpoint pre-cached
try:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(
        repo_id="depth-anything/Depth-Anything-V2-Base",
        filename="depth_anything_v2_vitb.pth",
    )
    print("DAv2 vitb cached:", p)
except Exception as e:
    print("DAv2 cache check warning:", e)

# Verify TripoSG weights baked at expected local_dir (module_a_retune uses cwd=/opt/TripoSG,
# scripts/inference_triposg.py uses snapshot_download(local_dir='pretrained_weights/...'))
import os as _os
_TRIPO_WEIGHTS = "/opt/TripoSG/pretrained_weights/TripoSG"
_RMBG_WEIGHTS  = "/opt/TripoSG/pretrained_weights/RMBG-1.4"
for p in (_TRIPO_WEIGHTS, _RMBG_WEIGHTS):
    if _os.path.isdir(p) and _os.listdir(p):
        n = sum(_os.path.getsize(_os.path.join(dp, f))
                for dp, _dn, fn in _os.walk(p) for f in fn)
        print(f"OK weights {p}: {n/1024/1024:.1f} MB, {len(_os.listdir(p))} entries")
    else:
        print(f"!! MISSING weights dir {p}")
        sys.exit(3)

print("=== VERIFY DONE ===")
