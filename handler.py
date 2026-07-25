"""
RunPod Serverless handler for opcreative-3d-portrait pipeline_v14.

Job input shape (JSON):
{
    "input": {
        "image_b64": "<base64 PNG>",
        "mesh_b64": "<base64 GLB>",          # optional; TripoSG raw mesh
        "base_mesh_b64": "<base64 GLB>",     # optional; Kent's base for alignment
        "run_module_a": false,
        "run_module_cd": true,
        "run_module_e": true,
        "cd_dav2_model": "base",
        "cd_face_weight": 0.7,
        "cd_emboss_strength": 0.010,
        "e_uv_bake_res": 2048,
        ...
    }
}

Returns:
{
    "output": {
        "glb_b64":        "<base64 final GLB>",
        "relief_b64":     "<base64 face-relief-only GLB>",
        "transform":      {...JSON sidecar...},
        "log_tail":       "<last 40 lines of pipeline log>",
        "timing":         {"stage_a": s, "stage_cd": s, "stage_e": s, "total": s},
        "meta":           {"gpu": ..., "torch": ..., ...}
    }
}
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import runpod


# Make TripoSG importable
sys.path.insert(0, "/opt/TripoSG")


def _b64_to_file(b64_str: str, tmp: Path, name: str) -> Path:
    p = tmp / name
    p.write_bytes(base64.b64decode(b64_str))
    return p


def _file_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    return {"torch": "?", "cuda": "?", "device": "?"}


def handler(job: dict) -> dict:
    """Handle a single inference job."""
    t0 = time.time()
    job_input = job.get("input", {}) or {}
    try:
        # ---- Validate inputs ----
        img_b64 = job_input.get("image_b64")
        if not img_b64:
            return {"error": "missing 'image_b64' in input"}
        mesh_b64 = job_input.get("mesh_b64")
        base_b64 = job_input.get("base_mesh_b64")

        # ---- Stage into a tmpdir ----
        with tempfile.TemporaryDirectory(prefix="rp_") as tmp_dir:
            tmp = Path(tmp_dir)
            in_img = _b64_to_file(img_b64, tmp, "input.png")
            in_mesh = _b64_to_file(mesh_b64, tmp, "input.glb") if mesh_b64 else None
            in_base = _b64_to_file(base_b64, tmp, "base.glb") if base_b64 else None
            output = tmp / "output.glb"
            log_path = tmp / "run.log"

            # ---- Redirect stdout/stderr to log file ----
            from contextlib import redirect_stdout, redirect_stderr
            with open(log_path, "w", buffering=1) as log_f, \
                    redirect_stdout(log_f), redirect_stderr(log_f):
                sys.path.insert(0, "/app")
                from pipeline_v14 import PipelineV14Config, run_pipeline_v14

                cfg_kwargs = dict(
                    input_image=in_img,
                    input_mesh=in_mesh,
                    base_mesh=in_base,
                    output=output,
                    workdir=Path("/models"),

                    run_module_a=bool(job_input.get("run_module_a", False)),
                    run_module_cd=bool(job_input.get("run_module_cd", True)),
                    run_module_e=bool(job_input.get("run_module_e", True)),

                    a_num_inference_steps=int(job_input.get("a_num_inference_steps", 75)),
                    a_guidance_scale=float(job_input.get("a_guidance_scale", 7.0)),
                    a_faces=int(job_input.get("a_faces", 500_000)),
                    a_seed=int(job_input.get("a_seed", 42)),
                    a_timeout_s=int(job_input.get("a_timeout_s", 300)),

                    cd_dav2_model=str(job_input.get("cd_dav2_model", "base")),
                    cd_mask_expand=float(job_input.get("cd_mask_expand", 1.2)),
                    cd_face_weight=float(job_input.get("cd_face_weight", 0.7)),
                    cd_umeyama_clamp_min=float(job_input.get("cd_umeyama_clamp_min", 0.5)),
                    cd_umeyama_clamp_max=float(job_input.get("cd_umeyama_clamp_max", 1.5)),
                    cd_emboss_strength=float(job_input.get("cd_emboss_strength", 0.010)),

                    e_uv_bake_res=int(job_input.get("e_uv_bake_res", 2048)),
                    e_dot_density_scale=float(job_input.get("e_dot_density_scale", 1.0)),
                    e_dot_size_px=int(job_input.get("e_dot_size_px", 3)),
                    e_dot_luminance_curve=str(job_input.get("e_dot_luminance_curve", "linear")),

                    dry_run=False,
                )

                cfg = PipelineV14Config(**cfg_kwargs)
                run_pipeline_v14(cfg)

            # ---- Collect outputs ----
            stem = output.stem
            relief_path = output.parent / f"{stem}_face_relief_only.glb"
            transform_path = output.parent / f"{stem}_transform.json"

            result = {
                "meta": {
                    **_gpu_info(),
                    "handler_wall_s": round(time.time() - t0, 2),
                },
            }
            if output.exists():
                result["glb_b64"] = _file_to_b64(output)
                result["glb_size_bytes"] = output.stat().st_size
            if relief_path.exists():
                result["relief_b64"] = _file_to_b64(relief_path)
                result["relief_size_bytes"] = relief_path.stat().st_size
            if transform_path.exists():
                result["transform"] = json.loads(transform_path.read_text())

            # Log tail
            if log_path.exists():
                lines = log_path.read_text(errors="replace").splitlines()
                result["log_tail"] = "\n".join(lines[-40:])

            return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "handler_wall_s": round(time.time() - t0, 2),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
