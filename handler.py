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
import urllib.request
from pathlib import Path

import runpod


# Make TripoSG importable
sys.path.insert(0, "/opt/TripoSG")


def _b64_to_file(b64_str: str, tmp: Path, name: str) -> Path:
    p = tmp / name
    p.write_bytes(base64.b64decode(b64_str))
    return p


_UPLOAD_LOG = []  # collected per-file upload attempts for return in output


def _upload_catbox(path: Path, timeout: int = 120) -> str:
    """Upload file to catbox.moe permanent, return URL. Try requests then urllib then curl."""
    if not path.exists() or path.stat().st_size == 0:
        _UPLOAD_LOG.append(f"{path.name}: skip empty/missing")
        return ""
    # 1. requests (usually present via transformers)
    try:
        import requests
        with open(path, "rb") as f:
            r = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (path.name, f, "application/octet-stream")},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=timeout,
            )
        out = (r.text or "").strip()
        _UPLOAD_LOG.append(f"{path.name}: requests http={r.status_code} out_head={out[:80]!r}")
        if out.startswith("http"):
            return out
    except Exception as e:
        _UPLOAD_LOG.append(f"{path.name}: requests EXC {type(e).__name__}: {str(e)[:120]}")
    # 2. subprocess curl fallback
    try:
        import subprocess
        r = subprocess.run(
            ["curl", "-sS", "--connect-timeout", "30", "--max-time", str(timeout),
             "-F", "reqtype=fileupload",
             "-F", f"fileToUpload=@{path}",
             "-A", "Mozilla/5.0",
             "https://catbox.moe/user/api.php"],
            capture_output=True, timeout=timeout + 15, text=True,
        )
        out = (r.stdout or "").strip()
        _UPLOAD_LOG.append(f"{path.name}: curl rc={r.returncode} out_head={out[:80]!r}")
        if out.startswith("http"):
            return out
    except Exception as e:
        _UPLOAD_LOG.append(f"{path.name}: curl EXC {type(e).__name__}: {str(e)[:120]}")
    return ""


def _upload_return_url_or_b64(path: Path, max_inline_bytes: int = 3_000_000) -> dict:
    """Small files (<=3MB) inline as b64. Large files upload to catbox and return URL.
    If upload FAILS for a large file, return sentinel (do NOT inline b64 - would blow
    RunPod response cap and get whole output dropped)."""
    if not path.exists():
        return {}
    sz = path.stat().st_size
    if sz <= max_inline_bytes:
        return {"b64": _file_to_b64(path), "size_bytes": sz}
    url = _upload_catbox(path)
    if url:
        return {"url": url, "size_bytes": sz}
    # Do NOT inline b64 - would trigger RunPod response cap drop
    return {"size_bytes": sz, "upload_failed": True}


def _url_to_file(url: str, tmp: Path, name: str, timeout: int = 60) -> Path:
    """Download URL to local file. Used to bypass /run 10MB payload limit."""
    p = tmp / name
    req = urllib.request.Request(url, headers={"User-Agent": "cowork-3d-v14/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        p.write_bytes(resp.read())
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
    global _UPLOAD_LOG
    _UPLOAD_LOG = []  # reset per-job upload log
    t0 = time.time()
    job_input = job.get("input", {}) or {}
    try:
        # ---- Validate inputs ----
        img_b64 = job_input.get("image_b64")
        img_url = job_input.get("image_url")
        if not img_b64 and not img_url:
            return {"error": "missing 'image_b64' or 'image_url' in input"}
        mesh_b64 = job_input.get("mesh_b64")
        mesh_url = job_input.get("mesh_url")
        base_b64 = job_input.get("base_mesh_b64")
        base_url = job_input.get("base_mesh_url")

        # ---- Stage into a tmpdir ----
        with tempfile.TemporaryDirectory(prefix="rp_") as tmp_dir:
            tmp = Path(tmp_dir)
            if img_b64:
                in_img = _b64_to_file(img_b64, tmp, "input.png")
            else:
                in_img = _url_to_file(img_url, tmp, "input.png")
            if mesh_b64:
                in_mesh = _b64_to_file(mesh_b64, tmp, "input.glb")
            elif mesh_url:
                in_mesh = _url_to_file(mesh_url, tmp, "input.glb")
            else:
                in_mesh = None
            if base_b64:
                in_base = _b64_to_file(base_b64, tmp, "base.glb")
            elif base_url:
                in_base = _url_to_file(base_url, tmp, "base.glb")
            else:
                in_base = None
            output = tmp / "output.glb"
            log_path = tmp / "run.log"

            # ---- Redirect stdout/stderr to log file ----
            from contextlib import redirect_stdout, redirect_stderr
            with open(log_path, "w", buffering=1) as log_f, \
                    redirect_stdout(log_f), redirect_stderr(log_f):
                sys.path.insert(0, "/app")
                pipeline_version = str(job_input.get("pipeline_version", "v14")).lower()
                if pipeline_version == "v18":
                    from pipeline_v18 import PipelineV18Config as _PipelineCfg
                    from pipeline_v18 import run_pipeline_v18 as _run_pipeline
                elif pipeline_version == "v17":
                    from pipeline_v17 import PipelineV17Config as _PipelineCfg
                    from pipeline_v17 import run_pipeline_v17 as _run_pipeline
                elif pipeline_version == "v16":
                    from pipeline_v16 import PipelineV16Config as _PipelineCfg
                    from pipeline_v16 import run_pipeline_v16 as _run_pipeline
                elif pipeline_version == "v15":
                    from pipeline_v15 import PipelineV15Config as _PipelineCfg
                    from pipeline_v15 import run_pipeline_v15 as _run_pipeline
                else:
                    from pipeline_v14 import PipelineV14Config as _PipelineCfg
                    from pipeline_v14 import run_pipeline_v14 as _run_pipeline

                if pipeline_version == "v18":
                    if in_mesh is None:
                        raise ValueError("pipeline_version=v18 requires mesh_b64 or mesh_url")
                    cfg_kwargs = dict(
                        input_image=in_img, input_mesh=in_mesh, output=output,
                        workdir=Path("/models"),
                        run_module_cd=bool(job_input.get("run_module_cd", True)),
                        run_module_e_ssle=bool(job_input.get("run_module_e_ssle", True)),
                        run_module_wrap=bool(job_input.get("run_module_wrap", True)),
                        cd_dav2_model=str(job_input.get("cd_dav2_model", "base")),
                        cd_face_weight=float(job_input.get("cd_face_weight", 0.7)),
                        cd_emboss_strength=float(job_input.get("cd_emboss_strength", 0.025)),
                        target_height_mm=float(job_input.get("target_height_mm", 70.0)),
                        target_count=int(job_input.get("target_count", 300_000)),
                        accept_low=int(job_input.get("accept_low", 150_000)),
                        accept_high=int(job_input.get("accept_high", 450_000)),
                        n_candidates=int(job_input.get("n_candidates", 2_500_000)),
                        r_min_mm=float(job_input.get("r_min_mm", 0.18)),
                        r_min_floor_mm=float(job_input.get("r_min_floor_mm", 0.15)),
                        w_floor=float(job_input.get("w_floor", 0.08)),
                        w_gamma=float(job_input.get("w_gamma", 1.6)),
                        w_back=float(job_input.get("w_back", 0.30)),
                        face_boost=float(job_input.get("face_boost", 1.3)),
                        normal_jitter_mm=float(job_input.get("normal_jitter_mm", 0.15)),
                        front_axis=str(job_input.get("front_axis", "+Z")),
                        seed=int(job_input.get("seed", 42)),
                        wrap_atlas_res=int(job_input.get("wrap_atlas_res", 4096)),
                        wrap_target_faces=int(job_input.get("wrap_target_faces", 60_000)),
                        wrap_xatlas_timeout_s=int(job_input.get("wrap_xatlas_timeout_s", 180)),
                        wrap_clahe_clip=float(job_input.get("wrap_clahe_clip", 3.0)),
                        wrap_clahe_tile=int(job_input.get("wrap_clahe_tile", 16)),
                        dry_run=False,
                    )
                elif pipeline_version == "v17":
                    # V17 SSLE: output is .ply (redirect from output.glb -> output.ply)
                    if in_mesh is None:
                        raise ValueError("pipeline_version=v17 requires mesh_b64 or mesh_url")
                    output_ply = output.with_suffix(".ply")
                    cfg_kwargs = dict(
                        input_image=in_img,
                        input_mesh=in_mesh,
                        output=output_ply,
                        workdir=Path("/models"),
                        run_module_cd=bool(job_input.get("run_module_cd", True)),
                        run_module_e_ssle=bool(job_input.get("run_module_e_ssle", True)),
                        cd_dav2_model=str(job_input.get("cd_dav2_model", "base")),
                        cd_mask_expand=float(job_input.get("cd_mask_expand", 1.2)),
                        cd_face_weight=float(job_input.get("cd_face_weight", 0.7)),
                        cd_umeyama_clamp_min=float(job_input.get("cd_umeyama_clamp_min", 0.5)),
                        cd_umeyama_clamp_max=float(job_input.get("cd_umeyama_clamp_max", 1.5)),
                        cd_emboss_strength=float(job_input.get("cd_emboss_strength", 0.025)),
                        target_height_mm=float(job_input.get("target_height_mm", 70.0)),
                        target_count=int(job_input.get("target_count", 300_000)),
                        accept_low=int(job_input.get("accept_low", 150_000)),
                        accept_high=int(job_input.get("accept_high", 450_000)),
                        n_candidates=int(job_input.get("n_candidates", 2_500_000)),
                        r_min_mm=float(job_input.get("r_min_mm", 0.18)),
                        r_min_floor_mm=float(job_input.get("r_min_floor_mm", 0.15)),
                        w_floor=float(job_input.get("w_floor", 0.08)),
                        w_gamma=float(job_input.get("w_gamma", 1.6)),
                        w_back=float(job_input.get("w_back", 0.30)),
                        face_boost=float(job_input.get("face_boost", 1.3)),
                        normal_jitter_mm=float(job_input.get("normal_jitter_mm", 0.15)),
                        front_axis=str(job_input.get("front_axis", "+Z")),
                        seed=int(job_input.get("seed", 42)),
                        dry_run=False,
                    )
                elif pipeline_version == "v16":
                    # V16 has a different config surface than V14/V15
                    if in_mesh is None:
                        raise ValueError("pipeline_version=v16 requires mesh_b64 or mesh_url")
                    cfg_kwargs = dict(
                        input_image=in_img,
                        input_mesh=in_mesh,
                        output=output,
                        workdir=Path("/models"),
                        run_module_cd=bool(job_input.get("run_module_cd", True)),
                        run_module_e_engrave=bool(job_input.get("run_module_e_engrave", True)),
                        cd_dav2_model=str(job_input.get("cd_dav2_model", "base")),
                        cd_mask_expand=float(job_input.get("cd_mask_expand", 1.2)),
                        cd_face_weight=float(job_input.get("cd_face_weight", 0.7)),
                        cd_umeyama_clamp_min=float(job_input.get("cd_umeyama_clamp_min", 0.5)),
                        cd_umeyama_clamp_max=float(job_input.get("cd_umeyama_clamp_max", 1.5)),
                        cd_emboss_strength=float(job_input.get("cd_emboss_strength", 0.010)),
                        e_emboss_strength=float(job_input.get("e_emboss_strength", 0.004)),
                        e_dot_count=int(job_input.get("e_dot_count", 20000)),
                        e_dot_min_radius_px=float(job_input.get("e_dot_min_radius_px", 3.0)),
                        e_influence_radius_px=float(job_input.get("e_influence_radius_px", 6.0)),
                        e_lum_curve=str(job_input.get("e_lum_curve", "linear")),
                        e_lum_gamma=float(job_input.get("e_lum_gamma", 2.2)),
                        e_front_axis=str(job_input.get("e_front_axis", "+Z")),
                        e_seed=int(job_input.get("e_seed", 42)),
                        dry_run=False,
                    )
                else:
                    # V14 / V15 config surface
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
                    if pipeline_version == "v15":
                        cfg_kwargs["f_weld_tol_frac"] = float(job_input.get("f_weld_tol_frac", 0.01))
                        cfg_kwargs["f_smooth_iterations"] = int(job_input.get("f_smooth_iterations", 3))
                        cfg_kwargs["f_smooth_ring_depth"] = int(job_input.get("f_smooth_ring_depth", 2))
                        cfg_kwargs["f_smooth_lambda"] = float(job_input.get("f_smooth_lambda", 0.5))
                cfg = _PipelineCfg(**cfg_kwargs)
                _run_pipeline(cfg)

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
                info = _upload_return_url_or_b64(output)
                if "url" in info: result["glb_url"] = info["url"]
                if "b64" in info: result["glb_b64"] = info["b64"]
                if info: result["glb_size_bytes"] = info["size_bytes"]
                if info.get("upload_failed"): result["glb_upload_failed"] = True
            if relief_path.exists():
                info = _upload_return_url_or_b64(relief_path)
                if "url" in info: result["relief_url"] = info["url"]
                if "b64" in info: result["relief_b64"] = info["b64"]
                if info: result["relief_size_bytes"] = info["size_bytes"]
            if transform_path.exists():
                result["transform"] = json.loads(transform_path.read_text())
            # V15 extra: head_only sidecar written by module_f_graft
            head_only_path = output.parent / f"{output.stem}_head_only.glb"
            if head_only_path.exists():
                result["head_only_b64"] = _file_to_b64(head_only_path)
                result["head_only_size_bytes"] = head_only_path.stat().st_size
            # V17/V18: SSLE + wrap texture outputs. V18c: use URL upload for large files
            # to bypass RunPod ~10-20MB response cap. Files <=3MB stay inline (b64).
            ply_path = output.with_suffix(".ply")
            xyz_path = output.with_suffix(".xyz")
            obj_path = output.with_suffix(".obj")
            mtl_path = output.with_suffix(".mtl")
            stl_path = output.with_suffix(".stl")
            tex_path = output.parent / f"{output.stem}_texture.png"
            for name, p in [
                ("ply", ply_path), ("xyz", xyz_path), ("obj", obj_path),
                ("mtl", mtl_path), ("stl", stl_path), ("texture_png", tex_path),
            ]:
                info = _upload_return_url_or_b64(p)
                if not info:
                    continue
                if "url" in info:
                    result[f"{name}_url"] = info["url"]
                if "b64" in info:
                    result[f"{name}_b64"] = info["b64"]
                result[f"{name}_size_bytes"] = info["size_bytes"]
                if info.get("upload_failed"):
                    result[f"{name}_upload_failed"] = True

            # Log tail
            if log_path.exists():
                lines = log_path.read_text(errors="replace").splitlines()
                result["log_tail"] = "\n".join(lines[-40:])

            # Upload attempt diagnostics (V18c: catbox reachability from RunPod worker)
            if _UPLOAD_LOG:
                result["upload_log"] = _UPLOAD_LOG[-40:]

            # Summary always present so we know handler completed even if all uploads dropped
            result["_v18_summary"] = {
                "files_collected": {
                    k: v for k, v in {
                        "glb": result.get("glb_size_bytes"),
                        "ply": result.get("ply_size_bytes"),
                        "xyz": result.get("xyz_size_bytes"),
                        "obj": result.get("obj_size_bytes"),
                        "mtl": result.get("mtl_size_bytes"),
                        "stl": result.get("stl_size_bytes"),
                        "texture_png": result.get("texture_png_size_bytes"),
                    }.items() if v is not None
                },
                "urls_returned": [k for k in result if k.endswith("_url")],
                "b64_returned": [k for k in result if k.endswith("_b64")],
            }

            return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "handler_wall_s": round(time.time() - t0, 2),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
