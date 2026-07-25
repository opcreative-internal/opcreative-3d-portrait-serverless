"""
Pipeline v19 -- V18 combo pack with camera-space UV wrap fix (V19 spec).

V18 xatlas ortho bake mis-placed face pixels onto torso. V19 replaces the wrap step
with camera-space direct UV (Fable 5 v3 verdict Q1) + neutral back-face + 8192 CLAHE
+ Sobel normal map. Rest of pipeline unchanged from V18 (v11 C+D face relief + V17
SSLE PLY-RGB point cloud).

Xatlas retained behind flag e_use_camera_uv=true (default) for rollback via
e_use_camera_uv=false which reverts to V18 module_wrap_texture.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineV19Config:
    input_image: Path
    input_mesh: Path
    output: Path
    workdir: Path = Path("/workspace")

    run_module_cd: bool = True
    run_module_e_ssle: bool = True
    run_module_wrap: bool = True

    # C+D
    cd_dav2_model: str = "base"
    cd_mask_expand: float = 1.2
    cd_face_weight: float = 0.7
    cd_umeyama_clamp_min: float = 0.5
    cd_umeyama_clamp_max: float = 1.5
    cd_emboss_strength: float = 0.025

    # SSLE PLY (V17/V18 pattern)
    target_height_mm: float = 70.0
    target_count: int = 300_000
    accept_low: int = 150_000
    accept_high: int = 450_000
    n_candidates: int = 2_500_000
    r_min_mm: float = 0.18
    r_min_floor_mm: float = 0.15
    w_floor: float = 0.08
    w_gamma: float = 1.6
    w_back: float = 0.30
    face_boost: float = 1.3
    normal_jitter_mm: float = 0.15
    front_axis: str = "+Z"
    seed: int = 42

    # Wrap (V19: camera-UV by default; xatlas rollback via flag)
    e_use_camera_uv: bool = True
    wrap_atlas_res: int = 8192              # V19 uplift 4096 -> 8192
    wrap_auto_detect_view_dir: bool = True  # V20 fix: pick +Z or -Z by head-normal count
    wrap_front_normal_threshold: float = 0.15
    wrap_neutral_border_frac: float = 0.05
    wrap_clahe_clip: float = 3.0
    wrap_clahe_tile: int = 16
    wrap_sobel_ksize: int = 3
    wrap_sobel_strength: float = 1.0
    # V18 xatlas legacy knobs (used only when e_use_camera_uv=false)
    wrap_target_faces: int = 60_000
    wrap_xatlas_timeout_s: int = 180

    dry_run: bool = False


def run_pipeline_v19(cfg: PipelineV19Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    parent = cfg.output.parent
    stem = cfg.output.stem
    print("=" * 66)
    print("  Pipeline v19 -- camera-space UV wrap + 8192 CLAHE + Sobel normal")
    print("=" * 66)
    if cfg.dry_run: return cfg.output

    t_all = time.time()
    if not Path(cfg.input_mesh).exists():
        raise FileNotFoundError(f"input_mesh missing: {cfg.input_mesh}")
    current_mesh = Path(cfg.input_mesh)

    # C+D face relief
    if cfg.run_module_cd:
        print(f"\n>>>>> STAGE C+D (emboss={cfg.cd_emboss_strength}) <<<<<", flush=True)
        from pipeline_v11_depth_umeyama import Config as V11Config, run_pipeline as v11_run
        cd_out = parent / f"{stem}_stageCD.glb"
        v11_cfg = V11Config(
            input_image=cfg.input_image, input_mesh=current_mesh, output=cd_out,
            dav2_model=cfg.cd_dav2_model, mask_expand=cfg.cd_mask_expand,
            face_weight=cfg.cd_face_weight,
            umeyama_clamp_min=cfg.cd_umeyama_clamp_min,
            umeyama_clamp_max=cfg.cd_umeyama_clamp_max,
            emboss_strength=cfg.cd_emboss_strength,
        )
        current_mesh = v11_run(v11_cfg)

    # SSLE point cloud (unchanged from V18)
    if cfg.run_module_e_ssle:
        print(f"\n>>>>> STAGE E SSLE (PLY+XYZ with RGB) <<<<<", flush=True)
        from module_e_ssle import ModuleESSLEConfig, run_module_e_ssle
        ply_out = parent / f"{stem}.ply"
        e_cfg = ModuleESSLEConfig(
            input_mesh=current_mesh, input_image=cfg.input_image, output_ply=ply_out,
            target_height_mm=cfg.target_height_mm,
            target_count=cfg.target_count, accept_low=cfg.accept_low,
            accept_high=cfg.accept_high, n_candidates=cfg.n_candidates,
            r_min_mm=cfg.r_min_mm, r_min_floor_mm=cfg.r_min_floor_mm,
            w_floor=cfg.w_floor, w_gamma=cfg.w_gamma, w_back=cfg.w_back,
            face_boost=cfg.face_boost, normal_jitter_mm=cfg.normal_jitter_mm,
            front_axis=cfg.front_axis, seed=cfg.seed,
        )
        run_module_e_ssle(e_cfg)

    # Wrap texture
    if cfg.run_module_wrap:
        wrap_stem = parent / stem
        if cfg.e_use_camera_uv:
            print(f"\n>>>>> STAGE WRAP V19 (camera-space UV, atlas={cfg.wrap_atlas_res}) <<<<<", flush=True)
            from module_wrap_texture_v19 import ModuleWrapV19Config, run_module_wrap_v19
            w_cfg = ModuleWrapV19Config(
                input_mesh=current_mesh, input_image=cfg.input_image, out_stem=wrap_stem,
                target_height_mm=cfg.target_height_mm,
                atlas_res=cfg.wrap_atlas_res,
                front_axis=cfg.front_axis,
                auto_detect_view_dir=cfg.wrap_auto_detect_view_dir,
                front_normal_threshold=cfg.wrap_front_normal_threshold,
                neutral_border_frac=cfg.wrap_neutral_border_frac,
                clahe_clip=cfg.wrap_clahe_clip, clahe_tile=cfg.wrap_clahe_tile,
                sobel_ksize=cfg.wrap_sobel_ksize, sobel_strength=cfg.wrap_sobel_strength,
            )
            run_module_wrap_v19(w_cfg)
        else:
            print(f"\n>>>>> STAGE WRAP V18 (xatlas rollback) <<<<<", flush=True)
            from module_wrap_texture import ModuleWrapConfig, run_module_wrap
            w_cfg = ModuleWrapConfig(
                input_mesh=current_mesh, input_image=cfg.input_image, out_stem=wrap_stem,
                target_height_mm=cfg.target_height_mm,
                atlas_res=cfg.wrap_atlas_res,
                xatlas_target_faces=cfg.wrap_target_faces,
                xatlas_timeout_s=cfg.wrap_xatlas_timeout_s,
                front_axis=cfg.front_axis,
                clahe_clip=cfg.wrap_clahe_clip, clahe_tile=cfg.wrap_clahe_tile,
            )
            run_module_wrap(w_cfg)

    try:
        cd_intermediate = parent / f"{stem}_stageCD.glb"
        if cd_intermediate.exists() and cd_intermediate != cfg.output:
            cd_intermediate.unlink()
    except Exception as e:
        print(f"  cleanup warn: {e}", flush=True)

    glb_final = parent / f"{stem}.glb"
    if glb_final.exists():
        cfg.output = glb_final

    print(f"\n[PIPELINE v19 DONE] total {time.time()-t_all:.1f}s", flush=True)
    return cfg.output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="pipeline_v19")
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--no-camera-uv", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = PipelineV19Config(input_image=a.input_image, input_mesh=a.input_mesh, output=a.output,
                            e_use_camera_uv=(not a.no_camera_uv), dry_run=a.dry_run)
    run_pipeline_v19(cfg)
