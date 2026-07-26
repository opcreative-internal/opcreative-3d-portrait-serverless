"""
Pipeline v21 -- V19 combo + explicit wrap direction (kill auto-detect) + input preproc.

Delta from V19:
  * wrap step calls module_wrap_texture_v21.run_module_wrap_v21 instead of V18's
    module_wrap_texture with wrap_auto_detect_view_dir.
  * New config fields: wrap_direction, flip_h, flip_v, brightness, contrast.
  * cfg.wrap_auto_detect_view_dir REMOVED (Fable v3 verdict Q1: kill it).

Backwards compat:
  * If run_module_wrap=False, V21 = V19 (E-SSLE PLY output only). Safe rollback.
  * pipeline_v19.py stays as-is for V20.1 delivery cycle (Kent decision: bundle
    riêng V21b commit).
"""
from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineV21Config:
    input_image: Path
    input_mesh: Path
    output: Path
    workdir: Path = Path("/workspace")

    run_module_cd: bool = True
    run_module_e_ssle: bool = True
    run_module_wrap: bool = True

    # C+D face relief (unchanged from V19)
    cd_dav2_model: str = "base"
    cd_mask_expand: float = 1.2
    cd_face_weight: float = 0.7
    cd_umeyama_clamp_min: float = 0.5
    cd_umeyama_clamp_max: float = 1.5
    cd_emboss_strength: float = 0.025

    # SSLE PLY (unchanged from V19)
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
    front_axis: str = "+Z"      # retained for E-SSLE (independent of wrap direction)
    seed: int = 42

    # Wrap (V21: EXPLICIT direction, no auto-detect)
    wrap_direction: str = "front"    # front | back | left | right
    flip_h: bool = False
    flip_v: bool = False
    brightness: float = 0.0          # -100..100 applied to input image before bake
    contrast: float = 0.0            # -100..100
    wrap_atlas_res: int = 8192
    wrap_clahe_clip: float = 3.0
    wrap_clahe_tile: int = 16
    wrap_target_faces: int = 60_000
    wrap_xatlas_timeout_s: int = 180

    dry_run: bool = False


def run_pipeline_v21(cfg: PipelineV21Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    parent = cfg.output.parent
    stem = cfg.output.stem
    print("=" * 66)
    print("  Pipeline v21 -- explicit wrap_direction + input B/C/flip preproc")
    print("=" * 66)
    print(f"  wrap_direction={cfg.wrap_direction}  flip_h={cfg.flip_h}  flip_v={cfg.flip_v}")
    print(f"  brightness={cfg.brightness}  contrast={cfg.contrast}")
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

    # SSLE point cloud (unchanged from V19)
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

    # WRAP TEXTURE V21 (explicit direction, no auto-detect)
    if cfg.run_module_wrap:
        print(f"\n>>>>> STAGE WRAP V21 (dir={cfg.wrap_direction}) <<<<<", flush=True)
        from module_wrap_texture_v21 import ModuleWrapV21Config, run_module_wrap_v21
        wrap_stem = parent / stem
        w_cfg = ModuleWrapV21Config(
            input_mesh=current_mesh, input_image=cfg.input_image, out_stem=wrap_stem,
            target_height_mm=cfg.target_height_mm,
            atlas_res=cfg.wrap_atlas_res,
            xatlas_target_faces=cfg.wrap_target_faces,
            xatlas_timeout_s=cfg.wrap_xatlas_timeout_s,
            clahe_clip=cfg.wrap_clahe_clip, clahe_tile=cfg.wrap_clahe_tile,
            wrap_direction=cfg.wrap_direction,
            flip_h=cfg.flip_h, flip_v=cfg.flip_v,
            brightness=cfg.brightness, contrast=cfg.contrast,
        )
        run_module_wrap_v21(w_cfg)

    # Copy wrap GLB to canonical cfg.output path (handler expects output.glb)
    wrap_glb = cfg.output.parent / f"{cfg.output.stem}.glb"
    if wrap_glb.exists() and wrap_glb != cfg.output:
        try:
            shutil.copy2(wrap_glb, cfg.output)
        except shutil.SameFileError:
            pass

    print(f"\n[pipeline_v21] total wall: {time.time() - t_all:.2f}s", flush=True)
    return cfg.output


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--wrap-direction", default="front",
                    choices=["front", "back", "left", "right"])
    ap.add_argument("--flip-h", action="store_true")
    ap.add_argument("--flip-v", action="store_true")
    ap.add_argument("--brightness", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = PipelineV21Config(
        input_image=args.image, input_mesh=args.mesh, output=args.output,
        wrap_direction=args.wrap_direction,
        flip_h=args.flip_h, flip_v=args.flip_v,
        brightness=args.brightness, contrast=args.contrast,
        dry_run=args.dry_run,
    )
    run_pipeline_v21(cfg)
