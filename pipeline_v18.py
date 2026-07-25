"""
Pipeline v18 -- combo pack: V17 SSLE PLY (with RGB) + QA preview mesh + textures.

Kent's dual-purpose:
  1. PLY point cloud (with per-vert RGB bonus) = laser input for CrystalMe3D Pro
  2. OBJ+MTL+PNG + STL + GLB = QA preview so Kent can verify face detail
     before wasting a crystal blank

Chain:
  1. V11 C+D face relief (deeper emboss 0.025)
  2. V17 SSLE point cloud with per-vertex RGB -> .ply + .xyz
  3. WRAP TEXTURE grayscale CLAHE ortho bake -> .obj + .mtl + _texture.png + .stl + .glb
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
class PipelineV18Config:
    input_image: Path
    input_mesh: Path
    output: Path                       # base path (extension replaced per format)
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

    # SSLE point cloud (V17 params)
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

    # Wrap (mesh + texture)
    wrap_atlas_res: int = 4096
    wrap_target_faces: int = 60_000
    wrap_xatlas_timeout_s: int = 180
    wrap_clahe_clip: float = 3.0
    wrap_clahe_tile: int = 16

    dry_run: bool = False


def run_pipeline_v18(cfg: PipelineV18Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    parent = cfg.output.parent
    stem = cfg.output.stem
    print("=" * 66)
    print("  Pipeline v18 -- SSLE combo pack (PLY + OBJ/MTL/PNG + STL + GLB)")
    print("=" * 66)
    if cfg.dry_run: return cfg.output

    t_all = time.time()
    if not Path(cfg.input_mesh).exists():
        raise FileNotFoundError(f"input_mesh missing: {cfg.input_mesh}")
    current_mesh = Path(cfg.input_mesh)

    # ---- Stage C+D
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

    # ---- Stage E SSLE (PLY + XYZ with RGB)
    if cfg.run_module_e_ssle:
        print(f"\n>>>>> STAGE E SSLE (PLY + XYZ with RGB) <<<<<", flush=True)
        from module_e_ssle import ModuleESSLEConfig, run_module_e_ssle
        ply_out = parent / f"{stem}.ply"
        e_cfg = ModuleESSLEConfig(
            input_mesh=current_mesh, input_image=cfg.input_image, output_ply=ply_out,
            target_height_mm=cfg.target_height_mm,
            target_count=cfg.target_count, accept_low=cfg.accept_low,
            accept_high=cfg.accept_high, n_candidates=cfg.n_candidates,
            r_min_mm=cfg.r_min_mm, r_min_floor_mm=cfg.r_min_floor_mm,
            w_floor=cfg.w_floor, w_gamma=cfg.w_gamma, w_back=cfg.w_back,
            face_boost=cfg.face_boost,
            normal_jitter_mm=cfg.normal_jitter_mm,
            front_axis=cfg.front_axis, seed=cfg.seed,
        )
        run_module_e_ssle(e_cfg)

    # ---- Stage WRAP TEXTURE (OBJ + MTL + PNG + STL + GLB)
    if cfg.run_module_wrap:
        print(f"\n>>>>> STAGE WRAP TEXTURE (QA preview package) <<<<<", flush=True)
        from module_wrap_texture import ModuleWrapConfig, run_module_wrap
        wrap_stem = parent / stem
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

    # Cleanup intermediate
    try:
        cd_intermediate = parent / f"{stem}_stageCD.glb"
        if cd_intermediate.exists() and cd_intermediate != cfg.output:
            cd_intermediate.unlink()
    except Exception as e:
        print(f"  cleanup warn: {e}", flush=True)

    # Ensure the output field points to something useful (GLB, best all-in-one)
    glb_final = parent / f"{stem}.glb"
    if glb_final.exists():
        cfg.output = glb_final

    print(f"\n[PIPELINE v18 DONE] total {time.time()-t_all:.1f}s", flush=True)
    print(f"  primary output -> {cfg.output}", flush=True)
    return cfg.output


def build_argparser():
    ap = argparse.ArgumentParser(prog="pipeline_v18")
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    ap.add_argument("--target-count", type=int, default=300_000)
    ap.add_argument("--target-height-mm", type=float, default=70.0)
    ap.add_argument("--atlas-res", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg = PipelineV18Config(
        input_image=a.input_image, input_mesh=a.input_mesh, output=a.output,
        workdir=a.workdir,
        target_count=a.target_count, target_height_mm=a.target_height_mm,
        wrap_atlas_res=a.atlas_res, dry_run=a.dry_run,
    )
    run_pipeline_v18(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
