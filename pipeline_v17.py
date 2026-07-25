"""
Pipeline v17 -- SSLE (subsurface laser engraving) point cloud for CrystalMe3D Pro.

Chain:
  1. Input mesh (TripoSG raw) + input photo.
  2. Module C+D (pipeline_v11) -- adds face relief. Same topology, deeper emboss
     than V14 (cd_emboss_strength=0.025 default) so the sampled points show real
     nose/eye/ear geometry in 3D parallax.
  3. Module E SSLE -- surface-samples mesh, luminance-weighted blue-noise
     elimination, writes ASCII PLY (Kent's CrystalMe engraver ingests PLY natively).

Output artifacts:
    person_2_v17.ply           -- primary (ASCII PLY, mm, Y-up)
    person_2_v17.xyz           -- fallback (raw text x y z per line)
    person_2_v17_intermediate.glb -- C+D output (audit; deleted at end)

Kent's product: CrystalMe3D Pro (crystalme3d.com) -- subsurface laser engraves
internal micro-fracture points inside a clear crystal cube. Density modulation
carries the portrait tone; the mesh shape carries the parallax/silhouette.
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
class PipelineV17Config:
    input_image: Path
    input_mesh: Path
    output: Path                       # will be .ply
    workdir: Path = Path("/workspace")

    run_module_cd: bool = True
    run_module_e_ssle: bool = True

    # C+D (deeper emboss for V17 vs V14's 0.010)
    cd_dav2_model: str = "base"
    cd_mask_expand: float = 1.2
    cd_face_weight: float = 0.7
    cd_umeyama_clamp_min: float = 0.5
    cd_umeyama_clamp_max: float = 1.5
    cd_emboss_strength: float = 0.025

    # SSLE knobs (Fable 5 v2 locked)
    cube_w_mm: float = 60.0
    cube_d_mm: float = 60.0
    cube_h_mm: float = 80.0
    margin_mm: float = 5.0
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

    dry_run: bool = False


def run_pipeline_v17(cfg: PipelineV17Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    parent = cfg.output.parent
    stem = cfg.output.stem
    if cfg.output.suffix.lower() != ".ply":
        cfg.output = cfg.output.with_suffix(".ply")

    print("=" * 66)
    print("  Pipeline v17 -- SSLE point cloud for CrystalMe3D Pro")
    print("=" * 66)
    print(f"  input image  : {cfg.input_image}")
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  output       : {cfg.output}")
    print(f"  cube (mm)    : {cfg.cube_w_mm} x {cfg.cube_d_mm} x {cfg.cube_h_mm}")
    print(f"  target       : {cfg.target_count} points")
    print("=" * 66, flush=True)
    if cfg.dry_run: print("[dry-run] validated"); return cfg.output

    t_all = time.time()
    if not Path(cfg.input_mesh).exists():
        raise FileNotFoundError(f"input_mesh missing: {cfg.input_mesh}")
    current_mesh = Path(cfg.input_mesh)

    # ---- Stage C+D (deeper face relief for V17)
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
        print(f"      C+D result: {current_mesh}", flush=True)

    # ---- Stage E SSLE (point cloud + PLY)
    if cfg.run_module_e_ssle:
        print(f"\n>>>>> STAGE E SSLE <<<<<", flush=True)
        from module_e_ssle import ModuleESSLEConfig, run_module_e_ssle
        e_cfg = ModuleESSLEConfig(
            input_mesh=current_mesh, input_image=cfg.input_image,
            output_ply=cfg.output,
            cube_w_mm=cfg.cube_w_mm, cube_d_mm=cfg.cube_d_mm, cube_h_mm=cfg.cube_h_mm,
            margin_mm=cfg.margin_mm,
            target_count=cfg.target_count, accept_low=cfg.accept_low,
            accept_high=cfg.accept_high, n_candidates=cfg.n_candidates,
            r_min_mm=cfg.r_min_mm, r_min_floor_mm=cfg.r_min_floor_mm,
            w_floor=cfg.w_floor, w_gamma=cfg.w_gamma, w_back=cfg.w_back,
            face_boost=cfg.face_boost,
            normal_jitter_mm=cfg.normal_jitter_mm,
            front_axis=cfg.front_axis, seed=cfg.seed,
        )
        run_module_e_ssle(e_cfg)

    # Cleanup intermediate GLB
    try:
        cd_intermediate = parent / f"{stem}_stageCD.glb"
        if cd_intermediate.exists() and cd_intermediate != cfg.output:
            cd_intermediate.unlink()
    except Exception as e:
        print(f"  cleanup warn: {e}", flush=True)

    print(f"\n[PIPELINE v17 DONE] total elapsed: {time.time()-t_all:.1f}s")
    print(f"  final -> {cfg.output}", flush=True)
    return cfg.output


def build_argparser():
    ap = argparse.ArgumentParser(prog="pipeline_v17")
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    ap.add_argument("--no-run-module-cd", dest="run_module_cd", action="store_false")
    ap.add_argument("--no-run-module-e-ssle", dest="run_module_e_ssle", action="store_false")
    ap.add_argument("--target-count", type=int, default=300_000)
    ap.add_argument("--cube-w-mm", type=float, default=60.0)
    ap.add_argument("--cube-h-mm", type=float, default=80.0)
    ap.add_argument("--cd-emboss-strength", type=float, default=0.025)
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(run_module_cd=True, run_module_e_ssle=True)
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg = PipelineV17Config(
        input_image=a.input_image, input_mesh=a.input_mesh, output=a.output,
        workdir=a.workdir,
        run_module_cd=a.run_module_cd, run_module_e_ssle=a.run_module_e_ssle,
        target_count=a.target_count,
        cube_w_mm=a.cube_w_mm, cube_h_mm=a.cube_h_mm,
        cd_emboss_strength=a.cd_emboss_strength,
        dry_run=a.dry_run,
    )
    run_pipeline_v17(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
