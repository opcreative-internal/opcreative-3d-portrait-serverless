"""
Pipeline v16 -- true 3D engrave (single unified GLB with photo-driven geometry).

Rewrite from V15 (which was V14 + head/body graft producing 2 separate meshes).
V16 keeps the mesh unified throughout the chain and adds photo-luminance-driven
GEOMETRY displacement (bumps as verts moved along normals) instead of texture.

Chain:
  1. Load input mesh (TripoSG raw) -- already single, connected topology.
  2. Module C+D (pipeline_v11) -- adds face relief to head region. Topology
     preserved: same verts, same faces, some head verts displaced along depth.
  3. Module E ENGRAVE (module_e_engrave) -- adds body/face luminance dot
     displacement. Same topology, more verts displaced.
  4. Export as ONE GLB (no scene, no split).

Output artifacts (single-file V16 spec):
    person_2_v16.glb           -- unified mesh with C+D face relief + engraved dots
    person_2_v16_run.log       -- pipeline stdout (audit)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineV16Config:
    input_image: Path
    output: Path
    input_mesh: Path
    workdir: Path = Path("/workspace")

    run_module_cd: bool = True
    run_module_e_engrave: bool = True

    # Module C+D (locked from V14)
    cd_dav2_model: str = "base"
    cd_mask_expand: float = 1.2
    cd_face_weight: float = 0.7
    cd_umeyama_clamp_min: float = 0.5
    cd_umeyama_clamp_max: float = 1.5
    cd_emboss_strength: float = 0.010

    # Module E Engrave (V16 new)
    e_emboss_strength: float = 0.004
    e_dot_count: int = 20000
    e_dot_min_radius_px: float = 3.0
    e_influence_radius_px: float = 6.0
    e_lum_curve: str = "linear"
    e_lum_gamma: float = 2.2
    e_front_axis: str = "+Z"
    e_seed: int = 42

    dry_run: bool = False


def run_pipeline_v16(cfg: PipelineV16Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    parent = cfg.output.parent
    stem = cfg.output.stem
    print("=" * 66)
    print("  Pipeline v16 -- unified 3D engrave")
    print("=" * 66)
    print(f"  input image  : {cfg.input_image}")
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  output       : {cfg.output}")
    print(f"  cd on={cfg.run_module_cd}  e_engrave on={cfg.run_module_e_engrave}")
    print("=" * 66, flush=True)
    if cfg.dry_run:
        print("[dry-run] validated"); return cfg.output

    t_all = time.time()
    if not cfg.input_mesh or not Path(cfg.input_mesh).exists():
        raise FileNotFoundError(f"input_mesh missing: {cfg.input_mesh}")
    current_mesh = cfg.input_mesh

    # ---- Stage C+D (face relief on head)
    if cfg.run_module_cd:
        print(f"\n>>>>> STAGE C+D <<<<<", flush=True)
        from pipeline_v11_depth_umeyama import Config as V11Config, run_pipeline as v11_run
        cd_out = parent / f"{stem}_stageCD.glb"
        v11_cfg = V11Config(
            input_image=cfg.input_image,
            input_mesh=current_mesh,
            output=cd_out,
            dav2_model=cfg.cd_dav2_model,
            mask_expand=cfg.cd_mask_expand,
            face_weight=cfg.cd_face_weight,
            umeyama_clamp_min=cfg.cd_umeyama_clamp_min,
            umeyama_clamp_max=cfg.cd_umeyama_clamp_max,
            emboss_strength=cfg.cd_emboss_strength,
        )
        current_mesh = v11_run(v11_cfg)
        print(f"      C+D result: {current_mesh}", flush=True)

    # ---- Stage E ENGRAVE (geometry dot displacement)
    if cfg.run_module_e_engrave:
        print(f"\n>>>>> STAGE E ENGRAVE <<<<<", flush=True)
        from module_e_engrave import ModuleEEngraveConfig, run_module_e_engrave
        e_cfg = ModuleEEngraveConfig(
            input_mesh=current_mesh,
            input_image=cfg.input_image,
            output_mesh=cfg.output,
            emboss_strength=cfg.e_emboss_strength,
            dot_count=cfg.e_dot_count,
            dot_min_radius_px=cfg.e_dot_min_radius_px,
            influence_radius_px=cfg.e_influence_radius_px,
            lum_curve=cfg.e_lum_curve,
            lum_gamma=cfg.e_lum_gamma,
            front_axis=cfg.e_front_axis,
            seed=cfg.e_seed,
        )
        run_module_e_engrave(e_cfg)
    else:
        # No engrave: just copy the C+D output as the final
        if current_mesh != cfg.output:
            shutil.copy2(current_mesh, cfg.output)

    # Cleanup intermediates
    try:
        cd_intermediate = parent / f"{stem}_stageCD.glb"
        if cd_intermediate.exists() and cd_intermediate != cfg.output:
            cd_intermediate.unlink()
    except Exception as e:
        print(f"  cleanup warn: {e}", flush=True)

    print(f"\n[PIPELINE v16 DONE] total elapsed: {time.time()-t_all:.1f}s")
    print(f"  final -> {cfg.output}", flush=True)
    return cfg.output


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline_v16")
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--input-mesh", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    ap.add_argument("--no-run-module-cd", dest="run_module_cd", action="store_false")
    ap.add_argument("--no-run-module-e-engrave", dest="run_module_e_engrave", action="store_false")
    ap.add_argument("--emboss-strength", type=float, default=0.004)
    ap.add_argument("--dot-count", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(run_module_cd=True, run_module_e_engrave=True)
    return ap


def main(argv=None) -> int:
    a = build_argparser().parse_args(argv)
    cfg = PipelineV16Config(
        input_image=a.input_image, input_mesh=a.input_mesh, output=a.output,
        workdir=a.workdir,
        run_module_cd=a.run_module_cd, run_module_e_engrave=a.run_module_e_engrave,
        e_emboss_strength=a.emboss_strength, e_dot_count=a.dot_count,
        dry_run=a.dry_run,
    )
    run_pipeline_v16(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
