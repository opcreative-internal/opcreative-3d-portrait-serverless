"""
Pipeline v15 -- V14 + Module F head-body graft (single seamless mesh).

V14 output was a Scene with two disjoint meshes (textured body + skin-color
head). V15 runs the full V14 chain then applies Module F to weld the head/body
boundary and Laplacian-smooth the seam ring so the mesh reads as a single
figure with both face relief detail and photo texture.

Outputs (all in cfg.output.parent):
    person_2_v15.glb                        -- single seamless merged mesh
    person_2_v15_face_relief_only.glb       -- head-only cut (from V14)
    person_2_v15_body_textured.glb          -- V14 Scene output (kept as debug ref)
    person_2_v15_transform.json             -- V14 fit-transform sidecar
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
class PipelineV15Config:
    input_image: Path
    output: Path
    input_mesh: Optional[Path] = None
    base_mesh: Optional[Path] = None
    workdir: Path = Path("/workspace")

    # V14 stage toggles (defaults per V14 §4)
    run_module_a: bool = False
    run_module_cd: bool = True
    run_module_e: bool = True

    a_num_inference_steps: int = 75
    a_guidance_scale: float = 7.0
    a_faces: int = 500_000
    a_seed: int = 42
    a_timeout_s: int = 300

    cd_dav2_model: str = "base"
    cd_mask_expand: float = 1.2
    cd_face_weight: float = 0.7
    cd_umeyama_clamp_min: float = 0.5
    cd_umeyama_clamp_max: float = 1.5
    cd_emboss_strength: float = 0.010

    e_uv_bake_res: int = 2048
    e_dot_density_scale: float = 1.0
    e_dot_size_px: int = 3
    e_dot_luminance_curve: str = "linear"

    # Module F knobs
    f_weld_tol_frac: float = 0.01
    f_smooth_iterations: int = 3
    f_smooth_ring_depth: int = 2
    f_smooth_lambda: float = 0.5

    dry_run: bool = False


def run_pipeline_v15(cfg: PipelineV15Config) -> Path:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("  Pipeline v15 -- V14 + Module F head-body graft")
    print("=" * 66)
    print(f"  input image  : {cfg.input_image}")
    print(f"  input mesh   : {cfg.input_mesh}")
    print(f"  output       : {cfg.output}")
    print("=" * 66, flush=True)
    if cfg.dry_run:
        print("[dry-run] validated"); return cfg.output

    t_all = time.time()
    parent = cfg.output.parent
    stem = cfg.output.stem  # e.g. person_2_v15

    # V14 writes to sidecars named from output stem, so give V14 its own tmp
    # output stem then copy final V14 assets to v15-named files after Module F.
    v14_out = parent / (stem + "_v14tmp.glb")

    # --------- Stage 1: full V14 chain ---------
    print(f"\n>>>>> V14 CHAIN <<<<<", flush=True)
    from pipeline_v14 import PipelineV14Config, run_pipeline_v14
    v14_cfg = PipelineV14Config(
        input_image=cfg.input_image,
        input_mesh=cfg.input_mesh,
        base_mesh=cfg.base_mesh,
        output=v14_out,
        workdir=cfg.workdir,
        run_module_a=cfg.run_module_a,
        run_module_cd=cfg.run_module_cd,
        run_module_e=cfg.run_module_e,
        a_num_inference_steps=cfg.a_num_inference_steps,
        a_guidance_scale=cfg.a_guidance_scale,
        a_faces=cfg.a_faces,
        a_seed=cfg.a_seed,
        a_timeout_s=cfg.a_timeout_s,
        cd_dav2_model=cfg.cd_dav2_model,
        cd_mask_expand=cfg.cd_mask_expand,
        cd_face_weight=cfg.cd_face_weight,
        cd_umeyama_clamp_min=cfg.cd_umeyama_clamp_min,
        cd_umeyama_clamp_max=cfg.cd_umeyama_clamp_max,
        cd_emboss_strength=cfg.cd_emboss_strength,
        e_uv_bake_res=cfg.e_uv_bake_res,
        e_dot_density_scale=cfg.e_dot_density_scale,
        e_dot_size_px=cfg.e_dot_size_px,
        e_dot_luminance_curve=cfg.e_dot_luminance_curve,
    )
    v14_final = run_pipeline_v14(v14_cfg)

    # Sidecars V14 emitted
    v14_relief = v14_out.with_name(v14_out.stem + "_face_relief_only.glb")
    v14_transform = v14_out.with_name(v14_out.stem + "_transform.json")

    if not v14_final.exists() or not v14_relief.exists():
        raise FileNotFoundError(
            f"V14 chain didn't produce expected outputs: {v14_final} / {v14_relief}"
        )

    # --------- Stage 2: Module F head-body graft ---------
    print(f"\n>>>>> MODULE F (graft) <<<<<", flush=True)
    from module_f_graft import ModuleFConfig, run_module_f
    f_cfg = ModuleFConfig(
        face_relief_glb=v14_relief,
        body_scene_glb=v14_final,
        output_glb=cfg.output,
        weld_tol_frac=cfg.f_weld_tol_frac,
        smooth_iterations=cfg.f_smooth_iterations,
        smooth_ring_depth=cfg.f_smooth_ring_depth,
        smooth_lambda=cfg.f_smooth_lambda,
    )
    run_module_f(f_cfg)

    # --------- Rename debug refs to v15 stems + copy sidecars ---------
    # Keep V14 scene output as debug reference
    body_ref = parent / (stem + "_body_textured.glb")
    if v14_final.exists() and body_ref != v14_final:
        try: shutil.copy2(v14_final, body_ref)
        except Exception as e: print(f"  copy body_ref warning: {e}", flush=True)

    v15_relief = parent / (stem + "_face_relief_only.glb")
    if v14_relief.exists() and v15_relief != v14_relief:
        try: shutil.copy2(v14_relief, v15_relief)
        except Exception as e: print(f"  copy relief warning: {e}", flush=True)

    v15_transform = parent / (stem + "_transform.json")
    if v14_transform.exists() and v15_transform != v14_transform:
        try: shutil.copy2(v14_transform, v15_transform)
        except Exception as e: print(f"  copy transform warning: {e}", flush=True)

    print(f"\n[PIPELINE v15 DONE] total elapsed: {time.time()-t_all:.1f}s")
    print(f"  final -> {cfg.output}", flush=True)
    return cfg.output


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline_v15")
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--input-mesh", type=Path, default=None)
    ap.add_argument("--base-mesh", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    ap.add_argument("--run-module-a", action="store_true")
    ap.add_argument("--no-run-module-cd", dest="run_module_cd", action="store_false")
    ap.add_argument("--no-run-module-e", dest="run_module_e", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(run_module_cd=True, run_module_e=True)
    return ap


def main(argv=None) -> int:
    a = build_argparser().parse_args(argv)
    cfg = PipelineV15Config(
        input_image=a.input_image, input_mesh=a.input_mesh, base_mesh=a.base_mesh,
        output=a.output, workdir=a.workdir,
        run_module_a=a.run_module_a,
        run_module_cd=a.run_module_cd, run_module_e=a.run_module_e,
        dry_run=a.dry_run,
    )
    run_pipeline_v15(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
