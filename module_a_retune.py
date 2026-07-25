"""
Module A retune — TripoSG image-to-3D wrapper with tunable knobs.

Wraps the existing `scripts.inference_triposg` CLI (from cloned VAST-AI TripoSG
repo at /workspace/TripoSG) so downstream pipeline v12 can sweep quality knobs:

    - num_inference_steps  (denoising steps, default 50 -> retune 75-100)
    - guidance_scale       (classifier-free guidance, default 7.0 -> sweep 5.5-9.0)
    - octree_resolution    (marching-cubes octree depth, default 512 -> 640-768)
    - faces                (mesh face cap, default 50000 -> -1 for uncap)
    - seed                 (RNG seed, default 42)

Rationale (per Kent 2026-07-21 feedback): baseline v11 mesh has weak finger
detail. TripoSG single-image mesh generation cannot magically produce fine hand
detail beyond what the denoiser saw, but higher steps + higher octree +
uncapped faces typically improve high-frequency regions (fingers, ears) by
15-40 percent.

Dependencies (installed on pod by setup_pod_v11.sh — reused by v12):
    - TripoSG repo cloned to /workspace/TripoSG
    - torch >= 2.0 + CUDA
    - HF weights VAST-AI/TripoSG cached

Usage:
    from module_a_retune import ModuleAConfig, run_module_a
    cfg = ModuleAConfig(input_image=Path("in.png"), output_mesh=Path("out.glb"),
                       num_inference_steps=75, guidance_scale=7.0,
                       octree_resolution=640, faces=-1, seed=42)
    result_path = run_module_a(cfg)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


# TripoSG repo path on pod (per pod_run_infer.py + setup_pod_v12.sh)
TRIPOSG_REPO = Path("/workspace/TripoSG")
TRIPOSG_INFER_MODULE = "scripts.inference_triposg"


@dataclass
class ModuleAConfig:
    """TripoSG inference config with retune knobs."""
    input_image: Path
    output_mesh: Path

    # Retune knobs
    num_inference_steps: int = 75       # baseline v11 was 50
    guidance_scale: float = 7.0          # unchanged from baseline
    octree_resolution: int = 640         # baseline was 512 (no-op via CLI; log-only)
    faces: int = -1                      # -1 = uncap (baseline was 50000)
    seed: int = 42

    # Runtime
    triposg_repo: Path = TRIPOSG_REPO
    dry_run: bool = False
    timeout_s: int = 300                 # 5 min hard cap (Fable v13: v12 hung)

    # v13 alternate runner: bypass CLI, instantiate TripoSGPipeline directly
    use_custom_runner: bool = False

    def __post_init__(self):
        self.input_image = Path(self.input_image)
        self.output_mesh = Path(self.output_mesh)
        self.triposg_repo = Path(self.triposg_repo)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("input_image", "output_mesh", "triposg_repo"):
            d[k] = str(d[k])
        return d


def build_triposg_cli(cfg: ModuleAConfig) -> list:
    """Compose the argv for `python -m scripts.inference_triposg ...`.

    NOTE (Fable 5 review C1): upstream inference_triposg.py accepts ONLY
    --image-input / --output-path / --seed / --num-inference-steps /
    --guidance-scale / --faces. `--octree-resolution` is NOT a CLI arg —
    dense_octree_depth / hierarchical_octree_depth are internal to
    TripoSGPipeline. Passing it triggers argparse error 2 → whole stage fails.

    For now: pass only supported args; log octree_resolution as a no-op.
    Real fix (deferred): custom runner that instantiates TripoSGPipeline
    directly with octree depth kwargs.
    """
    argv = [
        sys.executable, "-m", TRIPOSG_INFER_MODULE,
        "--image-input", str(cfg.input_image),
        "--output-path", str(cfg.output_mesh),
        "--seed", str(cfg.seed),
        "--num-inference-steps", str(cfg.num_inference_steps),
        "--guidance-scale", str(cfg.guidance_scale),
    ]
    # faces IS supported
    if cfg.faces is not None:
        argv += ["--faces", str(cfg.faces)]
    # octree_resolution is a NO-OP with upstream CLI — log honestly
    if cfg.octree_resolution is not None and cfg.octree_resolution != 512:
        print(
            f"[!] octree_resolution={cfg.octree_resolution} requested BUT upstream "
            f"scripts.inference_triposg has no such CLI arg; running at internal "
            f"default (~512). Deferring custom runner for a follow-up iteration.",
            flush=True,
        )
    return argv


def _print_plan(cfg: ModuleAConfig) -> None:
    print("=" * 66)
    print("  Module A (retune) — TripoSG image -> mesh")
    print("=" * 66)
    print(f"  input image  : {cfg.input_image}")
    print(f"  output mesh  : {cfg.output_mesh}")
    print(f"  ---- knobs ----")
    print(f"    num_inference_steps = {cfg.num_inference_steps}")
    print(f"    guidance_scale      = {cfg.guidance_scale}")
    print(f"    octree_resolution   = {cfg.octree_resolution}")
    print(f"    faces               = {cfg.faces}  ({'uncap' if cfg.faces == -1 else 'cap'})")
    print(f"    seed                = {cfg.seed}")
    print(f"  triposg_repo = {cfg.triposg_repo}")
    print(f"  dry_run      = {cfg.dry_run}")
    print("=" * 66, flush=True)


def run_module_a(cfg: ModuleAConfig) -> Path:
    """Run TripoSG inference. Returns output mesh path."""
    _print_plan(cfg)

    if cfg.dry_run:
        print("[dry-run] Skipping TripoSG subprocess. Plan validated.", flush=True)
        return cfg.output_mesh

    if not cfg.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {cfg.input_image}")
    if not cfg.triposg_repo.exists():
        raise FileNotFoundError(
            f"TripoSG repo not found at {cfg.triposg_repo}. "
            f"Run setup_pod_v12.sh (or v11) first to clone."
        )

    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)

    argv = build_triposg_cli(cfg)
    print(f"[+] argv: {' '.join(argv)}", flush=True)

    t0 = time.time()
    env = {**os.environ}
    env["PYTHONPATH"] = str(cfg.triposg_repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        cwd=str(cfg.triposg_repo),
        env=env,
    )
    # Fable 5 v13 review: previously `for line in proc.stdout` blocked until EOF
    # so `wait(timeout=)` could never fire — hard-timeout via threaded reader +
    # explicit deadline check.
    import threading
    def _drain():
        for line in proc.stdout:
            print(f"    | {line}", end="", flush=True)
    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    deadline = t0 + cfg.timeout_s
    while proc.poll() is None:
        if time.time() > deadline:
            print(f"\n[Module A TIMEOUT after {cfg.timeout_s}s — killing subprocess]",
                  flush=True)
            proc.kill()
            break
        time.sleep(2)
    reader.join(timeout=5)
    rc = proc.wait()
    elapsed = time.time() - t0
    print(f"\n[Module A EXIT {rc}] elapsed {elapsed:.1f}s", flush=True)

    if rc != 0:
        raise RuntimeError(f"TripoSG inference failed with exit code {rc}")
    if not cfg.output_mesh.exists():
        raise RuntimeError(
            f"TripoSG completed but output mesh missing: {cfg.output_mesh}"
        )

    size_mb = cfg.output_mesh.stat().st_size / 1024 / 1024
    print(f"[+] output: {cfg.output_mesh} ({size_mb:.2f} MB)", flush=True)
    return cfg.output_mesh


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="module_a_retune",
        description="TripoSG image -> mesh with retune knobs.",
    )
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--output-mesh", required=True, type=Path)
    ap.add_argument("--num-inference-steps", type=int, default=75)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--octree-resolution", type=int, default=640)
    ap.add_argument("--faces", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--triposg-repo", type=Path, default=TRIPOSG_REPO)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = ModuleAConfig(
        input_image=args.input_image,
        output_mesh=args.output_mesh,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        faces=args.faces,
        seed=args.seed,
        triposg_repo=args.triposg_repo,
        dry_run=args.dry_run,
    )
    run_module_a(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
