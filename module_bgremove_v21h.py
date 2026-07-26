"""V21h Bug D (Fable v3): background removal before wrap bake.

Uses TripoSG's already-baked RMBG-1.4 weights at /opt/TripoSG/pretrained_weights/RMBG-1.4.
No new pip dependency — RMBG-1.4 is a transformers-compatible model.

Falls back to no-op if RMBG unavailable (returns input unchanged).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


_MODEL = None
_MODEL_TRIED = False


def _load_rmbg():
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        # RMBG-1.4 on HF is transformers-compatible with trust_remote_code
        from transformers import AutoModelForImageSegmentation
        import torch
        weights_dir = "/opt/TripoSG/pretrained_weights/RMBG-1.4"
        model = AutoModelForImageSegmentation.from_pretrained(
            weights_dir, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _MODEL = (model, device)
        print(f"      [bgremove] RMBG-1.4 loaded from {weights_dir} on {device}", flush=True)
        return _MODEL
    except Exception as e:
        print(f"      [bgremove] load failed: {e} -- BG remove will no-op", flush=True)
        return None


def remove_background(pil_img, gray_replace: int = 128):
    """RGB PIL image -> RGB PIL image with background replaced by neutral gray.

    Prevents background pixels from being baked onto silhouette-adjacent mesh faces
    (Fable v3 root cause #4). Neutral gray = 128 keeps texture atlas from having
    hard color edges at the silhouette.
    """
    m = _load_rmbg()
    if m is None:
        return pil_img
    model, device = m
    try:
        import torch
        from PIL import Image
        from torchvision import transforms as T
        img = pil_img.convert("RGB")
        arr = np.array(img)
        orig_h, orig_w = arr.shape[:2]
        # RMBG-1.4 uses 1024x1024 by default
        tf = T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
        inp = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        # V21h.1 Fable v3 fix: BriaRMBG forward returns ([d1..d6 masks], [features]).
        # Previous code took out[-1][0] = first FEATURE MAP shape (1,64,1024,1024) which
        # cannot become a 2D mask (OpenCV `m->dims <= 2` assertion on 3D array).
        # Correct: out[0][0] = d1 (finest mask, already sigmoided inside model).
        if isinstance(out, (list, tuple)):
            mask_logits = out[0][0] if isinstance(out[0], (list, tuple)) else out[0]
        else:
            mask_logits = out
        # d1 is already sigmoided; DO NOT double-sigmoid (that squashes to [0.5, 0.73]).
        # Official BriaRMBG postproc is min-max normalize.
        mask = mask_logits.squeeze().float().cpu().numpy()
        if mask.ndim > 2:
            mask = mask[0] if mask.shape[0] == 1 else mask.mean(axis=0)
        mn = float(mask.min()); mx = float(mask.max())
        mask = (mask - mn) / (mx - mn + 1e-8)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        # resize back to original
        import cv2
        mask_up = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_up = np.clip(mask_up, 0.0, 1.0)[..., None]   # H,W,1
        composited = arr.astype(np.float32) * mask_up + gray_replace * (1.0 - mask_up)
        composited = np.clip(composited, 0, 255).astype(np.uint8)
        # V21h.3 Fable v3: emit RGBA so wrap bake can use alpha for subject bbox mapping
        alpha_u8 = np.clip(mask_up.squeeze() * 255.0, 0, 255).astype(np.uint8)
        rgba = np.concatenate([composited, alpha_u8[..., None]], axis=2)
        out_img = Image.fromarray(rgba, mode='RGBA')
        # simple diagnostics
        fg = float(mask_up.mean())
        print(f"      [bgremove] mask fg_fraction={fg:.3f} (replaced BG with gray={gray_replace})",
              flush=True)
        return out_img
    except Exception as e:
        print(f"      [bgremove] inference failed: {e} -- returning original", flush=True)
        return pil_img
