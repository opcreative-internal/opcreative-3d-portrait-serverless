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
        # RMBG returns a list of feature maps; the last one is the mask logits
        if isinstance(out, (list, tuple)):
            mask_logits = out[-1] if not isinstance(out[-1], (list, tuple)) else out[-1][0]
        else:
            mask_logits = out
        mask = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()
        # resize back to original
        import cv2
        mask_up = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_up = np.clip(mask_up, 0.0, 1.0)[..., None]   # H,W,1
        composited = arr.astype(np.float32) * mask_up + gray_replace * (1.0 - mask_up)
        composited = np.clip(composited, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(composited)
        # simple diagnostics
        fg = float(mask_up.mean())
        print(f"      [bgremove] mask fg_fraction={fg:.3f} (replaced BG with gray={gray_replace})",
              flush=True)
        return out_img
    except Exception as e:
        print(f"      [bgremove] inference failed: {e} -- returning original", flush=True)
        return pil_img
