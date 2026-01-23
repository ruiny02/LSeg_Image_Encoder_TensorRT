#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import clip

# make sure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# reuse realtime code path (keeps behavior identical)
from realtime.lseg_realtime import (  # noqa: E402
    TRTEncoder,
    compute_mask_from_features,
    draw_legend,
    ensure_onnx_and_engine,
    get_new_palette,
    infer_backbone_from_path,
    load_torch_encoder,
    preprocess_frame_to_tensor,
)


def _comma_split(s: str) -> List[str]:
    labels = [x.strip() for x in s.split(",")]
    return [x for x in labels if x]


def _gather_images(inp: str) -> List[Path]:
    p = Path(inp)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if p.is_file():
        return [p]
    if p.is_dir():
        return [x for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in exts]
    # glob support
    return [Path(x) for x in sorted(Path().glob(inp)) if Path(x).is_file()]


def _alpha_blend_bgr(base_bgr: np.ndarray, mask_idx: np.ndarray, palette_bgr: np.ndarray, alpha: float) -> np.ndarray:
    color = palette_bgr[mask_idx]  # (H,W,3) uint8 BGR
    out = (base_bgr.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LSeg offline inference for image folder")

    p.add_argument("--input", required=True, help="image file / directory / glob (e.g. samples/*.jpg)")
    p.add_argument("--out_dir", default="outputs/image_folder", help="output directory")

    p.add_argument("--weights", required=True, help="ckpt path")
    p.add_argument("--backend", choices=["trt", "torch"], default="trt")
    p.add_argument("--backbone", choices=["vit", "rn101", "auto"], default="auto")
    p.add_argument("--engine", default=None, help="TRT engine path (if backend=trt)")
    p.add_argument("--no_build", action="store_true", help="do not auto-build onnx/engine")

    p.add_argument("--labels", default="person, background", help="comma separated labels")
    p.add_argument("--clip_model", default="ViT-B/32")
    p.add_argument("--prompt", choices=["plain", "a_photo_of_a"], default="plain")

    p.add_argument("--in_w", type=int, default=512)
    p.add_argument("--in_h", type=int, default=288)

    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--no_legend", action="store_true")
    p.add_argument("--save_index", action="store_true", help="save *_index.png uint8 mask")

    # only used if auto-building TRT
    p.add_argument("--workspace", type=int, default=1 << 30)
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--int8", action="store_true", default=False)
    p.add_argument("--calib_dir", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (container runtime 문제).")

    labels = _comma_split(args.labels)
    if len(labels) < 2:
        raise ValueError("--labels 는 최소 2개 필요")

    images = _gather_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images found: {args.input}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_hw: Tuple[int, int] = (int(args.in_h), int(args.in_w))
    palette = get_new_palette(len(labels))

    device = torch.device("cuda")

    # CLIP text features
    clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    clip_labels: Sequence[str] = labels
    if args.prompt == "a_photo_of_a":
        clip_labels = [f"a photo of a {l}" for l in labels]

    with torch.no_grad():
        tok = clip.tokenize(list(clip_labels)).to(device)
        text_features = clip_model.encode_text(tok)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

    backbone = args.backbone
    if backbone == "auto":
        backbone = infer_backbone_from_path(args.weights)

    engine_path = None
    trt_encoder = None
    torch_encoder = None

    if args.backend == "trt":
        if args.engine:
            engine_path = args.engine
            if not os.path.exists(engine_path):
                raise FileNotFoundError(engine_path)
        else:
            if args.no_build:
                raise ValueError("backend=trt + --no_build면 --engine 필수")
            _, engine_path = ensure_onnx_and_engine(
                weights=args.weights,
                backbone=backbone,
                workspace=args.workspace,
                fp16=args.fp16,
                int8=args.int8,
                calib_dir=args.calib_dir,
                input_hw=input_hw,
                engine_out=None,
            )
        trt_encoder = TRTEncoder(engine_path, input_shape=(1, 3, input_hw[0], input_hw[1]))
        print(f"[TRT ] engine: {engine_path}")
    else:
        torch_encoder = load_torch_encoder(args.weights, backbone)
        print("[TORCH] ckpt encoder loaded")

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    enc_ms_all: List[float] = []
    tot_ms_all: List[float] = []

    for i, img_path in enumerate(images, 1):
        t0 = time.perf_counter()

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] skip unreadable: {img_path}")
            continue
        h, w = bgr.shape[:2]

        x_cpu = preprocess_frame_to_tensor(bgr, input_hw=input_hw)
        x = x_cpu.to(device=device, non_blocking=True)

        torch.cuda.synchronize()
        ev0.record()
        if args.backend == "trt":
            feat = trt_encoder(x)  # type: ignore[misc]
        else:
            feat = torch_encoder(x)  # type: ignore[misc]
        ev1.record()
        torch.cuda.synchronize()
        enc_ms = float(ev0.elapsed_time(ev1))

        mask = compute_mask_from_features(feat, text_features)  # (Hf,Wf) GPU
        mask_np = mask.detach().to("cpu").numpy().astype(np.uint8)
        mask_full = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = _alpha_blend_bgr(bgr, mask_full, palette, float(args.alpha))
        if not args.no_legend:
            draw_legend(overlay, labels, palette)

        stem = img_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), palette[mask_full])  # color mask
        if args.save_index:
            cv2.imwrite(str(out_dir / f"{stem}_index.png"), mask_full)

        tot_ms = (time.perf_counter() - t0) * 1000.0
        enc_ms_all.append(enc_ms)
        tot_ms_all.append(tot_ms)

        print(f"[{i:04d}/{len(images):04d}] {img_path.name}  enc={enc_ms:.2f}ms  total={tot_ms:.2f}ms")

    if enc_ms_all:
        enc = np.array(enc_ms_all, dtype=np.float64)
        tot = np.array(tot_ms_all, dtype=np.float64)
        fps = 1000.0 / tot.mean() if tot.mean() > 0 else float("nan")
        print("\n==================== SUMMARY ====================")
        print(f"[INFO] backend   : {args.backend}")
        print(f"[INFO] weights   : {args.weights}")
        if engine_path:
            print(f"[INFO] engine    : {engine_path}")
        print(f"[INFO] images    : {len(enc_ms_all)}")
        print(f"[INFO] input_hw  : {input_hw[0]}x{input_hw[1]}")
        print(f"[INFO] labels({len(labels)}): {', '.join(labels)}")
        print("-------------------------------------------------")
        print(f"[RESULT] Encoder : Avg={enc.mean():.3f} ms ± {enc.std():.3f} ms")
        print(f"[RESULT] Total   : Avg={tot.mean():.3f} ms ± {tot.std():.3f} ms   (FPS={fps:.2f})")
        print(f"[OUT] {out_dir.resolve()}")
        print("=================================================\n")


if __name__ == "__main__":
    main()
