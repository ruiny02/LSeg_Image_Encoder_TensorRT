#!/usr/bin/env python3

"""LSeg real-time demo (USB camera) with optional TensorRT.

Goals (per user request)
  * Read frames continuously from /dev/videoX (1280x720)
  * Preprocess to fixed network input (1,3,288,512)
  * Run LSeg image-encoder
      - backend=torch (PyTorch)
      - backend=trt   (TensorRT engine)
  * Compute pixel-wise label mask via CLIP text features (cosine similarity)
  * Alpha-blend overlay + optional legend + FPS/inference time text
  * Print benchmark stats to console (similar style to inferenceTimeTester)

Notes
  * This script uses OpenCV VideoCapture with a GStreamer pipeline.
  * Display is done via cv2.imshow (X11). In Docker, mount /tmp/.X11-unix and set DISPLAY.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 3rd-party
import clip
import tensorrt as trt
# TensorRT <-> torch dtype helpers (for correct I/O buffer types)
_TRt_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype not in _TRt_TO_TORCH:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return _TRt_TO_TORCH[dtype]


def _pick_single(names: List[str], what: str) -> str:
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly 1 {what} tensor, got: {names}")
    return names[0]


# Repo modules (PyTorch checkpoints)
from modules.lseg_module import LSegModule
from modules.lseg_module_zs import LSegModuleZS


# ---------------------------
# Utilities
# ---------------------------


def _comma_split(s: str) -> List[str]:
    labels = [x.strip() for x in s.split(',')]
    labels = [x for x in labels if x]
    return labels


def get_new_palette(num_cls: int) -> np.ndarray:
    """Return palette as (num_cls, 3) uint8 in BGR order (for OpenCV)."""
    if num_cls <= 0:
        raise ValueError("num_cls must be >= 1")
    pal = np.zeros((num_cls, 3), dtype=np.uint8)
    for j in range(num_cls):
        lab = j
        r = g = b = 0
        i = 0
        while lab > 0:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        # OpenCV uses BGR
        pal[j] = (b, g, r)
    return pal


def draw_legend(
    img_bgr: np.ndarray,
    labels: Sequence[str],
    palette_bgr: np.ndarray,
    origin: Tuple[int, int] = (10, 10),
    box: int = 16,
    pad: int = 6,
    font_scale: float = 0.5,
    max_items: int = 30,
) -> None:
    """Draw a simple legend (color box + label text) onto img_bgr in-place."""
    h, w = img_bgr.shape[:2]
    x0, y0 = origin
    y = y0

    n = min(len(labels), max_items)
    for idx in range(n):
        color = tuple(int(c) for c in palette_bgr[idx])
        # box
        cv2.rectangle(img_bgr, (x0, y), (x0 + box, y + box), color, thickness=-1)
        cv2.rectangle(img_bgr, (x0, y), (x0 + box, y + box), (0, 0, 0), thickness=1)

        # text with outline for readability
        text = labels[idx]
        tx = x0 + box + pad
        ty = y + box - 2
        cv2.putText(
            img_bgr,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        y += box + 6
        if y + box + 6 > h:
            break


def build_gstreamer_pipeline(
    device: str,
    width: int,
    height: int,
    fps: int,
    prefer_mjpeg: bool = True,
) -> str:
    """Create a GStreamer pipeline string usable by OpenCV VideoCapture."""
    # Many Logitech webcams support MJPEG at 720p; decoding is usually faster.
    if prefer_mjpeg:
        return (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGRx ! "
            "appsink drop=1 sync=false"
        )

    # Fallback to raw (YUY2 etc)
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "appsink drop=1 sync=false"
    )


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open /dev/videoX using a GStreamer pipeline (MJPEG-first, raw fallback)."""
    pipeline_mjpeg = build_gstreamer_pipeline(device, width, height, fps, prefer_mjpeg=True)
    cap = cv2.VideoCapture(pipeline_mjpeg, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap

    pipeline_raw = build_gstreamer_pipeline(device, width, height, fps, prefer_mjpeg=False)
    cap = cv2.VideoCapture(pipeline_raw, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap

    raise RuntimeError(
        "Failed to open camera with GStreamer pipelines. "
        f"Tried MJPEG:\n{pipeline_mjpeg}\n\nTried RAW:\n{pipeline_raw}\n"
    )


def preprocess_frame_to_tensor(
    frame_bgr: np.ndarray,
    input_hw: Tuple[int, int],
) -> torch.Tensor:
    """BGR uint8 frame -> CPU torch tensor float32 in NCHW normalized to [-1,1]."""
    in_h, in_w = input_hw
    # Handle BGRx from some GStreamer pipelines
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = frame_bgr[:, :, :3]

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to network input
    resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0)  # NCHW
    return torch.from_numpy(x).contiguous()  # float32 CPU


def ensure_onnx_and_engine(
    *,
    weights: str,
    backbone: str,
    workspace: int,
    fp16: bool,
    int8: bool,
    calib_dir: Optional[str],
    input_hw: Tuple[int, int],
    engine_out: Optional[str] = None,
) -> Tuple[str, str]:
    """Create ONNX and TRT engine if missing. Returns (onnx_path, engine_path)."""
    weights = str(weights)
    tag = Path(weights).stem
    if backbone == "vit":
        base = f"lseg_img_enc_vit_{tag}"
        onnx_script = "conversion/model_to_onnx.py"
    elif backbone in ("rn101", "resnet", "resnet101"):
        base = f"lseg_img_enc_rn101_{tag}"
        onnx_script = "conversion/model_to_onnx_zs.py"
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    onnx_path = f"models/onnx_engines/{base}.onnx"
    if not os.path.exists(onnx_path):
        print(f"[ONNX] building: {onnx_path}")
        subprocess.run(["python3", onnx_script, "--weights", weights], check=True)
    else:
        print(f"[ONNX] exists : {onnx_path}")

    # Select ONNX precision for TRT build (Jetson memory saver: FP16 initializers)
    onnx_for_trt = onnx_path
    if fp16 or int8:
        onnx_fp16_path = f"models/onnx_engines/{base}_fp16.onnx"
        # Rebuild fp16 ONNX if missing or older than the fp32 ONNX
        if (not os.path.exists(onnx_fp16_path)) or (os.path.getmtime(onnx_fp16_path) < os.path.getmtime(onnx_path)):
            print(f"[ONNX] converting to FP16: {onnx_fp16_path}")
            subprocess.run(["python3", "conversion/onnx_fp16.py", "--input", onnx_path, "--output", onnx_fp16_path], check=True)
        onnx_for_trt = onnx_fp16_path

    # TRT engine
    in_h, in_w = input_hw
    trt_cmd = [
        "python3",
        "conversion/onnx_to_trt.py",
        "--onnx",
        onnx_for_trt,
        "--workspace",
        str(int(workspace)),
        "--min_hw",
        str(in_h),
        str(in_w),
        "--opt_hw",
        str(in_h),
        str(in_w),
        "--max_hw",
        str(in_h),
        str(in_w),
    ]
    trt_cmd += ["--fp16"] if fp16 else ["--no-fp16"]
    if int8:
        trt_cmd += ["--int8"]
        if calib_dir:
            trt_cmd += ["--calib_dir", calib_dir]
    if engine_out:
        trt_cmd += ["--engine", engine_out]

    # If engine_out not provided, we need to locate the expected auto-named engine.
    # We do a glob search before/after build.
    trt_dir = Path("models") / "trt_engines"
    trt_dir.mkdir(parents=True, exist_ok=True)
    if engine_out:
        engine_path = engine_out
        if not os.path.exists(engine_path):
            print(f"[TRT ] building: {engine_path}")
            subprocess.run(trt_cmd, check=True)
        else:
            print(f"[TRT ] exists : {engine_path}")
        return onnx_path, engine_path

    candidates = sorted(trt_dir.glob(f"{base}__*.trt"))
    if candidates:
        # Use the newest candidate
        engine_path = str(candidates[-1])
        print(f"[TRT ] exists : {engine_path}")
        return onnx_path, engine_path

    print("[TRT ] building (auto-named under models/trt_engines/)")
    subprocess.run(trt_cmd, check=True)
    candidates = sorted(trt_dir.glob(f"{base}__*.trt"))
    if not candidates:
        raise RuntimeError("TRT build finished but no engine file was found.")
    engine_path = str(candidates[-1])
    return onnx_path, engine_path


# ---------------------------
# Backends
# ---------------------------


class TRTEncoder:
    """TensorRT engine runner that binds torch CUDA tensors directly (no DtoH copy).

    Why this exists:
      * Engines can be built as FP16 → I/O tensors may be FP16.
      * If you bind a torch.float32 buffer to an FP16 tensor address (or vice versa),
        you will get corrupted outputs (segmentation looks random or collapses).
      * Therefore, we query TensorRT I/O dtypes and allocate matching torch buffers.
    """

    def __init__(self, engine_path: str, input_shape: Tuple[int, int, int, int]):
        self.engine_path = engine_path
        self.input_shape = tuple(input_shape)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.engine = engine

        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")

        # Robust I/O discovery (do NOT assume index 0=input, 1=output).
        io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        in_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        out_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.in_name = _pick_single(in_names, "input")
        self.out_name = _pick_single(out_names, "output")

        # DTypes
        self.in_dtype_trt = engine.get_tensor_dtype(self.in_name)
        self.out_dtype_trt = engine.get_tensor_dtype(self.out_name)
        self.in_dtype_torch = _trt_dtype_to_torch(self.in_dtype_trt)
        self.out_dtype_torch = _trt_dtype_to_torch(self.out_dtype_trt)

        # Set shape once (fixed in this project), engine may still have dynamic profiles.
        self.context.set_input_shape(self.in_name, self.input_shape)
        self.out_shape = tuple(self.context.get_tensor_shape(self.out_name))

        # Output buffer (torch CUDA tensor). Reallocated if shape changes.
        self._out = torch.empty(self.out_shape, device="cuda", dtype=self.out_dtype_torch)

        print(
            f"[TRT ] I/O: input={self.in_name} {self.in_dtype_trt} {self.input_shape}  | "
            f"output={self.out_name} {self.out_dtype_trt} {self.out_shape}"
        )

    def __call__(self, x_cuda: torch.Tensor) -> torch.Tensor:
        if not x_cuda.is_cuda:
            raise ValueError("TRTEncoder expects a CUDA tensor")

        # Match engine input dtype (common: FP32 or FP16)
        if x_cuda.dtype != self.in_dtype_torch:
            x_cuda = x_cuda.to(dtype=self.in_dtype_torch)

        # Support dynamic shapes if engine profile allows.
        if tuple(x_cuda.shape) != self.input_shape:
            self.context.set_input_shape(self.in_name, tuple(x_cuda.shape))
            self.input_shape = tuple(x_cuda.shape)
            new_out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            if new_out_shape != self.out_shape:
                self.out_shape = new_out_shape
                self._out = torch.empty(self.out_shape, device="cuda", dtype=self.out_dtype_torch)

        if not x_cuda.is_contiguous():
            x_cuda = x_cuda.contiguous()

        self.context.set_tensor_address(self.in_name, int(x_cuda.data_ptr()))
        self.context.set_tensor_address(self.out_name, int(self._out.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        return self._out

def load_torch_encoder(weights: str, backbone: str) -> torch.nn.Module:
    """Load LSeg image-encoder from ckpt (PyTorch)."""
    weights = str(weights)
    if backbone == "vit":
        net = LSegModule.load_from_checkpoint(
            checkpoint_path=weights,
            map_location="cpu",
            backbone="clip_vitl16_384",
            aux=False,
            num_features=256,
            crop_size=480,
            readout="project",
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            ignore_index=255,
            dropout=0.0,
            scale_inv=False,
            augment=False,
            no_batchnorm=False,
            widehead=True,
            widehead_hr=False,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        ).net
    elif backbone in ("rn101", "resnet", "resnet101"):
        net = LSegModuleZS.load_from_checkpoint(
            checkpoint_path=weights,
            map_location="cpu",
            data_path="data/",
            dataset="ade20k",
            backbone="clip_resnet101",
            aux=False,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=255,
            dropout=0.0,
            scale_inv=False,
            augment=False,
            no_batchnorm=False,
            widehead=False,
            widehead_hr=False,
            arch_option=0,
            use_pretrained="True",
            strict=False,
            logpath="fewshot/logpath_4T/",
            fold=0,
            block_depth=0,
            nshot=1,
            finetune_mode=False,
            activation="lrelu",
        ).net
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    net.eval().cuda()
    return net


@torch.no_grad()
def compute_mask_from_features(
    image_features: torch.Tensor,  # (1,C,Hf,Wf)
    text_features: torch.Tensor,  # (N,C)
    *,
    out_hw: Optional[Tuple[int, int]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return argmax mask on GPU with shape (Hf, Wf).

    Notes:
      * LSeg-style inference uses cosine similarity between per-pixel embeddings and text embeddings.
      * Some exported / TensorRT engines may output FP16 tensors; we upcast + re-normalize for stability.
    """
    if image_features.dtype != torch.float32:
        image_features = image_features.float()
    if text_features.dtype != torch.float32:
        text_features = text_features.float()

    # Ensure contiguous for einsum/matmul
    if not image_features.is_contiguous():
        image_features = image_features.contiguous()

    # (Safety) Normalize again. (Some checkpoints already normalize inside the network.)
    image_features = image_features / (image_features.norm(dim=1, keepdim=True) + eps)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)

    # similarity: (B=1, N, Hf, Wf)
    sim = torch.einsum("nc,bchw->bnhw", text_features, image_features)

    # IMPORTANT: upsample logits first, then argmax.
    # Doing argmax at low-res and nearest-upsample the integer mask
    # produces blocky boundaries + speckle artifacts.
    if out_hw is not None:
        sim = F.interpolate(
            sim, size=(int(out_hw[0]), int(out_hw[1])),
            mode="bilinear", align_corners=False
        )

    mask = torch.argmax(sim, dim=1)[0]  # (Hf, Wf)
    return mask

def infer_backbone_from_path(weights: str) -> str:
    p = weights.lower()
    if "resnet" in p or "rn101" in p or "/resnet" in p:
        return "rn101"
    if "/vit" in p or "vit" in p:
        return "vit"
    # Default: vit
    return "vit"


# ---------------------------
# Benchmark
# ---------------------------


@dataclass
class Stats:
    frames: int
    warmup: int
    encoder_ms: List[float]
    total_ms: List[float]

    def add(self, encoder_ms: float, total_ms: float) -> None:
        self.encoder_ms.append(float(encoder_ms))
        self.total_ms.append(float(total_ms))

    def summary(self) -> dict:
        enc = np.array(self.encoder_ms, dtype=np.float64)
        tot = np.array(self.total_ms, dtype=np.float64)
        out = {
            "frames": len(enc),
            "enc_avg_ms": float(enc.mean()) if len(enc) else float("nan"),
            "enc_std_ms": float(enc.std()) if len(enc) else float("nan"),
            "tot_avg_ms": float(tot.mean()) if len(tot) else float("nan"),
            "tot_std_ms": float(tot.std()) if len(tot) else float("nan"),
        }
        if len(tot) and out["tot_avg_ms"] > 0:
            out["fps"] = 1000.0 / out["tot_avg_ms"]
        else:
            out["fps"] = float("nan")
        return out


def print_stats(
    *,
    backend: str,
    weights: str,
    engine: Optional[str],
    input_hw: Tuple[int, int],
    cam_hw: Tuple[int, int],
    labels: Sequence[str],
    stats: Stats,
) -> None:
    s = stats.summary()
    print("\n==================== BENCHMARK ====================")
    print(f"[INFO] backend     : {backend}")
    print(f"[INFO] weights     : {weights}")
    if engine:
        print(f"[INFO] engine      : {engine}")
    print(f"[INFO] camera_hw   : {cam_hw[0]}x{cam_hw[1]}")
    print(f"[INFO] input_hw    : {input_hw[0]}x{input_hw[1]}  (NCHW=1,3,H,W)")
    print(f"[INFO] labels({len(labels)}): {', '.join(labels)}")
    print(f"[INFO] warmup      : {stats.warmup} frames")
    print(f"[INFO] measured    : {s['frames']} frames")
    print("---------------------------------------------------")
    print(f"[RESULT] Encoder : Avg={s['enc_avg_ms']:.3f} ms ± {s['enc_std_ms']:.3f} ms")
    print(f"[RESULT] Total   : Avg={s['tot_avg_ms']:.3f} ms ± {s['tot_std_ms']:.3f} ms   (FPS={s['fps']:.2f})")
    print("===================================================\n")


# ---------------------------
# Main
# ---------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSeg real-time demo (GStreamer + X11)")

    p.add_argument("--device", type=str, default="/dev/video0", help="Camera device (e.g., /dev/video0)")
    p.add_argument("--cam_w", type=int, default=1280)
    p.add_argument("--cam_h", type=int, default=720)
    p.add_argument("--cam_fps", type=int, default=30)

    p.add_argument("--in_w", type=int, default=512, help="Network input width")
    p.add_argument("--in_h", type=int, default=288, help="Network input height")

    p.add_argument(
        "--labels",
        type=str,
        default="person, chair, desk, monitor, keyboard, background",
        help="Comma separated label list",
    )
    p.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model name used to encode text labels (e.g., ViT-B/32, RN50x16, RN101, ViT-B/16, ViT-L/14)",
    )
    p.add_argument(
        "--prompt",
        choices=["plain", "a_photo_of_a"],
        default="plain",
        help="Prompt template for CLIP text embeddings",
    )
    p.add_argument("--alpha", type=float, default=0.45, help="Alpha blending ratio for mask overlay")
    p.add_argument("--no_legend", action="store_true", help="Disable legend drawing")
    p.add_argument("--window", type=str, default="LSeg", help="OpenCV window title")
    p.add_argument("--no_display", action="store_true", help="Run without cv2.imshow (benchmark only)")

    p.add_argument(
        "--backend",
        choices=["trt", "torch"],
        default="trt",
        help="Inference backend for image encoder",
    )

    # Inputs for backend
    p.add_argument("--weights", type=str, required=True, help="Path to .ckpt (used for torch and/or to build ONNX/TRT)")
    p.add_argument(
        "--backbone",
        choices=["vit", "rn101", "auto"],
        default="auto",
        help="Which checkpoint family this is (auto tries to infer from path)",
    )
    p.add_argument("--engine", type=str, default=None, help="Path to TensorRT engine (required if backend=trt and --no_build)")
    p.add_argument("--no_build", action="store_true", help="Do not auto-build ONNX/TRT when missing")

    # TRT build options
    p.add_argument("--workspace", type=int, default=1 << 30, help="TRT workspace bytes")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--int8", action="store_true", default=False, help="Build/use INT8 engine (needs --calib_dir for build)")
    p.add_argument("--calib_dir", type=str, default=None, help="INT8 calibration images directory")

    # Benchmark control
    p.add_argument("--warmup", type=int, default=30, help="Warmup frames")
    p.add_argument("--frames", type=int, default=300, help="Number of measured frames (after warmup)")
    p.add_argument("--max_frames", type=int, default=0, help="Hard stop after N frames (0=disabled). If set, overrides --warmup/--frames")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this container. Check --gpus/--runtime settings.")

    cam_hw = (int(args.cam_h), int(args.cam_w))
    input_hw = (int(args.in_h), int(args.in_w))

    labels = _comma_split(args.labels)
    if len(labels) < 2:
        raise ValueError("Provide at least 2 labels (comma-separated).")

    palette = get_new_palette(len(labels))

    backbone = args.backbone
    if backbone == "auto":
        backbone = infer_backbone_from_path(args.weights)

    # Load CLIP text features (once)
    device = torch.device("cuda")
    clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    with torch.no_grad():
        clip_labels = labels
        if args.prompt == "a_photo_of_a":
            clip_labels = [f"a photo of a {l}" for l in labels]

        # Print prompts once (helps debug "segmentation looks wrong")
        print(f"[CLIP] model={args.clip_model}  prompt={args.prompt}")
        if len(clip_labels) <= 30:
            for i, (raw, prm) in enumerate(zip(labels, clip_labels)):
                print(f"  - {i:02d}: '{raw}' -> '{prm}'")

        text_tokens = clip.tokenize(clip_labels).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

    # Backend init
    engine_path = None
    trt_encoder = None
    torch_encoder = None

    if args.backend == "trt":
        if args.engine:
            engine_path = args.engine
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"Engine not found: {engine_path}")
        else:
            if args.no_build:
                raise ValueError("backend=trt requires --engine unless auto-build is enabled")
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
        print(f"[TRT ] loaded engine: {engine_path}")
    else:
        torch_encoder = load_torch_encoder(args.weights, backbone)
        print("[TORCH] loaded ckpt encoder")

    # Camera
    cap = open_camera(args.device, args.cam_w, args.cam_h, args.cam_fps)
    print(f"[CAM ] opened: {args.device} ({args.cam_w}x{args.cam_h}@{args.cam_fps})")

    # Benchmark containers
    stats = Stats(frames=int(args.frames), warmup=int(args.warmup), encoder_ms=[], total_ms=[])

    # Timing helpers
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    # Run loop
    frame_idx = 0
    measured = 0
    warmup_left = int(args.warmup)
    max_measured = int(args.frames)

    if args.max_frames and int(args.max_frames) > 0:
        # Interpret as total frames including warmup.
        warmup_left = 0
        max_measured = int(args.max_frames)

    t_last = time.perf_counter()
    fps_smooth = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[WARN] camera read failed, exiting")
            break
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = frame_bgr[:, :, :3]

        t0 = time.perf_counter()

        # Preprocess (CPU)
        x_cpu = preprocess_frame_to_tensor(frame_bgr, input_hw)
        x = x_cpu.to(device, non_blocking=False)

        # Encoder inference (GPU)
        ev_start.record()
        if args.backend == "trt":
            assert trt_encoder is not None
            feat = trt_encoder(x)
        else:
            assert torch_encoder is not None
            feat = torch_encoder(x)
        ev_end.record()

        # Postprocess: logits upsample -> argmax (GPU) -> mask (CPU)
        # Only upsample to camera res when we actually display (keeps benchmark overhead lower).
        out_hw = (args.cam_h, args.cam_w) if not args.no_display else None
        mask = compute_mask_from_features(feat, text_features, out_hw=out_hw)
        mask_cpu = mask.to("cpu", non_blocking=False).numpy().astype(np.uint8)

        # GPU time
        torch.cuda.synchronize()
        enc_ms = float(ev_start.elapsed_time(ev_end))

        # Visualization (CPU)
        if not args.no_display:
            # mask_cpu is already camera resolution because we upsampled logits before argmax.
            color_mask = palette[mask_cpu]  # (H,W,3) BGR
            blended = cv2.addWeighted(frame_bgr, 1.0 - args.alpha, color_mask, args.alpha, 0)

            # Legend
            if not args.no_legend:
                draw_legend(blended, labels, palette)

            # FPS (smoothed)
            now = time.perf_counter()
            dt = max(1e-6, now - t_last)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth == 0.0 else (fps_smooth * 0.9 + inst_fps * 0.1)
            t_last = now

            text = f"{args.backend.upper()}  enc={enc_ms:.1f}ms  fps={fps_smooth:.1f}"
            cv2.putText(
                blended,
                text,
                (10, args.cam_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                blended,
                text,
                (10, args.cam_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window, blended)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # Warmup / measure
        frame_idx += 1
        if warmup_left > 0:
            warmup_left -= 1
            continue

        stats.add(enc_ms, total_ms)
        measured += 1

        if measured >= max_measured:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print_stats(
        backend=args.backend,
        weights=args.weights,
        engine=engine_path,
        input_hw=input_hw,
        cam_hw=(args.cam_h, args.cam_w),
        labels=labels,
        stats=stats,
    )


if __name__ == "__main__":
    main()
