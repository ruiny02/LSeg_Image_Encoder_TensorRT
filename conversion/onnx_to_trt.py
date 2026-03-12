"""conversion/onnx_to_trt.py

ONNX -> TensorRT engine builder for LSeg image-encoder.

Key goals for this repo usage:
  * Real-time demo: fixed input (N,C,H,W) = (1,3,288,512)
  * Optional: dynamic profile for offline benchmark (e.g., 256..1024)
  * Default precision: FP16
  * Optional INT8 (with entropy calibration)

Notes
  * TensorRT engines are GPU / driver / TensorRT-version specific.
    Build the engine on the target machine (especially on Jetson).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _hw_pair(v: Sequence[int]) -> Tuple[int, int]:
    if len(v) != 2:
        raise ValueError(f"Expected 2 integers (H W), got: {v}")
    h, w = int(v[0]), int(v[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid H/W: {h}x{w}")
    return h, w


def _sorted_image_files(calib_dir: str) -> List[str]:
    p = Path(calib_dir)
    if not p.exists() or not p.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [str(x) for x in p.rglob("*") if x.suffix.lower() in exts]
    files.sort()
    return files


@dataclass(frozen=True)
class ProfileHW:
    min_hw: Tuple[int, int]
    opt_hw: Tuple[int, int]
    max_hw: Tuple[int, int]


class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Simple image-folder based INT8 calibrator (EntropyCalibrator2).

    Preprocess matches LSeg encoder input:
      * BGR image read by OpenCV -> RGB
      * resize to (W,H)
      * normalize: (x/255 - 0.5) / 0.5
      * NCHW float32
"""

    def __init__(
        self,
        image_paths: List[str],
        batch_size: int,
        input_hw: Tuple[int, int],
        cache_file: str,
        max_batches: int,
    ):
        super().__init__()

        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        self.image_paths = image_paths
        self.batch_size = batch_size
        self.input_h, self.input_w = input_hw
        self.cache_file = cache_file
        self.max_batches = max_batches

        # Lazy imports so FP16-only builds don't require OpenCV/pycuda at import time.
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        import pycuda.driver as cuda

        self._cv2 = cv2
        self._np = np
        self._cuda = cuda

        self._batch_index = 0
        self._num_batches = min(
            max_batches,
            (len(self.image_paths) + self.batch_size - 1) // self.batch_size,
        )
        if self._num_batches <= 0:
            raise ValueError(
                "No calibration batches available. Provide calib images (jpg/png) via --calib_dir."
            )

        # Allocate one device buffer for the whole batch.
        nbytes = self.batch_size * 3 * self.input_h * self.input_w * 4  # float32
        self.device_input = self._cuda.mem_alloc(nbytes)

    def get_batch_size(self) -> int:  # type: ignore[override]
        return self.batch_size

    def _preprocess_one(self, path: str) -> "self._np.ndarray":
        img_bgr = self._cv2.imread(path, self._cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read calib image: {path}")
        img_rgb = self._cv2.cvtColor(img_bgr, self._cv2.COLOR_BGR2RGB)
        img_rgb = self._cv2.resize(
            img_rgb,
            (self.input_w, self.input_h),
            interpolation=self._cv2.INTER_AREA,
        )
        x = img_rgb.astype(self._np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = self._np.transpose(x, (2, 0, 1))  # CHW
        return x

    def get_batch(self, names: Sequence[str]) -> Optional[List[int]]:  # type: ignore[override]
        if self._batch_index >= self._num_batches:
            return None

        start = self._batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[start:end]

        # If last batch is smaller, pad by repeating the last sample.
        if len(batch_paths) < self.batch_size:
            batch_paths = batch_paths + [batch_paths[-1]] * (self.batch_size - len(batch_paths))

        batch = self._np.stack([self._preprocess_one(p) for p in batch_paths], axis=0)  # NCHW
        batch = self._np.ascontiguousarray(batch, dtype=self._np.float32)
        self._cuda.memcpy_htod(self.device_input, batch)
        self._batch_index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:  # type: ignore[override]
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:  # type: ignore[override]
        _ensure_parent_dir(self.cache_file)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(
    onnx_path: str,
    engine_path: str,
    profile_hw: ProfileHW,
    workspace_size: int,
    use_fp16: bool,
    use_int8: bool,
    disable_timing_cache: bool,
    gpu_fallback: bool,
    debug_mode: bool,
    use_sparse: bool,
    use_cublas: bool,
    use_cudnn: bool,
    int8_calib_dir: Optional[str] = None,
    int8_calib_cache: Optional[str] = None,
    int8_calib_batch_size: int = 8,
    int8_calib_max_batches: int = 20,
) -> None:
    """Build a TensorRT engine and write it to engine_path."""

    onnx_path = str(onnx_path)
    engine_path = str(engine_path)
    _ensure_parent_dir(engine_path)

    min_h, min_w = profile_hw.min_hw
    opt_h, opt_w = profile_hw.opt_hw
    max_h, max_w = profile_hw.max_hw

    # Basic sanity checks.
    if not (min_h <= opt_h <= max_h and min_w <= opt_w <= max_w):
        raise ValueError(
            f"Invalid profile HW ordering: min={min_h}x{min_w}, opt={opt_h}x{opt_w}, max={max_h}x{max_w}"
        )

    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser, \
        builder.create_builder_config() as config:

        # Workspace
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size))

        # Precision flags
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.clear_flag(trt.BuilderFlag.FP16)

        if use_sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        else:
            config.clear_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        if disable_timing_cache:
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        else:
            config.clear_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

        if gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        if debug_mode:
            config.set_flag(trt.BuilderFlag.DEBUG)

        # INT8 (optional)
        if use_int8:
            if not builder.platform_has_fast_int8:
                raise RuntimeError("This platform reports no fast INT8 support.")
            config.set_flag(trt.BuilderFlag.INT8)

            if not int8_calib_cache:
                int8_calib_cache = "models/trt_engines/int8_calibration.cache"

            if not int8_calib_dir:
                raise RuntimeError(
                    "INT8 build requested but --calib_dir not provided. "
                    "Provide a folder with jpg/png images." 
                )
            calib_images = _sorted_image_files(int8_calib_dir)
            if not calib_images:
                raise RuntimeError(f"No calibration images found under: {int8_calib_dir}")

            # Use OPT resolution for calibration.
            calibrator = Int8EntropyCalibrator(
                image_paths=calib_images,
                batch_size=int8_calib_batch_size,
                input_hw=(opt_h, opt_w),
                cache_file=int8_calib_cache,
                max_batches=int8_calib_max_batches,
            )
            config.int8_calibrator = calibrator

        # Tactic sources
        # NOTE: Calling set_tactic_sources() restricts TensorRT to ONLY the specified sources.
        # Transformer blocks often need CUBLAS_LT; without it you can hit
        #   'Could not find any implementation for node ...'
        tactic_mask = 0
        if use_cublas:
            tactic_mask |= 1 << int(trt.TacticSource.CUBLAS)
            if hasattr(trt.TacticSource, 'CUBLAS_LT'):
                tactic_mask |= 1 << int(trt.TacticSource.CUBLAS_LT)
        if use_cudnn:
            tactic_mask |= 1 << int(trt.TacticSource.CUDNN)
        if tactic_mask:
            config.set_tactic_sources(tactic_mask)

        # Parse ONNX
        print(f"🔍 parsing ONNX: {onnx_path}")
        if not parser.parse_from_file(onnx_path.encode()):
            print("❌ ONNX parse-error(s):")
            for i in range(parser.num_errors):
                print(f"   ▶ {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()

        # batch fixed to 1 for this project
        profile.set_shape(
            input_tensor.name,
            (1, 3, min_h, min_w),
            (1, 3, opt_h, opt_w),
            (1, 3, max_h, max_w),
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                "❌ buildSerializedNetwork returned None (profile/shape constraint violation)"
            )

        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"✅ TensorRT engine saved: {engine_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")

    parser.add_argument("--onnx", required=True, help="Path to input ONNX file")
    parser.add_argument(
        "--engine",
        default=None,
        help="Optional explicit output engine path. If omitted, auto-named under models/trt_engines/",
    )

    # Profile (H W)
    parser.add_argument(
        "--min_hw",
        nargs=2,
        type=int,
        default=[288, 512],
        metavar=("H", "W"),
        help="Min input resolution (H W). Default: 288 512",
    )
    parser.add_argument(
        "--opt_hw",
        nargs=2,
        type=int,
        default=[288, 512],
        metavar=("H", "W"),
        help="Opt input resolution (H W). Default: 288 512",
    )
    parser.add_argument(
        "--max_hw",
        nargs=2,
        type=int,
        default=[288, 512],
        metavar=("H", "W"),
        help="Max input resolution (H W). Default: 288 512",
    )

    # Resources
    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 29,
        help="Workspace size in bytes (default: 1<<29)",
    )

    # Precision
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Enable FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16")
    parser.add_argument(
        "--int8",
        action="store_true",
        default=False,
        help="Enable INT8 (requires --calib_dir or existing calib cache)",
    )

    # INT8 calibration
    parser.add_argument(
        "--calib_dir",
        default=None,
        help="Folder with jpg/png images for INT8 calibration",
    )
    parser.add_argument(
        "--calib_cache",
        default=None,
        help="Calibration cache file path (default: models/trt_engines/int8_calibration.cache)",
    )
    parser.add_argument(
        "--calib_batch",
        type=int,
        default=8,
        help="INT8 calibration batch size (default: 8)",
    )
    parser.add_argument(
        "--calib_max_batches",
        type=int,
        default=20,
        help="Max number of calibration batches (default: 20)",
    )

    # Other builder flags
    parser.add_argument("--sparse", dest="sparse", action="store_true", default=True, help="Enable sparse weights")
    parser.add_argument("--no-sparse", dest="sparse", action="store_false", help="Disable sparse weights")
    parser.add_argument("--disable-timing-cache", action="store_true", default=False, help="Disable timing cache")
    parser.add_argument("--gpu-fallback", action="store_true", default=False, help="GPU fallback in INT8")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug builder flag")

    # Tactic sources
    parser.add_argument("--use-cublas", dest="use_cublas", action="store_true", default=True, help="Enable cuBLAS tactics")
    parser.add_argument("--no-cublas", dest="use_cublas", action="store_false", help="Disable cuBLAS tactics")
    parser.add_argument("--use-cudnn", dest="use_cudnn", action="store_true", default=True, help="Enable cuDNN tactics")
    parser.add_argument("--no-cudnn", dest="use_cudnn", action="store_false", help="Disable cuDNN tactics")

    args = parser.parse_args()

    # Resolve output path
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    trt_dir = project_dir / "models" / "trt_engines"
    trt_dir.mkdir(parents=True, exist_ok=True)

    base = Path(args.onnx).stem

    min_hw = _hw_pair(args.min_hw)
    opt_hw = _hw_pair(args.opt_hw)
    max_hw = _hw_pair(args.max_hw)
    profile_hw = ProfileHW(min_hw=min_hw, opt_hw=opt_hw, max_hw=max_hw)

    # Engine name tag
    prec_tag = "int8" if args.int8 else ("fp16" if args.fp16 else "fp32")

    flags = [
        prec_tag,
        "fp16" if (args.int8 and args.fp16) else None,  # allow mixed precision tag
        "sparse" if args.sparse else None,
        "noTC" if args.disable_timing_cache else None,
        "gpuFB" if args.gpu_fallback else None,
        "dbg" if args.debug else None,
        "cublas" if args.use_cublas else None,
        "cudnn" if args.use_cudnn else None,
        f"min{min_hw[0]}x{min_hw[1]}",
        f"opt{opt_hw[0]}x{opt_hw[1]}",
        f"max{max_hw[0]}x{max_hw[1]}",
        f"ws{int(args.workspace) >> 20}MiB",
    ]
    flags = [f for f in flags if f]
    suffix = "_".join(flags)

    engine_path = args.engine or str(trt_dir / f"{base}__{suffix}.trt")

    build_engine(
        onnx_path=args.onnx,
        engine_path=engine_path,
        profile_hw=profile_hw,
        workspace_size=int(args.workspace),
        use_fp16=bool(args.fp16),
        use_int8=bool(args.int8),
        disable_timing_cache=bool(args.disable_timing_cache),
        gpu_fallback=bool(args.gpu_fallback),
        debug_mode=bool(args.debug),
        use_sparse=bool(args.sparse),
        use_cublas=bool(args.use_cublas),
        use_cudnn=bool(args.use_cudnn),
        int8_calib_dir=args.calib_dir,
        int8_calib_cache=args.calib_cache,
        int8_calib_batch_size=int(args.calib_batch),
        int8_calib_max_batches=int(args.calib_max_batches),
    )

    print(f"\n✅ Engine saved as: {engine_path}")


if __name__ == "__main__":
    main()
