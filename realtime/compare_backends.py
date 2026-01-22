#!/usr/bin/env python3

"""Run LSeg real-time benchmark for multiple backends (torch vs trt) sequentially.

This is a thin wrapper around realtime/lseg_realtime.py.

Typical usage (x86, /dev/video4):
  python3 realtime/compare_backends.py \
    --device /dev/video4 \
    --weights models/weights/ViT/demo_e200.ckpt \
    --labels "person, chair, desk, monitor, keyboard, background" \
    --no_display

It will execute:
  1) backend=torch
  2) backend=trt (and auto-build engine if missing)

and print both benchmark logs to console.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare torch vs TensorRT backends (sequential runs)")

    p.add_argument("--device", type=str, default="/dev/video0")
    p.add_argument("--cam_w", type=int, default=1280)
    p.add_argument("--cam_h", type=int, default=720)
    p.add_argument("--cam_fps", type=int, default=30)
    p.add_argument("--in_w", type=int, default=512)
    p.add_argument("--in_h", type=int, default=288)

    p.add_argument("--labels", type=str, default="person, chair, desk, monitor, keyboard, background")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--no_legend", action="store_true")

    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--backbone", choices=["vit", "rn101", "auto"], default="auto")

    p.add_argument("--workspace", type=int, default=1 << 30)
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--int8", action="store_true", default=False)
    p.add_argument("--calib_dir", type=str, default=None)

    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--no_display", action="store_true")

    p.add_argument(
        "--backends",
        type=str,
        default="torch,trt",
        help="Comma separated backends to run sequentially. Default: torch,trt",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "realtime" / "lseg_realtime.py"

    backends = _split_csv(args.backends)
    if not backends:
        raise ValueError("No backends specified")

    base_cmd = [
        "python3",
        str(script),
        "--device",
        args.device,
        "--cam_w",
        str(args.cam_w),
        "--cam_h",
        str(args.cam_h),
        "--cam_fps",
        str(args.cam_fps),
        "--in_w",
        str(args.in_w),
        "--in_h",
        str(args.in_h),
        "--labels",
        args.labels,
        "--alpha",
        str(args.alpha),
        "--weights",
        args.weights,
        "--backbone",
        args.backbone,
        "--workspace",
        str(args.workspace),
        "--warmup",
        str(args.warmup),
        "--frames",
        str(args.frames),
    ]
    if args.no_legend:
        base_cmd.append("--no_legend")
    if args.no_display:
        base_cmd.append("--no_display")

    # TRT build flags will be ignored by torch backend, but keep them consistent.
    if args.fp16:
        base_cmd.append("--fp16")
    else:
        base_cmd.append("--no_fp16")
    if args.int8:
        base_cmd.append("--int8")
        if args.calib_dir:
            base_cmd += ["--calib_dir", args.calib_dir]

    for b in backends:
        print("\n" + "=" * 80)
        print(f"[RUN] backend={b}")
        print("[CMD] " + shlex.join(base_cmd + ["--backend", b]))
        print("=" * 80 + "\n")
        subprocess.run(base_cmd + ["--backend", b], check=True)


if __name__ == "__main__":
    main()
