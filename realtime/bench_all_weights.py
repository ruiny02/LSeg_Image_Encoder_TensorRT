#!/usr/bin/env python3

"""Benchmark all checkpoints under models/weights for both backends.

This script searches recursively for '*.ckpt' under --weights_dir (default: models/weights)
and runs realtime/compare_backends.py for each checkpoint.

Recommended to run with --no_display for faster and less noisy measurements.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark all weights (torch vs trt)")

    p.add_argument("--weights_dir", type=str, default="models/weights")

    p.add_argument("--device", type=str, default="/dev/video0")
    p.add_argument("--cam_w", type=int, default=1280)
    p.add_argument("--cam_h", type=int, default=720)
    p.add_argument("--cam_fps", type=int, default=30)
    p.add_argument("--in_w", type=int, default=512)
    p.add_argument("--in_h", type=int, default=288)

    p.add_argument("--labels", type=str, default="person, chair, desk, monitor, keyboard, background")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--no_legend", action="store_true")

    p.add_argument("--workspace", type=int, default=1 << 30)
    p.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--int8", action="store_true", default=False)
    p.add_argument("--calib_dir", type=str, default=None)

    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--no_display", action="store_true")

    p.add_argument("--backends", type=str, default="torch,trt")

    return p.parse_args()


def find_ckpts(weights_dir: str) -> List[str]:
    p = Path(weights_dir)
    ckpts = [str(x) for x in p.rglob("*.ckpt")]
    ckpts.sort()
    return ckpts


def main() -> None:
    args = parse_args()

    ckpts = find_ckpts(args.weights_dir)
    if not ckpts:
        raise RuntimeError(f"No .ckpt files found under: {args.weights_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "realtime" / "compare_backends.py"

    for i, ckpt in enumerate(ckpts, start=1):
        print("\n" + "#" * 100)
        print(f"[CKPT] {i}/{len(ckpts)}  {ckpt}")
        print("#" * 100 + "\n")

        cmd = [
            "python3",
            str(runner),
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
            ckpt,
            "--workspace",
            str(args.workspace),
            "--warmup",
            str(args.warmup),
            "--frames",
            str(args.frames),
            "--backends",
            args.backends,
        ]
        if args.no_legend:
            cmd.append("--no_legend")
        if args.no_display:
            cmd.append("--no_display")

        if args.fp16:
            cmd.append("--fp16")
        else:
            cmd.append("--no_fp16")

        if args.int8:
            cmd.append("--int8")
            if args.calib_dir:
                cmd += ["--calib_dir", args.calib_dir]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
