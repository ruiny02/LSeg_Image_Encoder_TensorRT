#!/usr/bin/env python3

"""Capture calibration images from a USB webcam using GStreamer (OpenCV).

Used for TensorRT INT8 entropy calibration.

Example (x86):
  python3 tools/capture_calib_images.py \
    --device /dev/video4 \
    --out_dir calib_images \
    --count 300

Example (Jetson):
  python3 tools/capture_calib_images.py \
    --device /dev/video0 \
    --out_dir calib_images \
    --count 300
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2


def build_gstreamer_pipeline(device: str, width: int, height: int, fps: int, prefer_mjpeg: bool = True) -> str:
    if prefer_mjpeg:
        return (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1 sync=false"
        )
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=false"
    )


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    p1 = build_gstreamer_pipeline(device, width, height, fps, True)
    cap = cv2.VideoCapture(p1, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    p2 = build_gstreamer_pipeline(device, width, height, fps, False)
    cap = cv2.VideoCapture(p2, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    raise RuntimeError(f"Failed to open camera. Tried:\n{p1}\n\n{p2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture calibration images for TensorRT INT8")

    p.add_argument("--device", type=str, default="/dev/video0")
    p.add_argument("--cam_w", type=int, default=1280)
    p.add_argument("--cam_h", type=int, default=720)
    p.add_argument("--cam_fps", type=int, default=30)

    p.add_argument("--out_dir", type=str, default="calib_images")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--stride", type=int, default=1, help="Save every Nth frame (default: 1)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.device, args.cam_w, args.cam_h, args.cam_fps)
    print(f"[CAM] opened {args.device} ({args.cam_w}x{args.cam_h}@{args.cam_fps})")
    print(f"[OUT] {out_dir.resolve()}")

    saved = 0
    idx = 0
    t0 = time.perf_counter()

    while saved < int(args.count):
        ok, frame = cap.read()
        if not ok:
            print("[WARN] read failed")
            break
        idx += 1
        if args.stride > 1 and (idx % int(args.stride) != 0):
            continue

        # jpg is fine for calibration data
        out_path = out_dir / f"frame_{saved:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

        if saved % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"saved {saved}/{args.count}  ({saved/dt:.1f} img/s)")

    cap.release()
    print(f"Done. Saved: {saved} images")


if __name__ == "__main__":
    main()
