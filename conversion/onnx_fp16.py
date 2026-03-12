#!/usr/bin/env python3
"""Convert an ONNX model's FP32 initializers/constants to FP16.

Why:
  Jetson-class devices can run out of memory while building TensorRT engines
  when the ONNX contains very large FP32 weights (e.g., ViT-L/16 ~1.2GB of consts).
  Converting weights to FP16 roughly halves that memory.

This script:
  - Converts graph.initializer tensors (FLOAT -> FLOAT16)
  - Converts Constant node tensor attributes (FLOAT -> FLOAT16)
  - Leaves graph inputs/outputs as-is (usually FLOAT) for compatibility

Usage:
  python3 conversion/onnx_fp16.py --input in.onnx --output out_fp16.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def _convert_tensorproto_fp32_to_fp16(t) -> bool:
    """In-place convert TensorProto FLOAT -> FLOAT16. Returns True if converted."""
    if t is None:
        return False
    if t.data_type != TensorProto.FLOAT:
        return False

    arr = numpy_helper.to_array(t)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    arr_fp16 = arr.astype(np.float16)
    new_t = numpy_helper.from_array(arr_fp16, name=t.name)
    t.CopyFrom(new_t)
    return True


def _convert_graph_fp16(g) -> Tuple[int, int]:
    init_cnt = 0
    const_cnt = 0

    for init in g.initializer:
        if _convert_tensorproto_fp32_to_fp16(init):
            init_cnt += 1

    for node in g.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                if _convert_tensorproto_fp32_to_fp16(attr.t):
                    const_cnt += 1
            elif attr.type == onnx.AttributeProto.GRAPH:
                a, b = _convert_graph_fp16(attr.g)
                init_cnt += a
                const_cnt += b
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    a, b = _convert_graph_fp16(sg)
                    init_cnt += a
                    const_cnt += b

    return init_cnt, const_cnt


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert ONNX FP32 weights/constants to FP16")
    ap.add_argument("--input", required=True, help="Input ONNX path")
    ap.add_argument("--output", required=True, help="Output ONNX path")
    ap.add_argument(
        "--external_data",
        action="store_true",
        help="Save weights as external data (out.onnx + out.onnx.data).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX] load : {in_path}")
    model = onnx.load(str(in_path))

    init_cnt, const_cnt = _convert_graph_fp16(model.graph)
    print(f"[ONNX] converted: initializers={init_cnt}, Constant-tensors={const_cnt}")

    try:
        onnx.checker.check_model(model)
        print("[ONNX] checker: OK")
    except Exception as e:
        print(f"[ONNX] checker: WARNING (still saving): {e}")

    if args.external_data:
        data_name = out_path.name + ".data"
        print(f"[ONNX] save (external data): {out_path}  +  {data_name}")
        onnx.save_model(
            model,
            str(out_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )
    else:
        print(f"[ONNX] save : {out_path}")
        onnx.save(model, str(out_path))


if __name__ == "__main__":
    main()
