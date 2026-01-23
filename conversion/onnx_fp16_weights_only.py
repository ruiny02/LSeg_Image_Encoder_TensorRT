#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = onnx.load(str(in_path))

    cnt = 0
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            cnt += 1

    print(f"[ONNX] converted initializers(FP32->FP16): {cnt}")
    onnx.save(m, str(out_path))
    print(f"[ONNX] saved: {out_path}")

if __name__ == "__main__":
    main()
