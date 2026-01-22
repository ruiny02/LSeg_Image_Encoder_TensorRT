import argparse
import torch
import torch.onnx
import os, sys

# ── 스크립트 위치의 상위 폴더(=프로젝트 루트)를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.lseg_module import LSegModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/weights/ViT/demo_e200.ckpt",
                        help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=320,
                        help="Export dummy input size (HxW). Smaller size reduces peak memory on Jetson.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for export. Use cpu to avoid GPU OOM on Jetson.")
    parser.add_argument("--use_pretrained", dest="use_pretrained", action="store_true",
                        help="Load CLIP/timm pretrained weights. Set false on Jetson to save RAM.")
    parser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false",
                        help="Skip pretrained backbone weights (Jetson-friendly).")
    parser.set_defaults(use_pretrained=None)
    parser.add_argument("--half", action="store_true", default=False,
                        help="Export in FP16 (CUDA only) to cut memory use.")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--no-constant-folding", dest="const_fold", action="store_false",
                        help="Disable constant folding to lower peak RAM.")
    parser.add_argument("--external-data", action="store_true", default=False,
                        help="Store large initializers in external .onnx_data to avoid >2GB graphs.")
    parser.set_defaults(const_fold=True)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_path = args.weights
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]

    # Respect env override USE_PRETRAINED=0|1
    env_pretrained = os.getenv("USE_PRETRAINED")
    use_pretrained = args.use_pretrained
    if use_pretrained is None:
        # default False to lower Jetson RAM unless env explicitly requests it
        use_pretrained = False
        if env_pretrained is not None:
            use_pretrained = env_pretrained not in ("0", "false", "False")
    else:
        if env_pretrained is not None:
            use_pretrained = env_pretrained not in ("0", "false", "False")

    # Propagate to encoder construction
    os.environ["USE_PRETRAINED"] = "1" if use_pretrained else "0"

    print(f"[ONNX] device={device}, img_size={args.img_size}, pretrained={use_pretrained}, "
          f"half={args.half}, external_data={args.external_data}, const_fold={args.const_fold}, opset={args.opset}")

    model = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone="clip_vitl16_384",
        aux=False,
        num_features=256,
        crop_size=args.img_size,
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
        map_location="cpu",
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    ).net.to(device)

    # Half precision export is useful on Jetson if GPU is used
    if args.half and device.type == "cuda":
        model = model.half()

    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    if args.half and device.type == "cuda":
        dummy_input = dummy_input.half()

    onnx_filename = f"models/onnx_engines/lseg_img_enc_vit_{tag}.onnx"

    os.makedirs(os.path.dirname(onnx_filename), exist_ok=True)

    # inference_mode cuts activation saves → much lower RAM/VRAM during export
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_filename,
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            dynamic_axes={
                "input":  {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=args.const_fold,
            external_data=args.external_data,
        )
    print(f"✅ Dynamic ONNX 저장: {onnx_filename} (img_size={args.img_size}, device={device})")


if __name__ == "__main__":
    main()
