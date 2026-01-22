import argparse
import os
import sys
import torch
import torch.onnx

# 프로젝트 루트 경로를 import 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.lseg_module_zs import LSegModuleZS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='models/weights/Resnet/fss_rn101.ckpt',
        help='Path to checkpoint'
    )
    parser.add_argument("--img_size", type=int, default=480,
                        help="Dummy input size (HxW). Reduce on Jetson to save memory.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for export.")
    parser.add_argument("--use_pretrained", dest="use_pretrained", action="store_true",
                        help="Load pretrained backbone weights.")
    parser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false",
                        help="Skip pretrained backbone weights (Jetson-friendly).")
    parser.set_defaults(use_pretrained=None)
    parser.add_argument("--half", action="store_true", default=False,
                        help="Export in FP16 (CUDA only) to reduce memory.")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("--no-constant-folding", dest="const_fold", action="store_false",
                        help="Disable constant folding to lower peak RAM.")
    parser.add_argument("--external-data", action="store_true", default=False,
                        help="Store large initializers in external .onnx_data.")
    parser.set_defaults(const_fold=True)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_path = args.weights
    # 체크포인트 파일명 그대로 태그로 사용
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]

    env_pretrained = os.getenv("USE_PRETRAINED")
    use_pretrained = args.use_pretrained
    if use_pretrained is None:
        use_pretrained = False
        if env_pretrained is not None:
            use_pretrained = env_pretrained not in ("0", "false", "False")
    else:
        if env_pretrained is not None:
            use_pretrained = env_pretrained not in ("0", "false", "False")
    os.environ["USE_PRETRAINED"] = "1" if use_pretrained else "0"

    print(f"[ONNX-ZS] device={device}, img_size={args.img_size}, pretrained={use_pretrained}, "
          f"half={args.half}, external_data={args.external_data}, const_fold={args.const_fold}, opset={args.opset}")

    # 기본값 그대로 사용
    module = LSegModuleZS.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_path='data/',
        dataset='ade20k',
        backbone='clip_resnet101',
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
        map_location='cpu',
        arch_option=0,
        use_pretrained='True',
        strict=False,
        logpath='fewshot/logpath_4T/',
        fold=0,
        block_depth=0,
        nshot=1,
        finetune_mode=False,
        activation='lrelu',
    )

    model = module.net.to(device).eval()
    if args.half and device.type == "cuda":
        model = model.half()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    if args.half and device.type == "cuda":
        dummy_input = dummy_input.half()

    # non-zs와 동일한 onnx 파일명 및 경로
    onnx_filename = f"models/onnx_engines/lseg_img_enc_rn101_{tag}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=['input'],
        output_names=['output'],
        opset_version=args.opset,
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
        do_constant_folding=args.const_fold,
        external_data=args.external_data,
    )

    print(f"✅ Dynamic ONNX 저장: {onnx_filename}")
