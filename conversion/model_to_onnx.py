import argparse
import torch
import torch.nn as nn
import torch.onnx
import os, sys
# ── 스크립트 위치의 상위 폴더(=프로젝트 루트)를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.lseg_module import LSegModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/weights/ViT/demo_e200.ckpt", help="Path to checkpoint")
    args = parser.parse_args()

    # ✅ 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # ✅ 체크포인트 경로 (데이터셋 관련 매개변수는 필요 없으므로 제거 가능)
    checkpoint_path = args.weights

    # ✅ 체크포인트 경로에서 태그 추출 (예: demo_e200.ckpt → ade20k, fss_l16.ckpt → fss)
    # 체크포인트 파일명 그대로 태그로 사용
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]

    model = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
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
        map_location="cpu",
        #map_location=device,
        arch_option=0,
        block_depth=0,
        activation="lrelu"
    ).net
    # ).net.to(device)

    model.eval()

    #dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    dummy_input = torch.randn(1, 3, 480, 480)
    onnx_filename = f"models/onnx_engines/lseg_img_enc_vit_{tag}.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        # 동적 배치 + 동적 공간축
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    )
print(f"✅ Dynamic ONNX 저장: {onnx_filename}")
