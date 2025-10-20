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
    args = parser.parse_args()

    checkpoint_path = args.weights
    # 체크포인트 파일명 그대로 태그로 사용
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]

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

    model = module.net.eval()
    dummy_input = torch.randn(1, 3, 480, 480)

    # non-zs와 동일한 onnx 파일명 및 경로
    onnx_filename = f"models/onnx_engines/lseg_img_enc_rn101_{tag}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        }
    )

    print(f"✅ Dynamic ONNX 저장: {onnx_filename}")
