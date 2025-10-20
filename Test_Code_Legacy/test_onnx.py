import onnxruntime as ort
import numpy as np

# ONNX Runtime 세션 생성
session = ort.InferenceSession("lseg_img_enc_vit_ade20k.onnx")

# 더미 입력 생성 (예: 배치 크기 1, 3채널, 384x384 이미지)
dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)

# 추론 실행 (입력 이름은 모델에 정의된 이름을 사용해야 합니다)
outputs = session.run(None, {"input": dummy_input})
print("출력 텐서의 shape:", outputs[0].shape)
