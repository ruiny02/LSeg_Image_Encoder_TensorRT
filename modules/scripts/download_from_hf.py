from huggingface_hub import hf_hub_download

REPO_ID = "joonyeol99/LSeg_ViT-to-ONNX"

# Hugging Face에서 파일 다운로드
onnx_path = hf_hub_download(repo_id=REPO_ID, filename="output/models/lseg_image_encoder.onnx")
#trt_path = hf_hub_download(repo_id=REPO_ID, filename="../lseg_image_encoder.trt")
ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="modules/demo_e200.ckpt")

print(f"ONNX 모델 경로: {onnx_path}")
#print(f"TRT 모델 경로: {trt_path}")
print(f"Checkpoint 파일 경로: {ckpt_path}")
