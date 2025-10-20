from huggingface_hub import HfApi

REPO_ID = "joonyeol99/LSeg_ViT-to-ONNX"
api = HfApi()
# api.upload_file(
#     path_or_fileobj="models/lseg_image_encoder.onnx",
#     path_in_repo="lseg_image_encoder.onnx",
#     repo_id=REPO_ID
# )

# api.upload_file(
#     path_or_fileobj="lseg_image_encoder.trt",
#     path_in_repo="lseg_image_encoder.trt",
#     repo_id=REPO_ID
# )

# api.upload_file(
#     path_or_fileobj="models/demo_e200.ckpt",
#     path_in_repo="demo_e200.ckpt",
#     repo_id=REPO_ID
# )


api.upload_file(
    path_or_fileobj="models/lseg_image_encoder_128.onnx",
    path_in_repo="lseg_image_encoder_128.onnx",
    repo_id=REPO_ID
)

api.upload_file(
    path_or_fileobj="models/lseg_image_encoder_128.trt",
    path_in_repo="lseg_image_encoder_128.trt",
    repo_id=REPO_ID
)

api.upload_file(
    path_or_fileobj="models/lseg_image_encoder_320.onnx",
    path_in_repo="lseg_image_encoder_320.onnx",
    repo_id=REPO_ID
)