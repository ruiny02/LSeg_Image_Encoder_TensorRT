import os
import argparse
import torch
import clip
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while lab > 0:
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i += 1
            lab >>= 3
    return pallete

def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(new_palette)
    patches = []
    if out_label_flag and labels is not None:
        u_index = np.unique(npimg)
        for index in u_index:
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0,
                         new_palette[index * 3 + 1] / 255.0,
                         new_palette[index * 3 + 2] / 255.0]
            patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(patch)
    return out_img, patches

def main(args):
    IMG_SIZE = int(args.size)  # ONNX 추출 시 사용한 이미지 사이즈
    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 입력 이미지 로드 및 전처리
    image = Image.open(args.image).convert("RGB")
    input_image = image_transform(image)  # (C, H, W)
    input_image_batch = input_image.unsqueeze(0)  # (1, C, H, W)

    # ONNX 이미지 인코더 로드 및 추론
    ort_session = onnxruntime.InferenceSession(args.onnx)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image_batch.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    # ONNX 이미지 인코더 출력: (1, C, H, W) – 타입을 float32로 맞춤
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_features = torch.from_numpy(ort_outs[0]).to(device).float()

    # CLIP 텍스트 인코더 로드 및 텍스트 인코딩
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    labels = [label.strip() for label in args.labels.split(",")]
    text_tokens = clip.tokenize(labels).to(device)  # (N, token_length)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)  # (N, C)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

    # 코사인 유사도 계산 (두 벡터가 정규화되었으므로 내적으로 계산)
    B, C, H, W = image_features.shape
    image_features_flat = image_features.view(C, -1)  # (C, H*W)
    similarity = torch.matmul(text_features, image_features_flat)  # (N, H*W)
    similarity = similarity.view(1, len(labels), H, W)

    # 각 픽셀에서 가장 높은 유사도를 가진 라벨 선택 → segmentation mask
    segmentation_mask = torch.argmax(similarity, dim=1).squeeze(0).cpu().numpy()  # (H, W)

    # 색상 팔레트 적용
    new_palette = get_new_pallete(len(labels))
    mask_img, patches = get_new_mask_pallete(segmentation_mask, new_palette, out_label_flag=True, labels=labels)

    # 결과 시각화: 원본 이미지와 세분화 결과를 나란히 출력
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask_img)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")
    if patches:
        axes[1].legend(handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1), prop={'size': 8})

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSeg ONNX + CLIP Demo using Pyplot")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--labels", type=str, default="dog, grass, other", help="Comma separated list of labels")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file")
    parser.add_argument("--size", type=str, default=384, help="Input image size to Model")
    args = parser.parse_args()
    main(args)
