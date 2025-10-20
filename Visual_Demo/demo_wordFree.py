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
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer

# tokenizer를 통해 전체 vocab 불러오기
tokenizer = SimpleTokenizer()
vocab = tokenizer.encoder  # {서브워드: 토큰 id}
decoder = {v: k for k, v in vocab.items()}
vocab_size = len(decoder)
candidate_words = [tokenizer.decode([i]).strip() for i in range(vocab_size)]

def get_new_pallete(num_cls):
    # num_cls: 실제 클래스 개수 (예: 등장한 토큰 수)
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
    # PIL의 putpalette는 길이가 768인 리스트(256*3)를 필요로 하므로, 패딩 수행
    if len(new_palette) < 768:
        new_palette = new_palette + [0]*(768 - len(new_palette))
    elif len(new_palette) > 768:
        new_palette = new_palette[:768]
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(new_palette)
    patches = []
    if out_label_flag and labels is not None:
        u_index = np.unique(npimg)
        for index in u_index:
            if index in labels:
                label = labels[index]
                cur_color = [new_palette[index * 3] / 255.0,
                             new_palette[index * 3 + 1] / 255.0,
                             new_palette[index * 3 + 2] / 255.0]
                patch = mpatches.Patch(color=cur_color, label=label)
                patches.append(patch)
    return out_img, patches

def main(args):
    IMG_SIZE = int(args.size)  # ONNX 모델 입력 이미지 크기
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ONNX 출력: (1, C, H, W)
    image_features = torch.from_numpy(ort_outs[0]).to(device).float()

    # CLIP 텍스트 인코더 로드 및 초기화
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # 이미지 feature map flatten: (C, H*W)
    B, C, H, W = image_features.shape
    image_features_flat = image_features.view(C, -1)

    # 픽셀별 최고 유사도를 저장할 변수 초기화
    best_similarity = torch.full((H * W,), -float('inf'), device=device)
    best_indices = torch.zeros((H * W,), dtype=torch.long, device=device)

    batch_size = 512  # 토큰 배치 사이즈 (메모리 상황에 맞게 조절)

    # 전체 vocab을 배치 단위로 순회하며 각 토큰의 텍스트 임베딩과 이미지 feature map 간 코사인 유사도 계산
    for i in tqdm(range(0, vocab_size, batch_size), desc="Processing vocab batches"):
        batch_tokens = candidate_words[i : i + batch_size]
        text_inputs = clip.tokenize(batch_tokens).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.float()
        similarity_batch = torch.matmul(text_features, image_features_flat)
        batch_max_similarity, batch_max_indices = similarity_batch.max(dim=0)
        update_mask = batch_max_similarity > best_similarity
        best_similarity[update_mask] = batch_max_similarity[update_mask]
        best_indices[update_mask] = i + batch_max_indices[update_mask]
    
    # segmentation_mask: (H, W) – 원래 인덱스 값(전체 vocab 상의 index)
    segmentation_mask = best_indices.view(H, W).cpu().numpy()

    # 시각화를 위해 실제 등장한 클래스만 remapping (0 ~ N-1)
    unique_indices = np.unique(segmentation_mask)
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
    remapped_mask = np.vectorize(mapping.get)(segmentation_mask)
    # remapped_mask에 해당하는 레이블 (토큰 문자열)
    labels = {mapping[idx]: candidate_words[idx] for idx in unique_indices}

    # 찾은 단어들을 정렬하여 출력 (예: 인덱스 순서대로)
    found_words = [labels[i] for i in sorted(labels.keys())]
    print("Found words:", found_words)

    # 등장한 클래스 수에 맞게 팔레트 생성
    new_palette = get_new_pallete(len(unique_indices))
    mask_img, patches = get_new_mask_pallete(remapped_mask, new_palette, out_label_flag=True, labels=labels)

    # 결과 시각화: 원본 이미지와 픽셀 단위 분류 결과를 나란히 출력
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask_img)
    axes[1].set_title("Pixel-wise Classification (Full CLIP Vocab)")
    axes[1].axis("off")
    if patches:
        axes[1].legend(handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1), prop={'size': 8})

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSeg ONNX + CLIP Demo: Full CLIP Vocab Pixel-wise Classification")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 모델 파일 경로")
    parser.add_argument("--size", type=str, default=384, help="모델 입력 이미지 크기")
    args = parser.parse_args()
    main(args)
