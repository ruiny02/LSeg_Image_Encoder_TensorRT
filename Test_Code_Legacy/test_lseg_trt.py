import os, math, argparse
from math import ceil
import numpy as np
import torch, torch.nn.functional as F
import clip, tensorrt as trt
import pycuda.autoinit, pycuda.driver as cuda
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from modules.lseg_full import LSegFull      # 분류·후처리 모듈
from typing import Dict, List

#from modules.lseg_module import LSegModule
# from data.base import BaseDataset, test_batchify_fn
from data.ade20k import ADE20KSegmentation

txt_feat = None

# ——————————————————————————————
# 1) TRT 래퍼 & 어댑터
# ——————————————————————————————
class TRTWrapper:
    def __init__(self, engine_path, S=None):
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        buf = open(engine_path, 'rb').read()
        self.engine = runtime.deserialize_cuda_engine(buf)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.in_name, self.out_name = (
            self.engine.get_tensor_name(0),
            self.engine.get_tensor_name(1)
        )
        # 동적 입력을 지원하도록 버퍼 할당은 infer 시점으로 연기
        self.S           = S
        self.d_in        = None
        self.d_out       = None
        # 현재 할당된 버퍼 크기(바이트 단위)를 추적
        self.in_capacity  = 0
        self.out_capacity = 0

    def infer(self, x: torch.Tensor) -> np.ndarray:
        """
        x: torch.Tensor CPU (1,3,H,W), dtype=float32
        반환: numpy.ndarray (1,512,H/2,W/2), dtype=float32
        """
        arr = x.cpu().numpy().astype(np.float32)
        self.context.set_input_shape(self.in_name, tuple(arr.shape))
        # 필요시 버퍼 할당 (처음 또는 크기가 바뀐 경우)
        b, c, h, w = arr.shape
        in_bytes  = b * c * h * w * 4
        out_bytes = b * 512 * (h//2) * (w//2) * 4
        # 새로 할당이 필요하면 (capacity 비교)
        if self.d_in is None or in_bytes > self.in_capacity or out_bytes > self.out_capacity:
            # 기존 버퍼가 있으면 해제
            if self.d_in is not None:
                self.d_in.free()
                self.d_out.free()
            # 새로 할당
            self.d_in        = cuda.mem_alloc(in_bytes)
            self.d_out       = cuda.mem_alloc(out_bytes)
            self.in_capacity  = in_bytes
            self.out_capacity = out_bytes

        self.context.set_tensor_address(self.in_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))
        cuda.memcpy_htod_async(self.d_in, arr, self.stream)

        self.context.execute_async_v3(self.stream.handle)

        out_h, out_w = arr.shape[2]//2, arr.shape[3]//2
        out = np.empty((arr.shape[0],512,out_h,out_w), dtype=np.float32)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out

class TRTAdapter(torch.nn.Module):
    def __init__(self, wrapper: TRTWrapper):
        super().__init__()
        self.w = wrapper

    def forward(self, img: torch.Tensor):
        f1 = self.w.infer(img)
        f2 = self.w.infer(torch.flip(img, [-1]))
        f2 = np.flip(f2, axis=3)
        fused = f1 + f2
        return torch.from_numpy(fused).to(img.device)  # (1,512,H,W)

# ——————————————————————————————
# 2) 슬라이딩 윈도우 + Fusion-Head
# ——————————————————————————————
def pad_image(img, mean, std, cs):
    b,c,h,w = img.shape
    padh, padw = max(cs-h,0), max(cs-w,0)
    pad_vals = (-np.array(mean)/np.array(std)).tolist()
    out = img.new_zeros((b,c,h+padh,w+padw))
    for i in range(c):
        out[:,i] = F.pad(img[:,i], (0,padw,0,padh), value=pad_vals[i])
    return out

def sliding_inference(adapter: TRTAdapter,
                      scales_px: List[int], rel_scales: List[float],
                      img: torch.Tensor, txt_feat_t: torch.Tensor,
                      logit_scale, up_kwargs, base_size: int,
                      crop_size: int, mean, std, flip: bool=True):
    """
    txt_feat_t: (num_classes,512) GPU tensor
    logit_scale: scalar GPU tensor
    up_kwargs: dict for F.interpolate
    base_size, crop_size: int
    mean, std: list[float] for normalize
    """

    # sliding_inference 시작 부분에 추가
    txt_feat_t  = txt_feat_t.to(device=img.device, dtype=img.dtype)
    logit_scale = logit_scale.to(device=img.device, dtype=img.dtype)

    device = img.device
    b,_,H,W = img.shape
    stride = int(crop_size * 2/3)

    # 1) 최종 스코어 누적
    scores = torch.zeros((b, txt_feat_t.shape[0], H, W), device=device)

    # 이제 static 엔진은 없고, 모든 분기에서 이 동적 어댑터 하나만 사용
    dyn_adpt = adapter
    for S, rs in zip(scales_px, rel_scales):
        long_size = math.ceil(base_size * rs)
        # 2) 비율에 따른 리사이즈
        if H > W:
            nh, nw = long_size, int(W * long_size / H + 0.5)
        else:
            nh, nw = int(H * long_size / W + 0.5), long_size


        im_s = F.interpolate(img, size=(nh, nw), **up_kwargs)

        # 3) 두 가지 경로:
        if long_size <= crop_size:
            # 전체 이미지 한 번에 동적 엔진으로
            pad = pad_image(im_s, mean, std, long_size)
            tmp = adapter(pad)          # (1,512, nh//2, nw//2)
            fh, fw = nh//2, nw//2
            feat = tmp[:, :, :fh, :fw]
        else:
            # 슬라이딩 패치 전부 동적 엔진으로
            pad = pad_image(im_s, mean, std, crop_size)
            ph,pw = pad.shape[2], pad.shape[3]
            ph_f, pw_f = ph//2, pw//2
            feat = torch.zeros((b,512,ph_f,pw_f), device=device)
            cnt  = torch.zeros((b,1,ph_f,pw_f), device=device)
            for i in range(math.ceil((ph-crop_size)/stride)+1):
                for j in range(math.ceil((pw-crop_size)/stride)+1):
                    y0, x0 = i*stride, j*stride
                    # 실제 잘린 원본 패치 크기
                    orig_h = min(crop_size, ph - y0)
                    orig_w = min(crop_size, pw - x0)
                    patch = pad[:, :, y0:y0+orig_h, x0:x0+orig_w]
                    patch = pad_image(patch, mean, std, crop_size)
                    tmp = adapter(patch)     # dynamic
                        # 유효 영역만 크롭
                    h_f = math.ceil(orig_h / 2)
                    w_f = math.ceil(orig_w / 2)
                    tmp = tmp[:, :, :h_f, :w_f]
                    y0_f, x0_f = y0//2, x0//2
                    feat[:, :, y0_f:y0_f+h_f, x0_f:x0_f+w_f] += tmp
                    cnt [:, :, y0_f:y0_f+h_f, x0_f:x0_f+w_f] += 1
            feat = feat / cnt
            tmp = feat

        # 3) PyTorch 분류부
        # (이미 tmp는 Tensor)
        f = tmp[0]  # (512,fh,fw)
        logit = logit_scale * torch.tensordot(txt_feat_t, f, dims=([1],[0]))
        logit = logit.unsqueeze(0)  # (1,C,fh,fw)
        logit_up = F.interpolate(logit, size=(H,W), **up_kwargs)
        scores += logit_up

    # 6) flip 앙상블
    if flip:
        rev = sliding_inference(
                adapter, scales_px, rel_scales,
                img.flip(-1), txt_feat_t, logit_scale, 
                up_kwargs, base_size, crop_size, 
                mean, std, flip=False
            )
        scores += rev.flip(-1)

    return scores

# ——————————————————————————————
# 3) 스크립트 시작
# ——————————————————————————————
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--mode", choices=["save","eval","both"], default="both")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda"

    # 3.1) 분류·후처리 모듈 로드
    # → model_dir 아래 .ckpt 파일을 찾아서 그 경로를 넘겨줍니다.
    # ckpts = [f for f in os.listdir(args.model_dir) if f.endswith(".ckpt")]
    # assert ckpts, f"No .ckpt in {args.model_dir}"
    # ckpt_path = os.path.join(args.model_dir, ckpts[0])
    ckpt_path = "modules/demo_e200.ckpt"
    print(f"분류 모듈(LSegFull) 로드 중... ({ckpt_path})")
    full = LSegFull(
        checkpoint_path=ckpt_path,
        device=device
    )

    # (1) ADE20K 레이블 파일에서 직접 150개 레이블을 읽어서
    #     CLIP 텍스트 임베딩을 다시 계산합니다.
    label_file = os.path.join(args.data_root, "objectInfo150.txt")
    labels = [
        line.strip().split(',')[-1].split(';')[0]
        for line in open(label_file, "r")
    ][1:]  # 첫 줄은 헤더
    with torch.no_grad():
        text_inputs = clip.tokenize(labels).to(device)
        txt_feat = full.module.net.clip_pretrained.encode_text(text_inputs)
        txt_feat_t = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    # 파라미터 꺼내기
    logit_scale = full.logit_scale        # scalar GPU tensor
    up_kwargs   = full.up_kwargs          # {'mode':'bilinear','align_corners':True}
    base_size   = full.base_size          # 520
    crop_size   = full.crop_size          # 480
    mean, std   = full.module.mean, full.module.std


    # 1) 원본과 같은 스케일 정의
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    scales_px = [ceil(base_size * s) for s in scales]  # [260,390,520,650,780,910]
    rel_scales = [s/base_size for s in scales_px]

    # 2) 풀 이미지용 스케일과 슬라이딩용 엔진 목록
    full_scales = [S for S in scales_px if S <= crop_size]  # [260, 390]
    patch_scale = crop_size                                 # 480

    # 3) 단일 엔진만 로드 (메모리 절약)
    engine_name = "lseg_img_enc_vit_ade20k.trt"
    engine_path = os.path.join(args.model_dir, engine_name)
    assert os.path.exists(engine_path), f"{engine_name} 이 {args.model_dir}에 없습니다."
    # crop_size 를 S로 넘겨주면, pad_image(..., crop_size) 와 딱 맞습니다.
    wrapper = TRTWrapper(engine_path, S=None)
    adapter = TRTAdapter(wrapper)
    print(f"[DEBUG] Loaded dynamic engine: {engine_path}")

    # 3.4) DataLoader
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3),
    ])
    ds = ADE20KSegmentation(
        root=args.data_root, split="val", mode="val",
        transform=val_tf, target_transform=None,
        # module.xxx 대신 변수 사용
        base_size=base_size, crop_size=crop_size
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=4,
    )
    # CLIP 텍스트 임베딩 개수로 conf 크기 결정
    # txt_feat_t 은 (num_classes,512) GPU tensor
    num_classes = txt_feat_t.shape[0]

    # 5) 평가 루프
    print("평가 루프 시작!")
    conf   = np.zeros((num_classes, num_classes), dtype=np.int64)
    total, correct = 0,0
    pbar = tqdm(loader, total=len(loader))
    for idx, (img_t, gt_m, fn) in enumerate(pbar):
        img = img_t.to(device)            # 이제 img_t 는 텐서!
        with torch.no_grad():
            scores = sliding_inference(
            adapter, scales_px, rel_scales,
            img, txt_feat_t, logit_scale, up_kwargs,
            base_size, crop_size, mean, std, flip=True
        )

        pred = scores.argmax(dim=1).squeeze(0).cpu().numpy()
        gt   = gt_m.squeeze(0).numpy()
        valid = gt>0
        gi, pi = gt[valid]-1, pred[valid]
        total   += valid.sum(); correct += (pi==gi).sum()
        for g,p in zip(gi,pi): conf[g,p]+=1

        if args.mode in ("save","both"):
            # basename = os.path.basename(ds.images[idx])  # 원본 이미지 파일명
            # Image.fromarray(pred.astype(np.uint8))\
            #      .save(os.path.join(args.out_dir, basename.replace('.jpg','.png')))
            basename = fn
            Image.fromarray(pred.astype(np.uint8))\
                .save(os.path.join(args.out_dir, basename.replace('.jpg','.png')))

        pix  = correct/total
        ious = []
        for c in range(num_classes):
            tp    = conf[c, c]
            denom = conf[c, :].sum() + conf[:, c].sum() - tp
            if denom > 0:
                ious.append(tp / denom)
            else:
                ious.append(0.0)
        mIoU = np.mean(ious)
        pbar.set_description(f"pixAcc:{pix:.4f}, mIoU:{mIoU:.4f}")

    if args.mode in ("eval","both"):
        print(f"\n최종 pixAcc: {pix:.4f}, mIoU: {mIoU:.4f}")
        print("클래스별 IoU:", ious)

if __name__=="__main__":
    main()
