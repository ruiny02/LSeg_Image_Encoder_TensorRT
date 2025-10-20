#!/usr/bin/env python3
import argparse, os
import torch, torch.nn.functional as F
from tqdm import tqdm
import clip
import tensorrt as trt
import pycuda.autoinit, pycuda.driver as cuda
import numpy as np

# ─── FewShot 평가용 유틸리티 ───────────────────────────────────
from fewshot_data.common.logger import AverageMeter, Logger
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.common import utils
from fewshot_data.data.dataset import FSSDataset

# ─── Zero‐Shot 모듈 ────────────────────────────────────────────
from lseg_module_zs import LSegModuleZS          # :contentReference[oaicite:2]{index=2}
from lsegmentation_module_zs import LSegmentationModuleZS  # :contentReference[oaicite:3]{index=3}

# ─── Adapter 베이스 클래스 ────────────────────────────────────
class AdapterBase(torch.nn.Module):
    def _inherit(self, module):
        # LSegModuleZS/LSegmentationModuleZS 가 가진 전처리·후처리 cfg 상속
        self.base_size  = module.base_size
        self.crop_size  = module.crop_size
        self.mean       = getattr(module, "mean", [0.5,0.5,0.5])
        self.std        = getattr(module, "std",  [0.5,0.5,0.5])
        self._up_kwargs = getattr(module, "_up_kwargs",
                            getattr(module, "up_kwargs",
                                    {'mode':'bilinear','align_corners':True}))
        # MultiEvalModule 호환
        self.evaluate = self.forward

# ─── PyTorch 모드 Adapter ─────────────────────────────────────
class PTAdapterZS(AdapterBase):
    def __init__(self, module: LSegModuleZS):
        super().__init__()
        self.module = module
        self._inherit(module)
        # 내부 네트워크는 LSegModuleZS.net(x, class_info) 그대로 호출
        self.net = module.net

    def forward(self, img, class_info):
        img = img.to(next(self.net.parameters()).device)
        with torch.no_grad():
            logits = self.net(img, class_info)    # (B, Ncls, H, W)
        return logits

# ─── TensorRT 모드 Wrapper + Adapter ─────────────────────────
class TRTWrapperZS:
    def __init__(self, engine_path: str):
        # TensorRT 엔진 로드
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        buf = open(engine_path, "rb").read()
        self.engine = runtime.deserialize_cuda_engine(buf)
        self.ctx    = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # binding 이름 추출
        names = [ self.engine.get_tensor_name(i)
                  for i in range(self.engine.num_io_tensors) ]
        # 입력은 0번, 출력 중 채널==512인 바인딩을 feat_name 으로 잡음
        self.in_name   = names[0]
        self.feat_name = next(n for n in names[1:]
                              if self.engine.get_tensor_shape(n)[1]==512)
        self.d_ptr, self.cap = {}, {}

    def _alloc(self, name, nbytes):
        if name in self.cap and self.cap[name]>=nbytes: return
        if name in self.d_ptr: self.d_ptr[name].free()
        self.d_ptr[name] = cuda.mem_alloc(nbytes)
        self.cap[name]   = nbytes

    def infer(self, x: torch.Tensor):
        arr = x.cpu().numpy().astype(np.float32)
        b,c,h,w = arr.shape
        # 입력 shape 설정
        self.ctx.set_input_shape(self.in_name, (b,c,h,w))
        # 512ch 출력 크기 얻기
        b2,C2,H2,W2 = self.ctx.get_tensor_shape(self.feat_name)
        # 버퍼 할당
        self._alloc(self.in_name, arr.nbytes)
        self._alloc(self.feat_name, b2*C2*H2*W2*4)
        # 바인딩 & 실행
        self.ctx.set_tensor_address(self.in_name,  int(self.d_ptr[self.in_name]))
        self.ctx.set_tensor_address(self.feat_name,int(self.d_ptr[self.feat_name]))
        cuda.memcpy_htod_async(self.d_ptr[self.in_name], arr, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        feat = np.empty((b2,C2,H2,W2), dtype=np.float32)
        cuda.memcpy_dtoh_async(feat, self.d_ptr[self.feat_name], self.stream)
        self.stream.synchronize()
        return torch.from_numpy(feat)

class TRTAdapterZS(AdapterBase):
    def __init__(self, engine_path: str, module: LSegModuleZS):
        super().__init__()
        self.w = TRTWrapperZS(engine_path)
        self._inherit(module)

        # ZS 분류 파라미터 상속
        clip_model = module.net.clip_pretrained   # CLIP 비전+텍스트 모델
        self.encode_text = clip_model.encode_text
        self.logit_scale = module.net.logit_scale
        # 클래스별 텍스트 임베딩 미리 계산
        labels = module.get_labels(module.args.benchmark)
        prompts = [['others',lbl] for lbl in labels]
        toks = clip.tokenize(prompts).to(self.logit_scale.device)
        with torch.no_grad():
            tfeat = self.encode_text(toks).float()
        self.txt_feat = tfeat / tfeat.norm(dim=-1,keepdim=True)

    def forward(self, img, class_info=None):
        feat = self.w.infer(img).to(self.logit_scale.device)  # (B,512,hf,wf)
        B,C,hf,wf = feat.shape
        # cosine 유사도
        f = F.normalize(feat,1).permute(0,2,3,1).reshape(-1,C)  # (BP,512)
        logits = (f @ self.txt_feat.T)*self.logit_scale          # (BP,Ncls)
        logits = logits.view(B,hf,wf,-1).permute(0,3,1,2)        # (B,Ncls,hf,wf)
        # upsample to orig
        logits = F.interpolate(logits, size=img.shape[-2:], **self._up_kwargs)
        return logits

# ─── 모델 빌드 함수 ──────────────────────────────────────────
def build_model(args):
    device = 'cuda' if args.cuda else 'cpu'
    # 1) TensorRT 모드
    if args.trt_engine:
        module = LSegModuleZS.load_from_checkpoint(
            args.weights, map_location='cpu', **vars(args)
        ).to(device).eval()
        return TRTAdapterZS(args.trt_engine, module).to(device).eval()
    # 2) PyTorch 모드
    else:
        module = LSegModuleZS.load_from_checkpoint(
            args.weights, map_location='cpu', **vars(args)
        ).to(device).eval()
        return PTAdapterZS(module).to(device).eval()

# ─── 테스트 파이프라인 ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str, required=True, help='ZS ckpt path')
    parser.add_argument('--datapath',   type=str, default='fewshot_data/Datasets_HSN')
    parser.add_argument('--benchmark',  type=str, default='fss')
    parser.add_argument('--nshot',      type=int, default=0)
    parser.add_argument('--fold',       type=int, default=0)
    parser.add_argument('--cuda',       action='store_true', default=True)
    parser.add_argument('--trt_engine', type=str, default='', help='TRT engine path')
    # LSegModuleZS kwargs
    parser.add_argument('--backbone',   type=str, default='clip_vitl16_384')
    parser.add_argument('--num_features',type=int,default=256)
    parser.add_argument('--base_lr',    type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=0)
    parser.add_argument('--aux',        action='store_false')
    parser.add_argument('--nshot',      type=int, default=0)
    parser.add_argument('--finetune_mode',action='store_false')
    parser.add_argument('--ignore_index',type=int, default=255)
    parser.add_argument('--no_scaleinv',action='store_false',dest='scale_inv')
    parser.add_argument('--widehead',   action='store_true')
    parser.add_argument('--widehead_hr',action='store_true')
    parser.add_argument('--arch_option',type=int, default=0)
    parser.add_argument('--block_depth',type=int, default=0)
    parser.add_argument('--activation', choices=['relu','lrelu','tanh'],default='relu')
    parser.add_argument('--use_pretrained', type=str, default='True')
    args = parser.parse_args()

    # 1) 모델 준비
    model = build_model(args)

    # 2) 평가 유틸 초기화
    Evaluator.initialize()
    FSSDataset.initialize(
        img_size=480,
        datapath=args.datapath,
        use_original_imgsize=False
    )
    dataloader = FSSDataset.build_dataloader(
        args.benchmark, args.batch_size, 0, args.fold, 'test', args.nshot
    )
    meter = AverageMeter(dataloader.dataset)

    # 3) 테스트 루프
    for idx, batch in enumerate(tqdm(dataloader, desc="Test")):
        batch = utils.to_cuda(batch)
        img  = batch['query_img']
        cls  = batch['class_id']
        pred = model(img, cls)                  # (B,Ncls,H,W)
        area_inter, area_union = Evaluator.classify_prediction(
            pred.argmax(1), batch['query_mask']
        )
        meter.update(area_inter, area_union, cls, loss=None)
        meter.write_process(idx, len(dataloader), 0, write_batch_idx=50)

    # 4) 결과 출력
    meter.write_result('Test', 0)
    miou, fbiou = meter.compute_iou()
    print(f"Final mIoU: {miou:.2f}, FB-IoU: {fbiou:.2f}")

if __name__ == "__main__":
    main()
