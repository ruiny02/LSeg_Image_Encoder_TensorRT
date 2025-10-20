# lseg_full.py
import torch
import torch.nn.functional as F
import clip
from modules.lseg_module import LSegModule  # 원본 전체 네트워크

class LSegFull:
    """
    PyTorch 기반 분류·후처리 모듈.
    TRT로 추출된 인코더 출력을 받아, Fusion-Head와 Upsample을 수행합니다.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # 1) 전체 LSegModule 로드
        self.module = LSegModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cpu",
            backbone="clip_vitl16_384",
            num_features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            readout="ignore",
        )
        # now move the actual PyTorch submodules to GPU and eval them
        self.module.net        = self.module.net.to(device).eval()
        self.module.logit_scale= self.module.logit_scale.to(device)

        # 3) 분류용 파라미터
        self.logit_scale = self.module.logit_scale.to(device)  # scalar
        self.up_kwargs   = self.module._up_kwargs              # e.g. {'mode':'bilinear','align_corners':True}
        self.base_size   = self.module.base_size               # 520
        self.crop_size   = self.module.crop_size               # 480

    def classify(self, feat: torch.Tensor, orig_size: tuple) -> torch.Tensor:
        """
        인코더 출력 feat (1,512,h,w)을 받아
        1×1 conv 헤드 + logit_scale + 업샘플 → (1,num_classes,H,W)
        """
        # Fusion-Head 적용
        logits = self.module.net.scratch.head1(feat)       # head1 == 1×1 conv → (1,out_c,h,w)
        logits = logits * self.logit_scale                 # 온도 스케일링
        # 업샘플 (원본 H,W)
        logits = F.interpolate(logits, size=orig_size, **self.up_kwargs)
        return logits

    def forward(self, feat: torch.Tensor, orig_size: tuple, flip: bool = True) -> torch.Tensor:
        """
        sliding_inference와 동일한 flip 앙상블 기능을 제공하는 래퍼
        """
        # 기본 분류
        score = self.classify(feat, orig_size)
        if flip:
            # 좌우 뒤집어서 재분류
            rev_feat = torch.flip(feat, dims=[3])
            rev_score = self.classify(rev_feat, orig_size)
            score = score + torch.flip(rev_score, dims=[3])
        return score
