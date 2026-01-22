import os
import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import clip

# ------------------------------
# Activation hook (readout은 'ignore' 모드로 고정)
# ------------------------------
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

# ------------------------------
# 최소한의 readout 연산: "ignore" 모드 (Slice)
# ------------------------------
class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index
    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.GELU()
        )
    
    def forward(self, x):
        # x의 shape: (B, N, C)라고 가정합니다.
        # 첫 번째 토큰(readout 토큰)을 추출하여 나머지 토큰들과 결합합니다.
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        concatenated = torch.cat((x[:, self.start_index:], readout), dim=-1)
        return self.project(concatenated)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [ProjectReadout(vit_features, start_index) for _ in features]
    else:
        assert False, "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"
    return readout_oper

# ------------------------------
# ViT feature extraction functions
# ------------------------------
def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    # ViT 백본의 forward_flex() 실행 (포지셔널 임베딩 적용 등)
    glob = pretrained.model.forward_flex(x)
    # 등록한 hook을 통해 각 블록의 출력을 획득
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]
    # 초기 postprocess 단계 (최소화된 처리: 앞쪽 두 모듈만 사용)
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)
    # 텐서가 3D라면 재구성 (patch_size에 맞게)
    if layer_1.ndim == 3:
        # reshape 시에도 Python int 제거, grid_h/w 사용
        grid_h = h // pretrained.model.patch_size[1]
        grid_w = w // pretrained.model.patch_size[0]
        layer_1 = layer_1.reshape(layer_1.shape[0], layer_1.shape[1], grid_h, grid_w)
    if layer_2.ndim == 3:
        grid_h = h // pretrained.model.patch_size[1]
        grid_w = w // pretrained.model.patch_size[0]
        layer_2 = layer_2.reshape(layer_2.shape[0], layer_2.shape[1], grid_h, grid_w)
    if layer_3.ndim == 3:
        grid_h = h // pretrained.model.patch_size[1]
        grid_w = w // pretrained.model.patch_size[0]
        layer_3 = layer_3.reshape(layer_3.shape[0], layer_3.shape[1], grid_h, grid_w)
    if layer_4.ndim == 3:
        grid_h = h // pretrained.model.patch_size[1]
        grid_w = w // pretrained.model.patch_size[0]
        layer_4 = layer_4.reshape(layer_4.shape[0], layer_4.shape[1], grid_h, grid_w)
    # 후처리 단계 (나머지 모듈 적용)
    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)
    return layer_1, layer_2, layer_3, layer_4

def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :]
    )

    # 사전학습된 ViT의 패치 그리드 크기를 직접 계산
    # patch_embed.img_size: (height, width) of the pretrained grid
    # patch_size: [patch_height, patch_width]
    img_h, img_w     = self.patch_embed.img_size
    patch_h, patch_w = self.patch_size
    gs_old = img_h // patch_h

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb

def forward_flex(self, x):
    b, c, h, w = x.shape
    # grid_h, grid_w를 텐서 연산으로 먼저 계산해 dynamic_axes 와 호환되게
    grid_h = h // self.patch_size[1]
    grid_w = w // self.patch_size[0]
    pos_embed = self._resize_pos_embed(self.pos_embed, grid_h, grid_w)
    B = x.shape[0]
    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x

# ------------------------------
# Pretrained backbone 생성 함수들 (최소화된 버전)
# ------------------------------
def _get_clip_device():
    # Override via env CLIP_DEVICE=cpu|cuda; default prefers CPU for low-memory export.
    env = os.getenv("CLIP_DEVICE")
    if env in ("cpu", "cuda"):
        return env
    return "cpu"  # safer default on Jetson during export


def _make_pretrained_clip_vitl16_384(pretrained, use_readout="project", hooks=None, enable_attention_hooks=False):
    clip_pretrained = None
    if pretrained:
        clip_pretrained, _ = clip.load("ViT-B/32", device=_get_clip_device(), jit=False)
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)
    hooks = [5, 11, 17, 23] if hooks is None else hooks
    pretrained = _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained

def _make_pretrained_clipRN50x16_vitl16_384(pretrained, use_readout="project", hooks=None, enable_attention_hooks=False):
    clip_pretrained = None
    if pretrained:
        clip_pretrained, _ = clip.load("RN50x16", device=_get_clip_device(), jit=False)
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)
    hooks = [5, 11, 17, 23] if hooks is None else hooks
    pretrained = _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained

def _make_pretrained_clip_vitb32_384(pretrained, use_readout="project", hooks=None, enable_attention_hooks=False):
    clip_pretrained = None
    if pretrained:
        clip_pretrained, _ = clip.load("ViT-B/32", device=_get_clip_device(), jit=False)
    model = timm.create_model("vit_base_patch32_384", pretrained=pretrained)
    hooks = [2, 5, 8, 11] if hooks is None else hooks
    pretrained = _make_vit_b32_backbone(
        model, 
        features=[96, 192, 384, 768], 
        hooks=hooks, 
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained

def _make_vit_b32_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="project",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
    pretrained.activations = activations
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True),
    )
    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True),
    )
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0),
    )
    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0),
        nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1),
    )
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained

def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="project",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
    pretrained.activations = activations
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True),
    )
    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True),
    )
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0),
    )
    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0),
        nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1),
    )
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained
