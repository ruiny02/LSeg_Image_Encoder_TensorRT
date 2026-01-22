import os
import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import clip
from torchvision import models

# ------------------------------
# Activation hook 등록 및 관련 함수들
# ------------------------------
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

# ------------------------------
# Readout 연산 정의
# ------------------------------
class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index
    def forward(self, x):
        return x[:, self.start_index:]

class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index
    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.GELU()
        )
    def forward(self, x):
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
        assert False, "wrong operation for readout token, use_readout must be 'ignore', 'add', or 'project'"
    return readout_oper

# # ------------------------------
# # ViT 백본의 이미지 인코더 관련 함수들
# # ------------------------------
# def forward_vit(pretrained, x):
#     b, c, h, w = x.shape
#     # forward_flex()를 통해 global feature를 계산하며 hook에 의해 각 레이어의 출력을 저장
#     _ = pretrained.model.forward_flex(x)
#     layer_1 = activations["1"]
#     layer_2 = activations["2"]
#     layer_3 = activations["3"]
#     layer_4 = activations["4"]
#     # 초기 post-process: 등록된 act_postprocess의 앞부분만 사용
#     layer_1 = pretrained.act_postprocess1[0:2](layer_1)
#     layer_2 = pretrained.act_postprocess2[0:2](layer_2)
#     layer_3 = pretrained.act_postprocess3[0:2](layer_3)
#     layer_4 = pretrained.act_postprocess4[0:2](layer_4)
#     # 만약 unflatten이 필요하다면 (토큰이 3D 텐서인 경우) 변환
#     if layer_1.ndim == 3:
#         unflatten = nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]]))
#         layer_1 = unflatten(layer_1)
#     if layer_2.ndim == 3:
#         unflatten = nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]]))
#         layer_2 = unflatten(layer_2)
#     if layer_3.ndim == 3:
#         unflatten = nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]]))
#         layer_3 = unflatten(layer_3)
#     if layer_4.ndim == 3:
#         unflatten = nn.Unflatten(2, torch.Size([h // pretrained.model.patch_size[1], w // pretrained.model.patch_size[0]]))
#         layer_4 = unflatten(layer_4)
#     # 후처리 단계: act_postprocess의 나머지 모듈 적용
#     layer_1 = pretrained.act_postprocess1[2:](layer_1)
#     layer_2 = pretrained.act_postprocess2[2:](layer_2)
#     layer_3 = pretrained.act_postprocess3[2:](layer_3)
#     layer_4 = pretrained.act_postprocess4[2:](layer_4)
#     return layer_1, layer_2, layer_3, layer_4

# def _resize_pos_embed(self, posemb, gs_h, gs_w):
#     posemb_tok, posemb_grid = posemb[:, : self.start_index], posemb[0, self.start_index:]
#     gs_old = int(math.sqrt(len(posemb_grid)))
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#     return posemb

# def forward_flex(self, x):
#     b, c, h, w = x.shape
#     pos_embed = self._resize_pos_embed(self.pos_embed, h // self.patch_size[1], w // self.patch_size[0])
#     B = x.shape[0]
#     # 만약 patch_embed가 백본을 내장하고 있다면 이를 사용
#     if hasattr(self.patch_embed, "backbone"):
#         x = self.patch_embed.backbone(x)
#         if isinstance(x, (list, tuple)):
#             x = x[-1]
#     x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
#     # dist_token이 존재하는 경우 처리
#     if getattr(self, "dist_token", None) is not None:
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#     else:
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#     x = x + pos_embed
#     x = self.pos_drop(x)
#     for blk in self.blocks:
#         x = blk(x)
#     x = self.norm(x)
#     return x

# # ------------------------------
# # ViT 백본 구성 함수 (_zs 버전용)
# # ------------------------------
# def _make_vit_b16_backbone(
#     model,
#     features=[96, 192, 384, 768],
#     size=[384, 384],
#     hooks=[2, 5, 8, 11],
#     vit_features=768,
#     use_readout="ignore",
#     start_index=1,
#     enable_attention_hooks=False,
# ):
#     pretrained = nn.Module()
#     pretrained.model = model
#     pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
#     pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
#     pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
#     pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
#     pretrained.activations = activations
#     readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
#     pretrained.act_postprocess1 = nn.Sequential(
#         readout_oper[0],
#         Transpose(1, 2),
#         nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
#         nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1),
#         nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4)
#     )
#     pretrained.act_postprocess2 = nn.Sequential(
#         readout_oper[1],
#         Transpose(1, 2),
#         nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
#         nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1),
#         nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2)
#     )
#     pretrained.act_postprocess3 = nn.Sequential(
#         readout_oper[2],
#         Transpose(1, 2),
#         nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
#         nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1)
#     )
#     pretrained.act_postprocess4 = nn.Sequential(
#         readout_oper[3],
#         Transpose(1, 2),
#         nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
#         nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1),
#         nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1)
#     )
#     pretrained.model.start_index = start_index
#     pretrained.model.patch_size = [16, 16]
#     pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
#     pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
#     return pretrained


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained

def _get_clip_device():
    env = os.getenv("CLIP_DEVICE")
    if env in ("cpu", "cuda"):
        return env
    return "cpu"


def _make_pretrained_clip_rn101(use_pretrained):
    # CLIP 모델 로드 (ResNet101용) — 사전학습 생략 시 None으로 두어 메모리 절약
    clip_pretrained = None
    if use_pretrained:
        clip_pretrained, _ = clip.load("ViT-B/32", device=_get_clip_device(), jit=False)
    resnet = models.resnet101(pretrained=use_pretrained)
    pretrained = _make_resnet_backbone(resnet)
    return clip_pretrained, pretrained
