import os
import torch
import torch.nn as nn

from .lseg_vit import (
    _make_pretrained_clip_vitl16_384,
    _make_pretrained_clip_vitb32_384,
    _make_pretrained_clipRN50x16_vitl16_384,
    forward_vit,
)

def _make_encoder(
    backbone,
    features,
    use_pretrained=True,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="project",
    enable_attention_hooks=False,
):  
    # 환경변수 USE_PRETRAINED=0 로 설정하면 사전학습 가중치 로드를 생략해 메모리 사용을 낮춘다.
    env_use_pretrained = os.getenv("USE_PRETRAINED")
    if env_use_pretrained is not None:
        use_pretrained = env_use_pretrained not in ("0", "false", "False")

    if backbone == "clip_vitl16_384": 
        clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        ) 
    elif backbone == "clipRN50x16_vitl16_384":
        clip_pretrained, pretrained = _make_pretrained_clipRN50x16_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif backbone == "clip_vitb32_384":
        clip_pretrained, pretrained = _make_pretrained_clip_vitb32_384(
            use_pretrained, 
            hooks=hooks, 
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        ) 
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return clip_pretrained, pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class Interpolate(nn.Module):
    """단순 인터폴레이션 모듈."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )



class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super(ResidualConvUnit_custom, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=not bn)
        self.activation = activation
        self.bn = bn
        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(x, out)

class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features if not expand else features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output
