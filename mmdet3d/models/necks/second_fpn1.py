# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS


class WeightedConcat(nn.Module):
    def __init__(self, num_inputs, dimension=1):
        super(WeightedConcat, self).__init__()
        self.d = dimension
        # 设置可学习参数，用于加权特征融合
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # 归一化权重
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # 加权特征融合
        weighted_x = [weight[i] * x[i] for i in range(len(x))]
        return torch.cat(weighted_x, self.d)

@MODELS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to
            [dict(type='Kaiming', layer='ConvTranspose2d'),
             dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)].
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=[
                     dict(type='Kaiming', layer='ConvTranspose2d'),
                     dict(
                         type='Constant',
                         layer='NaiveSyncBatchNorm2d',
                         val=1.0)
                 ]):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        # 添加 WeightedConcat 模块
        self.weighted_concat = WeightedConcat(num_inputs=len(out_channels), dimension=1)

    def forward(self, x):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = self.weighted_concat(ups)
        else:
            out = ups[0]

        return [out]
