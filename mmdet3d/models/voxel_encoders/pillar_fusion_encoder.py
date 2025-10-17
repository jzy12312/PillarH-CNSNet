import torch
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer, build_activation_layer

from mmdet3d.models.voxel_encoders.pillar_hist_se import PillarHist
from mmdet3d.registry import MODELS


@MODELS.register_module()
class PillarHistFusion(nn.Module):
    def __init__(self,
                 pillar_feature_net_cfg,
                 num_bins=64,
                 z_range=(-3.0, 1.0),
                 feat_channels=64,
                 fusion_cfg=dict(type='MLP',
                                 in_channels=128,
                                 hidden_channels=[128, 64],
                                 norm_cfg=dict(type='BN1d'),
                                 act_cfg=dict(type='ReLU'),
                                 dropout=0.1)):
        super().__init__()

        from .pillar_encoder import PillarFeatureNet
        self.pfn = PillarFeatureNet(**pillar_feature_net_cfg)
        self.hist = PillarHist(num_bins=num_bins,
                               z_range=z_range,
                               feat_channels=feat_channels)

        # 构建 MLP 融合模块
        self.fusion = self._build_mlp(**fusion_cfg)

    def _build_mlp(self,
                   type,
                   in_channels,
                   hidden_channels,
                   norm_cfg=None,
                   act_cfg=None,
                   dropout=0.0):
        """构建一个简单的前向 MLP."""
        assert type == 'MLP'
        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(hidden_channels):
            layers.append(nn.Linear(in_ch, out_ch))
            if norm_cfg is not None:
                layers.append(build_norm_layer(norm_cfg, out_ch)[1])
            if act_cfg is not None:
                layers.append(build_activation_layer(act_cfg))
            if dropout > 1e-3:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, features, num_points, coors):
        pfn_feat = self.pfn(features, num_points, coors)      # (M, 64)
        hist_feat = self.hist(features, num_points, coors)    # (M, 64)
        fused = torch.cat([pfn_feat, hist_feat], dim=1)       # (M, 128)
        return self.fusion(fused)                             # (M, 64)