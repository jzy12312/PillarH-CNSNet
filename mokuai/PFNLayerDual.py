import math           # 计算最大公约数 gcd 用
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer   # mmcv 的归一化层构造工具

class PFNLayerDual(nn.Module):
    """
    用 1-D DualConv（1×1 Conv1d + 3×1 Conv1d，分组卷积）替换 PFNLayer 的 nn.Linear。
    输入/输出形状与原 PFNLayer 完全一致：
        输入: (B, N, C_in)   ->   输出: (B, N, C_out)
    """
    def __init__(self,
                 in_channels: int,      # 每个点的特征维度
                 out_channels: int,     # 经过本层后要输出的特征维度
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer: bool = False,  # True -> 最后一层，需要按 pillar 做池化
                 mode: str = 'max',     # 池化方式 'max' 或 'avg'
                 groups: int = 4):      # 分组卷积的组数（默认 4）
        super().__init__()
        self.last_layer = last_layer
        self.mode = mode

        # -------------------------------------------------
        # 1. 让 groups 始终合法：既要 ≤ in_channels，又要能整除
        #    math.gcd 求最大公约数，保证一定能整除
        # -------------------------------------------------
        groups = max(1, min(groups, in_channels))
        groups = math.gcd(groups, in_channels)

        # -------------------------------------------------
        # 2. DualConv 双分支：1×1 卷积 + 3×1 卷积（分组）
        #    out_channels 被拆成两半，最后再 concat
        # -------------------------------------------------
        half_out = out_channels // 2
        # 1×1 卷积：通道间信息交互
        self.conv1x1 = nn.Conv1d(in_channels, half_out, kernel_size=1, groups=groups)
        # 3×1 卷积：局部时序/邻域特征
        self.conv3x1 = nn.Conv1d(in_channels, half_out, kernel_size=3, padding=1, groups=groups)

        # -------------------------------------------------
        # 3. 归一化：对拼接后的 2*half_out 个通道做 BN
        # -------------------------------------------------
        norm_channels = half_out * 2
        _, self.norm = build_norm_layer(norm_cfg, norm_channels)

        # -------------------------------------------------
        # 4. 激活函数：最后一层不需要 relu
        # -------------------------------------------------
        self.relu = nn.ReLU(inplace=True) if not last_layer else None

    def forward(self, features, num_points=None):
        """
        前向传播
        Args:
            features: (B, N, C_in)  每个 pillar 内的点特征
            num_points: (B,)        每个 pillar 的实际点数（做 avg pool 时用）
        Returns:
            out: (B, N, C_out)  或  (B, 1, C_out) 如果是 last_layer
        """
        # ---------------------------------------------
        # 1. 把 (B, N, C) -> (B, C, N) 适应 Conv1d
        # ---------------------------------------------
        x = features.transpose(1, 2).contiguous()

        # ---------------------------------------------
        # 2. 双分支卷积 & 拼接
        # ---------------------------------------------
        branch1 = self.conv1x1(x)   # (B, half_out, N)
        branch2 = self.conv3x1(x)   # (B, half_out, N)
        x = torch.cat([branch1, branch2], dim=1)  # (B, out_channels, N)

        # ---------------------------------------------
        # 3. BN + ReLU
        # ---------------------------------------------
        x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)

        # ---------------------------------------------
        # 4. 把 (B, C, N) 还原成 (B, N, C)
        # ---------------------------------------------
        x = x.transpose(1, 2).contiguous()

        # ---------------------------------------------
        # 5. 如果是最后一层，要按 pillar 做池化
        #    - max: 直接取最大
        #    - avg: 用实际点数做平均
        # ---------------------------------------------
        if self.last_layer:
            if self.mode == 'max':
                x = x.max(dim=1, keepdim=True)[0]          # (B, 1, C_out)
            elif self.mode == 'avg':
                x = x.sum(dim=1, keepdim=True) / num_points.unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f'Unsupported pooling mode: {self.mode}')
        return x