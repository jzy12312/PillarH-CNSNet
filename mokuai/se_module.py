# se_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, N = x.size()  # x: [B, C, N]
        y = self.avg_pool(x).view(B, C)  # 全局平均池化，压缩空间维度
        y = self.fc(y).view(B, C, 1)  # 生成通道注意力权重
        return x * y.expand_as(x)  # 将注意力权重应用到原始特征上