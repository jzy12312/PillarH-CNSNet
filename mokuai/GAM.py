# 导入必要的库
import torch
import torch.nn as nn


# 定义GAM_Attention类，继承自nn.Module
class GAM_Attention(nn.Module):
    # 初始化方法
    def __init__(self, in_channels, out_channels, rate=4):
        # 调用父类的构造函数
        super(GAM_Attention, self).__init__()

        # 定义通道注意力机制
        self.channel_attention = nn.Sequential(
            # 第一个线性层，将输入通道数压缩到in_channels/rate
            nn.Linear(in_channels, int(in_channels / rate)),
            # ReLU激活函数，增加非线性
            nn.ReLU(inplace=True),
            # 第二个线性层，将通道数映射回in_channels
            nn.Linear(int(in_channels / rate), in_channels)
        )

        # 定义空间注意力机制
        self.spatial_attention = nn.Sequential(
            # 第一个卷积层，将输入通道数压缩到in_channels/rate，使用7x7卷积
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            # 批归一化，规范化卷积层的输出
            nn.BatchNorm2d(int(in_channels / rate)),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 第二个卷积层，将通道数变换到out_channels
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            # 批归一化
            nn.BatchNorm2d(out_channels)
        )

    # 定义前向传播方法
    def forward(self, x):
        # 获取输入张量的批量大小、通道数、高度和宽度
        b, c, h, w = x.shape

        # 变换输入张量的维度为 (b, h, w, c)，方便进行通道注意力计算
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)

        # 通过通道注意力网络计算通道注意力权重
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)

        # 变换回 (b, c, h, w) 维度
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        # 逐元素相乘应用通道注意力权重
        x = x * x_channel_att

        # 计算空间注意力，并应用sigmoid激活
        x_spatial_att = self.spatial_attention(x).sigmoid()

        # 逐元素相乘应用空间注意力
        out = x * x_spatial_att

        # 返回最终的输出
        return out