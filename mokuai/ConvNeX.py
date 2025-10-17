from torch import nn
from torch.nn import LayerNorm

class ConvNeXt(nn.Module):
    def __init__(self, dim ):
        super().__init__()
        # 分组卷积+大卷积核
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 在1x1之前使用唯一一次LN做归一化
        self.norm = LayerNorm(dim, eps=1e-6)
        # 全连接层跟1x1conv等价，但pytorch计算上fc略快
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 整个block只使用唯一一次激活层
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # 由于用FC来做1x1conv，所以需要调换通道顺序
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input +x
        return x