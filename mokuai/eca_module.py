import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        t = max(t, 3)  # 确保最小卷积核大小为3
        self.conv = nn.Conv1d(channels, channels, kernel_size=t, padding=(t//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        y = self.conv(y.unsqueeze(-1)).squeeze(-1)

        y = self.sigmoid(y)

        return x * y.unsqueeze(-1)