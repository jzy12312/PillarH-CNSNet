import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.reduce_layers = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth_layers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])

    def forward(self, features):
        # features is a list of feature maps from different scales
        # each feature map has shape [N, C, H, W]
        last_feature = features[-1]
        for idx, x in enumerate(features[:-1]):
            diffY = last_feature.size()[2] - x.size()[2]
            diffX = last_feature.size()[3] - x.size()[3]

            x = F.upsample(x, size=last_feature.size()[2:], mode='nearest')
            x = torch.cat([x, last_feature], dim=1)
            x = self.reduce_layers[idx](x)
            x = self.smooth_layers[idx](x)
            last_feature = x
        return last_feature