import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPELayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.conv_layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
            shape = inputs.shape
            inputs = inputs.view(shape[0], shape[3], shape[1], shape[2])
            max_point_per_voxel = shape[2]
        x = self.conv_layer1(inputs)
        x = self.bn1(x)
        x = F.relu(x)
        max_feature = torch.max(x, dim=-1, keepdim=True)[0]
        attention_score = F.softmax(self.conv_layer2(inputs), dim=-1)
        avg_feature = x * attention_score
        avg_feature = torch.sum(avg_feature, dim=-1, keepdim=True)
        feature = (avg_feature + max_feature) / 2.0
        feature = feature.transpose(1, 2)
        return feature