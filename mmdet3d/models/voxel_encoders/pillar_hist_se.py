# mmdet3d/models/voxel_encoders/pillar_hist_se.py
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

@MODELS.register_module()
class PillarHist(nn.Module):
    def __init__(self,
                 in_channels=4,
                 num_bins=64,
                 z_range=(-3.0, 1.0),
                 feat_channels=64,):
        super().__init__()
        self.in_channels = in_channels
        self.num_bins = num_bins
        self.z_min, self.z_max = z_range
        self.bin_size = (self.z_max - self.z_min) / num_bins
        self.out_dim = num_bins * 3  # count + mean_z + mean_r

        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, features, num_points, coors):
        M, N, C = features.shape
        device = features.device
        mask = torch.arange(N, device=device).unsqueeze(0) < num_points.unsqueeze(1)

        z = features[..., 2]
        r = features[..., 3]
        bin_idx = ((z - self.z_min) / self.bin_size).long()
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        hist_feats = torch.zeros(M, self.num_bins * 3, device=device)

        for b in range(self.num_bins):
            bin_mask = (bin_idx == b) & mask
            count = bin_mask.sum(dim=1).float()
            mean_z = (z * bin_mask).sum(dim=1) / (count + 1e-5)
            mean_r = (r * bin_mask).sum(dim=1) / (count + 1e-5)

            hist_feats[:, b * 3 + 0] = count
            hist_feats[:, b * 3 + 1] = mean_z
            hist_feats[:, b * 3 + 2] = mean_r

        pillar_feat = self.mlp(hist_feats)  # (M, feat_channels)

        return pillar_feat