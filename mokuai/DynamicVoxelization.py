import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointpillars.ops import Voxelization

class DynamicVoxelization(Voxelization):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, density_threshold=0.5):
        super().__init__(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)
        self.density_threshold = density_threshold  # 密度阈值，用于决定是否调整体素大小

    @torch.no_grad()
    def forward(self, pts):
        """
        动态体素化前向传播
        :param pts: (N, C) 点云数据
        :return: 动态体素化后的点云
        """
        # 计算点云的密度
        voxel_grid = self.voxelize(pts)
        voxel_density = self.calculate_density(voxel_grid)

        # 根据密度调整体素大小
        adjusted_voxel_size = self.adjust_voxel_size(voxel_density)

        # 使用调整后的体素大小重新体素化
        adjusted_voxel_grid = self.voxelize(pts, voxel_size=adjusted_voxel_size)

        return adjusted_voxel_grid

    def calculate_density(self, voxel_grid):
        """
        计算体素网格的密度
        :param voxel_grid: (M, C) 体素网格
        :return: (M, ) 体素密度
        """
        voxel_density = torch.sum(voxel_grid, dim=1) / self.max_num_points
        return voxel_density

    def adjust_voxel_size(self, voxel_density):
        """
        根据体素密度调整体素大小
        :param voxel_density: (M, ) 体素密度
        :return: (3, ) 调整后的体素大小
        """
        # 如果密度低于阈值，则增大体素大小
        if torch.mean(voxel_density) < self.density_threshold:
            adjusted_voxel_size = self.voxel_size * 2
        else:
            adjusted_voxel_size = self.voxel_size
        return adjusted_voxel_size

    def voxelize(self, pts, voxel_size=None):
        """
        体素化点云
        :param pts: (N, C) 点云数据
        :param voxel_size: (3, ) 体素大小，默认使用初始化时的体素大小
        :return: (M, C) 体素化后的点云
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        voxel_grid = super().forward(pts, voxel_size=voxel_size)
        return voxel_grid