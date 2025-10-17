import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.registry import MODELS
from mmdet3d.datasets import build_dataloader, build_dataset
import os


def extract_pillar_features(config_path, checkpoint_path, pcd_path):
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    model.eval()

    # 假设你已经注册 PillarHistEncoder 和 PillarFusionEncoder
    data = dict(points=[torch.from_numpy(np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4))])
    data = model.data_preprocessor(data, training=False)

    voxel_dict = data['inputs']['voxels']
    voxel_features = voxel_dict['voxels']
    num_points = voxel_dict['num_points']
    coors = voxel_dict['coors']

    # 提取两种特征
    pfn_encoder = model.voxel_encoder.pfn
    hist_encoder = model.voxel_encoder.hist

    with torch.no_grad():
        pfn_feat = pfn_encoder(voxel_features, num_points, coors).cpu()
        hist_feat = hist_encoder(voxel_features, num_points, coors).cpu()

    return pfn_feat, hist_feat, coors.cpu()


def plot_feature_heatmap(pfn_feat, hist_feat, coors, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 归一化
    pfn_norm = (pfn_feat - pfn_feat.min()) / (pfn_feat.max() - pfn_feat.min())
    hist_norm = (hist_feat - hist_feat.min()) / (hist_feat.max() - hist_feat.min())

    # 投影到2D平面（BEV）
    x = coors[:, 3].numpy()
    y = coors[:, 2].numpy()

    # 选择第0通道作为示例
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=x, y=y, hue=pfn_norm[:, 0], palette='viridis', s=20)
    plt.title('PillarFeatureNet Heatmap (Channel 0)')
    plt.gca().set_aspect('equal')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=x, y=y, hue=hist_norm[:, 0], palette='plasma', s=20)
    plt.title('PillarHistEncoder Heatmap (Channel 0)')
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pillar_feature_comparison.png')
    plt.close()
    print(f"Saved heatmap to {output_dir}/pillar_feature_comparison.png")


if __name__ == '__main__':
    config = 'configs/pointpillars/pointpillars_kitti-3class-fusion.py'
    checkpoint = 'work_dirs/pointpillars_fusion/latest.pth'
    pcd = 'demo/data/kitti/kitti_000008.bin'
    out_dir = 'work_dirs/viz'

    pfn, hist, coors = extract_pillar_features(config, checkpoint, pcd)
    plot_feature_heatmap(pfn, hist, coors, out_dir)