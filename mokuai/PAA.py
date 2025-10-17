import torch
from torch import nn
import torch.nn.functional as F

class PCAttention(nn.Module):
    def __init__(self, gate_channels, reduction_rate, pool_types=['max', 'mean'], activation=nn.ReLU(),
                 channel_mean=False):
        super(PCAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_rate),
            activation,
            nn.Linear(gate_channels // reduction_rate, gate_channels)
        )
        self.pool_types = pool_types
        print('self.pool_types', self.pool_types)

        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None)) #if not channel_mean else GetChannelMean(keepdim=True)
        print('self.max_pool, self.avg_pool',self.max_pool, self.avg_pool)
    def forward(self, x):
        '''
        # shape [n_voxels, channels, n_points] for point-wise attention
        # shape [n_voxels, n_points, channels] for channels-wise attention
        '''
        attention_sum = None
        for pool_type in self.pool_types:
            # [n_voxels, 1, n_points]
            if pool_type == 'max':
                max_pool = self.max_pool(x)
                attention_raw = self.mlp(max_pool)
            elif pool_type == 'mean':
                avg_pool = self.avg_pool(x)
                attention_raw = self.mlp(avg_pool)
            if attention_sum is None:
                attention_sum = attention_raw
            else:
                attention_sum += attention_raw
        scale = torch.sigmoid(attention_sum).permute(0, 2, 1)
        # scale = attention_sum.permute(0, 2, 1)
        return scale


class GetChannelMean(nn.Module):
    def __init__(self, keepdim=True):
        super(GetChannelMean, self).__init__()
        self.keepdim = keepdim

    def forward(self, x):  # [n_voxels, n_points, n_channels]
        x = x.permute(0, 2, 1)  # [n_voxels, n_channels, n_points]
        sum = x.sum(dim=-1)
        cnt = (x != 0).sum(-1).type(torch.float)
        cnt[cnt == 0] = 1  # replace 0 to 1 to avoid divide by 0
        mean = torch.true_divide(sum, cnt)  # [n_voxels, n_channels]
        if self.keepdim:
            return mean.unsqueeze(-1).permute(0, 2, 1)  # [n_voxels, n_channels, 1] -> [n_voxels, 1, n_channels]
        return mean


class TaskAware(nn.Module):
    def __init__(self, channels, reduction_rate=8, k=2, pool_types=['max','mean']):
        super(TaskAware, self).__init__()
        self.channels = channels
        self.k = k
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction_rate),
                                nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction_rate, 2 * k * channels),
                                )
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        # self.get_mean = GetChannelMean(keepdim=False)

    def get_relu_coefs(self, x):  # [n_voxels, n_points, n_channels]
        mean = torch.mean(x, dim=1)
        theta_mean = self.fc(mean)
        theta = theta_mean
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x): # [n_voxels, n_points, n_channels]
        assert x.shape[2] == self.channels
        theta = self.get_relu_coefs(x)  # [n_voxels, n_channels * 2 * k]
        relu_coefs = theta.view(-1, self.channels, 2 * self.k) * self.lambdas + self.init_v

        # BxCxL -> LxCxBx1
        x_perm = x.permute(1,0,2).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0,1)
        return result


class PAASubModules(nn.Module):
    def __init__(self, n_points, n_channels, reduction_rate=4, channel_att=True,
                 point_att=True, task_aware=True, pool_types=['max', 'mean']):
        super(PAASubModules, self).__init__()
        self.name = 'PAASubModules'
        self.use_channel_att = channel_att
        self.use_point_att = point_att
        self.use_task_aware = task_aware
        print('channel_att, point_att, task_aware',channel_att,point_att, task_aware)
        if point_att:
            self.point_att = PCAttention(n_points, reduction_rate=reduction_rate, activation=nn.ReLU(),
                                         pool_types=pool_types)
        if channel_att:
            self.channel_att = PCAttention(n_channels, reduction_rate=reduction_rate, activation=nn.ReLU(),
                                           channel_mean=True, pool_types=pool_types)
        if task_aware:
            self.task_aware = TaskAware(n_channels, reduction_rate=reduction_rate, pool_types=pool_types)

    def forward(self, x):  # shape [n_voxels, n_points, n_channels]

        point_weight = self.point_att(x.permute(0, 2, 1)) \
            if self.use_point_att else torch.tensor(1.)  # [n_voxels, n_points, 1]
        channel_weight = self.channel_att(x).permute(0, 2, 1) \
            if self.use_channel_att else torch.tensor(1.)  # [n_voxels, 1, n_channels]

        if torch.any(point_weight != 1.) or torch.any(channel_weight != 1.):
            beta = torch.mul(channel_weight, point_weight)
            attention = beta
            x = x * attention  # shape [n_voxels, n_points, n_channels]

        if self.use_task_aware:
            x = self.task_aware(x)  # shape [n_voxels, n_points, n_channels]

        return x


class AttentionModule(nn.Module):
    def __init__(self, n_points, n_channels, reduction_rate=4):
        super(AttentionModule, self).__init__()
        self.name = 'PFNLayerDA'
        self.point_att = PCAttention(n_points, reduction_rate=reduction_rate, activation=nn.ReLU())
        self.channel_att = PCAttention(n_channels, reduction_rate=reduction_rate, activation=nn.ReLU(),
                                       channel_mean=True)
        self.task_aware = TaskAware(n_channels, reduction_rate=reduction_rate)

    def forward(self, x):  # shape [n_voxels, n_points, n_channels]
        point_weight = self.point_att(x.permute(0, 2, 1))  # [n_voxels, n_points, 1]
        channel_weight = self.channel_att(x).permute(0, 2, 1)  # [n_voxels, 1, n_channels]
        beta = torch.mul(channel_weight, point_weight)
        attention = torch.sigmoid(beta)
        x = x * attention  # shape [n_voxels, n_points, n_channels]
        out = self.task_aware(x).permute(0, 2, 1)  # shape [n_voxels, n_points, n_channels]
        return out


class PAAModule(nn.Module):
    def __init__(self, dim_channels=9, dim_points=100, reduction_rate=8, boost_channels=64,
                 residual=False, relu=True, channel_att=True, point_att=True, task_aware=True,
                 pool_types=['max', 'mean']):
        super(PAAModule, self).__init__()
        self.residual = residual
        self.att_module1 = PAASubModules(n_points=dim_points, n_channels=dim_channels,
                                         reduction_rate=reduction_rate,
                                         channel_att=channel_att,
                                         point_att=point_att,
                                         task_aware=task_aware,
                                         pool_types=pool_types)  # linear last

        if residual:
            self.fc1 = nn.Sequential(
                nn.Linear(dim_channels, boost_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(dim_channels * 2, boost_channels),
                nn.ReLU(inplace=True) if relu else nn.Identity(),
            )

    def forward(self, x):  # [n_voxels, n_points, n_channels]
        out1 = self.att_module1(x)
        if self.residual:
            out1 = out1 + x
        else:
            out1 = torch.cat([out1, x], dim=2)  # [n_voxels, n_points, 2*n_channels]
        out1 = self.fc1(out1)  # Linear last

        return out1