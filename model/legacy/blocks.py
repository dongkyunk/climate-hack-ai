import torch
import torch.nn as nn


def get_activation(activation: str):
    if activation == "none":
        return lambda x: x
    elif activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsuported activation: {activation}")


def get_normalization_1d(norm: str, norm_dim: int):
    if norm == "none":
        return lambda x: x
    elif norm == "bn":
        return nn.BatchNorm1d(num_features=norm_dim)
    elif norm == "in":
        return nn.InstanceNorm1d(num_features=norm_dim)
    else:
        raise ValueError(f"Unsupported normalization: {norm}")


def get_normalization_2d(norm, norm_dim = None):
    if norm == "none":
        return lambda x: x
    elif norm == "bn":
        return nn.BatchNorm2d(num_features=norm_dim)
    elif norm == "in":
        return nn.InstanceNorm2d(num_features=norm_dim)
    elif norm == "adain":
        return AdaptiveInstanceNorm2D
    elif norm == "ln":
        return nn.LayerNorm(norm_dim)
    else:
        raise ValueError(f"Unsupported normalization: {norm}")


def AdaptiveInstanceNorm2D(x, adain_weight, adain_bias):
    size = x.size()

    # feature의 평균과 분산 계산
    feature_mean, feature_std = calc_mean_std(x)

    # mean, std를 이용해서 정규화 수행
    normalized_feat = (x - feature_mean.expand(size)) / feature_std.expand(size)  # (N, C, H, W)

    # normalize 이후에 adain statistics를 가지도록 설정
    adain_feat = normalized_feat * adain_weight.expand(size) + adain_bias.expand(size)
    return adain_feat


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)  # check feature map size
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps # prevent varaince to be 0
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_sn=False, use_bias=True, norm="none", act='relu'):
        super(LinearBlock, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)
        
        self.normalization = get_normalization_1d(norm=norm, norm_dim=out_dim)
        self.activation = get_activation(act)

    
    def forward(self, x):
        x = self.fc(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ConvBlock_2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode="zeros", use_sn=False, norm="none", activation="none"):
        super(ConvBlock_2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,            
        )
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        
        self.normalization = get_normalization_2d(norm, out_dim)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ResBlock_2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode="zeros", use_sn=False, norm="none", activation="relu"):
        super(ResBlock_2d, self).__init__()
        self.res_block = nn.Sequential(
                ConvBlock_2d(in_dim, out_dim, kernel_size, stride, padding, padding_mode, use_sn, norm, activation),
                ConvBlock_2d(out_dim, out_dim, kernel_size, stride, padding, padding_mode, use_sn, norm, "none")
        ) 

    def forward(self, x):
        _x = self.res_block(x)
        return x + _x


class AdaINResBlock(nn.Module):
    def __init__(self, adain_index, in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode="zeros", use_sn=False, norm="adain", activation="relu"):
        super(AdaINResBlock, self).__init__()

        self.dim_adain = out_dim
        self.adain_index = adain_index

        self.adaptive_res_block1 = ConvBlock_2d(in_dim, out_dim, kernel_size, stride, padding, padding_mode, use_sn)
        self.adaptive_res_block2 = ConvBlock_2d(in_dim, out_dim, kernel_size, stride, padding, padding_mode, use_sn)

        self.relu = get_activation('relu')
    

    def forward(self, x, mu, sigma):
        idx_head1 = 2 * self.adain_index * self.dim_adain
        mu_1 = mu[..., idx_head1 : idx_head1 + self.dim_adain].unsqueeze(-1).unsqueeze(-1)
        sigma_1 = sigma[..., idx_head1 : idx_head1 + self.dim_adain].unsqueeze(-1).unsqueeze(-1)

        idx_head2 = (2 * self.adain_index + 1) * self.dim_adain
        mu_2 = mu[..., idx_head2 : idx_head2 + self.dim_adain].unsqueeze(-1).unsqueeze(-1)
        sigma_2 = sigma[..., idx_head2 : idx_head2 + self.dim_adain].unsqueeze(-1).unsqueeze(-1)

        _x = self.adaptive_res_block1(x)
        _x = get_normalization_2d("adain")(_x, mu_1, sigma_1)
        _x = self.relu(_x)
        _x = self.adaptive_res_block2(_x)
        _x = get_normalization_2d("adain")(_x, mu_2, sigma_2)
        return x + _x


class MLP_AdaIN(nn.Module):
    def __init__(self, n_adain, in_dim, hid_dim=512, channel=512):
        super(MLP_AdaIN, self).__init__()

        self.mlp_layer = nn.Sequential(
            LinearBlock(in_dim, hid_dim, use_sn=False, act='relu'),
            LinearBlock(hid_dim, hid_dim, use_sn=False, act='relu')
        )
        self.mu = LinearBlock(int(hid_dim*2), n_adain * channel, use_sn=False)
        self.sigma = LinearBlock(int(hid_dim*2), n_adain * channel, use_sn=False)
        

    def forward(self, x, font_idx_embd):
        x = self.mlp_layer(x)
        x = torch.cat([x, font_idx_embd], dim=-1)

        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma



"""
Patch Encoders / Decoders as used by DeepMind in their sonnet repo example:
https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

# -----------------------------------------------------------------------------

class DeepMindResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            DeepMindResBlock(2*n_hid, 2*n_hid//4),
            DeepMindResBlock(2*n_hid, 2*n_hid//4),
        )

        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            DeepMindResBlock(2*n_hid, 2*n_hid//4),
            DeepMindResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)