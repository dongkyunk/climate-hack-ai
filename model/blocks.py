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
