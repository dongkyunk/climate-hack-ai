import torch
import timm
import torch.nn as nn
from einops import rearrange
from model.blocks import ConvBlock_2d, ResBlock_2d


class CnnImageEncoder(nn.Module):
    def __init__(self):
        super(CnnImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            ConvBlock_2d(1, 64, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(64, 128, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(128, 256, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(256, 512, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            *[ResBlock_2d(512, 512, padding_mode="reflect", norm="none", activation="relu") for _ in range(3)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        return x.squeeze()


class CnnImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 8*8*512)
        self.generator = nn.Sequential(
            *[ResBlock_2d(512, 512, padding_mode="reflect", norm="none",
                          activation="relu") for _ in range(3)],
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(512, 256, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="bn", activation="relu"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(256, 128, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="bn", activation="relu"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(128, 64, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="bn", activation="relu"),
            ConvBlock_2d(64, 1, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="sigmoid"),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)
        return self.generator(x).squeeze()


class ImageGPT(nn.Module):
    def __init__(self):
        super(ImageGPT, self).__init__()
        self.image_encoder = CnnImageEncoder()
        self.image_decoder = CnnImageDecoder()
        transformer_layer = nn.TransformerEncoderLayer(512, 8)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)
        self.input_embd = nn.Parameter(torch.randn(12, 512))
        self.output_embd = nn.Parameter(torch.randn(24, 512))

    def forward(self, x):
        batch_size = x.size(0)
        x = x/1024
        x = rearrange(x, 'b s h w -> (b s) 1 h w')
        x = self.image_encoder(x)
        x = rearrange(x, '(b s) c -> b s c', b=batch_size)
        x = x + self.input_embd.repeat(batch_size, 1, 1)
        x = torch.cat([x, self.output_embd.repeat(batch_size, 1, 1)], dim=1)
        x = self.transformer(x)[:, 12:36]
        x = rearrange(x, 'b s c  -> (b s) c', b=batch_size)
        x = self.image_decoder(x)
        x = rearrange(x, '(b s) h w  -> b s h w', b=batch_size)
        return x * 1024