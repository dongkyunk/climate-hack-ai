import torch
import torch.nn as nn
from einops import rearrange
from model.blocks import ConvBlock_2d, ResBlock_2d, LinearBlock, AdaINResBlock
from perceiver_pytorch import PerceiverIO


class CnnImageEncoder(nn.Module):
    def __init__(self):
        super(CnnImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            ConvBlock_2d(1, 32, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(32, 64, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(64, 128, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(128, 256, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(256, 512, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            *[ResBlock_2d(512, 512, padding_mode="reflect", norm="none", activation="relu") for _ in range(3)]
        )
        self.fc = nn.Linear(8*8*512, 512)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
        # pooled = self.pool(x)
        # return pooled.squeeze(), x



class CnnImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mu_mlp = nn.Sequential(
        #     LinearBlock(512, 1024, act='relu'),
        #     LinearBlock(1024, 4096, act='relu'),
        # )
        # self.sigma_mlp = nn.Sequential(
        #     LinearBlock(512, 1024, act='relu'),
        #     LinearBlock(1024, 4096, act='relu'),
        # )
        # self.adain_blocks = nn.ModuleList(
        #     [AdaINResBlock(idx, 512, 512, padding_mode="reflect")
        #      for idx in range(4)]
        # )
        self.fc = nn.Linear(512, 512*8*8)
        self.generator = nn.Sequential(
            *[ResBlock_2d(512, 512, padding_mode="reflect", norm="none",
                          activation="relu") for _ in range(3)],
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(512, 256, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(256, 128, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(128, 64, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(64, 1, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="tanh"),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.generator(x)
        return x.squeeze()
        # mu, sigma = self.mu_mlp(x), self.sigma_mlp(x)
        # for adain_block in self.adain_blocks:
        #     img_embd = adain_block(img_embd, mu, sigma)
        # return self.generator(img_embd).squeeze()


class ImageGPT(nn.Module):
    def __init__(self):
        super(ImageGPT, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(512, 8, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)
        self.input_embd = nn.Parameter(torch.randn(36, 512))
        self.mask = torch.triu(torch.ones(36, 36), diagonal=1)
        self.mask[:12] = 0
        self.mask[:,:12] = 0

    def forward(self, x):
        batch_size = x.size(0)
        x = x/1024
        skip = x[:,11:35]
        x = rearrange(x, 'b s h w -> (b s) 1 h w')
        x, image_feat = self.image_encoder(x)
        x = rearrange(x, '(b s) c -> b s c', b=batch_size)
        x = x + self.input_embd.repeat(batch_size, 1, 1)
        x = self.transformer(x, self.mask.type_as(x))[:, 11:35]
        x = rearrange(x, 'b s c  -> (b s) c', b=batch_size)

        image_feat = rearrange(image_feat, '(b s) c h w -> b s c h w', b=batch_size)
        image_feat = image_feat[:, 11:35] - image_feat[:, 10:34]
        image_feat = rearrange(image_feat, 'b s c h w -> (b s) c h w', b=batch_size)
        
        x = self.image_decoder(x, image_feat)
        x = rearrange(x, '(b s) h w  -> b s h w', b=batch_size)
        x = x + skip
        x = x*1024
        return x, skip*1024


class ImageTransformer(nn.Module):
    def __init__(self):
        super(ImageTransformer, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(128, 8, batch_first=True)
        self.fc_in = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 512)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)
        self.input_embd = nn.Parameter(torch.randn(16*16*9, 128))
        self.mask = torch.triu(torch.ones(16*16*9, 16*16*9), diagonal=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = rearrange(x, 'b c s h w -> b (s h w) c')
        x = self.fc_in(x)
        x = x + self.input_embd.repeat(batch_size, 1, 1)
        x = self.transformer(x, self.mask.type_as(x))
        x = self.fc_out(x)
        x = x.view(batch_size, 512, 9, 16, 16)
        return x


class CnnBaseline(nn.Module):
    def __init__(self):
        super(CnnBaseline, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock_2d(12, 64, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(64, 128, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(128, 256, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(256, 512, kernel_size=3, stride=2, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            *[ResBlock_2d(512, 512, padding_mode="reflect", norm="none", activation="relu") for _ in range(3)]
        )
        self.decoder = nn.Sequential(
            *[ResBlock_2d(512, 512, padding_mode="reflect", norm="none",
                          activation="relu") for _ in range(3)],
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(512, 256, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(256, 128, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock_2d(128, 64, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="relu"),
            ConvBlock_2d(64, 24, kernel_size=3, stride=1, padding=1,
                         padding_mode="reflect", norm="none", activation="tanh"),
        )


    def forward(self, x):
        x = x/1024
        skip = x[:,11].unsqueeze(1).repeat(1, 24, 1, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x*1024
        skip = skip*1024
        return x, skip




# class PerceiverModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.perceiver = PerceiverIO(
#             dim = 512,                    # dimension of sequence to be encoded
#             queries_dim = 32,            # dimension of decoder queries
#             logits_dim = 100,            # dimension of final logits
#             depth = 6,                   # depth of net
#             num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
#             latent_dim = 512,            # latent dimension
#             cross_heads = 1,             # number of heads for cross attention. paper said 1
#             latent_heads = 8,            # number of heads for latent self attention, 8
#             cross_dim_head = 64,         # number of dimensions per cross attention head
#             latent_dim_head = 64,        # number of dimensions per latent self attention head
#             weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
#         )