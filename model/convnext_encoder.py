import torch
from torch import nn
from inspect import isfunction
from einops import rearrange
from model.conv_gru import ConvGRU
# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, mult = 2, norm = True):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)
        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)


class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        dim=32,    
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
        channels=1,
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out, dim_in),
                ConvNextBlock(dim_in, dim_in),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, 1, 1)
        )

        self.conv_grus_enc = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.conv_grus_enc.append(
                ConvGRU(input_size=dim_out, hidden_sizes=dim_out, kernel_sizes=3, n_layers=12)
            )            

        self.conv_grus_dec = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.conv_grus_dec.append(
                ConvGRU(input_size=dim_out, hidden_sizes=dim_out, kernel_sizes=3, n_layers=24)
            )            


    def forward(self, x):
        h = []
        b = x.shape[0]
        x = rearrange(x, 'b t h w -> (b t) 1 h w')
        for convnext, convnext2, downsample in self.downs:
            x = convnext(x)
            x = convnext2(x)
            # x = attn(x)
            h.append(x)
            x = downsample(x)

        for i, conv_gru in enumerate(self.conv_grus_enc):
            h[i] = rearrange(h[i], '(b t) c h w -> t b c h w', b=b)
            h[i] = conv_gru(torch.zeros_like(h[i][0]), h[i])[-1]

        for i, conv_gru in enumerate(self.conv_grus_dec):
            h[i] = conv_gru(h[i])
            h[i] = torch.stack(h[i], dim = 0)
            h[i] = rearrange(h[i], 't b c h w -> (b t) c h w')

        x = torch.zeros_like(h[-1])
        for convnext, convnext2, upsample in self.ups:
            #x = torch.cat((x, h.pop()), dim=1)
            x = x + h.pop()
            x = convnext(x)
            x = convnext2(x)
            # x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = rearrange(x, '(b t) 1 h w -> b t h w', b=b)

        return x[:, :, 32:96, 32:96]
