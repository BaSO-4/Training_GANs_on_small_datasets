import math
import torch
from torch import nn
from torch.nn import functional as F

def make_noise(batch, resolution, device):
    return torch.randn(batch, 1, resolution, resolution, device=device)

blur_kernel = [1, 3, 3, 1]

def normalize_kernel(kernel):
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel / kernel.sum()
    return kernel

blur_kernel = normalize_kernel(blur_kernel)
blur_kernel = blur_kernel[None, None]


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim,demodulate=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate

        fan_in = in_channels * kernel_size * kernel_size
        self.scale = 1 / math.sqrt(fan_in)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        N, C, H, W = x.shape
        style = self.style(style).view(N, 1, C, 1, 1)
        weight = self.weight * self.scale
        weight = weight * style
        if self.demodulate:
            demod = torch.rsqrt((weight ** 2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(N, self.out_channels, 1, 1, 1)
        weight = weight.view(N * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, N * self.in_channels, H, W)
        padding = self.kernel_size // 2
        out = F.conv2d(x, weight, padding=padding, groups=N)
        out = out.view(N, self.out_channels, H, W)
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight.view(1, -1, 1, 1) * noise


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.blur = Blur(blur_kernel)
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.noise1 = NoiseInjection(out_channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        self.noise2 = NoiseInjection(out_channels)
        self.act2 = nn.LeakyReLU(0.2)
        self.to_rgb = ModulatedConv2d(out_channels, 3, 1, style_dim, demodulate=False)

    def forward(self, x, style, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.blur(x)
        x = self.conv1(x, style)
        x = self.noise1(x, noise)
        x = self.act1(x)
        x = self.conv2(x, style)
        x = self.noise2(x, noise)
        x = self.act2(x)
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)

    def forward(self, x, style):
        return self.conv(x, style)


class Blur(nn.Module):
    def __init__(self, kernel, pad=(1,1)):
        super().__init__()
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        kH, kW = self.kernel.shape[-2:]
        pad_h, pad_w = self.pad
        if (H + 2 * pad_h) < kH or (W + 2 * pad_w) < kW:
            return x
        kernel = self.kernel.expand(C, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.pad, groups=C)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, style_dim, num_layers=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        w = self.mapping(z)
        return w


class Generator(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, channels=[512,512,256,128,64,32,16,8]):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, style_dim)
        self.constant_input = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.initial_noise = NoiseInjection(channels[0])
        self.initial_act = nn.LeakyReLU(0.2)
        self.initial_conv = ModulatedConv2d(channels[0], channels[0], 3, style_dim)
        self.to_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

        in_c = channels[0]
        for out_c in channels:
            self.blocks.append(GenBlock(in_c, out_c, style_dim, upsample=(in_c!=out_c)))
            self.to_rgb_layers.append(ToRGB(out_c, style_dim))
            in_c = out_c

    def forward(self, z):
        styles = self.mapping(z)
        batch = z.shape[0]
        x = self.constant_input.repeat(batch, 1, 1, 1)
        x = self.initial_noise(x)
        x = self.initial_act(x)
        x = self.initial_conv(x, styles)
        x = self.initial_act(x)
        rgb = None
        for block, to_rgb in zip(self.blocks, self.to_rgb_layers):
            x = block(x, styles)
            rgb_new = to_rgb(x, styles)
            if rgb is None:
                rgb = rgb_new
            else:
                target_h, target_w = rgb_new.shape[-2:]
                rgb = F.interpolate(rgb, size=(target_h, target_w), mode='nearest') + rgb_new
        return rgb


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        if downsample:
            self.blur = Blur(blur_kernel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        if self.downsample:
            x = self.blur(x)
            H, W = x.shape[-2], x.shape[-1]
            if H >= 2 and W >= 2:
                x = F.avg_pool2d(x, kernel_size=2)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels=[8,16,32,64,128,256,512,512]):
        super().__init__()
        self.from_rgb = nn.Conv2d(3, channels[-1], kernel_size=1)
        self.blocks = nn.ModuleList()
        in_c = channels[-1]
        for out_c in reversed(channels[:-1]):
            self.blocks.append(DiscBlock(in_c, out_c, downsample=True))
            in_c = out_c
        self.final_conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        self.final_act  = nn.LeakyReLU(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(in_c, 1)

    def forward(self, x):
        x = self.from_rgb(x)           
        for block in self.blocks:
            x = block(x)               
        x = self.final_conv(x)
        x = self.final_act(x)
        x = self.avgpool(x)           
        x = x.view(x.size(0), -1)     
        x = self.fc(x)                
        return x


def get_generator(l_dim=512, s_dim=512):
    return Generator(latent_dim=l_dim, style_dim=s_dim)


def get_discriminator():
    return Discriminator()
