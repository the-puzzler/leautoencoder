import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def skip(self, x):
        x = F.pad(x, (0, x.size(-1) % 2, 0, x.size(-2) % 2))
        x = F.pixel_unshuffle(x, downscale_factor=2)
        batch_size, channels, height, width = x.shape
        group_size = channels // self.out_channels
        x = x.view(batch_size, self.out_channels, group_size, height, width).mean(dim=2)
        return x

    def forward(self, x):
        return self.act(self.skip(x) + self.block(x))


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def skip(self, x):
        return self.skip_proj(F.interpolate(x, scale_factor=2, mode="nearest"))

    def forward(self, x):
        return self.act(self.skip(x) + self.block(x))


class FlattenMLPSummary(nn.Module):
    def __init__(self, in_channels, latent_channels, latent_hw):
        super().__init__()
        self.latent_channels = latent_channels
        self.flat_dim = in_channels * latent_hw * latent_hw
        self.proj = nn.Linear(self.flat_dim, latent_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.flatten(1)
        x = self.proj(x)
        return x.view(batch_size, self.latent_channels, 1, 1)


class FlattenMLPExpand(nn.Module):
    def __init__(self, latent_channels, out_channels, latent_hw):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.latent_hw = latent_hw
        self.flat_dim = out_channels * latent_hw * latent_hw
        self.proj = nn.Linear(latent_channels, self.flat_dim)
        self.act = nn.SiLU()

    def forward(self, z):
        batch_size = z.size(0)
        z = z.flatten(1)
        z = self.proj(z)
        z = z.view(batch_size, self.out_channels, self.latent_hw, self.latent_hw)
        return self.act(z)


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dim=64,
        latent_channels=16,
        output_size=32,
        pooled_latent=False,
        collapse_style="oneshot",
        expand_style="oneshot",
        pooled_map_channels=None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.pooled_latent = pooled_latent
        self.latent_hw = max(1, output_size // 8)
        self.output_size = output_size
        pooled_map_channels = hidden_dim if pooled_map_channels is None else pooled_map_channels
        self.spatial_latent_channels = pooled_map_channels if pooled_latent else latent_channels
        self.code_channels = latent_channels if pooled_latent else self.spatial_latent_channels
        skip_channels = 8 * hidden_dim
        if skip_channels % self.spatial_latent_channels != 0 or skip_channels % max(1, self.spatial_latent_channels // 4) != 0:
            raise ValueError(
                "incompatible hidden_dim/pooled_map_channels for residual skip paths: "
                f"hidden_dim={hidden_dim}, pooled_map_channels={self.spatial_latent_channels}"
            )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder_features = nn.Sequential(
            ResidualDownBlock(hidden_dim, hidden_dim * 2),
            ResidualDownBlock(hidden_dim * 2, self.spatial_latent_channels),
        )
        if pooled_latent:
            self.latent_summary = self.build_latent_summary(collapse_style)
            self.latent_expand = self.build_latent_expand(expand_style)
        self.latent_norm = nn.BatchNorm1d(self.code_channels)
        self.decoder = nn.Sequential(
            ResidualUpBlock(self.spatial_latent_channels, hidden_dim * 2),
            ResidualUpBlock(hidden_dim * 2, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def build_latent_summary(self, collapse_style):
        if collapse_style == "oneshot":
            return nn.Sequential(
                nn.Conv2d(self.spatial_latent_channels, self.spatial_latent_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.spatial_latent_channels, self.code_channels, kernel_size=self.latent_hw, stride=self.latent_hw),
            )
        elif collapse_style == "progressive":
            if self.latent_hw & (self.latent_hw - 1) != 0:
                raise ValueError(f"progressive collapse expects power-of-two latent_hw, got {self.latent_hw}")
            layers = [
                nn.Conv2d(self.spatial_latent_channels, self.spatial_latent_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            ]
            for _ in range(int(math.log2(self.latent_hw))):
                layers.append(ResidualDownBlock(self.spatial_latent_channels, self.spatial_latent_channels))
            if self.code_channels != self.spatial_latent_channels:
                layers.append(nn.Conv2d(self.spatial_latent_channels, self.code_channels, kernel_size=1))
            return nn.Sequential(*layers)
        elif collapse_style == "mlp":
            return FlattenMLPSummary(self.spatial_latent_channels, self.code_channels, self.latent_hw)
        raise ValueError(f"unknown collapse_style: {collapse_style}")

    def build_latent_expand(self, expand_style):
        if expand_style == "oneshot":
            return nn.Sequential(
                nn.ConvTranspose2d(self.code_channels, self.spatial_latent_channels, kernel_size=self.latent_hw, stride=self.latent_hw),
                nn.SiLU(),
            )
        elif expand_style == "progressive":
            if self.latent_hw & (self.latent_hw - 1) != 0:
                raise ValueError(f"progressive expand expects power-of-two latent_hw, got {self.latent_hw}")
            layers = []
            if self.code_channels != self.spatial_latent_channels:
                layers.extend([nn.Conv2d(self.code_channels, self.spatial_latent_channels, kernel_size=1), nn.SiLU()])
            for _ in range(int(math.log2(self.latent_hw))):
                layers.append(ResidualUpBlock(self.spatial_latent_channels, self.spatial_latent_channels))
            return nn.Sequential(*layers)
        elif expand_style == "mlp":
            return FlattenMLPExpand(self.code_channels, self.spatial_latent_channels, self.latent_hw)
        raise ValueError(f"unknown expand_style: {expand_style}")

    def encode(self, x, update_latent_norm=True):
        z = self.encoder_features(self.stem(x))
        if self.pooled_latent:
            z = self.latent_summary(z)
        batch_size, _, latent_h, latent_w = z.shape
        z = z.view(batch_size, self.code_channels, -1)
        if not update_latent_norm:
            z = F.batch_norm(
                z,
                None,
                None,
                self.latent_norm.weight,
                self.latent_norm.bias,
                training=True,
                momentum=0.0,
                eps=self.latent_norm.eps,
            )
        else:
            z = self.latent_norm(z)
        return z.view(batch_size, self.code_channels, latent_h, latent_w)

    def decode(self, z):
        if self.pooled_latent:
            z = self.latent_expand(z)
        x = self.head(self.decoder(z))
        if x.size(-1) != self.output_size or x.size(-2) != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
