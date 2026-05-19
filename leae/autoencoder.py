import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 2:
            raise ValueError("ResidualDownBlock expects stride=2")
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def skip(self, x):
        pad_h = x.size(-2) % 2
        pad_w = x.size(-1) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = F.pixel_unshuffle(x, downscale_factor=2)
        batch_size, channels, height, width = x.shape
        if channels % self.out_channels != 0:
            raise ValueError(f"cannot average {channels} channels into {self.out_channels}")
        group_size = channels // self.out_channels
        x = x.view(batch_size, self.out_channels, group_size, height, width).mean(dim=2)
        return x

    def forward(self, x):
        return self.act(self.skip(x) + self.block(x))


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def skip(self, x):
        x = F.pixel_shuffle(x, upscale_factor=2)
        channels = x.size(1)
        if self.out_channels % channels != 0:
            raise ValueError(f"cannot duplicate {channels} channels into {self.out_channels}")
        return x.repeat(1, self.out_channels // channels, 1, 1)

    def forward(self, x):
        return self.act(self.skip(x) + self.block(x))


class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, latent_channels=16, output_size=32):
        super().__init__()
        assert latent_channels % 4 == 0 and (8 * hidden_dim) % latent_channels == 0 and (8 * hidden_dim) % (latent_channels // 4) == 0, "incompatible hidden_dim/latent_channels for residual skip paths"
        self.latent_channels = latent_channels
        self.latent_hw = 4
        self.output_size = output_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder_features = nn.Sequential(
            ResidualDownBlock(hidden_dim, hidden_dim * 2, stride=2),
            ResidualDownBlock(hidden_dim * 2, latent_channels, stride=2),
        )
        self.flat_latent_dim = latent_channels * self.latent_hw * self.latent_hw
        self.latent_norm = nn.BatchNorm1d(self.flat_latent_dim)
        self.decoder = nn.Sequential(
            ResidualUpBlock(latent_channels, hidden_dim * 2),
            ResidualUpBlock(hidden_dim * 2, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, update_latent_norm=True):
        z = self.encoder_features(self.stem(x))
        batch_size = z.size(0)
        z = z.flatten(1)
        if isinstance(self.latent_norm, nn.BatchNorm1d) and not update_latent_norm:
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
        return z.view(batch_size, self.latent_channels, self.latent_hw, self.latent_hw)

    def decode(self, z):
        x = self.head(self.decoder(z))
        if x.size(-1) != self.output_size or x.size(-2) != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
