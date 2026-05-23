import torch


def latent_to_sigreg_samples(latent):
    if latent.ndim != 4:
        raise ValueError(f"expected latent with shape [batch, channels, height, width], got {tuple(latent.shape)}")
    batch_size, channels, height, width = latent.shape
    if height == 1 and width == 1:
        return latent.view(batch_size, channels)
    return latent.permute(2, 3, 0, 1).reshape(height * width, batch_size, channels)


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, *projs):
        if len(projs) == 1:
            proj = projs[0]
        else:
            proj = torch.stack(projs, dim=1)
        if proj.ndim not in {2, 3}:
            raise ValueError(f"expected projection tensor with 2 or 3 dims, got shape {tuple(proj.shape)}")

        A = torch.randn(proj.size(-1), 256, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
