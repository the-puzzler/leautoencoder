import torch

def make_synthetic(n=12000, size=28, seed=0):
    """Random filled circles, rings, and rectangles on black backgrounds."""
    rng = torch.Generator().manual_seed(seed)
    imgs = torch.zeros(n, 1, size, size)
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    for i in range(n):
        mode = torch.randint(0, 3, (1,), generator=rng).item()
        cx = torch.randint(5, size - 5, (1,), generator=rng).item()
        cy = torch.randint(5, size - 5, (1,), generator=rng).item()
        r = torch.randint(3, 9, (1,), generator=rng).item()
        v = (torch.rand(1, generator=rng) * 0.7 + 0.3).item()
        if mode == 0:
            mask = ((xs - cx).float() ** 2 + (ys - cy).float() ** 2) <= r ** 2
        elif mode == 1:
            d = ((xs - cx).float() ** 2 + (ys - cy).float() ** 2).sqrt()
            mask = (d >= r - 1.5) & (d <= r + 1.5)
        else:
            mask = (xs >= cx - r) & (xs <= cx + r) & (ys >= cy - r) & (ys <= cy + r)
        imgs[i, 0][mask] = v
    return imgs
