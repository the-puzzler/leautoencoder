import torch


def make_pixel_mask(images, mask_ratio=0.7):
    batch_size, _, height, width = images.shape
    num_pixels = height * width
    num_masked = int(num_pixels * mask_ratio)
    pixel_mask = torch.ones(batch_size, num_pixels, device=images.device, dtype=images.dtype)

    for i in range(batch_size):
        masked_idx = torch.randperm(num_pixels, device=images.device)[:num_masked]
        pixel_mask[i, masked_idx] = 0

    return pixel_mask.view(batch_size, 1, height, width)


def apply_mask(images, mask):
    return images * mask
