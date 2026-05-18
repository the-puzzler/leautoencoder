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


def make_channel_mask(images, mask_ratio=0.7):
    batch_size, channels, _, _ = images.shape
    keep = (torch.rand(batch_size, channels, 1, 1, device=images.device) > mask_ratio).to(images.dtype)
    return keep


def make_patch_mask(images, patch_size=4, mask_ratio=0.7):
    batch_size, _, height, width = images.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    patches_h = height // patch_size
    patches_w = width // patch_size
    num_patches = patches_h * patches_w
    num_masked = int(num_patches * mask_ratio)
    patch_mask = torch.ones(batch_size, num_patches, device=images.device, dtype=images.dtype)

    for i in range(batch_size):
        masked_idx = torch.randperm(num_patches, device=images.device)[:num_masked]
        patch_mask[i, masked_idx] = 0

    patch_mask = patch_mask.view(batch_size, patches_h, patches_w)
    patch_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return patch_mask.unsqueeze(1)
