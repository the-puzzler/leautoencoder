import torch
import torch.nn.functional as F


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


def fill_mask_with_image_average(images, mask):
    mask = mask.to(device=images.device, dtype=images.dtype)
    if mask.shape != images.shape[:1] + (1,) + images.shape[2:]:
        raise ValueError(
            f"expected mask shape {(images.size(0), 1, images.size(2), images.size(3))}, got {tuple(mask.shape)}"
        )
    image_average = images.mean(dim=(-2, -1), keepdim=True)
    return images * mask + image_average * (1.0 - mask)


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


def sample_square_crop_boxes(images, crop_ratio=0.5):
    _, _, height, width = images.shape
    crop_size = max(1, int(min(height, width) * crop_ratio))
    max_top = height - crop_size
    max_left = width - crop_size
    top = torch.randint(max_top + 1, (images.size(0),), device=images.device)
    left = torch.randint(max_left + 1, (images.size(0),), device=images.device)
    return top, left, crop_size


def apply_square_crop(images, top, left, crop_size):
    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")

    ys = torch.linspace(0, crop_size - 1, height, device=device, dtype=dtype)
    xs = torch.linspace(0, crop_size - 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    top = top.to(device=device, dtype=dtype).view(batch_size, 1, 1)
    left = left.to(device=device, dtype=dtype).view(batch_size, 1, 1)
    sample_y = top + grid_y.unsqueeze(0)
    sample_x = left + grid_x.unsqueeze(0)

    if height > 1:
        sample_y = (sample_y / (height - 1)) * 2 - 1
    else:
        sample_y = torch.zeros_like(sample_y)
    if width > 1:
        sample_x = (sample_x / (width - 1)) * 2 - 1
    else:
        sample_x = torch.zeros_like(sample_x)

    grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)


def apply_square_crop_raw(images, top, left, crop_size):
    batch_size, _, height, width = images.shape
    device = images.device

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if crop_size > height or crop_size > width:
        raise ValueError(f"crop_size {crop_size} exceeds image size {(height, width)}")

    top = top.to(device=device, dtype=torch.long)
    left = left.to(device=device, dtype=torch.long)
    offsets = torch.arange(crop_size, device=device)
    y_idx = top[:, None] + offsets[None, :]
    x_idx = left[:, None] + offsets[None, :]
    gather_y = y_idx[:, None, :, None].expand(-1, images.size(1), -1, width)
    cropped = torch.gather(images, dim=2, index=gather_y)
    gather_x = x_idx[:, None, None, :].expand(-1, images.size(1), crop_size, -1)
    return torch.gather(cropped, dim=3, index=gather_x)
