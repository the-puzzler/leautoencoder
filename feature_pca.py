import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from leae.autoencoder import Autoencoder
from leae.prep_data import load_data


def make_model():
    return Autoencoder(in_channels=3, hidden_dim=128, latent_channels=32, output_size=128)


def find_latest_checkpoint(log_root: str | Path = "logs") -> Path:
    checkpoints = sorted(Path(log_root).rglob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_*.pt files found under {log_root}")
    return checkpoints[-1]


def extract_feature_map(model, images, layer):
    if layer == "latent":
        return model.encode(images)
    stem = model.stem(images)
    if layer == "stem":
        return stem
    x = stem
    for idx, block in enumerate(model.encoder_features):
        x = block(x)
        if layer == f"block{idx + 1}":
            return x
    raise ValueError(f"unknown layer: {layer}")


def collect_pca_features(model, loader, device, max_batches, layer):
    feature_chunks = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            z = extract_feature_map(model, images, layer)
            features = z.permute(0, 2, 3, 1).reshape(-1, z.size(1))
            feature_chunks.append(features)
    if not feature_chunks:
        raise RuntimeError("no features collected for PCA")
    return torch.cat(feature_chunks, dim=0)


def fit_pca(features):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    _, _, v = torch.pca_lowrank(centered, q=3, center=False)
    return mean, v[:, :3]


def project_feature_map(z, mean, components):
    b, c, h, w = z.shape
    features = z.permute(0, 2, 3, 1).reshape(-1, c)
    proj = (features - mean) @ components
    return proj.view(b, h, w, 3).permute(0, 3, 1, 2)


def project_feature_map_per_image(z):
    batch_proj = []
    for feature_map in z:
        c, h, w = feature_map.shape
        features = feature_map.permute(1, 2, 0).reshape(-1, c)
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        _, _, v = torch.pca_lowrank(centered, q=3, center=False)
        proj = centered @ v[:, :3]
        batch_proj.append(proj.view(h, w, 3).permute(2, 0, 1))
    return torch.stack(batch_proj, dim=0)


def normalize_pca_rgb(pca_rgb):
    flat = pca_rgb.permute(1, 0, 2, 3).reshape(3, -1)
    lo = torch.quantile(flat, 0.01, dim=1).view(1, 3, 1, 1)
    hi = torch.quantile(flat, 0.99, dim=1).view(1, 3, 1, 1)
    denom = (hi - lo).clamp_min(1e-6)
    return ((pca_rgb - lo) / denom).clamp(0.0, 1.0)


def make_visual_grid(images, recon, pca_rgb, per_image_pca_rgb):
    pca_up = F.interpolate(pca_rgb, size=images.shape[-2:], mode="nearest")
    per_image_pca_up = F.interpolate(per_image_pca_rgb, size=images.shape[-2:], mode="nearest")
    return torch.cat([images, recon, pca_up, per_image_pca_up], dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--output-dir", type=str, default="diagnostics/feature_pca")
    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--pca-batches", type=int, default=16)
    parser.add_argument("--layer", type=str, default="block1", choices=["stem", "block1", "block2", "latent"])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(args.log_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loader, test_loader = load_data(
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
        dataset_name=args.dataset,
    )
    pca_loader = train_loader if args.split == "train" else test_loader

    features = collect_pca_features(model, pca_loader, device, max_batches=args.pca_batches, layer=args.layer)
    mean, components = fit_pca(features)

    vis_images = None
    vis_recon = None
    vis_pca = None
    vis_per_image_pca = None
    with torch.no_grad():
        for images, _ in pca_loader:
            images = images.to(device, non_blocking=True)
            z = model.encode(images)
            recon = model.decode(z)
            feature_map = extract_feature_map(model, images, args.layer)
            pca_rgb = project_feature_map(feature_map, mean, components)
            per_image_pca_rgb = project_feature_map_per_image(feature_map)
            vis_images = images[: args.num_images].cpu()
            vis_recon = recon[: args.num_images].cpu()
            vis_pca = pca_rgb[: args.num_images].cpu()
            vis_per_image_pca = per_image_pca_rgb[: args.num_images].cpu()
            break

    if vis_images is None:
        raise RuntimeError("no images available for visualization")

    vis_pca = normalize_pca_rgb(vis_pca)
    vis_per_image_pca = normalize_pca_rgb(vis_per_image_pca)
    grid = make_visual_grid(vis_images, vis_recon, vis_pca, vis_per_image_pca)

    out_dir = Path(args.output_dir) / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_pca_rgb.png"
    save_image(grid, out_path, nrow=args.num_images, normalize=False)

    print(f"checkpoint={checkpoint_path}")
    print(f"saved={out_path}")
    print(f"images={tuple(vis_images.shape)} pca_map={tuple(vis_pca.shape)} per_image_pca_map={tuple(vis_per_image_pca.shape)} layer={args.layer}")


if __name__ == "__main__":
    main()
