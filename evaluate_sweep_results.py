import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

from leae.autoencoder import Autoencoder
from leae.prep_data import load_data


def make_model():
    return Autoencoder(in_channels=3, hidden_dim=64, latent_channels=256, output_size=32)


def find_run_dirs(root: Path):
    return sorted(path for path in root.glob("*/*") if path.is_dir())


def find_latest_checkpoint(run_dir: Path):
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {run_dir}")
    return checkpoints[-1]


def find_latest_test_image(run_dir: Path):
    images = sorted((run_dir / "reconstructions").glob("test_*.png"))
    return images[-1] if images else None


def load_config(run_dir: Path):
    config_path = run_dir / "config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}


def build_feature_extractor(device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.eval()
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"}).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return extractor, mean, std


def inception_features(extractor, mean, std, images):
    images = ((images.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    images = (images - mean) / std
    features = extractor(images)["features"].flatten(1)
    return features


def covariance(features):
    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.T @ centered / max(features.size(0) - 1, 1)
    return mean, cov


def matrix_sqrt_psd(matrix):
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(0).sqrt()
    return (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    sigma1_sqrt = matrix_sqrt_psd(sigma1)
    middle = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    covmean = matrix_sqrt_psd(middle)
    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid.clamp_min(0).cpu())


def evaluate_run(run_dir: Path, device, test_loader, extractor, mean, std, max_batches: int | None):
    checkpoint_path = find_latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = make_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_mse = 0.0
    total_count = 0
    real_features = []
    recon_features = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            recon = model(images)

            batch_size_actual = images.size(0)
            total_mse += F.mse_loss(recon, images, reduction="sum").item()
            total_count += images.numel()

            real_features.append(inception_features(extractor, mean, std, images).cpu())
            recon_features.append(inception_features(extractor, mean, std, recon).cpu())

    real_features = torch.cat(real_features, dim=0)
    recon_features = torch.cat(recon_features, dim=0)
    mu_real, sigma_real = covariance(real_features)
    mu_recon, sigma_recon = covariance(recon_features)
    fid = frechet_distance(mu_real, sigma_real, mu_recon, sigma_recon)

    config = load_config(run_dir)
    variant = config.get("variant", {})
    return {
        "variant": run_dir.parent.name,
        "run_id": run_dir.name,
        "run_dir": run_dir.as_posix(),
        "checkpoint": checkpoint_path.name,
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "epochs": config.get("epochs", ""),
        "crop_ratios": json.dumps(variant.get("crop_ratios", [])),
        "crops_per_sample": variant.get("crops_per_sample", ""),
        "sample_mode": variant.get("sample_mode", ""),
        "test_mse": total_mse / total_count,
        "fid": fid,
        "latest_test_image": (find_latest_test_image(run_dir) or "").as_posix() if find_latest_test_image(run_dir) else "",
    }


def write_summary(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    fieldnames = [
        "variant",
        "run_id",
        "run_dir",
        "checkpoint",
        "checkpoint_step",
        "epochs",
        "crop_ratios",
        "crops_per_sample",
        "sample_mode",
        "test_mse",
        "fid",
        "latest_test_image",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["test_mse"], row["fid"])))
    return summary_csv


def plot_summary(rows, output_dir: Path):
    rows = sorted(rows, key=lambda row: row["test_mse"])
    labels = [row["variant"] for row in rows]
    mse = [row["test_mse"] for row in rows]
    fid = [row["fid"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].bar(labels, mse, color="#2b6cb0")
    axes[0].set_ylabel("Test Reconstruction MSE")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(labels, fid, color="#c05621")
    axes[1].set_ylabel("FID")
    axes[1].tick_params(axis="x", rotation=45)

    fig.savefig(output_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.scatter(mse, fid, s=80, color="#1a202c")
    for row in rows:
        ax.annotate(row["variant"], (row["test_mse"], row["fid"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Test Reconstruction MSE")
    ax.set_ylabel("FID")
    fig.savefig(output_dir / "metrics_scatter.png", dpi=150)
    plt.close(fig)


def make_reconstruction_montage(rows, output_dir: Path):
    rows = [row for row in rows if row["latest_test_image"]]
    if not rows:
        return None

    cols = 2
    rows_count = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(12, 4 * rows_count), constrained_layout=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, row in zip(axes, rows):
        image = Image.open(row["latest_test_image"])
        ax.imshow(image)
        ax.set_title(f"{row['variant']}\nMSE={row['test_mse']:.4f} FID={row['fid']:.2f}")
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")

    montage_path = output_dir / "reconstruction_montage.png"
    fig.savefig(montage_path, dpi=150)
    plt.close(fig)
    return montage_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sweep runs with comparable reconstruction metrics.")
    parser.add_argument("--sweep-root", default="logs/crop_sweeps")
    parser.add_argument("--output-dir", default="logs/crop_sweeps_eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for faster approximate evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweep_root = Path(args.sweep_root)
    run_dirs = find_run_dirs(sweep_root)
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {sweep_root}")
    _, test_loader = load_data(batch_size=args.batch_size, test_batch_size=args.batch_size, pin_memory=device.type == "cuda")
    extractor, mean, std = build_feature_extractor(device)

    print(f"evaluating {len(run_dirs)} runs on {device}")
    rows = []
    for run_dir in run_dirs:
        row = evaluate_run(run_dir, device, test_loader, extractor, mean, std, args.max_batches)
        rows.append(row)
        print(f"{row['variant']}: mse={row['test_mse']:.6f} fid={row['fid']:.3f}")

    output_dir = Path(args.output_dir)
    summary_csv = write_summary(rows, output_dir)
    plot_summary(rows, output_dir)
    montage_path = make_reconstruction_montage(rows, output_dir)
    print(f"wrote {summary_csv}")
    if montage_path:
        print(f"wrote {montage_path}")


if __name__ == "__main__":
    main()
