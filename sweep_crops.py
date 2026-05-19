import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from leae.autoencoder import Autoencoder
from leae.logging import TrainingLogger
from leae.masking import apply_square_crop, sample_square_crop_boxes
from leae.prep_data import load_data
from leae.sigreg import SIGReg


@dataclass(frozen=True)
class CropVariant:
    name: str
    crop_ratios: tuple[float, ...]
    crops_per_sample: int
    sample_mode: str


def make_model():
    return Autoencoder(in_channels=3, hidden_dim=64, latent_channels=256, output_size=32)


def save_checkpoint(model, optimizer, run_dir, percent, epoch, global_step):
    checkpoint_path = Path(run_dir) / f"checkpoint_{percent}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "percent": percent,
        },
        checkpoint_path,
    )


def build_variants():
    fixed_ratios = [0.1, 0.2, 0.3, 0.5, 0.7]
    variants = [CropVariant(name=f"fixed_{ratio:.2f}", crop_ratios=(ratio,), crops_per_sample=1, sample_mode="fixed") for ratio in fixed_ratios]
    variants.extend(
        [
            CropVariant(name="vary_0.10_0.30", crop_ratios=(0.1, 0.2, 0.3), crops_per_sample=1, sample_mode="random"),
            CropVariant(name="vary_0.20_0.50", crop_ratios=(0.2, 0.3, 0.4, 0.5), crops_per_sample=1, sample_mode="random"),
            CropVariant(name="vary_0.10_0.70", crop_ratios=(0.1, 0.2, 0.3, 0.5, 0.7), crops_per_sample=1, sample_mode="random"),
            CropVariant(name="multi2_fixed_0.20", crop_ratios=(0.2,), crops_per_sample=2, sample_mode="fixed"),
            CropVariant(name="multi2_fixed_0.30", crop_ratios=(0.3,), crops_per_sample=2, sample_mode="fixed"),
            CropVariant(name="multi2_fixed_0.50", crop_ratios=(0.5,), crops_per_sample=2, sample_mode="fixed"),
            CropVariant(name="multi3_fixed_0.30", crop_ratios=(0.3,), crops_per_sample=3, sample_mode="fixed"),
            CropVariant(name="multi2_mixed_small_large", crop_ratios=(0.2, 0.5), crops_per_sample=2, sample_mode="cycle"),
            CropVariant(name="multi3_mixed_0.10_0.30_0.50", crop_ratios=(0.1, 0.3, 0.5), crops_per_sample=3, sample_mode="cycle"),
            CropVariant(name="multi4_random_0.10_0.50", crop_ratios=(0.1, 0.2, 0.3, 0.5), crops_per_sample=4, sample_mode="random"),
        ]
    )
    return variants


def choose_crop_ratio(variant, crop_idx, device):
    if variant.sample_mode == "fixed":
        return variant.crop_ratios[0]
    if variant.sample_mode == "cycle":
        return variant.crop_ratios[crop_idx % len(variant.crop_ratios)]
    if variant.sample_mode == "random":
        idx = torch.randint(len(variant.crop_ratios), (1,), device=device).item()
        return variant.crop_ratios[idx]
    raise ValueError(f"unknown sample_mode={variant.sample_mode}")


def compute_crop_losses(model, sigreg, images, recon, variant, sigreg_weight):
    mse_losses = []
    sigreg_losses = []

    for crop_idx in range(variant.crops_per_sample):
        crop_ratio = choose_crop_ratio(variant, crop_idx, images.device)
        top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=crop_ratio)
        crop_x = apply_square_crop(images, top, left, crop_size)
        crop_rec_x = apply_square_crop(recon, top, left, crop_size)
        crop_z = model.encode(crop_x).flatten(1)
        crop_rec_z = model.encode(crop_rec_x).flatten(1)
        mse_losses.append(F.mse_loss(crop_z, crop_rec_z))
        sigreg_losses.append(sigreg(crop_z) + sigreg(crop_rec_z))

    mse_loss = torch.stack(mse_losses).mean()
    sigreg_loss = sigreg_weight * torch.stack(sigreg_losses).mean()
    return mse_loss, sigreg_loss


def write_run_config(run_dir, variant, args):
    config = {
        "variant": asdict(variant),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sigreg_weight": args.sigreg_weight,
        "num_log_images": args.num_log_images,
        "metric_log_every": args.metric_log_every,
        "image_log_every": args.image_log_every,
    }
    Path(run_dir, "config.json").write_text(json.dumps(config, indent=2) + "\n")


def train_variant(variant, args, device, train_loader, test_loader, sweep_root):
    model = make_model().to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logger = TrainingLogger(log_dir=sweep_root / variant.name, num_images=args.num_log_images, image_value_range=(-1, 1))
    run_dir = logger.log_dir
    write_run_config(run_dir, variant, args)

    total_steps = args.epochs * len(train_loader)
    checkpoint_percents = [0, 20, 40, 60, 80, 100]
    next_checkpoint_idx = 0
    global_step = 0
    train_bar = tqdm(total=total_steps, desc=variant.name, leave=True)

    save_checkpoint(model, optimizer, run_dir, checkpoint_percents[next_checkpoint_idx], epoch=0, global_step=0)
    next_checkpoint_idx += 1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_sigreg = 0.0
        train_count = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            recon = model.decode(model.encode(images))
            mse_loss, sigreg_loss = compute_crop_losses(model, sigreg, images, recon, variant, args.sigreg_weight)
            loss = mse_loss + sigreg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_mse += mse_loss.item() * batch_size
            train_sigreg += sigreg_loss.item() * batch_size
            train_count += batch_size
            train_bar.update()

            while (
                next_checkpoint_idx < len(checkpoint_percents)
                and global_step * 100 >= checkpoint_percents[next_checkpoint_idx] * total_steps
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    run_dir,
                    checkpoint_percents[next_checkpoint_idx],
                    epoch=epoch,
                    global_step=global_step,
                )
                next_checkpoint_idx += 1

            image_path = ""
            if global_step % args.image_log_every == 0:
                image_path = logger.log_images("train", epoch, global_step, images, recon)

            if global_step % args.metric_log_every == 0:
                logger.log_metrics("train", epoch, global_step, loss.item(), mse_loss.item(), sigreg_loss.item(), image_path=image_path)
                if image_path:
                    logger.plot_metrics()

        model.eval()
        test_loss = 0.0
        test_mse = 0.0
        test_sigreg = 0.0
        test_count = 0
        test_images = None
        test_recon = None

        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"{variant.name}-test-{epoch:02d}", leave=False):
                images = images.to(device, non_blocking=True)
                recon = model.decode(model.encode(images))
                mse_loss, sigreg_loss = compute_crop_losses(model, sigreg, images, recon, variant, args.sigreg_weight)
                loss = mse_loss + sigreg_loss

                batch_size = images.size(0)
                test_loss += loss.item() * batch_size
                test_mse += mse_loss.item() * batch_size
                test_sigreg += sigreg_loss.item() * batch_size
                test_count += batch_size
                test_images = images
                test_recon = recon

        test_image_path = logger.log_images("test", epoch, global_step, test_images, test_recon)
        logger.log_metrics(
            "test",
            epoch,
            global_step,
            test_loss / test_count,
            test_mse / test_count,
            test_sigreg / test_count,
            image_path=test_image_path,
        )
        logger.plot_metrics()

        print(
            f"{variant.name} "
            f"epoch {epoch:02d} "
            f"train_loss={train_loss / train_count:.4f} "
            f"train_mse={train_mse / train_count:.4f} "
            f"train_sigreg={train_sigreg / train_count:.4f} "
            f"test_loss={test_loss / test_count:.4f} "
            f"test_mse={test_mse / test_count:.4f} "
            f"test_sigreg={test_sigreg / test_count:.4f}"
        )

    train_bar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a sweep of crop-based LEAE training variants.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--metric-log-every", type=int, default=10)
    parser.add_argument("--image-log-every", type=int, default=100)
    parser.add_argument("--num-log-images", type=int, default=8)
    parser.add_argument("--log-dir", default="logs/crop_sweeps")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variant names to run.")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = build_variants()
    if args.variants:
        wanted = set(args.variants)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"unknown variants: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(batch_size=args.batch_size, pin_memory=device.type == "cuda")
    sweep_root = Path(args.log_dir)
    sweep_root.mkdir(parents=True, exist_ok=True)

    print("running variants:")
    for variant in variants:
        print(f"  {variant.name}: ratios={variant.crop_ratios} crops_per_sample={variant.crops_per_sample} mode={variant.sample_mode}")

    for variant in variants:
        train_variant(variant, args, device, train_loader, test_loader, sweep_root)


if __name__ == "__main__":
    main()
