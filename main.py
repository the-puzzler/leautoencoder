import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from leae.autoencoder import Autoencoder
from leae.logging import TrainingLogger
from leae.masking import apply_square_crop, make_inpainted_input, sample_square_crop_boxes
from leae.prep_data import load_data
from leae.sigreg import SIGReg, latent_to_sigreg_samples

ae = Autoencoder(
    in_channels=3,
    hidden_dim=128,
    latent_channels=128,
    output_size=128,
    pooled_latent=True,
    collapse_style="mlp",
    expand_style="mlp",
    pooled_map_channels=128,
)


def save_checkpoint(model, optimizer, log_dir, percent, epoch, global_step):
    checkpoint_path = Path(log_dir) / f"checkpoint_{percent}.pt"
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


def save_branch_images(image_dir, split, epoch, step, images, clean_recon, masked_recon, num_images, value_range):
    images = images[:num_images].detach().cpu()
    clean_recon = clean_recon[:num_images].detach().cpu()
    masked_recon = masked_recon[:num_images].detach().cpu()
    image_path = image_dir / f"{split}_epoch{epoch:02d}_step{step:06d}.png"
    grid = torch.cat([images, clean_recon, masked_recon], dim=0)
    save_image(grid, image_path, nrow=max(1, images.size(0)), normalize=True, value_range=value_range)
    return image_path.as_posix()


def main():
    epochs = 50
    batch_size = 128
    metric_log_every = 10
    image_log_every = 500
    test_every = 2000
    sigreg_weight = 0.03
    crop_ratio = 0.15
    inpaint_mask_ratio = 0.35
    latent_match_weight = 1.0
    clean_crop_weight = 1.0
    masked_crop_weight = 1.0
    log_dir = "logs"
    num_log_images = 8
    learning_rate = 1e-3
    dataset_name = "celeba"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(
        batch_size=batch_size,
        pin_memory=device.type == "cuda",
        dataset_name=dataset_name,
    )
    model = ae.to(device)
    target_encoder = copy.deepcopy(model).to(device)
    for param in target_encoder.parameters():
        param.requires_grad_(False)
    target_encoder.eval()
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger = TrainingLogger(log_dir=log_dir, num_images=num_log_images, image_value_range=(0, 1))
    run_dir = logger.log_dir
    train_bar = tqdm(total=epochs * len(train_loader), desc="train", leave=True)
    total_steps = epochs * len(train_loader)
    global_step = 0
    checkpoint_percents = [0, 20, 40, 60, 80, 100]
    next_checkpoint_idx = 0

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, run_dir, checkpoint_percents[next_checkpoint_idx], epoch=0, global_step=0)
    next_checkpoint_idx += 1

    def crop_resize_views(images, clean_recon, masked_recon):
        top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=crop_ratio)
        crop_images = apply_square_crop(images, top, left, crop_size)
        crop_clean_recon = apply_square_crop(clean_recon, top, left, crop_size)
        crop_masked_recon = apply_square_crop(masked_recon, top, left, crop_size)
        return crop_images, crop_clean_recon, crop_masked_recon

    def judge_crop_loss(crop_images, crop_recon):
        with torch.no_grad():
            target_z = target_encoder.encode(crop_images, update_latent_norm=False)
        recon_z = target_encoder.encode(crop_recon, update_latent_norm=False)
        return F.mse_loss(target_z, recon_z)

    def sync_target_encoder():
        target_encoder.load_state_dict(model.state_dict())
        target_encoder.eval()

    def run_test(epoch, global_step):
        model.eval()
        target_encoder.eval()
        test_loss = 0.0
        test_mse = 0.0
        test_sigreg = 0.0
        test_count = 0
        test_images = None
        test_clean_recon = None
        test_masked_recon = None

        for images, _ in tqdm(test_loader, desc=f"test  {epoch:02d}", leave=False):
            images = images.to(device, non_blocking=True)
            masked_images = make_inpainted_input(images, mask_ratio=inpaint_mask_ratio)
            with torch.no_grad():
                z_clean = model.encode(images)
                clean_recon = model.decode(z_clean)
                z_masked = model.encode(masked_images)
                masked_recon = model.decode(z_masked)
                crop_images, crop_clean_recon, crop_masked_recon = crop_resize_views(images, clean_recon, masked_recon)
                latent_loss = F.mse_loss(z_masked, z_clean)
                clean_crop_loss = judge_crop_loss(crop_images, crop_clean_recon)
                masked_crop_loss = judge_crop_loss(crop_images, crop_masked_recon)
                mse_loss = (
                    latent_match_weight * latent_loss
                    + clean_crop_weight * clean_crop_loss
                    + masked_crop_weight * masked_crop_loss
                ) / (latent_match_weight + clean_crop_weight + masked_crop_weight)
            sigreg_loss = 0.5 * sigreg_weight * (
                sigreg(latent_to_sigreg_samples(z_clean)) + sigreg(latent_to_sigreg_samples(z_masked))
            )
            loss = mse_loss + sigreg_loss
            test_loss += loss.item() * images.size(0)
            test_mse += mse_loss.item() * images.size(0)
            test_sigreg += sigreg_loss.item() * images.size(0)
            test_count += images.size(0)
            test_images = images
            test_clean_recon = clean_recon
            test_masked_recon = masked_recon

        test_image_path = save_branch_images(
            logger.image_dir,
            "test",
            epoch,
            global_step,
            test_images,
            test_clean_recon,
            test_masked_recon,
            num_log_images,
            (0, 1),
        )
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
            f"test  step={global_step:06d} "
            f"loss={test_loss / test_count:.4f} "
            f"mse={test_mse / test_count:.4f} "
            f"sigreg={test_sigreg / test_count:.4f}"
        )
        model.train()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_sigreg = 0.0
        train_count = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masked_images = make_inpainted_input(images, mask_ratio=inpaint_mask_ratio)

            z_clean = model.encode(images)
            clean_recon = model.decode(z_clean)
            z_masked = model.encode(masked_images)
            masked_recon = model.decode(z_masked)

            crop_images, crop_clean_recon, crop_masked_recon = crop_resize_views(images, clean_recon, masked_recon)
            latent_loss = F.mse_loss(z_masked, z_clean.detach())
            clean_crop_loss = judge_crop_loss(crop_images, crop_clean_recon)
            masked_crop_loss = judge_crop_loss(crop_images, crop_masked_recon)
            mse_loss = (
                latent_match_weight * latent_loss
                + clean_crop_weight * clean_crop_loss
                + masked_crop_weight * masked_crop_loss
            ) / (latent_match_weight + clean_crop_weight + masked_crop_weight)
            sigreg_loss = 0.5 * sigreg_weight * (
                sigreg(latent_to_sigreg_samples(z_clean)) + sigreg(latent_to_sigreg_samples(z_masked))
            )
            loss = mse_loss + sigreg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sync_target_encoder()
            global_step += 1
            train_loss += loss.item() * images.size(0)
            train_mse += mse_loss.item() * images.size(0)
            train_sigreg += sigreg_loss.item() * images.size(0)
            train_count += images.size(0)
            train_bar.update()

            while next_checkpoint_idx < len(checkpoint_percents) and global_step * 100 >= checkpoint_percents[next_checkpoint_idx] * total_steps:
                save_checkpoint(model, optimizer, run_dir, checkpoint_percents[next_checkpoint_idx], epoch=epoch, global_step=global_step)
                next_checkpoint_idx += 1

            image_path = ""
            if global_step % image_log_every == 0:
                image_path = save_branch_images(
                    logger.image_dir,
                    "train",
                    epoch,
                    global_step,
                    images,
                    clean_recon,
                    masked_recon,
                    num_log_images,
                    (0, 1),
                )

            if global_step % metric_log_every == 0:
                logger.log_metrics("train", epoch, global_step, loss.item(), mse_loss.item(), sigreg_loss.item(), image_path=image_path)
                if image_path:
                    logger.plot_metrics()

            if global_step % test_every == 0:
                run_test(epoch, global_step)

        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss / train_count:.4f} "
            f"train_mse={train_mse / train_count:.4f} "
            f"train_sigreg={train_sigreg / train_count:.4f}"
        )

    if global_step % test_every != 0:
        run_test(epochs, global_step)

    train_bar.close()


if __name__ == "__main__":
    main()
