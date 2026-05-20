from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from leae.autoencoder import Autoencoder
from leae.logging import TrainingLogger
from leae.masking import apply_square_crop, sample_square_crop_boxes
from leae.prep_data import load_data
from leae.sigreg import SIGReg, latent_to_sigreg_samples

ae = Autoencoder(in_channels=3, hidden_dim=64, latent_channels=8, output_size=32)


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

def main():
    epochs = 10
    metric_log_every = 10 # steps
    image_log_every = 500 # steps
    crop_ratio = 0.05
    sigreg_weight = 0.03
    log_dir = "logs"
    num_log_images = 8
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(batch_size=128, pin_memory=device.type == "cuda")
    model = ae.to(device)
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

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_sigreg = 0.0
        train_count = 0

        for images, _ in train_loader:
            """
            plan:
            x : image
            enc(x) --> z
            dec(z) --> rec_x
            crop(rec_x) --> crec_x
            crop(x) --> c_x
            enc(crec_x) --> crec_z
            enc(c_x) --> c_z
            Loss = mse(c_z, crec_z) + sigreg(z, crec_z, c_z)
            """
            images = images.to(device, non_blocking=True)
            z = model.encode(images)
            recon = model.decode(z)
            top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=crop_ratio)
            crop_x = apply_square_crop(images, top, left, crop_size)
            crop_rec_x = apply_square_crop(recon, top, left, crop_size)
            crop_z = model.encode(crop_x, update_latent_norm=False)
            crop_rec_z = model.encode(crop_rec_x, update_latent_norm=False)
            mse_loss = F.mse_loss(crop_z, crop_rec_z)
            sigreg_loss = sigreg_weight * (
                sigreg(latent_to_sigreg_samples(crop_z)) + sigreg(latent_to_sigreg_samples(crop_rec_z))
            )
            loss = mse_loss + sigreg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss += loss.item() * images.size(0)
            train_mse += mse_loss.item() * images.size(0)
            train_sigreg += sigreg_loss.item() * images.size(0)
            train_count += images.size(0)
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
            if global_step % image_log_every == 0:
                image_path = logger.log_images("train", epoch, global_step, images, recon)

            if global_step % metric_log_every == 0:
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
            for images, _ in tqdm(test_loader, desc=f"test  {epoch:02d}", leave=False):
                images = images.to(device, non_blocking=True)
                z = model.encode(images)
                recon = model.decode(z)
                top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=crop_ratio)
                crop_x = apply_square_crop(images, top, left, crop_size)
                crop_rec_x = apply_square_crop(recon, top, left, crop_size)
                crop_z = model.encode(crop_x, update_latent_norm=False)
                crop_rec_z = model.encode(crop_rec_x, update_latent_norm=False)
                mse_loss = F.mse_loss(crop_z, crop_rec_z)
                sigreg_loss = sigreg_weight * (
                    sigreg(latent_to_sigreg_samples(crop_z)) + sigreg(latent_to_sigreg_samples(crop_rec_z))
                )
                loss = mse_loss + sigreg_loss
                test_loss += loss.item() * images.size(0)
                test_mse += mse_loss.item() * images.size(0)
                test_sigreg += sigreg_loss.item() * images.size(0)
                test_count += images.size(0)
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
            f"epoch {epoch:02d} "
            f"train_loss={train_loss / train_count:.4f} "
            f"train_mse={train_mse / train_count:.4f} "
            f"train_sigreg={train_sigreg / train_count:.4f} "
            f"test_loss={test_loss / test_count:.4f} "
            f"test_mse={test_mse / test_count:.4f} "
            f"test_sigreg={test_sigreg / test_count:.4f}"
        )

    train_bar.close()


if __name__ == "__main__":
    main()
