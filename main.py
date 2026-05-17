import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from leae.autoencoder import Autoencoder
from leae.logging import TrainingLogger
from leae.masking import apply_mask, make_pixel_mask
from leae.sigreg import SIGReg
from leae.synth import make_synthetic

ae = Autoencoder(in_channels=1, hidden_dim=32, latent_channels=16, output_size=28)

def main():
    epochs = 10
    metric_log_every = 10 # steps
    image_log_every = 100 # steps
    mask_ratio = 0.7
    sigreg_weight = 0.1
    image_size = 28
    log_dir = "logs"
    num_log_images = 8
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_images = make_synthetic(n=12000, size=image_size, seed=0)
    train_dataset = TensorDataset(all_images[:10000], torch.zeros(10000))
    test_dataset = TensorDataset(all_images[10000:], torch.zeros(2000))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = ae.to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = TrainingLogger(log_dir=log_dir, num_images=num_log_images, image_value_range=(0, 1))
    train_bar = tqdm(total=epochs * len(train_loader), desc="train", leave=True)
    global_step = 0

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
            mask(rec_x) --> mrec_x
            mask(x) --> m_x
            enc(mrec_x) --> mrec_z
            enc(m_x) --> m_z
            Loss = mse(m_z, mrec_z) + sigreg(z, mrec_z, m_z)
            """
            images = images.to(device, non_blocking=True)
            z = model.encode(images)
            recon = model.decode(z)
            mask = make_pixel_mask(images, mask_ratio=mask_ratio)
            m_x = apply_mask(images, mask)
            mrec_x = apply_mask(recon, mask)
            m_z = model.encode(m_x)
            mrec_z = model.encode(mrec_x)
            z_flat = z.flatten(1)
            m_z_flat = m_z.flatten(1)
            mrec_z_flat = mrec_z.flatten(1)
            mse_loss = F.mse_loss(m_z_flat, mrec_z_flat)
            sigreg_loss = sigreg_weight * (sigreg(m_z_flat) + sigreg(mrec_z_flat))
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
                mask = make_pixel_mask(images, mask_ratio=mask_ratio)
                m_x = apply_mask(images, mask)
                mrec_x = apply_mask(recon, mask)
                m_z = model.encode(m_x)
                mrec_z = model.encode(mrec_x)
                z_flat = z.flatten(1)
                m_z_flat = m_z.flatten(1)
                mrec_z_flat = mrec_z.flatten(1)
                mse_loss = F.mse_loss(m_z_flat, mrec_z_flat)
                sigreg_loss = sigreg_weight * (sigreg(m_z_flat) + sigreg(mrec_z_flat))
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
