import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from leae.autoencoder import Autoencoder
from leae.logging import TrainingLogger
from leae.prep_data import load_data

ae = Autoencoder(in_channels=3, hidden_dim=64, latent_channels=32, output_size=32)


def main():
    epochs = 10
    metric_log_every = 10  # steps
    image_log_every = 1000  # steps
    log_dir = "logs"
    num_log_images = 8
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data(batch_size=64, pin_memory=device.type == "cuda")
    model = ae.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = TrainingLogger(log_dir=log_dir, num_images=num_log_images, image_value_range=(0, 1))
    train_bar = tqdm(total=epochs * len(train_loader), desc="train", leave=True)
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            recon = model(images)
            loss = F.mse_loss(recon, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)
            train_bar.update()

            image_path = ""
            if global_step % image_log_every == 0:
                image_path = logger.log_images("train", epoch, global_step, images, recon)

            if global_step % metric_log_every == 0:
                logger.log_metrics("train", epoch, global_step, loss.item(), loss.item(), 0.0, image_path=image_path)
                if image_path:
                    logger.plot_metrics()

        model.eval()
        test_loss = 0.0
        test_count = 0
        test_images = None
        test_recon = None

        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"test  {epoch:02d}", leave=False):
                images = images.to(device, non_blocking=True)
                recon = model(images)
                loss = F.mse_loss(recon, images)
                test_loss += loss.item() * images.size(0)
                test_count += images.size(0)
                test_images = images
                test_recon = recon

        test_image_path = logger.log_images("test", epoch, global_step, test_images, test_recon)
        logger.log_metrics(
            "test",
            epoch,
            global_step,
            test_loss / test_count,
            test_loss / test_count,
            0.0,
            image_path=test_image_path,
        )
        logger.plot_metrics()

        print(f"epoch {epoch:02d} train_loss={train_loss / train_count:.4f} test_loss={test_loss / test_count:.4f}")

    train_bar.close()


if __name__ == "__main__":
    main()
