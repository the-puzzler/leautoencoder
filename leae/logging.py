from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


class TrainingLogger:
    def __init__(self, log_dir="logs", num_images=8, image_value_range=(-1, 1), run_dir=None):
        root_dir = Path(log_dir)
        self.log_dir = Path(run_dir) if run_dir is not None else root_dir / str(self.next_run_id(root_dir))
        self.image_dir = self.log_dir / "reconstructions"
        self.metrics_path = self.log_dir / "metrics.tsv"
        self.metrics_plot_path = self.log_dir / "metrics.png"
        self.num_images = num_images
        self.image_value_range = image_value_range
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        if not self.metrics_path.exists():
            self.metrics_path.write_text("split\tepoch\tstep\tloss\tmse_loss\tsigreg_loss\timage_path\n")

    def next_run_id(self, root_dir):
        root_dir.mkdir(parents=True, exist_ok=True)
        run_ids = [int(path.name) for path in root_dir.iterdir() if path.is_dir() and path.name.isdigit()]
        return 0 if not run_ids else max(run_ids) + 1

    def log_metrics(self, split, epoch, step, loss, mse_loss, sigreg_loss, image_path=""):
        with self.metrics_path.open("a") as f:
            f.write(f"{split}\t{epoch}\t{step}\t{loss:.6f}\t{mse_loss:.6f}\t{sigreg_loss:.6f}\t{image_path}\n")

    def log_images(self, split, epoch, step, images, reconstructions):
        images = images[: self.num_images].detach().cpu()
        reconstructions = reconstructions[: self.num_images].detach().cpu()
        image_path = self.image_dir / f"{split}_epoch{epoch:02d}_step{step:06d}.png"
        grid = torch.cat([images, reconstructions], dim=0)
        save_image(
            grid,
            image_path,
            nrow=max(1, images.size(0)),
            normalize=True,
            value_range=self.image_value_range,
        )
        return image_path.as_posix()

    def plot_metrics(self):
        rows = self.metrics_path.read_text().strip().splitlines()
        if len(rows) <= 1:
            return

        train_steps = []
        train_loss = []
        train_mse = []
        train_sigreg = []
        test_steps = []
        test_loss = []
        test_mse = []
        test_sigreg = []

        for row in rows[1:]:
            split, _, step, loss, mse_loss, sigreg_loss, _ = row.split("\t")
            step = int(step)
            loss = float(loss)
            mse_loss = float(mse_loss)
            sigreg_loss = float(sigreg_loss)

            if split == "train":
                train_steps.append(step)
                train_loss.append(loss)
                train_mse.append(mse_loss)
                train_sigreg.append(sigreg_loss)
            elif split == "test":
                test_steps.append(step)
                test_loss.append(loss)
                test_mse.append(mse_loss)
                test_sigreg.append(sigreg_loss)

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        series = [
            ("loss", train_loss, test_loss),
            ("mse_loss", train_mse, test_mse),
            ("sigreg_loss", train_sigreg, test_sigreg),
        ]

        for ax, (label, train_values, test_values) in zip(axes, series):
            if train_steps:
                ax.plot(train_steps, train_values, label="train")
            if test_steps:
                ax.plot(test_steps, test_values, label="test")
            ax.set_ylabel(label)
            ax.legend()

        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(self.metrics_plot_path, dpi=150)
        plt.close(fig)
