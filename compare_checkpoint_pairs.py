from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from leae.autoencoder import Autoencoder
from leae.masking import fill_mask_with_image_average, make_inpainting_mask
from leae.prep_data import load_data

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "logs"
OUTPUT_DIR = LOG_ROOT / "pair_comparisons"

RUN_PAIRS = [
    {"main_run": 42, "baseline_run": 43, "latent_channels": 128, "method_label": "our method", "baseline_label": "baseline"},
    {"main_run": 44, "baseline_run": 45, "latent_channels": 512, "method_label": "our method", "baseline_label": "baseline"},
]

NUM_IMAGES = 8
MASK_RATIO = 0.35
SEED = 0


def build_model(latent_channels: int) -> Autoencoder:
    return Autoencoder(
        in_channels=3,
        hidden_dim=128,
        latent_channels=latent_channels,
        output_size=128,
        pooled_latent=True,
        collapse_style="mlp",
        expand_style="mlp",
        pooled_map_channels=128,
    )


def load_model(checkpoint_path: Path, latent_channels: int, device: torch.device) -> Autoencoder:
    model = build_model(latent_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def first_test_batch(device: torch.device) -> torch.Tensor:
    _, test_loader = load_data(
        batch_size=NUM_IMAGES,
        test_batch_size=NUM_IMAGES,
        pin_memory=device.type == "cuda",
        dataset_name="celeba",
        num_workers=0,
    )
    images, _ = next(iter(test_loader))
    return images.to(device)


def mask_boxes(mask: torch.Tensor) -> list[tuple[int, int, int, int]]:
    boxes = []
    for sample_mask in mask[:, 0]:
        ys, xs = torch.where(sample_mask < 0.5)
        top = int(ys.min().item())
        bottom = int(ys.max().item())
        left = int(xs.min().item())
        right = int(xs.max().item())
        boxes.append((left, top, right - left + 1, bottom - top + 1))
    return boxes


@torch.no_grad()
def main_clean_and_masked_recon(model: Autoencoder, images: torch.Tensor, masked_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clean_latent = model.encode(images)
    masked_latent = model.encode(masked_images)
    return model.decode(clean_latent), model.decode(masked_latent)


@torch.no_grad()
def baseline_clean_and_masked_recon(
    model: Autoencoder, images: torch.Tensor, masked_images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(images), model(masked_images)


def save_panel(
    output_path: Path,
    title: str,
    boxes: list[tuple[int, int, int, int]],
    row_batches: list[tuple[str, torch.Tensor]],
) -> None:
    rows = len(row_batches)
    cols = row_batches[0][1].size(0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1), squeeze=False)

    for row_idx, (row_label, batch) in enumerate(row_batches):
        batch = batch.detach().cpu().clamp(0.0, 1.0)
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            ax.imshow(batch[col_idx].permute(1, 2, 0).numpy())
            left, top, width, height = boxes[col_idx]
            ax.add_patch(Rectangle((left, top), width, height, linewidth=1.3, edgecolor="#ff8c00", facecolor="none"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(f"{col_idx + 1}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=50, fontsize=11)

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.9, bottom=0.02, wspace=0.02, hspace=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_pair_comparison(images: torch.Tensor, mask: torch.Tensor, pair: dict[str, int | str], device: torch.device) -> Path:
    masked_images = fill_mask_with_image_average(images, mask)
    main_checkpoint = LOG_ROOT / str(pair["main_run"]) / "checkpoint_100.pt"
    baseline_checkpoint = LOG_ROOT / str(pair["baseline_run"]) / "checkpoint_100.pt"
    main_model = load_model(main_checkpoint, pair["latent_channels"], device)
    baseline_model = load_model(baseline_checkpoint, pair["latent_channels"], device)

    main_clean, main_masked = main_clean_and_masked_recon(main_model, images, masked_images)
    baseline_clean, baseline_masked = baseline_clean_and_masked_recon(baseline_model, images, masked_images)

    output_path = OUTPUT_DIR / f"our_method_vs_baseline_latent{pair['latent_channels']}_checkpoint100.png"
    save_panel(
        output_path=output_path,
        title=(
            f"{pair['method_label'].title()} vs {pair['baseline_label'].title()} | "
            f"latent={pair['latent_channels']} | checkpoint=100% | CelebA test | shared mask_ratio={MASK_RATIO}"
        ),
        boxes=mask_boxes(mask),
        row_batches=[
            ("original", images),
            ("masked", masked_images),
            (f"{pair['method_label']} masked", main_masked),
            (f"{pair['baseline_label']} masked", baseline_masked),
            (f"{pair['method_label']} clean", main_clean),
            (f"{pair['baseline_label']} clean", baseline_clean),
        ],
    )
    return output_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = first_test_batch(device)
    torch.manual_seed(SEED)
    mask = make_inpainting_mask(images, mask_ratio=MASK_RATIO)

    for pair in RUN_PAIRS:
        output_path = make_pair_comparison(images, mask, pair, device)
        print(output_path)


if __name__ == "__main__":
    main()
