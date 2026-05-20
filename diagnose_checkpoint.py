import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from leae.autoencoder import Autoencoder
from leae.masking import apply_square_crop, sample_square_crop_boxes
from leae.prep_data import load_data
from leae.sigreg import SIGReg, latent_to_sigreg_samples


def make_model():
    return Autoencoder(in_channels=3, hidden_dim=64, latent_channels=256, output_size=32)


def find_latest_checkpoint(log_root: str | Path = "logs") -> Path:
    checkpoints = sorted(Path(log_root).rglob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_*.pt files found under {log_root}")
    return checkpoints[-1]


def compute_crop_objective(model, sigreg, images, crop_ratio, sigreg_weight):
    z = model.encode(images)
    recon = model.decode(z)
    top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=crop_ratio)
    crop_x = apply_square_crop(images, top, left, crop_size)
    crop_rec_x = apply_square_crop(recon, top, left, crop_size)
    crop_z = model.encode(crop_x)
    crop_rec_z = model.encode(crop_rec_x)
    mse_loss = F.mse_loss(crop_z, crop_rec_z)
    sigreg_loss = sigreg_weight * (
        sigreg(latent_to_sigreg_samples(crop_z)) + sigreg(latent_to_sigreg_samples(crop_rec_z))
    )
    loss = mse_loss + sigreg_loss
    return {
        "loss": loss,
        "mse_loss": mse_loss,
        "sigreg_loss": sigreg_loss,
        "recon": recon,
        "z": z,
        "crop_z": crop_z,
        "crop_rec_z": crop_rec_z,
        "crop_size": crop_size,
    }


def tensor_stats(prefix, tensor):
    flat = tensor.detach().flatten()
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std(unbiased=False)),
        f"{prefix}_abs_mean": float(flat.abs().mean()),
        f"{prefix}_abs_max": float(flat.abs().max()),
        f"{prefix}_rms": float(flat.square().mean().sqrt()),
    }


def batchnorm_stats(model):
    bn = model.latent_norm
    stats = {"latent_norm_type": type(bn).__name__}
    if isinstance(bn, torch.nn.BatchNorm1d):
        stats.update(
            {
                "bn_running_mean_abs_mean": float(bn.running_mean.abs().mean()),
                "bn_running_mean_abs_max": float(bn.running_mean.abs().max()),
                "bn_running_var_mean": float(bn.running_var.mean()),
                "bn_running_var_min": float(bn.running_var.min()),
                "bn_running_var_max": float(bn.running_var.max()),
                "bn_weight_mean": float(bn.weight.mean()),
                "bn_weight_std": float(bn.weight.std(unbiased=False)),
                "bn_bias_mean": float(bn.bias.mean()),
                "bn_bias_std": float(bn.bias.std(unbiased=False)),
            }
        )
    return stats


def gradient_stats(model):
    total_sq = 0.0
    max_abs = 0.0
    param_count = 0
    named = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        norm = float(grad.norm())
        total_sq += norm * norm
        max_abs = max(max_abs, float(grad.abs().max()))
        param_count += 1
        if name in {"stem.0.weight", "encoder_features.1.block.2.weight", "latent_norm.weight", "head.0.weight"}:
            named[f"grad_norm::{name}"] = norm
    return {
        "grad_total_norm": total_sq ** 0.5,
        "grad_abs_max": max_abs,
        "grad_tensors": param_count,
        **named,
    }


def parameter_stats(model):
    total_sq = 0.0
    max_abs = 0.0
    for _, param in model.named_parameters():
        data = param.detach()
        total_sq += float(data.norm()) ** 2
        max_abs = max(max_abs, float(data.abs().max()))
    return {
        "param_total_norm": total_sq ** 0.5,
        "param_abs_max": max_abs,
    }


def update_stats(model, pre_step_params):
    total_sq = 0.0
    max_abs = 0.0
    for name, param in model.named_parameters():
        delta = (param.detach() - pre_step_params[name]).detach()
        total_sq += float(delta.norm()) ** 2
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "update_total_norm": total_sq ** 0.5,
        "update_abs_max": max_abs,
    }


def optimizer_stats(optimizer):
    lr = optimizer.param_groups[0]["lr"]
    beta1, beta2 = optimizer.param_groups[0]["betas"]
    eps = optimizer.param_groups[0]["eps"]
    return {"lr": lr, "adam_beta1": beta1, "adam_beta2": beta2, "adam_eps": eps}


def run_probe(model, sigreg, images, crop_ratio, sigreg_weight, mode):
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(mode)
    with torch.no_grad():
        out = compute_crop_objective(model, sigreg, images, crop_ratio, sigreg_weight)
    return {
        f"{mode}_probe_loss": float(out["loss"]),
        f"{mode}_probe_mse": float(out["mse_loss"]),
        f"{mode}_probe_sigreg": float(out["sigreg_loss"]),
        f"{mode}_probe_crop_size": int(out["crop_size"]),
        **tensor_stats(f"{mode}_probe_z", out["z"]),
        **tensor_stats(f"{mode}_probe_crop_z", out["crop_z"]),
        **tensor_stats(f"{mode}_probe_crop_rec_z", out["crop_rec_z"]),
    }


def write_header(path):
    header = [
        "resume_step",
        "loss",
        "mse_loss",
        "sigreg_loss",
        "crop_size",
        "grad_total_norm",
        "grad_abs_max",
        "update_total_norm",
        "update_abs_max",
        "param_total_norm",
        "param_abs_max",
        "bn_running_mean_abs_mean",
        "bn_running_mean_abs_max",
        "bn_running_var_mean",
        "bn_running_var_min",
        "bn_running_var_max",
        "lr",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "has_nan_loss",
        "has_nan_grad",
        "has_nan_param",
        "grad_norm::stem.0.weight",
        "grad_norm::encoder_features.1.block.2.weight",
        "grad_norm::latent_norm.weight",
        "grad_norm::head.0.weight",
        "train_probe_loss",
        "train_probe_mse",
        "train_probe_sigreg",
        "train_probe_crop_size",
        "eval_probe_loss",
        "eval_probe_mse",
        "eval_probe_sigreg",
        "eval_probe_crop_size",
        "test_eval_probe_loss",
        "test_eval_probe_mse",
        "test_eval_probe_sigreg",
        "test_eval_probe_crop_size",
        "batch_z_mean",
        "batch_z_std",
        "batch_z_abs_mean",
        "batch_z_abs_max",
        "batch_z_rms",
        "batch_crop_z_mean",
        "batch_crop_z_std",
        "batch_crop_z_abs_mean",
        "batch_crop_z_abs_max",
        "batch_crop_z_rms",
        "batch_crop_rec_z_mean",
        "batch_crop_rec_z_std",
        "batch_crop_rec_z_abs_mean",
        "batch_crop_rec_z_abs_max",
        "batch_crop_rec_z_rms",
        "train_probe_z_mean",
        "train_probe_z_std",
        "train_probe_z_abs_mean",
        "train_probe_z_abs_max",
        "train_probe_z_rms",
        "train_probe_crop_z_mean",
        "train_probe_crop_z_std",
        "train_probe_crop_z_abs_mean",
        "train_probe_crop_z_abs_max",
        "train_probe_crop_z_rms",
        "train_probe_crop_rec_z_mean",
        "train_probe_crop_rec_z_std",
        "train_probe_crop_rec_z_abs_mean",
        "train_probe_crop_rec_z_abs_max",
        "train_probe_crop_rec_z_rms",
        "eval_probe_z_mean",
        "eval_probe_z_std",
        "eval_probe_z_abs_mean",
        "eval_probe_z_abs_max",
        "eval_probe_z_rms",
        "eval_probe_crop_z_mean",
        "eval_probe_crop_z_std",
        "eval_probe_crop_z_abs_mean",
        "eval_probe_crop_z_abs_max",
        "eval_probe_crop_z_rms",
        "eval_probe_crop_rec_z_mean",
        "eval_probe_crop_rec_z_std",
        "eval_probe_crop_rec_z_abs_mean",
        "eval_probe_crop_rec_z_abs_max",
        "eval_probe_crop_rec_z_rms",
        "test_eval_probe_z_mean",
        "test_eval_probe_z_std",
        "test_eval_probe_z_abs_mean",
        "test_eval_probe_z_abs_max",
        "test_eval_probe_z_rms",
        "test_eval_probe_crop_z_mean",
        "test_eval_probe_crop_z_std",
        "test_eval_probe_crop_z_abs_mean",
        "test_eval_probe_crop_z_abs_max",
        "test_eval_probe_crop_z_rms",
        "test_eval_probe_crop_rec_z_mean",
        "test_eval_probe_crop_rec_z_std",
        "test_eval_probe_crop_rec_z_abs_mean",
        "test_eval_probe_crop_rec_z_abs_max",
        "test_eval_probe_crop_rec_z_rms",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
    return header


def append_row(path, header, row):
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Resume a checkpoint with extra diagnostics.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to newest checkpoint under logs/.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--crop-ratio", type=float, default=0.1)
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override LR instead of using checkpoint optimizer state.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--output-dir", default="diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.output_dir) / checkpoint_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.tsv"
    jsonl_path = run_dir / "metrics.jsonl"
    header = write_header(metrics_path)

    model = make_model().to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if args.learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate

    train_loader, test_loader = load_data(batch_size=args.batch_size, pin_memory=device.type == "cuda")
    train_iter = iter(train_loader)
    train_probe_images, _ = next(iter(train_loader))
    test_probe_images, _ = next(iter(test_loader))
    train_probe_images = train_probe_images.to(device, non_blocking=True)
    test_probe_images = test_probe_images.to(device, non_blocking=True)

    start_step = int(checkpoint.get("global_step", 0))
    print(f"checkpoint={checkpoint_path} start_step={start_step} device={device}")
    print(f"diagnostics_dir={run_dir}")

    for local_step in range(1, args.steps + 1):
        try:
            images, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, _ = next(train_iter)

        images = images.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = compute_crop_objective(model, sigreg, images, args.crop_ratio, args.sigreg_weight)
        loss = out["loss"]
        pre_step_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        loss.backward()
        grad = gradient_stats(model)
        optimizer.step()
        update = update_stats(model, pre_step_params)

        resume_step = start_step + local_step
        row = {
            "resume_step": resume_step,
            "loss": float(out["loss"]),
            "mse_loss": float(out["mse_loss"]),
            "sigreg_loss": float(out["sigreg_loss"]),
            "crop_size": int(out["crop_size"]),
            **grad,
            **update,
            **parameter_stats(model),
            **batchnorm_stats(model),
            **optimizer_stats(optimizer),
            **tensor_stats("batch_z", out["z"]),
            **tensor_stats("batch_crop_z", out["crop_z"]),
            **tensor_stats("batch_crop_rec_z", out["crop_rec_z"]),
            "has_nan_loss": int(not torch.isfinite(loss)),
            "has_nan_grad": int(any(param.grad is not None and not torch.isfinite(param.grad).all() for param in model.parameters())),
            "has_nan_param": int(any(not torch.isfinite(param).all() for param in model.parameters())),
        }

        if local_step == 1 or local_step % args.probe_every == 0:
            row.update(run_probe(model, sigreg, train_probe_images, args.crop_ratio, args.sigreg_weight, mode="train"))
            row.update(run_probe(model, sigreg, train_probe_images, args.crop_ratio, args.sigreg_weight, mode="eval"))
            row.update({f"test_{k}": v for k, v in run_probe(model, sigreg, test_probe_images, args.crop_ratio, args.sigreg_weight, mode="eval").items()})

        append_row(metrics_path, header, row)
        with jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if local_step == 1 or local_step % args.log_every == 0:
            message = (
                f"step={resume_step} loss={row['loss']:.4f} mse={row['mse_loss']:.4f} sig={row['sigreg_loss']:.4f} "
                f"grad={row['grad_total_norm']:.4f} update={row['update_total_norm']:.6f} "
                f"bn_mean={row.get('bn_running_mean_abs_mean', float('nan')):.4f} "
                f"bn_var_max={row.get('bn_running_var_max', float('nan')):.4f}"
            )
            if "eval_probe_loss" in row:
                message += f" eval_probe={row['eval_probe_loss']:.4f} test_eval_probe={row['test_eval_probe_loss']:.4f}"
            print(message)


if __name__ == "__main__":
    main()
