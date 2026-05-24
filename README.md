# `leautoencoder`

A small repo for a self-teaching autoencoder.

Instead of training only with pixel reconstruction, the model also learns from its own latent judgments. The core idea is: if a masked view and a clean view come from the same image, their reconstructions should become consistent under the model's own encoder.

**Blog post:** [self-teaching-autoencoder](https://the-puzzler.github.io/share/self-teaching-autoencoder.html)

![Our method vs baseline at latent 512](logs/pair_comparisons/our_method_vs_baseline_latent512_checkpoint100.png)

Shown above: `our method` vs `baseline` from the `100%` checkpoint at latent size `512`.

## What This Repo Is

This repo trains autoencoders on `CelebA` center-cropped to `128x128`.

There are three main training entrypoints:

- `main.py`: the current symmetric self-teaching variant
- `main_regular.py`: the plain masked-image reconstruction baseline
- `main_clean_branch.py`: an experimental less-symmetric variant

The autoencoder itself lives in `leae/autoencoder.py`.

## The Main Idea

The model is trained on two views of the same image:

- a clean image
- a masked image, where a square region is removed and filled with the image average

Both views go through the same autoencoder. The model then uses an EMA-style target copy of its own encoder as a judge.

So this is "self-teaching" in a literal sense:

- the student is the current autoencoder
- the teacher is a slowly refreshed copy of the same encoder
- the learning signal comes from consistency in latent space, not just from pixels

## The Two Objectives

### 1. Current Repo: The More Symmetric JEPA-like Variant

This is the version in `main.py`.

Both clean and masked inputs are encoded and decoded:

```text
z_clean = E(x)
x_clean_hat = D(z_clean)

z_masked = E(mask(x))
x_masked_hat = D(z_masked)
```

Then the target encoder judges the reconstructions:

```text
consistency_loss =
    MSE(T(x_clean_hat), T(x_masked_hat))

clean_crop_loss =
    MSE(T(crop(x)), T(crop(x_clean_hat)))

masked_crop_loss =
    MSE(T(crop(x)), T(crop(x_masked_hat)))

mse_loss = average(consistency_loss, clean_crop_loss, masked_crop_loss)
```

And there is also a latent regularizer:

```text
sigreg_loss =
    0.5 * lambda * (SIGReg(z_clean) + SIGReg(z_masked))
```

Final objective:

```text
loss = mse_loss + sigreg_loss
```

Why this is the more symmetric version:

- both branches pass through the full autoencoder
- both reconstructions are judged in the same latent space
- both branches contribute equally to the latent consistency objective

This is the version the repo is currently centered on.

### 2. The Less Symmetric Variant

This lives in `main_clean_branch.py`.

Here the masked branch is pushed directly toward the clean latent:

```text
latent_loss = MSE(z_masked, stopgrad(z_clean))
```

It still keeps the crop-based teacher losses and the same `SIGReg` term, but the main latent matching is less symmetric because:

- the clean latent acts more like the target
- the masked latent is the thing being pulled toward it
- the clean branch is not judged in exactly the same way as the masked branch

In short:

- `main.py`: reconstruction-to-reconstruction latent consistency
- `main_clean_branch.py`: masked-latent-to-clean-latent matching

## Baseline

The baseline in `main_regular.py` is just a standard masked autoencoder:

```text
recon = model(masked_image)
loss = MSE(recon, image)
```

No self-teaching latent objective, no target encoder, no crop consistency.

## Running It

The repo assumes a working Python environment with PyTorch, torchvision, `datasets`, and `matplotlib`.

Basic runs:

```bash
./.venv/bin/python main.py
./.venv/bin/python main_regular.py
```

Paired experiment scripts:

```bash
./run_main_and_baseline_128_512.sh
./run_main_and_baseline_6x.sh
```

Logs and checkpoints are written under `logs/`.

## Repo Layout

- `main.py`: current symmetric self-teaching method
- `main_regular.py`: baseline masked autoencoder
- `main_clean_branch.py`: less-symmetric experimental variant
- `leae/autoencoder.py`: model definitions
- `leae/masking.py`: masking and crop helpers
- `leae/prep_data.py`: dataset loading
- `latent_brush_demo/`: static browser demo for latent editing

## Why This Repo Exists

This repo is for exploring a simple question:

Can an autoencoder learn useful reconstructions by being asked to stay self-consistent in latent space under masking, cropping, and reconstruction, instead of relying only on direct pixel loss?

If you want the longer writeup and results, the blog post is here:

**[https://the-puzzler.github.io/share/self-teaching-autoencoder.html](https://the-puzzler.github.io/share/self-teaching-autoencoder.html)**
