# Self-Teaching Autoencoder

A small experiment in training an autoencoder to teach itself.

Instead of relying only on pixel reconstruction loss, the model also asks a second question: if a clean view and a masked view come from the same image, can their reconstructions be made consistent in the model's own latent space?

**Blog post:** [self-teaching-autoencoder](https://the-puzzler.github.io/share/self-teaching-autoencoder.html)

![Our method vs baseline](our_method_vs_baseline_latent512_checkpoint100.png)

The image above compares the current method against a plain masked-autoencoder baseline at latent size `512` (96x compresion).

## Overview

This repo trains autoencoders on `CelebA`, center-cropped to `128x128`.

The main training scripts are:

- `main.py`: the current self-teaching method
- `main_regular.py`: the baseline masked-image reconstruction model

The model itself lives in `leae/autoencoder.py`.

## Core Idea

The model sees two versions of the same image:

- the original clean image
- a masked version with a square region removed and filled with the image average

Both views are passed through the same autoencoder. A slowly refreshed copy of the encoder then acts as a judge. The autoencoder is trained so that its reconstructions become mutually consistent under that judge.

That is why this is a self-teaching autoencoder:

- the student is the current autoencoder
- the teacher is a target copy of the same encoder

## Current Objective

The current repo is centered on the symmetric latent-consistency objective in `main.py`.

For an image `x`:

```text
z_clean = E(x)
x_clean_hat = D(z_clean)

z_masked = E(mask(x))
x_masked_hat = D(z_masked)
```

The target encoder `T` is then used to score the reconstructions:

```text
consistency_loss =
    MSE(T(x_clean_hat), T(x_masked_hat))

clean_crop_loss =
    MSE(T(crop(x)), T(crop(x_clean_hat)))

masked_crop_loss =
    MSE(T(crop(x)), T(crop(x_masked_hat)))
```

These are averaged into the main latent objective:

```text
mse_loss = average(consistency_loss, clean_crop_loss, masked_crop_loss)
```

There is also a latent regularizer on both clean and masked codes:

```text
sigreg_loss =
    0.5 * lambda * (SIGReg(z_clean) + SIGReg(z_masked))
```

Final training loss:

```text
loss = mse_loss + sigreg_loss
```

The important part is the symmetry:

- both clean and masked branches go through the full autoencoder
- both reconstructions are judged in the same latent space
- both branches help define what a "good" reconstruction is

## Baseline

The baseline in `main_regular.py` is intentionally simple:

```text
recon = model(masked_image)
loss = MSE(recon, image)
```

It does not use:

- a target encoder
- latent consistency between branches
- crop consistency losses
- `SIGReg`

So it serves as the direct "just reconstruct the image" comparison.

## Running

uv sync

uv run main.py

## Repo Layout

- `main.py`: current self-teaching training loop
- `main_regular.py`: baseline training loop
- `leae/autoencoder.py`: autoencoder architecture
- `leae/masking.py`: masking and crop helpers
- `leae/prep_data.py`: dataset loading
- `leae/sigreg.py`: latent regularizer
- `latent_brush_demo/`: browser demo for editing pooled latents

## Why This Exists

This repo explores a simple question:

Can an autoencoder learn stronger representations if it is trained to stay self-consistent under masking and reconstruction, instead of optimizing only direct pixel error?

For the longer writeup and results, see:

**[https://the-puzzler.github.io/share/self-teaching-autoencoder.html](https://the-puzzler.github.io/share/self-teaching-autoencoder.html)**
