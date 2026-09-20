# LDM weights and comparison protocol

## Download an official model

The downloader uses only Python's standard library. It fetches one official
CompVis ZIP and the matching **inference config** from a pinned upstream commit.
The ZIP contains `model.ckpt`; it does not contain the config.

```bash
python scripts/download_ldm.py --list
python scripts/download_ldm.py --model ffhq --dry-run
python scripts/download_ldm.py --model ffhq
```

Choose `ffhq`, `celebahq`, `lsun_churches`, or `lsun_bedrooms`. All four are
unconditional 256-pixel models from the [official model zoo](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/README.md#pretrained-ldms).
Use `--output-dir /path/to/weights` to change the default root.

```text
pretrained/ldm/ffhq/
├── model.ckpt
├── config.yaml
└── download.json
```

`download.json` records the source URLs, upstream config revision, download time,
archive hash, and both output-file hashes. A repeated invocation verifies the
existing files and reuses them without contacting the server. An incomplete or
modified destination is rejected rather than overwritten.

The inspected archives are about 2.2–2.5 GB; extraction temporarily needs space
for both the ZIP and the 2.4–2.7 GB checkpoint. Allow roughly 5.3 GB of free disk.
The temporary ZIP is removed after installation. Interrupted downloads are
cleaned up; rerunning starts the transfer again. Progress is printed during transfer.

No publisher SHA-256 is supplied in the model-zoo listing. Recorded hashes identify
the bytes received; they are not an independent publisher checksum. If you have a
trusted archive digest, pass `--sha256 <64-hex-characters>` to require a match.
The script also checks transfer length and ZIP CRC. It never unpickles the weights.

**Implemented here:** download, provenance, and local file verification.
**Still required:** a CompVis checkpoint loader, latent data/statistics path, and
LDM training/sampling integration. These weights cannot yet be passed directly to
this repository's `train.py` or `sample.py`, which load the local v1 NCSN++ format.
The upstream config is preserved exactly; no incompatible local JSON preset is created.

## First experiment: freeze the autoencoder and train the latent denoiser

If FFHQ data is available, start with **FFHQ LDM-VQ-4**. Its unconditional model
avoids prompt/guidance confounders and its default L2 noise-prediction loss fits
the present DSM comparison. LSUN-Churches uses `loss_type: l1`: changing that to
L2 only for the proposed arm would confound the comparison.
See the [FFHQ config](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/models/ldm/ffhq256/config.yaml),
[Churches config](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/models/ldm/lsun_churches256/config.yaml),
and [LDM loss implementation](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py).

1. Load the official model and reproduce its sampling behavior before modifying it.
   Record checkpoint hash, EMA choice, preprocessing, sampler, steps, eta, batch
   size, seed, sample count, and FID implementation. Re-evaluate the public model
   with the same FID pipeline used for the proposed model.
2. Freeze the pretrained first-stage encoder/decoder. Preserve the original U-Net
   architecture, timestep conditioning, noise schedule, and loss weighting.
3. Produce training latents using the exact first-stage path and checkpoint scale:
   `z = ldm.get_first_stage_encoding(ldm.encode_first_stage(images))`.
   Estimate `mu` and Fourier `P` on these **training** latents, not RGB images.
4. Train the arms below from the same randomly initialized U-Net weights, holding
   the autoencoder fixed. Pair training seeds, data order, optimizer, EMA, effective
   batch, and evaluation settings. Report the full pretrained LDM separately as
   a reference; its unknown/much larger training budget is not a matched-budget arm.

| Arm | Denoiser output interpretation | Purpose |
|---|---|---|
| Original epsilon prediction | `epsilon_hat = h_theta` | Native LDM baseline |
| Scalar Gaussian | Channelwise constant covariance | Gaussian reference without frequency-specific covariance |
| Fourier Gaussian, unscaled | Fourier reference plus unscaled residual | Isolate the reference term |
| Fourier Gaussian | Fourier reference plus frequency-scaled residual | Full proposal |

For a first implementation, baseline / scalar / full Fourier are the three main
arms. Add the unscaled control to isolate residual scaling. Use the same L2 loss
for these experiments; a matched L1 comparison can be a separately labeled extension.

## Where the proposed method enters LDM

Keep the native LDM epsilon-prediction interface. For
`z_t = alpha_t * z + sigma_t * epsilon`, turn the U-Net's raw output into

```text
scaled_score = sigma_t * s_G(z_t, t) + F^-1[b_t * F(h_theta(z_t, t))]
epsilon_hat  = -scaled_score
loss         = mean((epsilon_hat - epsilon)^2)
```

The existing [FourierGaussian](../fourier_score/method.py) implements the first
line. In a future adapter, use the pretrained LDM's `sqrt_alphas_cumprod[t]` and
`sqrt_one_minus_alphas_cumprod[t]` for alpha and sigma. Preserve its actual schedule
and integer timestep inputs; the local VE/NCSN++ process is not a substitute.
The adapted total epsilon must be returned in **both training and sampling**
(including DDIM), while the upstream `parameterization` remains `eps`.

The VQ first stage used by FFHQ returns continuous, pre-quantization features
from `VQModelInterface.encode`; estimate statistics on that exact representation,
not codebook indices. KL models sample their encoder posterior in the native
first-stage path. Replacing samples with posterior means or caching one draw
changes the latent data distribution; choose and document a common protocol for
all arms. Preserve the checkpoint's latent scaling rather than re-estimating it.
Sources: [first-stage encoding](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/diffusion/ddpm.py),
[VQ interface](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/autoencoder.py).

Cache identities should include the encoder/checkpoint hash, latent scaling,
posterior convention, data split, preprocessing/augmentation, and statistics floor.
Before training, inspect whether latent power varies across spatial frequencies;
this measures how much frequency information distinguishes the scalar and Fourier controls.

## A separate question: can a completed LDM improve with extra training?

Use three arms with the same frozen autoencoder and data:

| Arm | Role |
|---|---|
| Public pretrained LDM, no extra training | Starting quality |
| Original LDM continued for a fixed budget | Effect of extra training |
| Proposed parameterization adapted and trained for the same total budget | Additional effect of the proposal |

**Copying pretrained epsilon-prediction weights does not initialize a Gaussian
residual predictor with the same predictions.** Evaluate the model immediately
after any conversion and record its quality before further training.

A practical option is a short teacher-matching warm start: initialize a student
from the public U-Net and fit its **converted total epsilon prediction** to the
frozen public model's epsilon prediction. Then train on the original noise target.
This is an approximate initialization, not an exact weight conversion. Count its
compute in the proposed arm's budget and give the original baseline the same total
budget. Compare both update counts and elapsed GPU time; teacher inference has a cost.

An exactly prediction-preserving algebraic wrapper produces the same samples
before training under the same sampler/RNG. If its inverse transformations simply
cancel on every forward pass, it is still the original function and does not test
the proposed parameterization. Prefer the frozen-autoencoder, from-scratch
denoiser experiment above for the primary learning-efficiency claim.

## What to measure and how to interpret it

| Measurement | Interpretation |
|---|---|
| FID at matched updates and at matched training time | Quality and learning efficiency, including Fourier/statistics overhead |
| Updates/time to a fixed FID target | Whether the method reaches useful quality sooner |
| FID versus sampler NFE and generation time | Sampling quality/cost tradeoff after training |
| Noise × frequency DSM and generated-image spectra | Where errors change; diagnostic evidence, not a substitute for image quality |
| Repeated training seeds and fixed-seed sample grids | Variability and visible artifacts |

Use cheap pilot evaluations before final 50K-sample FID comparisons, with identical
sample counts within each comparison. Report the distinction between early learning
improvements and final quality. Improved high-noise or frequency-band DSM, faster
learning, and better FID are hypotheses to test; none follows automatically from
adding the Gaussian reference. If latent spectra are already close to flat, the
Fourier arm may provide little benefit over the scalar control.
