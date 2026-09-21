# Frozen-first-stage LDM experiments

`ldm.py` connects official checkpoint loading, latent caches/statistics, from-scratch
native U-Net training, EMA DDIM/DDPM sampling, frozen decoding, and image metrics.
It is separate from the pixel-space `train.py` / `sample.py` checkpoint format.

## Native presets and fidelity to the paper

The U-Net, first-stage encoder/decoder, diffusion schedule utilities and LitEma
come from [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4),
pinned at `a506df5756472e2ebaf9078affdde2c4f1502cd4`. Computational modules retain
upstream code with package imports redirected. No modern Diffusers architecture
is substituted. The released training YAML files are preserved under
`configs/ldm/upstream/` and their hashes are stored in each experiment.

Training defaults use the **effective batch, actual learning rate and iterations
from Table 12**, and sampling defaults use Table 1 of the
[LDM paper](https://arxiv.org/html/2112.10752). `base_learning_rate` in the face/VQ
YAMLs is multiplied by the effective batch in the original runner; it is not the
actual optimizer learning rate. Churches explicitly disables this LR scaling.

| Config | Frozen first stage | Latent C×H×W | Batch | Actual LR | Updates | Loss | DDIM steps |
|---|---|---|---:|---:|---:|---|---:|
| `configs/ldm/ffhq.json` | VQ-f4 | 3×64×64 | 42 | 8.4e-5 | 635,000 | L2 | 200 |
| `configs/ldm/celebahq.json` | VQ-f4 | 3×64×64 | 48 | 9.6e-5 | 410,000 | L2 | 500 |
| `configs/ldm/lsun_churches.json` | KL-f8 | 4×32×32 | 96 | 5e-5 | 500,000 | **L1** | 200 |
| `configs/ldm/lsun_bedrooms.json` | VQ-f4 | 3×64×64 | 48 | 9.6e-5 | 1,900,000 | L2 | 200 |

All native models use 1,000 diffusion steps, AdamW (betas 0.9/0.999, epsilon 1e-8,
weight decay 0.01), and native LitEma (maximum decay 0.9999, warm-start updates).
Churches retains its 10,000-update LR warmup starting at 1e-6 of the target LR.
The upstream schedule named `linear` interpolates **sqrt(beta)** and squares it;
it is not the pixel runner's linearly spaced beta schedule.

`configs/ldm/lsun_churches_l2.json` changes only the common loss to L2 and labels
the run `l2`. Use this for the proposal's DSM comparison. `upstream` keeps L1 for
all Churches arms; that is a separate L1 experiment, not DSM. Never compare an
L1 epsilon arm against an L2 Fourier arm as a parameterization-only experiment.

Microbatch size defaults to 1; gradients are accumulated to the full effective
batch above. The final batch of an epoch may be shorter, as in the upstream
DataLoader. All arms must use the same microbatch. FP32 and TF32 disabled are
explicit. Optional `training.gradient_checkpointing=true` uses the native U-Net
checkpointing flag. Runtime and maximum memory still depend on hardware.

## Install and download

```bash
uv sync --locked --extra ldm --extra metrics
python scripts/download_ldm.py --model lsun_churches
uv run --locked --extra ldm python ldm.py inspect -c configs/ldm/lsun_churches.json

# Weights + original dataset + official split lists, in the preset paths:
uv run --locked --extra datasets python scripts/download_ldm.py \
  --model lsun_churches --with-data
```

The downloader supports all four models above. It writes `model.ckpt`, matching
inference `config.yaml`, and `download.json` under `pretrained/ldm/MODEL/`.
The manifest records pinned URLs, hashes and download time; repeat downloads
verify and reuse the files. The loader uses tensor-only `torch.load` and strict
computational state keys. Native pretrained denoiser import explicitly selects
raw or EMA tensors; training starts with a **new randomly initialized U-Net**.

Allow about 5.3 GB temporary disk space per full checkpoint download. No publisher
checksum is listed in the upstream model zoo; the recorded SHA-256 identifies
received bytes. `--sha256` can enforce an independently trusted archive digest.

## Dataset and split lists

`scripts/download_ldm_data.py` downloads or imports images and installs the exact
upstream splits into the preset paths below. `ldm.py prepare` then builds the
latent cache. Neither command changes the train/validation partition.

| Model | Image root | Training list | Validation list |
|---|---|---|---|
| FFHQ | `data/ffhq` | `data/ffhqtrain.txt` | `data/ffhqvalidation.txt` |
| CelebA-HQ | `data/celebahq` | `data/celebahqtrain.txt` | `data/celebahqvalidation.txt` |
| Churches | `data/lsun/churches` | `data/lsun/church_outdoor_train.txt` | `data/lsun/church_outdoor_val.txt` |
| Bedrooms | `data/lsun/bedrooms` | `data/lsun/bedrooms_train.txt` | `data/lsun/bedrooms_val.txt` |

```bash
# Show paths/sources without creating files or using the network.
python scripts/download_ldm_data.py --model ffhq --dry-run

# Dataset only; previously downloaded weights are reused separately.
uv run --locked --extra datasets python scripts/download_ldm_data.py --model ffhq
uv run --locked --extra datasets python scripts/download_ldm_data.py --model lsun_churches
uv run --locked --extra datasets python scripts/download_ldm_data.py --model lsun_bedrooms

# Reuse an existing download: original image directory, ZIP, or LSUN training LMDB.
uv run --locked --extra datasets python scripts/download_ldm_data.py \
  --model lsun_churches --source /datasets/church_outdoor_train_lmdb
python scripts/download_ldm_data.py --model ffhq --source /datasets/images1024x1024

# CelebA-HQ: the original .npy files from the upstream reconstruction procedure.
python scripts/download_ldm_data.py --model celebahq --source /datasets/celebahq
# A user-supplied ZIP URL containing those same .npy files is also supported:
# python scripts/download_ldm_data.py --model celebahq --url "$CELEBAHQ_NPY_ZIP_URL"

# Only install the official train/validation lists; do not fetch images.
python scripts/download_ldm_data.py --model celebahq --splits-only
```

FFHQ downloads the publisher's 1024×1024 PNGs using its metadata and checks each
image's MD5. It flattens the directory layout to the original CompVis filenames,
without resizing or changing image IDs. Publisher metadata and license files are
kept under `data/.downloads/ldm/ffhq/`. The CompVis split is 60,000/10,000; its
membership comes from the pinned lists, not from NVIDIA's sequential split.

LSUN downloads the publisher's **training** LMDB ZIP and exports its encoded
image bytes directly as `<key>.webp`, as the official exporter does. Both CompVis
train and validation come from this database: Churches has 121,227/5,000 images,
Bedrooms 3,028,042/5,000. The publisher's separate LSUN validation LMDB is not used.
The default is the publisher's HTTP URL from `fyu/lsun/download.py`; HTTPS at that
host currently has an invalid certificate. `--url` can select an alternate source
for the same archive and `--sha256` can enforce a trusted archive digest.

CelebA-HQ does not have a complete image archive linked by the original official
instructions: they describe reconstruction from CelebA and correction files.
Supply the resulting `imgHQXXXXX.npy` directory/ZIP with `--source`, or a URL to
that ZIP with `--url`. This installs all 30,000 arrays and the official 25,000/5,000
split. A differently numbered or JPEG-converted mirror is not silently substituted.
With `download_ldm.py --with-data`, pass this path as `--data-source`.

Downloads use `data/.downloads/ldm/`; HTTP range requests and Google Drive resume
support retain partial transfers. Existing completed download files are hash-checked.
Dataset import refuses conflicting existing images/splits, and writes
`data/<dataset>/download.json` only when all required files are installed. Completed
datasets are reused after checking official split hashes and nonempty image paths;
this fast reuse check does not hash the entire image collection. Local imports use
hard links when possible, otherwise copies, without moving the originals.
Google Drive quota or access failures remain explicit download errors.

Keep space for both the downloaded ZIP/LMDB and exported images. FFHQ images alone
are about 89 GB; LSUN Bedrooms is a much larger preparation than Churches. The
downloader does not remove its source cache automatically. `--data-dir` changes
the data prefix; when using it, also set the experiment's `data.root`,
`data.train_list`, and `data.validation_list` accordingly.

See the [upstream dataset preparation](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4#data-preparation)
and [taming face datasets](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/data/faceshq.py).
Every list contains one path relative to its image root. Duplicate entries and
overlapping train/validation paths are rejected. Custom lists are supported via
`--set data.train_list=... --set data.validation_list=...`; report them as a
modified data protocol. No random holdout is silently substituted.

Faces follow shortest-side OpenCV bilinear resizing then center cropping,
without random flips, matching taming ImagePaths on the original square images.
CelebA-HQ also accepts the upstream uint8 `[1,3,H,W]` NumPy files. LSUN follows
center-square cropping, PIL bicubic resizing and training-only horizontal flips.
Images are converted to RGB in [-1,1]. The frozen first stage stays in eval mode.

## Cache and training-only statistics

```bash
uv run --locked --extra ldm python ldm.py prepare \
  -c configs/ldm/lsun_churches.json --device cuda
```

`data/ldm_cache/MODEL/` contains float32 memory-mapped `train.npy` and
`validation.npy`, `stats.pt`, and a completed manifest with file hashes. Cache
preparation is staged in a temporary directory and published only on success.
An existing incompatible/corrupted cache is rejected; use a new `cache.dir`.

* KL caches posterior **mean and clamped log variance**, not one latent draw.
  Each training/evaluation visit resamples the native diagonal Gaussian.
* VQ caches continuous **pre-quantization** features. Decode includes the native
  nearest-codebook lookup followed by post-quantization convolution and decoder.
* Training flips are encoded as separate pixel-space views. Flipping a cached
  latent is not assumed to equal encoding the flipped image.
* The checkpoint's latent scale is reused exactly. For `scale_by_std` models,
  a checkpoint without `scale_factor` is rejected; it is never re-estimated.
* Mean and Fourier power are calculated only on training latents **after scaling**.
  For KL, posterior noise variance is integrated analytically into the spectrum:
  each channel's spatial-average conditional variance contributes to every FFT
  mode. Statistics therefore match posterior sampling without Monte Carlo noise.
* Scalar and Fourier controls share the same mean and statistics cache. Covariance
  remains diagonal across latent channels; no whitening is silently introduced.

The identity includes checkpoint/configuration, scale convention, split-list
hashes, source file size/mtime fingerprints, preprocessing, augmentation and power
floor. Cached arrays/statistics also have full content hashes. Source-image
fingerprints are fast metadata identities, not adversarial content checksums.
Cache preparation time is recorded separately for end-to-end cost accounting.
KL has roughly twice as many stored channels as a fixed draw; flips double train
storage again. Inspect available disk space before preparing a large LSUN set.

## Paired training and resume

```bash
# Inspect the 3 arms × 3 seeds before starting full paper-length runs.
uv run --locked --extra ldm python ldm.py compare \
  -c configs/ldm/lsun_churches_l2.json --seeds 42 43 44 --dry-run

# Explicit 10K-update pilot (not a paper-convergence claim).
uv run --locked --extra ldm python ldm.py compare \
  -c configs/ldm/lsun_churches_l2.json --device cuda --seeds 42 \
  --set training.iterations=10000 \
  --set 'name=churches_l2_pilot10k_{parameterization}_s{seed}'

# One arm, with the native full budget unless overridden.
uv run --locked --extra ldm python ldm.py train \
  -c configs/ldm/ffhq.json --device cuda --parameterization fourier_gaussian

uv run --locked --extra ldm python ldm.py train \
  -r saved/ldm_ffhq_upstream_fourier_gaussian_s42/last.pt
```

`compare` prepares/verifies the common cache once and runs each arm in a separate
process. Defaults are `epsilon`, `scalar_gaussian`, and `fourier_gaussian`;
`--parameterizations` can add `fourier_gaussian_unscaled`. Single-arm `train`
requires a completed cache. `--dry-run` reads configs only.

All paired runs start with the same raw U-Net weights, data order, posterior
samples, timesteps and noise. Native integer timesteps and the unchanged loss
weighting are retained. The adapter returns **total epsilon** in both training
and sampling:

```text
z_t = alpha_t z + sigma_t epsilon
scaled_score = sigma_t s_G + F^-1[b_t F(h_theta)]
epsilon_hat = -scaled_score
```

Only the denoiser is optimized; the first stage is not even resident during
cached training. `last.pt` saves optimizer, LitEma, consumed-batch cursor, RNGs,
statistics and immutable first-stage identity. `ema_STEP.pt` is inference-only.
Resume verifies source, config, cache identity, PyTorch version and device.
An interrupted partial optimizer step is never saved. Existing run directories
are not overwritten. Checkpoints reference the frozen weights instead of copying
them into every training snapshot; retain the original checkpoint for decoding.
Use `--set first_stage.checkpoint=/new/path/model.ckpt` for inference relocation;
the stored hash must still match.

`metrics.jsonl` records native training loss, validation epsilon MSE, noise ×
frequency diagnostics, learning rate, gradient norm and update throughput.
`optimizer_wall_seconds` measures synchronized updates including data access;
`training_wall_seconds` additionally includes setup, evaluation and checkpointing.
Cache preparation has its own time. Compare both matched updates and matched time.

## Native pretrained reference and trained samples

```bash
# Check the released baseline with the same sampler/decoder/metrics.
uv run --locked --extra ldm python ldm.py sample \
  -c configs/ldm/lsun_churches.json --pretrained --weights ema --device cuda \
  --num-samples 64 --batch-size 1 -o saved/churches_public_reference

uv run --locked --extra ldm python ldm.py sample \
  -r saved/ldm_ffhq_upstream_fourier_gaussian_s42/last.pt --device cuda \
  --num-samples 50000 -o saved/ffhq_fourier_50k

uv run --locked --extra ldm python ldm.py evaluate \
  -r saved/ldm_ffhq_upstream_fourier_gaussian_s42/last.pt \
  -o saved/ffhq_fourier_validation.json --device cuda
```

Sampling defaults to native uniform DDIM, eta=1.0, with 200 steps (500 for
CelebA-HQ). Native timestep offset `+1` and the first previous alpha are retained.
To avoid silently changing the number of evaluations, DDIM step counts must divide
1,000 and be smaller than 1,000 (e.g. 50/100/200/250/500). Full ancestral DDPM uses
`--set sampling.method=ddpm --steps 1000`. Latents are never clamped to RGB bounds.
All latent steps use the same epsilon adapter. Only decoded RGB is clamped and
converted to uint8, as in the original sample exporter.

Outputs include PNGs, uint8 NHWC NPZ shards, a preview and protocol metadata with
checkpoint hashes, EMA selection, actual sample count, NFE and elapsed time.
The public pretrained denoiser's training budget is not a matched experimental arm.

## Reconstruction and RGB FID

```bash
uv run --locked --extra ldm python ldm.py reconstruct \
  -c configs/ldm/lsun_churches.json --device cuda \
  --set evaluation.max_images=128 -o saved/churches_reconstruction

# Paper E.3.1 uses 50K generated images and the entire TRAINING split for FID.
uv run --locked --extra ldm python ldm.py export-real \
  -c configs/ldm/lsun_churches.json --split train -o saved/churches_real_train

uv run --locked --extra ldm --extra metrics python ldm.py fid \
  --real saved/churches_real_train/png --generated saved/churches_public_reference/png \
  --device cuda -o saved/churches_fid.json
```

Use 50K generated samples for final comparisons; the 64-image reference command
above only checks execution. Reconstruction exports original/reconstructed pairs
and PSNR; the `fid` command can also compare those folders to compute rFID.
Reconstruction FID is diagnostic, not a mathematical lower bound on generation FID.

The paper also uses **torch-fidelity**. This implementation records its installed
version and the evaluated image hashes/counts. Versions, preprocessing, real splits,
sampling settings and sample counts still must match across compared models.
Do not report smoke-test FID or a newly trained short pilot as reproduced paper FID.

Supported scope: the four unconditional released models above, on one device,
with a frozen pretrained first stage and denoiser training from scratch. Text/class
conditioning, pretrained-denoiser fine-tuning and automatic aggregate FID curves
are not implemented. The paper's long-training quality is an experimental result
to establish, not a consequence of matching configuration values.
