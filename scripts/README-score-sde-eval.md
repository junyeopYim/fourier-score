# Original Score-SDE CIFAR metrics on existing samples

`scripts/evaluate_score_sde.py` evaluates an already completed `sample.py` output
with the official TF-Hub Inception model and TF-GAN 2.0.0. It requires one visible
GPU, exactly 50,000 generated uint8 NHWC images, and the official 50,000-image
CIFAR-10 feature reference. It neither trains nor samples.

Keep TensorFlow separate from the training environment:

```bash
uv venv --python 3.12 .venv-score-sde-eval
uv pip install --python .venv-score-sde-eval/bin/python \
  -r scripts/requirements-score-sde-eval.txt
mkdir -p pretrained/score_sde/evaluation
curl -fL \
  https://raw.githubusercontent.com/yang-song/score_sde_pytorch/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/evaluation.py \
  -o pretrained/score_sde/evaluation/evaluation.py
uvx --from gdown==5.2.1 gdown 14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI \
  -O pretrained/score_sde/evaluation/cifar10_stats.npz

# Expose this environment's pip-installed CUDA libraries to the dynamic loader.
eval_cuda_libs=$(.venv-score-sde-eval/bin/python -c \
  'import pathlib,sys; print(":".join(str(p) for p in pathlib.Path(sys.prefix).glob("lib/python*/site-packages/nvidia/*/lib")))')
LD_LIBRARY_PATH="$eval_cuda_libs" .venv-score-sde-eval/bin/python \
  scripts/evaluate_score_sde.py \
  --samples saved/cifar10_fg_s42_samples50k \
  --stats pretrained/score_sde/evaluation/cifar10_stats.npz \
  --upstream pretrained/score_sde/evaluation/evaluation.py \
  --output saved/cifar10_fg_s42_tfgan50k
```

The runner verifies the pinned upstream source hash, exact image counts, shard
continuity, and PNG/NPZ equality at each shard's boundaries. It calls the original
`run_inception_jit`, using its internal preprocessing and TF-Hub graph. It checks
CPU/GPU feature agreement on eight images and disables TF32. TF-GAN receives all
50,000 logits for one global IS and all 50,000 pool activations for FID, matching
the aggregation in upstream `run_lib.py`. This IS has no split-based standard
deviation and should not be compared as though it were a training-seed mean.

Compatibility changes are limited to omitting the unused JAX import, enabling
legacy Keras layers, and loading TF-GAN's unmodified evaluation modules without
its obsolete training/Estimator initializers. The TensorFlow/CUDA runtime is
modern; the original paper used TensorFlow 2.4.

`result.json` records metrics, versions, source/model/reference hashes, input
identity, checkpoint and sampling settings, and remaining comparison limits.
`sample_manifest.json` records every input shard. The generated Inception
activations are retained separately in `generated_activations.npz`.

Existing `sample.py` images use round-to-nearest uint8, while upstream generation
truncates to uint8. The float originals were not retained, so this evaluation
preserves the saved pixels. It also preserves the selected checkpoint and
sampler. Matching the metric backend does not reproduce the paper's best-FID
checkpoint selection or change the existing sampling protocol.

Sources: [official evaluator](https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/evaluation.py),
[metric aggregation](https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/run_lib.py),
[50k configuration](https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/configs/default_cifar10_configs.py),
[official reference download](https://github.com/yang-song/score_sde_pytorch#how-to-evaluate).
