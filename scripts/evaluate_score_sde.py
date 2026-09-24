"""Evaluate existing CIFAR-10 uint8 samples with the original Score-SDE metrics.

Run in a separate TensorFlow environment; no training or sampling is performed.
The upstream evaluation.py and official reference statistics are explicit inputs.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import types


REVISION = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
EVALUATION_SHA256 = "6a9d880c02187bc8dc2c51f2f37ff0fde5ee4287a25316ff8e26a69b940194af"
STATS_URL = "https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view"
COUNT = 50_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def tfgan_evaluation_only():
    """Load unmodified TF-GAN metric modules without removed Estimator APIs.

    TF-GAN 2.0's top-level initializer imports training/Estimator code that is
    unrelated to metrics and incompatible with current TensorFlow. Restrict
    package initialization to its installed, unchanged evaluation modules.
    """
    distribution = importlib.metadata.distribution("tensorflow-gan")
    if distribution.version != "2.0.0":
        raise RuntimeError("Use tensorflow-gan==2.0.0, as in Score-SDE requirements")
    base = Path(distribution.locate_file("tensorflow_gan"))
    for name, directory in (
        ("tensorflow_gan", base),
        ("tensorflow_gan.python", base / "python"),
        ("tensorflow_gan.python.eval", base / "python/eval"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package
    metrics = importlib.import_module("tensorflow_gan.python.eval.classifier_metrics")
    sys.modules["tensorflow_gan"].eval = metrics
    return metrics, {
        name: sha256(base / "python/eval" / name)
        for name in ("classifier_metrics.py", "eval_utils.py")
    }


def load_upstream(path):
    if sha256(path) != EVALUATION_SHA256:
        raise ValueError("evaluation.py does not match the pinned official source")
    source = path.read_text()
    # Single-GPU evaluation calls run_inception_jit directly. JAX is only used
    # by the unused multi-device wrapper, so it need not own GPU memory here.
    # Preserve line numbers: AutoGraph inspects the original file on disk.
    source = source.replace("import jax\n", "# JAX is unused by the single-GPU entry point.\n")
    module = types.ModuleType("score_sde_official_evaluation")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def read_samples(directory):
    import numpy as np
    from PIL import Image

    settings = json.loads((directory / "settings.json").read_text())
    if settings.get("complete") is not True or settings.get("num_samples") != COUNT:
        raise ValueError("Require a completed 50,000-sample run")
    files = sorted(directory.glob("samples_*.npz"))
    arrays, manifest = [], []
    digest = hashlib.sha256()
    index = 0
    for shard, path in enumerate(files):
        if int(path.stem.split("_")[-1]) != shard:
            raise ValueError("Non-contiguous sample shards")
        with np.load(path, allow_pickle=False) as data:
            images = data["samples"]
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(f"Expected uint8 NHWC CIFAR images: {path}")
        if not len(images):
            raise ValueError(f"Empty sample shard: {path}")
        # Check both boundaries of every shard against the PNGs used previously.
        for offset in sorted({0, len(images) - 1}):
            with Image.open(directory / "png" / f"{index + offset:07d}.png") as image:
                if not np.array_equal(np.asarray(image), images[offset]):
                    raise ValueError("NPZ and existing evaluation PNGs differ")
        digest.update(images.tobytes(order="C"))
        manifest.append({"name": path.name, "count": len(images), "sha256": sha256(path)})
        arrays.append(images)
        index += len(images)
    if index != COUNT or len(list((directory / "png").glob("*.png"))) != COUNT:
        raise ValueError(f"Expected exactly {COUNT} images; NPZ count={index}")
    return np.concatenate(arrays), settings, {
        "path": str(directory.resolve()),
        "count": index,
        "uint8_nhwc_sha256": digest.hexdigest(),
        "settings_sha256": sha256(directory / "settings.json"),
        "shards": manifest,
    }


def evaluate(args):
    import numpy as np

    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Completed output already exists; choose a new output directory")

    def progress(phase, **fields):
        state = {"state": "running", "phase": phase, "updated_unix": time.time(), **fields}
        write_json(output / "status.json", state)
        print(json.dumps(state), flush=True)

    progress("verifying_existing_samples")
    samples, settings, manifest = read_samples(args.samples)
    write_json(output / "sample_manifest.json", manifest)
    stats_hash = sha256(args.stats)
    with np.load(args.stats, allow_pickle=False) as data:
        real_pool = data["pool_3"]
    if real_pool.shape != (COUNT, 2048) or not np.isfinite(real_pool).all():
        raise ValueError(f"Unexpected official CIFAR reference: {real_pool.shape}")

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    import tensorflow_hub as hub

    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError(f"This evaluator requires exactly one visible GPU: {devices}")
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    metrics, metric_hashes = tfgan_evaluation_only()
    upstream = load_upstream(args.upstream)
    progress("loading_official_inception")
    model_path = Path(hub.resolve(upstream.INCEPTION_TFHUB))
    with tf.device("/GPU:0"):
        model = upstream.get_inception_model(inceptionv3=False)

    pool, logits = [], []
    extraction_start = time.time()
    parity = None
    for offset in range(0, COUNT, args.batch_size):
        batch = samples[offset : offset + args.batch_size]
        with tf.device("/GPU:0"):
            features = upstream.run_inception_jit(batch, model, inceptionv3=False)
        for name, width in (("pool_3", 2048), ("logits", 1008)):
            tensor = features[name]
            if "GPU:0" not in tensor.device:
                raise RuntimeError(f"Inception did not run on the GPU: {tensor.device}")
            values = tensor.numpy()
            if values.shape != (len(batch), width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} activations: {values.shape}")
        pool.append(features["pool_3"].numpy())
        logits.append(features["logits"].numpy())
        if offset == 0:
            # Check the modern GPU execution against a CPU pass over the same
            # eight images, using the identical official graph/preprocessing.
            with tf.device("/CPU:0"):
                cpu = upstream.run_inception_jit(batch[:8], model, inceptionv3=False)
            parity = {}
            for name in ("pool_3", "logits"):
                a, b = features[name].numpy()[:8], cpu[name].numpy()
                np.testing.assert_allclose(a, b, rtol=5e-4, atol=5e-4)
                parity[name + "_max_abs"] = float(np.abs(a - b).max())
            write_json(output / "cpu_gpu_parity.json", parity)
        done = offset + len(batch)
        if offset == 0 or done == COUNT or done % (args.batch_size * 20) == 0:
            progress("extracting_inception", count=done, total=COUNT,
                     elapsed_seconds=time.time() - extraction_start)
    pool = np.concatenate(pool)
    logits = np.concatenate(logits)
    np.savez(output / "generated_activations.npz", pool_3=pool, logits=logits)
    progress("computing_fid_is", count=COUNT, total=COUNT)
    # The original run_lib.py computes both metrics over all 50k activations,
    # with one global IS calculation (no ten-way split mean/std).
    with tf.device("/CPU:0"):
        inception_score = float(metrics.classifier_score_from_logits(logits).numpy())
        fid = float(metrics.frechet_classifier_distance_from_activations(real_pool, pool).numpy())
    if not all(math.isfinite(value) for value in (fid, inception_score)):
        raise FloatingPointError("Nonfinite final metrics")
    result = {
        "metrics": {"frechet_inception_distance": fid, "inception_score": inception_score},
        "implementation": "score_sde_pytorch evaluation.py + tensorflow-gan 2.0.0",
        "upstream_revision": REVISION,
        "upstream_evaluation_sha256": EVALUATION_SHA256,
        "tfgan_metric_source_sha256": metric_hashes,
        "inception": {"url": upstream.INCEPTION_TFHUB,
                      "files": {str(p.relative_to(model_path)): sha256(p)
                                for p in sorted(model_path.rglob("*")) if p.is_file()}},
        "real": {"count": len(real_pool), "stats_path": str(args.stats.resolve()),
                 "stats_sha256": stats_hash, "official_url": STATS_URL},
        "generated": {key: value for key, value in manifest.items() if key != "shards"},
        "checkpoint_step": settings.get("checkpoint_step"),
        "checkpoint_sha256": settings.get("checkpoint_sha256"),
        "training_seed": settings["config"]["seed"],
        "parameterization": settings["config"]["loss"]["type"],
        "sampling": settings["config"]["sampling"],
        "device": tf.config.experimental.get_device_details(devices[0]),
        "batch_size": args.batch_size,
        "tf32": False,
        "cpu_gpu_parity": parity,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("tensorflow", "tensorflow-gan", "tensorflow-hub",
                      "tensorflow-probability", "tf-keras", "numpy", "setuptools")},
        "compatibility_changes": [
            "Call the original run_inception_jit on one GPU; omit unused JAX import.",
            "Use tf-keras's legacy layers API for the original flatten call.",
            "Load unchanged TF-GAN evaluation modules without unrelated Estimator training imports.",
        ],
        "comparison_limits": [
            "Reuses the same saved uint8 images; no resampling or re-quantization.",
            "These samples used round-to-nearest uint8; upstream generation truncates to uint8. Float originals were not retained.",
            "This is a fixed final checkpoint, not the paper's best-FID checkpoint selection.",
            "Modern TensorFlow/CUDA runtime; metric graph, reference stats and TF-GAN formulas follow upstream.",
        ],
        "wall_seconds": time.time() - started,
        "completed_unix": time.time(),
    }
    write_json(output / "result.json", result)
    (output / "failure.json").unlink(missing_ok=True)
    write_json(output / "status.json", {"state": "completed", "metrics": result["metrics"],
                                       "updated_unix": time.time()})
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 8:
        parser.error("batch-size must be at least eight for the parity check")
    try:
        evaluate(args)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "state": "failed", "error": f"{type(error).__name__}: {error}",
            "updated_unix": time.time(),
        }
        if not (args.output / "result.json").exists():
            write_json(args.output / "failure.json", failure)
            write_json(args.output / "status.json", failure)
        raise


if __name__ == "__main__":
    main()
