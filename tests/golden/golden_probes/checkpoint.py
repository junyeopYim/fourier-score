"""Checkpoint contracts: epoch-0 payloads, payload schemas, state_dict keys, overrides.

Protects the moves of the model / trainer / checkpoint modules (and the LDM
and GMM loaders that share them): files written by the epoch-0 code must keep
loading strictly and computing the same outputs, freshly written checkpoints
must keep their structure and run layout, and module/state_dict names must
not change.

Payload writers run only in the recorder, with the epoch-0 tree, and write
``tests/golden/payloads/checkpoint/``:

* ``pixel/<run>/``: ``configs/smoke.json`` runs (nf=8, 3 updates, CPU, one
  thread) through ``train.py``.  The gated ``fourier_gaussian`` run
  (log_sigma gate, normalized_residual) keeps ``last.pt`` and
  ``ema_000000003.pt``; the ``score`` run keeps its EMA snapshot.
* ``ldm/``: the ``tests/test_ldm.py::latent_experiment`` recipe
  (AutoencoderKL, copied below) with a micro U-Net, 2 updates and
  ``snapshot_every=2``: ``tiny_fourier_gaussian_s42/{last.pt,ema_000000002.pt}``
  plus ``native.yaml``, the image lists/images and the latent ``cache/``.
* ``gmm/<case>/<arm>_seed42/{checkpoint.pt,metrics.json}``: ``train_arm`` on a
  4x4 config smaller than the runners' smoke preset (a baseline and a gate).

Payload budget (< 3 MB): training ``last.pt`` payloads drop their
``optimizer`` entry (about half of each file).  Refactored code can never read
it: resuming requires the checkpoint's own ``source_sha256`` and inference
ignores it; ``payload_schema`` records the full structure of fresh
checkpoints instead.  For the same reason the score run keeps only its EMA
snapshot and the frozen first stage (``public.ckpt``, 2.6 MB for the recipe)
is not kept: ``load_trained`` never reads it and ``state_dict_keys`` pins the
first-stage key layout.

Reproducible bytes: writers train in fixed directories under the recorder's
fixed scratch root (saved configs and the latent-cache manifest embed those
paths), image mtimes are pinned (they enter the cache identity), and the
wall-clock fields in :data:`WALL_CLOCK_KEYS` are set to ``0.0`` in every
stored checkpoint and JSON file.  Every other entry is what epoch 0 saved.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

from golden_probes import (
    config_path, entrypoint, floats, json_digest, payload_writer, probe, shapes, symbol,
    tensor_digest,
)

GROUP = "checkpoint"

CONFIG = ("fourier_score.config",)
CHECKPOINTS = ("fourier_score.trainer.checkpoints", "fourier_score.checkpoints")
MODEL = ("fourier_score.model.model", "fourier_score.model")
GATES = ("fourier_score.gates", "fourier_score.method")
SAMPLING = ("fourier_score.model.sampling", "fourier_score.diffusion")
EMA = ("fourier_score.trainer.ema", "fourier_score.ema")
LDM_CONFIG = ("fourier_score.ldm.config",)
LDM_DATA = ("fourier_score.ldm.data",)
LDM_MODEL = ("fourier_score.ldm.model",)
LDM_TRAINING = ("fourier_score.ldm.training",)
LDM_FIRST_STAGE = ("fourier_score.ldm.first_stage",)
LDM_EVALUATION = ("fourier_score.ldm.evaluation",)
LDM_EMA = ("fourier_score.ldm.upstream.ema",)
GMM = ("fourier_score.gmm",)

# ------------------------------------------------------------------ recipes

PIXEL_ARMS = {
    "score": ("--parameterization", "score"),
    "fourier_gaussian_gated": (
        "--parameterization", "fourier_gaussian",
        "--set", "loss.objective=normalized_residual",
        "--set", "fourier.gate.mode=log_sigma",
    ),
}
PIXEL_KEEP = {"score": ("ema_000000003.pt",), "fourier_gaussian_gated": ("last.pt", "ema_000000003.pt")}
# Machine-specific paths inside saved configs; replaced before digesting.
PIXEL_PATHS = (("trainer", "save_dir"), ("fourier", "cache_dir"), ("data_loader", "args", "root"))
LDM_PATHS = (
    ("upstream_config",), ("first_stage", "checkpoint"), ("data", "root"), ("data", "train_list"),
    ("data", "validation_list"), ("cache", "dir"), ("training", "save_dir"),
)

# tests/test_ldm.py::latent_experiment U-Net, and the smallest GroupNorm32-valid
# variant used for payloads (~105k parameters instead of ~715k).
TINY_UNET = {
    "image_size": 4, "in_channels": 2, "out_channels": 2, "model_channels": 32,
    "attention_resolutions": [1, 2], "num_res_blocks": 1, "channel_mult": [1, 2], "num_heads": 4,
}
MICRO_UNET = {**TINY_UNET, "attention_resolutions": [], "num_res_blocks": 0, "channel_mult": [1]}

GMM_SEED = 42
GMM_LAMBDA = 1.0
# The runners' smoke preset (4x4, width 32, 12 steps, 3 bins) made smaller still.
GMM_TINY = dict(
    preset="golden-checkpoint", run_tag="golden-checkpoint", image_size=4, spectrum_lambdas=(GMM_LAMBDA,),
    distributions=("gmm",), seeds=(GMM_SEED,), steps=4, eval_every=2, batch_size=8, width=16, depth=2,
    n_noise_levels=3, val_per_noise=8, test_per_noise=16, cpu_threads=1, device="cpu",
    bank_version="gmm-gated-v1",
)
# Stand-in for the runners' provenance (python/torch/git/source hashes).
GMM_PROVENANCE = {"golden": "checkpoint-payload-v1", "source_sha256": "fixed", "torch": "fixed"}

# String values kept verbatim by describe(); every other leaf becomes its type.
VALUE_KEYS = frozenset({"format", "kind"})
# Timing fields zeroed in stored payloads (no loader or probe reads them).
WALL_CLOCK_KEYS = frozenset(
    {"training_wall_seconds", "optimizer_wall_seconds", "preparation_wall_seconds", "optimizer_seconds"}
)
# Pinned st_mtime_ns of the LDM recipe images (ImageList fingerprints include it).
IMAGE_MTIME_NS = 1_600_000_000_000_000_000


def payload_root(ctx, part):
    return ctx.payload_dir / GROUP / part


def fresh_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def scratch(ctx):
    return Path(tempfile.mkdtemp(dir=ctx.tmp))


def writer_dir(ctx, name):
    """Fixed-name work dir for a payload writer (its path ends up in the payload)."""
    return fresh_dir(ctx.tmp / f"payload-{name}")


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def zero_wall_clock(value):
    """Set every WALL_CLOCK_KEYS float to 0.0 at any depth, in place.

    In place so container types (state_dict OrderedDicts and their
    ``_metadata``) are saved back unchanged.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if k in WALL_CLOCK_KEYS and isinstance(v, float):
                value[k] = 0.0
            else:
                zero_wall_clock(v)
    elif isinstance(value, list):
        for v in value:
            zero_wall_clock(v)
    return value


def copy_json(src, dst):
    """JSON file with zeroed wall clock, written like fourier_score.utils.json_write."""
    obj = zero_wall_clock(json.loads(src.read_text()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def copy_checkpoint(src, dst, drop=()):
    """Checkpoint with zeroed wall clock and without the ``drop`` entries."""
    import torch

    ckpt = zero_wall_clock(torch.load(src, map_location="cpu", weights_only=True))
    for key in drop:
        del ckpt[key]
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)


def copy_without_optimizer(src, dst):
    copy_checkpoint(src, dst, drop=("optimizer",))


def train_pixel(ctx, work, arm):
    """``train.py -c configs/smoke.json`` in a subprocess; returns the run dir."""
    command = [
        *entrypoint(ctx, "train.py"), "-c", str(config_path("configs/smoke.json")),
        "--device", "cpu", "--set", f"trainer.save_dir={work / 'runs'}",
        "--set", f"fourier.cache_dir={work / 'stats'}", "--set", "backend.cpu_threads=1",
        *PIXEL_ARMS[arm],
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
    env.pop("PYTHONSAFEPATH", None)  # train.py must import the tree next to it
    done = subprocess.run(command, cwd=ctx.root, env=env, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"train.py ({arm}) failed:\n{done.stderr[-4000:]}")
    (run,) = (work / "runs").iterdir()
    return run


def ldm_recipe(root, work, kind="AutoencoderKL", unet=TINY_UNET):
    """Copy of tests/test_ldm.py::latent_experiment (config + spec part)."""
    import yaml

    defaults = symbol("defaults", *LDM_CONFIG)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    work.mkdir(parents=True, exist_ok=True)
    cfg = defaults("ffhq")
    source = yaml.safe_load((root / "configs/ldm/upstream/ffhq.yaml").read_text())
    p = source["model"]["params"]
    kl = kind == "AutoencoderKL"
    p.update(image_size=4, channels=2, timesteps=10, scale_by_std=kl)
    p["unet_config"]["params"] = copy.deepcopy(unet)
    p["first_stage_config"] = {
        "target": "ldm.models.autoencoder." + kind,
        "params": {
            "embed_dim": 2,
            "ddconfig": {
                "double_z": kl, "z_channels": 2, "resolution": 8, "in_channels": 3, "out_ch": 3,
                "ch": 32, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": [], "dropout": 0.0,
            },
            "lossconfig": {"target": "torch.nn.Identity"},
        },
    }
    if not kl:
        p["first_stage_config"]["params"]["n_embed"] = 16
    source["data"]["params"]["train"]["params"]["size"] = 8
    spec_path = work / "native.yaml"
    spec_path.write_text(yaml.safe_dump(source))
    cfg.update(upstream_config=str(spec_path), device="cpu", name="tiny_{parameterization}_s{seed}")
    cfg["sampling"].update(steps=5, num_samples=2, batch_size=2, decode_batch_size=1, eta=0.0)
    cfg["evaluation"].update(batch_size=2, max_images=2, noise_bins=2, frequency_bins=2)
    cfg["training"].update(
        iterations=2, batch_size=2, microbatch_size=1, save_dir=str(work / "runs"), save_every=1,
        snapshot_every=0, eval_every=1, log_every=1, console="quiet", lr=1e-3,
    )
    cfg["backend"]["cpu_threads"] = 1  # the recipe uses 2; probes run on one thread
    cfg["cache"].update(dir=str(work / "cache"), batch_size=2)
    return cfg, load_spec(cfg)


def ldm_materialize(work, cfg, spec):
    """Recipe images, split lists and the random frozen first stage (public.ckpt)."""
    import numpy as np
    from PIL import Image
    import torch

    FrozenFirstStage = symbol("FrozenFirstStage", *LDM_FIRST_STAGE)
    images = work / "images"
    images.mkdir()
    rng = np.random.default_rng(17)
    for i in range(6):
        Image.fromarray(rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)).save(images / f"{i}.png")
    for path in images.iterdir():
        os.utime(path, ns=(IMAGE_MTIME_NS, IMAGE_MTIME_NS))
    for split, ids in (("train", range(4)), ("validation", range(4, 6))):
        listing = work / f"{split}.txt"
        listing.write_text("".join(f"{i}.png\n" for i in ids))
        cfg["data"][split + "_list"] = str(listing)
    cfg["data"].update(root=str(images), random_flip=True)
    torch.manual_seed(123)
    stage = FrozenFirstStage(spec)
    state = {"first_stage_model." + k: v for k, v in stage.state_dict().items()}
    if spec["first_stage"]["kind"] == "AutoencoderKL":
        state["scale_factor"] = torch.tensor(0.7)
    checkpoint = work / "public.ckpt"
    torch.save({"state_dict": state}, checkpoint)
    cfg["first_stage"]["checkpoint"] = str(checkpoint)


def train_ldm(root, work):
    """Micro-U-Net fourier_gaussian run: 2 updates, EMA snapshot at update 2."""
    import torch

    prepare_cache = symbol("prepare_cache", *LDM_DATA)
    Trainer = symbol("Trainer", *LDM_TRAINING)
    cfg, spec = ldm_recipe(root, work, unet=MICRO_UNET)
    cfg["parameterization"] = "fourier_gaussian"
    cfg["training"]["snapshot_every"] = 2
    ldm_materialize(work, cfg, spec)
    prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda _: None)
    return Trainer(cfg, spec).train().parent


@contextmanager
def deterministic_algorithms():
    """The GMM runners enable deterministic algorithms around train_arm."""
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def gmm_setup():
    GMMConfig = symbol("GMMConfig", *GMM)
    BASELINE_ARMS = symbol("BASELINE_ARMS", *GMM)
    gated_arm = symbol("gated_arm", *GMM)
    arms = (next(a for a in BASELINE_ARMS if a.name == "score"), gated_arm("fourier"))
    cfg = GMMConfig(**GMM_TINY, methods=tuple(a.name for a in arms))
    family = symbol("MatchedMomentFamily", *GMM)(cfg, "gmm", GMM_LAMBDA)
    return cfg, family, arms


def train_gmm(family, arms, output):
    validation = symbol("make_bank", *GMM)(family, "validation")
    train_arm = symbol("train_arm", *GMM)
    with deterministic_algorithms():
        for arm in arms:
            train_arm(family, arm, GMM_SEED, validation, output, GMM_PROVENANCE)
    return [Path(family.case_id) / f"{arm.name}_seed{GMM_SEED}" for arm in arms]


# ------------------------------------------------------------------ writers


@payload_writer(GROUP, "pixel")
def write_pixel(ctx):
    target = fresh_dir(payload_root(ctx, "pixel"))
    for arm, names in PIXEL_KEEP.items():
        run = train_pixel(ctx, writer_dir(ctx, f"pixel-{arm}"), arm)
        for name in names:
            write = copy_without_optimizer if name == "last.pt" else copy_checkpoint
            write(run / name, target / run.name / name)


@payload_writer(GROUP, "ldm")
def write_ldm(ctx):
    target = fresh_dir(payload_root(ctx, "ldm"))
    work = writer_dir(ctx, "ldm")
    run = train_ldm(ctx.root, work)
    copy_without_optimizer(run / "last.pt", target / run.name / "last.pt")
    copy_checkpoint(run / "ema_000000002.pt", target / run.name / "ema_000000002.pt")
    for name in ("native.yaml", "train.txt", "validation.txt"):
        copy_file(work / name, target / name)
    for folder in ("images", "cache"):
        shutil.copytree(work / folder, target / folder, copy_function=shutil.copyfile)
    copy_json(work / "cache/manifest.json", target / "cache/manifest.json")


@payload_writer(GROUP, "gmm")
def write_gmm(ctx):
    target = fresh_dir(payload_root(ctx, "gmm"))
    work = writer_dir(ctx, "gmm")
    _cfg, family, arms = gmm_setup()
    for folder in train_gmm(family, arms, work):
        copy_checkpoint(work / folder / "checkpoint.pt", target / folder / "checkpoint.pt")
        copy_json(work / folder / "metrics.json", target / folder / "metrics.json")


# ------------------------------------------------------------------ helpers


def summary(tensor, n=4):
    """Digest plus a few floats so cross-machine tolerance checks still bite."""
    value = tensor.detach().cpu()
    flat = value.double().flatten()
    return {
        "digest": tensor_digest(value),
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "first": floats(flat[:n]),
    }


def state_summary(state):
    """Digest of a whole state dict and float sums over its floating tensors."""
    values = [v.detach().double() for v in state.values() if v.is_floating_point()]
    return {
        "digest": tensor_digest(dict(state)),
        "sum": float(sum(v.sum() for v in values)),
        "abs_sum": float(sum(v.abs().sum() for v in values)),
    }


def modules(model):
    """Module class names along named_modules() (they feed architecture hashes)."""
    return [[name, type(module).__name__] for name, module in model.named_modules()]


def same_tensors(a, b):
    import torch

    return list(a) == list(b) and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a
    )


def mark_training_ema_pairs(runs, states):
    """last.pt's EMA and the EMA snapshot of the same run/step load the same model."""
    for run, entry in runs.items():
        pair = [state for rel, state in states.items() if rel.startswith(run + "/")]
        if len(pair) == 2:
            entry["training_ema_equals_snapshot"] = same_tensors(*pair)


def normalize_paths(cfg, paths):
    cfg = copy.deepcopy(cfg)
    for path in paths:
        parent = cfg
        for key in path[:-1]:
            parent = parent[key]
        if parent.get(path[-1]) is not None:
            parent[path[-1]] = "<path>"
    return cfg


def builtin_types(value):
    """Python type names inside a saved config (must stay plain builtins)."""
    found = {type(value).__name__}
    if isinstance(value, dict):
        for k, v in value.items():
            found |= {type(k).__name__} | builtin_types(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            found |= builtin_types(v)
    return found


def describe_optimizer(state):
    """Optimizer layout without torch-version-specific param_group flags."""
    groups = [
        {"n_params": len(g["params"]), **{k: g[k] for k in ("lr", "betas", "eps", "weight_decay") if k in g}}
        for g in state["param_groups"]
    ]
    entries = [[str(k), describe(v)] for k, v in state["state"].items()]
    return {
        "keys": sorted(state),
        "param_groups": groups,
        "state_entries": len(entries),
        "state_entry_keys": sorted({k for v in state["state"].values() for k in v}),
        "state_layout_sha256": json_digest(entries),
    }


def describe(value, key=None):
    """Nested key sets and value types; strings only for VALUE_KEYS."""
    import torch

    if isinstance(value, torch.Tensor):
        return ["Tensor", str(value.dtype), list(value.shape)]
    if key == "environment" and isinstance(value, dict):
        return ["keys", sorted(value)]  # value types depend on the torch build
    if isinstance(value, dict) and set(value) == {"state", "param_groups"}:
        return describe_optimizer(value)
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            return ["dict", len(value), sorted({type(k).__name__ for k in value})]
        if len(value) > 12 and all(isinstance(v, torch.Tensor) for v in value.values()):
            return ["TensorDict", len(value), json_digest(shapes(value))]
        return {k: describe(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        name = type(value).__name__
        if value and all(isinstance(v, dict) for v in value):
            return [name + "[dict]", len(value), sorted({k for v in value for k in v})]
        return [name, len(value), sorted({type(v).__name__ for v in value})]
    if isinstance(value, str) and key in VALUE_KEYS:
        return ["str", value]
    return type(value).__name__


def load_weights_only(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def run_files(folder, ignore=("tensorboard",)):
    return sorted(
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and p.relative_to(folder).parts[0] not in ignore
    )


def jsonl_layout(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [[row.get("split"), row.get("step"), sorted(row)] for row in rows]


def quiet(function, *args, **kwargs):
    """Call a loader without its "source differs" warning (expected after moves)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def dotted(cfg, path):
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


# ------------------------------------------------------------------ 1. pixel


@probe(GROUP, "epoch0_pixel_load")
def epoch0_pixel_load(ctx):
    """Epoch-0 smoke last.pt / ema_*.pt through the current strict load_inference."""
    import torch

    load_inference = symbol("load_inference", *CHECKPOINTS)
    sample_batch = symbol("sample_batch", *SAMPLING)
    experiment_name = symbol("experiment_name", *CONFIG)
    root = payload_root(ctx, "pixel")
    files = sorted(p.relative_to(root).as_posix() for p in root.glob("*/*.pt"))
    exact, numeric, states = {"files": files, "runs": {}}, {}, {}
    for rel in files:
        model, cfg, device, ckpt = quiet(load_inference, root / rel, ["backend.cpu_threads=1"], device="cpu")
        state = model.state_dict()
        states[rel] = state
        run = exact["runs"].setdefault(rel.split("/")[0], {"state_dict": shapes(state)})
        g = torch.Generator().manual_seed(20260924)
        a = cfg["data_loader"]["args"]
        y = torch.randn(2, a["channels"], a["image_size"], a["image_size"], generator=g)
        level = model.process.level(torch.tensor([0.25, 0.75]), device)
        with torch.no_grad():
            score = model(y.to(device), level)
        samples, nfe = sample_batch(model, cfg, 2, device, torch.Generator().manual_seed(7))
        exact[rel] = {
            "payload_keys": sorted(ckpt),
            "format": ckpt["format"],
            "kind": ckpt["kind"],
            "step": ckpt["step"],
            "model_class": type(model).__name__,
            "experiment_name": experiment_name(cfg),
            "config_sha256": json_digest(normalize_paths(cfg, PIXEL_PATHS)),
            "loaded_config_equals_saved": normalize_paths(cfg, PIXEL_PATHS)
            == normalize_paths(ckpt["config"], PIXEL_PATHS),
            "state_dict_matches_run": shapes(state) == run["state_dict"],
            "stats_keys": sorted(ckpt["stats"]),
            "sample_nfe": nfe,
        }
        numeric[rel] = {"weights": state_summary(state), "score": summary(score), "samples": summary(samples)}
    mark_training_ema_pairs(exact["runs"], states)
    return {"exact": exact, "numeric": numeric}


# ------------------------------------------------------------------ 2. LDM


@probe(GROUP, "epoch0_ldm_load")
def epoch0_ldm_load(ctx):
    """Epoch-0 tiny LDM last.pt / EMA snapshot through the current load_trained."""
    import torch

    load_trained = symbol("load_trained", *LDM_TRAINING)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    sample_latents = symbol("sample_latents", *LDM_MODEL)
    evaluate_latents = symbol("evaluate_latents", *LDM_EVALUATION)
    image_splits = symbol("image_splits", *LDM_DATA)
    root = payload_root(ctx, "ldm")
    files = sorted(
        p.relative_to(root).as_posix() for p in [*root.glob("*/last.pt"), *root.glob("*/ema_*.pt")]
    )
    changes = [f"cache.dir={root / 'cache'}", "backend.cpu_threads=1"]
    exact, numeric, states = {"files": files, "payload_files": run_files(root), "runs": {}}, {}, {}
    for rel in files:
        model, cfg, device, state = quiet(load_trained, root / rel, changes, device="cpu")
        weights = model.state_dict()
        states[rel] = weights
        run = exact["runs"].setdefault(rel.split("/")[0], {"state_dict": shapes(weights)})
        spec_cfg = copy.deepcopy(cfg)
        spec_cfg["upstream_config"] = str(root / "native.yaml")
        g = torch.Generator().manual_seed(20260924)
        z = torch.randn(2, *model.spec["latent_shape"], generator=g)
        t = torch.tensor([1, 7])
        with torch.no_grad():
            eps = model(z.to(device), t.to(device))
        latents, calls = sample_latents(model, 2, cfg["sampling"], device, torch.Generator().manual_seed(7))
        metrics = evaluate_latents(model, cfg, state["cache"], device)
        exact[rel] = {
            "payload_keys": sorted(state),
            "format": state["format"],
            "kind": state["kind"],
            "step": state["step"],
            "model_class": type(model).__name__,
            "parameterization": cfg["parameterization"],
            "config_sha256": json_digest(normalize_paths(cfg, LDM_PATHS)),
            "spec_sha256": json_digest(state["spec"]),
            "spec_reproduced_by_load_spec": load_spec(spec_cfg) == state["spec"],
            "state_dict_matches_run": shapes(weights) == run["state_dict"],
            "sample_calls": calls,
        }
        numeric[rel] = {
            "weights": state_summary(weights),
            "epsilon": summary(eps),
            "samples": summary(latents),
            "evaluate_latents": metrics,
        }
    mark_training_ema_pairs(exact["runs"], states)
    data_cfg = copy.deepcopy(cfg)
    data_cfg["data"].update(
        root=str(root / "images"), train_list=str(root / "train.txt"), validation_list=str(root / "validation.txt")
    )
    splits = image_splits(data_cfg, state["spec"])
    exact["image_splits"] = {
        split: {"count": len(ds), "cache_rows": state["cache"]["arrays"][split]["shape"][0]}
        for split, ds in splits.items()
    }
    numeric["validation_images"] = summary(torch.stack([splits["validation"][i] for i in range(2)]))
    return {"exact": exact, "numeric": numeric}


# ------------------------------------------------------------------ 3. GMM


@probe(GROUP, "epoch0_gmm_load")
def epoch0_gmm_load(ctx):
    """Epoch-0 GMM checkpoints loaded like run_gmm_comparison.test_selected / prepare_arm."""
    from dataclasses import asdict
    import torch

    make_bank = symbol("make_bank", *GMM)
    make_model = symbol("make_model", *GMM)
    evaluate_model = symbol("evaluate_model", *GMM)
    tensor_state_hash = symbol("tensor_state_hash", *GMM)
    GMMArm = symbol("GMMArm", *GMM)
    cfg, family, arms = gmm_setup()
    banks = {split: make_bank(family, split) for split in ("validation", "test")}
    root = payload_root(ctx, "gmm")
    exact = {"case_id": family.case_id, "files": run_files(root)}
    numeric = {"banks": {split: bank.fingerprint for split, bank in banks.items()}}
    # Runner reuse compares configs without the scheduling fields.
    administrative = {"run_tag", "methods", "seeds", "spectrum_lambdas", "distributions"}
    for arm in arms:
        folder = root / family.case_id / f"{arm.name}_seed{GMM_SEED}"
        payload = load_weights_only(folder / "checkpoint.pt")
        state = payload["state"]
        model = make_model(family, arm, GMM_SEED)
        initial = tensor_state_hash(model.backbone.state_dict())
        model.load_state_dict(payload["ema"], strict=True)
        final = tensor_state_hash(model.backbone.state_dict())
        g = torch.Generator().manual_seed(20260924)
        y = torch.randn(3, 1, cfg.image_size, cfg.image_size, generator=g)
        level = model.process.level(torch.tensor([0.1, 0.5, 0.9]))
        with torch.no_grad():
            scaled = model.scaled_score(y, level)
        results = {split: evaluate_model(model, family, bank) for split, bank in banks.items()}
        saved_cfg = {k: v for k, v in payload["config"].items() if k not in administrative}
        new_cfg = {k: v for k, v in asdict(cfg).items() if k not in administrative}
        exact[arm.name] = {
            "payload_keys": sorted(payload),
            "state_keys": sorted(state),
            "method": state["method"],
            "arm_roundtrip": GMMArm(**state["arm"]) == arm,
            "arm": state["arm"],
            "config_matches": saved_cfg == new_cfg,
            "completed": state["completed"],
            "step": state["step"],
            "seed": state["seed"],
            "spectrum_lambda": state["spectrum_lambda"],
            "distribution": state["distribution"],
            "final_ema_backbone_sha256": final,
            "final_ema_matches_state": final == state["final_ema_backbone_sha256"],
            "metrics_json_equals_state": json.loads((folder / "metrics.json").read_text())
            == json.loads(json.dumps(state)),
            "model_state_dict": shapes(model.state_dict()),
            "archived_validation": [[m["step"], m["score_error"]] for m in state["validation"]],
        }
        numeric[arm.name] = {
            "initial_backbone_sha256": initial,
            "scaled_score": summary(scaled),
            **{
                split: {
                    "score_error": r["score_error"],
                    "reference_error": r["reference_error"],
                    "dsm_pixel_mean": r["dsm_pixel_mean"],
                    "relative_to_gaussian": r["relative_to_gaussian"],
                    "per_noise": [row["score_error"] for row in r["per_noise"]],
                    "by_frequency": r["score_error_by_frequency"],
                }
                for split, r in results.items()
            },
        }
    return {"exact": exact, "numeric": numeric}


# ------------------------------------------------------------------ 4. schema


def checkpoint_schema(path):
    ckpt = load_weights_only(path)  # raises unless weights_only=True works
    config = ckpt.get("config")
    return {
        "weights_only": True,
        "structure": describe(ckpt),
        "config_is_dict": type(config) is dict,
        "config_types": sorted(builtin_types(config)),
    }, ckpt


@probe(GROUP, "payload_schema")
def payload_schema(ctx):
    """Structure of freshly written pixel / LDM / GMM checkpoints and run dirs."""
    pixel_work = scratch(ctx)
    run = train_pixel(ctx, pixel_work, "fourier_gaussian_gated")
    pixel = {"run_name": run.name, "run_files": run_files(run), "stats_files": len(run_files(pixel_work / "stats"))}
    for name in sorted(p.name for p in run.glob("*.pt")):
        pixel[name], ckpt = checkpoint_schema(run / name)
        pixel[name]["config_equals_resolved_json"] = ckpt["config"] == json.loads(
            (run / "config.resolved.json").read_text()
        )
    pixel["architecture_json_keys"] = sorted(json.loads((run / "architecture.json").read_text()))
    pixel["environment_json_keys"] = sorted(json.loads((run / "environment.json").read_text()))
    pixel["metrics_jsonl"] = jsonl_layout(run / "metrics.jsonl")

    ldm_work = scratch(ctx)
    run = train_ldm(ctx.root, ldm_work)
    ldm = {"run_name": run.name, "run_files": run_files(run), "cache_files": run_files(ldm_work / "cache")}
    for name in sorted(p.name for p in run.glob("*.pt")):
        ldm[name], ckpt = checkpoint_schema(run / name)
        ldm[name]["config_equals_resolved_json"] = ckpt["config"] == json.loads(
            (run / "config.resolved.json").read_text()
        )
        ldm[name]["spec_equals_ldm_spec_json"] = ckpt["spec"] == json.loads((run / "ldm_spec.json").read_text())
    ldm["architecture_json_keys"] = sorted(json.loads((run / "architecture.json").read_text()))
    ldm["environment_json_keys"] = sorted(json.loads((run / "environment.json").read_text()))
    ldm["metrics_jsonl"] = jsonl_layout(run / "metrics.jsonl")

    gmm_work = scratch(ctx)
    _cfg, family, arms = gmm_setup()
    (folder,) = train_gmm(family, arms[1:], gmm_work)
    gmm = {"output_files": run_files(gmm_work)}
    gmm["checkpoint.pt"], ckpt = checkpoint_schema(gmm_work / folder / "checkpoint.pt")
    metrics = json.loads((gmm_work / folder / "metrics.json").read_text())
    gmm["metrics.json"] = describe(metrics)
    gmm["metrics_json_equals_state"] = metrics == json.loads(json.dumps(ckpt["state"]))
    return {"exact": {"pixel": pixel, "ldm": ldm, "gmm": gmm}}


# ------------------------------------------------------------------ 5. keys


def placeholder_stats(channels, size):
    import torch

    return {"mean": torch.zeros(channels, size, size), "power": torch.ones(channels, size, size)}


def layout(model):
    return {"state_dict": shapes(model.state_dict()), "modules": modules(model)}


def layout_delta(full, base):
    """Entries absent from ``base``; shared entries must keep base's order and shapes."""
    known = {json.dumps(entry) for entry in base}
    return {
        "extra": [entry for entry in full if json.dumps(entry) not in known],
        "shared_equals_base": [entry for entry in full if json.dumps(entry) in known] == base,
    }


def relative_layout(model, base):
    return {k: layout_delta(v, base[k]) for k, v in layout(model).items()}


@probe(GROUP, "state_dict_keys")
def state_dict_keys(ctx):
    """state_dict key/dtype/shape lists and module class names of every model family."""
    load_config = symbol("load_config", *CONFIG)
    build_model = symbol("build_model", *MODEL)
    architecture_report = symbol("architecture_report", *MODEL)
    OBJECTIVES = symbol("OBJECTIVES", *GATES)
    EMAClass = symbol("EMA", *EMA)

    base = [
        f"trainer.save_dir={ctx.tmp / 'runs'}", f"fourier.cache_dir={ctx.tmp / 'stats'}",
        "device=cpu", "backend.cpu_threads=1",
    ]
    variants = {name: [f"parameterization={name}"] for name in OBJECTIVES}
    variants["fourier_gaussian_gated"] = [
        "parameterization=fourier_gaussian", "loss.objective=normalized_residual", "fourier.gate.mode=log_sigma",
    ]
    # Full layouts for "score" (and LDM "epsilon"); other variants record only
    # their extra entries and that the shared ones are unchanged and in order.
    pixel = {"objectives": list(OBJECTIVES), "base": "score"}
    reference = None
    for name in ("score", *(n for n in variants if n != "score")):
        overrides = variants[name]
        cfg = load_config(str(config_path("configs/smoke.json")), [*base, *overrides])
        a = cfg["data_loader"]["args"]
        model = build_model(cfg, placeholder_stats(a["channels"], a["image_size"]), "cpu")
        ema = EMAClass(model, cfg["trainer"]["ema_decay"])
        entry = {
            "architecture_sha256": architecture_report(model.backbone)["architecture_sha256"],
            "ema_state_keys": sorted(ema.state_dict()),
        }
        if reference is None:
            reference = {**layout(model), "ema_shadow": list(ema.shadow)}
            pixel[name] = {**reference, **entry}
        else:
            pixel[name] = {
                **relative_layout(model, reference),
                "ema_shadow_equals_base": list(ema.shadow) == reference["ema_shadow"],
                **entry,
            }

    Denoiser = symbol("Denoiser", *LDM_MODEL)
    LitEma = symbol("LitEma", *LDM_EMA)
    FrozenFirstStage = symbol("FrozenFirstStage", *LDM_FIRST_STAGE)
    PARAMETERIZATIONS = symbol("PARAMETERIZATIONS", *LDM_CONFIG)
    ldm = {"parameterizations": list(PARAMETERIZATIONS), "base": "denoiser_epsilon"}
    for kind in ("AutoencoderKL", "VQModelInterface"):
        _cfg, spec = ldm_recipe(ctx.root, ctx.tmp / kind, kind)
        ldm["first_stage_" + kind] = layout(FrozenFirstStage(spec))
        if kind != "AutoencoderKL":
            continue
        stats = placeholder_stats(*spec["latent_shape"][:2])
        reference = None
        for name in ("epsilon", *(n for n in PARAMETERIZATIONS if n != "epsilon")):
            model = Denoiser(spec, name, stats if name != "epsilon" else None)
            buffers = shapes(dict(LitEma(model).named_buffers()))
            if reference is None:
                reference = {**layout(model), "litema_buffers": buffers}
                ldm["denoiser_" + name] = reference
            else:
                ldm["denoiser_" + name] = {
                    **relative_layout(model, reference),
                    "litema_buffers_equal_base": buffers == reference["litema_buffers"],
                }

    ToyScoreModel = symbol("ToyScoreModel", *GMM)
    GMMConfig = symbol("GMMConfig", *GMM)
    BASELINE_ARMS = symbol("BASELINE_ARMS", *GMM)
    gated = [
        symbol("gated_arm", *GMM)("fourier"),
        symbol("plateau_arm", *GMM)("fourier"),
        symbol("spectral_cap_arm", *GMM)("fourier"),
        *(symbol("shaped_gate_arm", *GMM)("fourier", mode) for mode in ("linear_sigma", "tanh_sigma")),
        *(symbol("log_gate_arm", *GMM)("fourier", mode) for mode in ("linear_log_sigma", "bounded_log_sigmoid")),
    ]
    cfg = GMMConfig(**GMM_TINY)
    gmm = {"arms": [arm.name for arm in (*BASELINE_ARMS, *gated)]}
    for arm in (*BASELINE_ARMS, *gated):
        gmm[arm.name] = layout(ToyScoreModel(cfg, placeholder_stats(1, cfg.image_size), arm))
    return {"exact": {"pixel": pixel, "ldm": ldm, "gmm": gmm}}


# ------------------------------------------------------------------ 6. overrides

PIXEL_ACCEPTED = (
    "device=cpu", "sampling.steps=6", "sampling.method=pc", "sampling.num_samples=3", "sampling.seed=5",
    "evaluation.max_images=2", "evaluation.frequency_bins=2", "evaluation.seed=3", "backend.cpu_threads=1",
    "backend.spectral_transform=matmul", "backend.tf32=false", "data_loader.args.root=relocated/data",
    "data_loader.args.download=true", "data_loader.args.num_workers=0",
)
PIXEL_REJECTED = (
    "parameterization=score", "loss.type=score", "loss.objective=dsm", "loss.reduction=half_sum",
    "process.sigma_max=3.0", "arch.args.nf=16", "fourier.gate.mode=none", "fourier.power_floor=0.1",
    "fourier.cache_dir=relocated/stats", "fourier.stats_batch_size=8", "trainer.iterations=5",
    "trainer.save_dir=relocated/runs", "trainer.ema_decay=0.5", "data_loader.args.batch_size=4",
    "data_loader.args.dataset=mnist", "data_loader.args.split_seed=1", "optimizer.args.lr=0.1",
    "seed=1", "name=renamed", "schema_version=1",
)
PIXEL_INVALID = (
    "sampling.method=bogus", "sampling.bogus=1", "sampling.steps", "evaluation.max_images=0",
    "backend.precision=fp16", "device=tpu",
)
LDM_ACCEPTED = (
    "device=cpu", "sampling.steps=2", "sampling.eta=1.0", "sampling.seed=5", "evaluation.max_images=1",
    "evaluation.frequency_bins=0", "backend.cpu_threads=1", "backend.spectral_transform=fft",
    "cache.dir=relocated/cache", "first_stage.checkpoint=relocated/model.ckpt",
)
LDM_REJECTED = (
    "parameterization=epsilon", "protocol=l2", "model=celebahq", "seed=1", "name=renamed",
    "upstream_config=relocated.yaml", "data.root=relocated/images", "data.random_flip=false",
    "cache.power_floor=0.1", "cache.batch_size=8", "cache.num_workers=1", "training.lr=0.1",
    "training.save_dir=relocated/runs", "training.ema_decay=0.5", "training.iterations=5",
)
LDM_INVALID = (
    "sampling.method=bogus", "sampling.bogus=1", "sampling.steps", "evaluation.max_images=0",
    "backend.precision=fp16",
)


def override_outcomes(load, path, accepted, rejected, invalid, *, cfg_index):
    def attempt(change):
        try:
            return None, quiet(load, path, [change])
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            return [type(error).__name__, str(error)], None

    out = {"accepted": {}, "rejected": {}, "invalid": {}}
    for change in accepted:
        error, loaded = attempt(change)
        out["accepted"][change] = error or dotted(loaded[cfg_index], change.split("=", 1)[0])
    for group, changes in (("rejected", rejected), ("invalid", invalid)):
        for change in changes:
            error, _ = attempt(change)
            out[group][change] = error or "accepted"
    return out


def tampered(src, dst, **changes):
    import torch

    ckpt = load_weights_only(src)
    ckpt.update(changes)
    torch.save(ckpt, dst)
    return dst


def tamper_outcomes(load, src, tmp):
    out = {}
    for label, changes in (("format", {"format": "bogus-format"}), ("kind", {"kind": "bogus"})):
        try:
            quiet(load, tampered(src, tmp / f"{label}.pt", **changes))
            out[label] = "accepted"
        except Exception as error:  # noqa: BLE001
            out[label] = [type(error).__name__, str(error)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load(tampered(src, tmp / "source.pt", source_sha256="0" * 64))
    out["source_mismatch_warnings"] = sorted(
        {f"{w.category.__name__}: {w.message}" for w in caught if "source" in str(w.message).lower()}
    )
    return out


@probe(GROUP, "inference_override_allowlist")
def inference_override_allowlist(ctx):
    """Which --set changes load_inference / ldm load_trained accept, and their errors."""
    load_inference = symbol("load_inference", *CHECKPOINTS)
    load_trained = symbol("load_trained", *LDM_TRAINING)
    pixel = payload_root(ctx, "pixel") / "synthetic_ve_fourier_gaussian_s0_normalized_residual_gate_s1_p4"
    ldm = payload_root(ctx, "ldm") / "tiny_fourier_gaussian_s42"
    return {
        "exact": {
            "load_inference": {
                **override_outcomes(
                    load_inference, pixel / "ema_000000003.pt", PIXEL_ACCEPTED, PIXEL_REJECTED, PIXEL_INVALID,
                    cfg_index=1,
                ),
                "tampered": tamper_outcomes(load_inference, pixel / "ema_000000003.pt", scratch(ctx)),
            },
            "load_trained": {
                **override_outcomes(
                    load_trained, ldm / "ema_000000002.pt", LDM_ACCEPTED, LDM_REJECTED, LDM_INVALID, cfg_index=1
                ),
                "tampered": tamper_outcomes(load_trained, ldm / "ema_000000002.pt", scratch(ctx)),
            },
        }
    }
