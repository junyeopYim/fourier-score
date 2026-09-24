"""Local-only contracts against the ORIGINAL checkout's real artifacts.

Every probe here needs ``ctx.golden_root`` (``FOURIER_GOLDEN_ROOT``, the live
``fourier-score`` checkout) and only ever READS it: whatever a loader might
write (statistics caches, latent caches, run directories) is pointed at a copy
under ``ctx.tmp``.

Probes
------
* ``real_checkpoints``: strict loads of real pixel, score-SDE, LDM and GMM
  checkpoints with the current loaders, a fixed-input forward digest, and
  whether the loader warned that the source hash differs.
* ``mnist_stats_identity``: MNIST metadata / ``stats_identity`` and a
  no-recompute load of a tmp copy of the cached statistics.
* ``mnist_pairing``: ``Trainer`` construction for the three MNIST seed-0 arms
  (same initial backbone hash as the live runs).
* ``stored_configs``: resume signatures / run names of every stored
  ``config.resolved.json``.  Runs appear while the live study proceeds, so the
  contract compares only runs present in both golden and current (see
  :func:`check_stored_configs`).
* ``ldm_cache_identity``: the latent-cache identity recomputed the way
  ``fourier_score/ldm/data.py`` does.
"""

from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import shutil
import warnings

import golden_probes as gp
from golden_probes import file_sha256, json_digest, probe, shapes, symbol, tensor_digest

GROUP = "local"
STORED_CONFIGS_KEY = f"{GROUP}.stored_configs"

# Live MNIST seed-0 runs (saved/mnist_ve_*_s0/architecture.json).
MNIST_INITIAL_BACKBONE_SHA256 = "9f34373e4723bcf281f7c122d59ed8b21016aafb68c67b2c611671a7cb224974"
MNIST_ARCHITECTURE_SHA256 = "ca9a40cad1d88ab58e83e2b7c7dbe97158440ef4297d2375865970dd91ee59be"
MNIST_ARMS = ("score", "scalar_gaussian", "fourier_gaussian")

# Finished pixel-space checkpoints (fourier-image-template-v1).  All were
# written by source revisions older than epoch 0, so loading must warn.
IMAGE_CHECKPOINTS = (
    "pretrained/score_sde/cifar10_ncsnpp_continuous/ema.pt",
    "saved/mnist_ve_score_s0/ema_000100000.pt",
    "saved/mnist_ve_score_s0/last.pt",
    "saved/recovered/mnist_ve_scalar_gaussian_s0/ema_000100000.pt",
    "saved/mnist_ve_fourier_gaussian_s0/ema_000100000.pt",
)
LDM_CHECKPOINT = "saved/ldm_native_churches_final_fg/ema_000000001.pt"
LDM_CACHE = "saved/ldm_native_final_cache"
LDM_RUN_CONFIG = "saved/ldm_native_churches_final_fg/config.resolved.json"
LDM_ARCHITECTURE = "saved/ldm_native_churches_final_fg/architecture.json"
LDM_FIRST_STAGE = "pretrained/ldm/lsun_churches/model.ckpt"
LDM_DATA = "saved/ldm_native_smoke_data"
LDM_CACHE_IDENTITY_KEYS = (
    "format",
    "representation",
    "settings",
    "sources",
    "first_stage_sha256",
    "encoding",
)
GMM_CHECKPOINT = (
    "saved/gmm_log_gates_20260924/gmm_lambda1/fourier_gate_loglinear_lo0p1_hi3_seed42/checkpoint.pt"
)
GMM_AUDIT = "assets/gmm_log_gate_comparison/checkpoint_audit.json"

FORWARD_SEED = 20260924

# ---------------------------------------------------------------- imports
# New module path first, epoch-0 path last.

CHECKPOINTS = ("fourier_score.trainer.checkpoints", "fourier_score.checkpoints")
CONFIG = ("fourier_score.config",)
DATA = ("fourier_score.data_loader.data_loaders", "fourier_score.data")
STATISTICS = ("fourier_score.data_loader.statistics", "fourier_score.data")
TRAINER = ("fourier_score.trainer.trainer", "fourier_score.training")
LDM_CONFIG = ("fourier_score.ldm.config",)
LDM_DATA_MODULE = ("fourier_score.ldm.data",)
LDM_TRAINING = ("fourier_score.ldm.training",)
DIGEST_JSON = ("fourier_score.provenance", "fourier_score.ldm.data")
GMM = ("fourier_score.gmm",)


# ---------------------------------------------------------------- helpers


def _golden(ctx):
    if ctx.golden_root is None:
        raise RuntimeError("local probes need ctx.golden_root (FOURIER_GOLDEN_ROOT)")
    return Path(ctx.golden_root)


def _require(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing golden artifact: {path}")
    return path


def _summary(tensor, head=4):
    """Digest plus a few floats that still bite under cross-machine tolerance."""
    t = tensor.detach().double().cpu()
    return {
        "sha256": tensor_digest(tensor),
        "shape": list(tensor.shape),
        "sum": float(t.sum()),
        "abs_sum": float(t.abs().sum()),
        "head": [float(v) for v in t.flatten()[:head]],
    }


def _load_quietly(fn, *args, **kwargs):
    """Call a loader, returning (result, emitted a source-hash warning?)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    source = any("source" in str(w.message).lower() for w in caught)
    return result, source


def _listing(folder):
    """{name: [size, mtime_ns]} of a directory (non-recursive)."""
    return {
        p.name: [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(Path(folder).iterdir())
    }


def _copy_mnist_stats(ctx, identity):
    """Copy the cached MNIST statistics into ctx.tmp; return the cache dir."""
    source = _require(_golden(ctx) / "data" / "stats" / f"{identity}.pt")
    cache = ctx.tmp / "stats"
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, cache / source.name)
    return cache


def _mnist_overrides(ctx, cache_dir, save_dir):
    return [
        f"data_loader.args.root={_golden(ctx) / 'data'}",
        "data_loader.args.download=false",
        f"fourier.cache_dir={cache_dir}",
        f"trainer.save_dir={save_dir}",
        "trainer.console=quiet",
        "trainer.tensorboard=false",
        "device=cpu",
    ]


def _mnist_config(ctx, overrides=()):
    load_config = symbol("load_config", *CONFIG)
    return load_config(str(ctx.root / "configs" / "mnist.json"), list(overrides))


def _mnist_identity(ctx):
    """(cfg, bundle, identity) for configs/mnist.json on the golden MNIST."""
    build_data = symbol("build_data", *DATA)
    stats_identity = symbol("stats_identity", *STATISTICS)
    cfg = _mnist_config(ctx, _mnist_overrides(ctx, ctx.tmp / "unused-stats", ctx.tmp / "runs"))
    bundle = build_data(cfg)
    return cfg, bundle, stats_identity(bundle.metadata)


def _metadata_facts(meta):
    import torch

    out = {}
    for key in sorted(meta):
        value = meta[key]
        if torch.is_tensor(value):
            out[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_digest(value),
                "head": value.flatten()[:5].tolist(),
                "sum": int(value.sum()),
            }
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------- 1. real checkpoints


def _image_forward(model, cfg):
    import torch

    d = cfg["data_loader"]["args"]
    g = torch.Generator().manual_seed(FORWARD_SEED)
    shape = (2, d["channels"], d["image_size"], d["image_size"])
    clean = torch.rand(shape, generator=g) * 2 - 1
    noise = torch.randn(shape, generator=g)
    process = model.process
    if process.kind == "ve":
        coordinate = torch.tensor([0.15, 0.65])
    else:
        coordinate = torch.tensor([10, 600])
    level = process.level(coordinate, torch.device("cpu"))
    y = level.alpha[:, None, None, None] * clean + level.sigma[:, None, None, None] * noise
    with torch.no_grad():
        score = model(y, level)
    return _summary(score)


def _image_checkpoint(ctx, rel):
    import torch

    load_inference = symbol("load_inference", *CHECKPOINTS)
    resume_signature = symbol("resume_signature", *CHECKPOINTS)
    validate = symbol("validate", *CONFIG)
    path = _require(_golden(ctx) / rel)
    (model, cfg, device, ckpt), warned = _load_quietly(load_inference, str(path), device="cpu")
    torch.set_num_threads(1)
    state = model.state_dict()
    exact = {
        "file_sha256": file_sha256(path),
        "format": ckpt["format"],
        "kind": ckpt["kind"],
        "step": ckpt["step"],
        "device": str(device),
        "objective": cfg["loss"]["type"],
        "process": cfg["process"]["type"],
        "has_reference": model.reference is not None,
        "n_state_tensors": len(state),
        "state_shapes_sha256": json_digest(shapes(state)),
        "source_hash_warning": warned,
    }
    if ckpt["kind"] == "training":
        # Resume path of train.py: validate(apply_overrides(validate(config), [])).
        signature = resume_signature(validate(validate(ckpt["config"])))
        exact["resume_signature_matches"] = ckpt["signature"] == signature
        exact["resume_signature_sha256"] = json_digest(signature)
    numeric = {"state": tensor_digest(state), "score": _image_forward(model, cfg)}
    del model, ckpt, state
    gc.collect()
    return exact, numeric


def _ldm_checkpoint(ctx):
    import torch

    load_trained = symbol("load_trained", *LDM_TRAINING)
    golden = _golden(ctx)
    path = _require(golden / LDM_CHECKPOINT)
    changes = [
        f"cache.dir={_require(golden / LDM_CACHE)}",
        f"first_stage.checkpoint={_require(golden / LDM_FIRST_STAGE)}",
    ]
    (model, cfg, device, state), warned = _load_quietly(load_trained, str(path), changes, device="cpu")
    torch.set_num_threads(1)
    weights = model.state_dict()
    exact = {
        "file_sha256": file_sha256(path),
        "format": state["format"],
        "kind": state["kind"],
        "step": state["step"],
        "device": str(device),
        "parameterization": cfg["parameterization"],
        "has_reference": model.reference is not None,
        "n_state_tensors": len(weights),
        "state_shapes_sha256": json_digest(shapes(weights)),
        "spec_sha256": json_digest(state["spec"]),
        "source_hash_warning": warned,
    }
    g = torch.Generator().manual_seed(FORWARD_SEED)
    z = torch.randn(1, *state["spec"]["latent_shape"], generator=g)
    with torch.no_grad():
        eps = model(z, torch.tensor([500]))
    numeric = {"state": tensor_digest(weights), "epsilon": _summary(eps)}
    del model, state, weights
    gc.collect()
    return exact, numeric


def _gmm_checkpoint(ctx):
    import torch

    GMMConfig = symbol("GMMConfig", *GMM)
    GMMArm = symbol("GMMArm", *GMM)
    MatchedMomentFamily = symbol("MatchedMomentFamily", *GMM)
    make_model = symbol("make_model", *GMM)
    tensor_state_hash = symbol("tensor_state_hash", *GMM)
    golden = _golden(ctx)
    path = _require(golden / GMM_CHECKPOINT)
    audit = json.loads(_require(golden / GMM_AUDIT).read_text())
    (entry,) = [e for e in audit if e["checkpoint"] == str(path)]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"]
    config = {k: tuple(v) if isinstance(v, list) else v for k, v in payload["config"].items()}
    cfg = GMMConfig(**config)
    arm = GMMArm(**state["arm"])
    family = MatchedMomentFamily(cfg, state["distribution"], state["spectrum_lambda"])
    model = make_model(family, arm, state["seed"])
    initial = tensor_state_hash(model.backbone.state_dict())
    model.load_state_dict(payload["ema"], strict=True)
    model.eval()
    final = tensor_state_hash(model.backbone.state_dict())
    weights = model.state_dict()
    exact = {
        "file_sha256_matches_audit": file_sha256(path) == entry["checkpoint_sha256"],
        "case_id": family.case_id,
        "arm": arm.name,
        "gate": arm.gate,
        "seed": state["seed"],
        "completed": state["completed"],
        "n_state_tensors": len(weights),
        "state_shapes_sha256": json_digest(shapes(weights)),
        "final_ema_matches_state": final == state["final_ema_backbone_sha256"],
        "final_ema_matches_audit": final == entry["final_ema_backbone_sha256"],
    }
    g = torch.Generator().manual_seed(FORWARD_SEED)
    y = torch.randn(3, 1, cfg.image_size, cfg.image_size, generator=g)
    level = model.process.level(torch.tensor([0.1, 0.5, 0.9]), torch.device("cpu"))
    with torch.no_grad():
        score = model(y * level.sigma[:, None, None, None], level)
    numeric = {
        # CPU init RNG: pairing with the stored run is a machine fact.
        "initial_matches_state": initial == state["initial_backbone_sha256"],
        "initial_matches_audit": initial == entry["initial_backbone_sha256"],
        "initial_backbone_sha256": initial,
        "final_ema_backbone_sha256": final,
        "state": tensor_digest(weights),
        "score": _summary(score),
    }
    return exact, numeric


@probe(GROUP, "real_checkpoints", local=True)
def real_checkpoints(ctx):
    exact, numeric = {}, {}
    for rel in IMAGE_CHECKPOINTS:
        exact[rel], numeric[rel] = _image_checkpoint(ctx, rel)
    exact[LDM_CHECKPOINT], numeric[LDM_CHECKPOINT] = _ldm_checkpoint(ctx)
    exact[GMM_CHECKPOINT], numeric[GMM_CHECKPOINT] = _gmm_checkpoint(ctx)
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 2. MNIST statistics identity


@probe(GROUP, "mnist_stats_identity", local=True)
def mnist_stats_identity(ctx):
    prepare_stats = symbol("prepare_stats", *STATISTICS)
    golden_file = _golden(ctx) / "data" / "stats"
    cfg, bundle, identity = _mnist_identity(ctx)
    exists = (golden_file / f"{identity}.pt").is_file()
    cache = _copy_mnist_stats(ctx, identity)
    cfg = copy.deepcopy(cfg)
    cfg["fourier"]["cache_dir"] = str(cache)
    before = _listing(cache)
    stats = prepare_stats(cfg, bundle)
    after = _listing(cache)
    stats_facts = {
        k: v for k, v in stats.items() if k in ("identity", "n_effective", "floor", "schema_version")
    }
    return {
        "exact": {
            "metadata": _metadata_facts(bundle.metadata),
            "stats_identity": identity,
            "identity_file_exists": exists,
            "identity_file_sha256": file_sha256(golden_file / f"{identity}.pt"),
            "n_full": len(bundle.full),
            "n_train": len(bundle.train),
            "n_validation": len(bundle.validation),
            "cache_files": sorted(after),
            "cache_unchanged": before == after,
            "loaded_keys": sorted(stats),
            "loaded": stats_facts,
            "loaded_metadata_matches": all(
                json_digest(_metadata_facts({k: stats[k]})) == json_digest(_metadata_facts({k: v}))
                for k, v in bundle.metadata.items()
            ),
        },
        "numeric": {
            "mean": _summary(stats["mean"]),
            "power": _summary(stats["power"]),
            "floored_fraction": float(stats["floored_fraction"]),
        },
    }


# ---------------------------------------------------------------- 3. MNIST pairing


@probe(GROUP, "mnist_pairing", local=True)
def mnist_pairing(ctx):
    Trainer = symbol("Trainer", *TRAINER)
    _, _, identity = _mnist_identity(ctx)
    cache = _copy_mnist_stats(ctx, identity)
    before = _listing(cache)
    golden = _golden(ctx)
    numeric, names, architecture, files = {}, {}, {}, {}
    for arm in MNIST_ARMS:
        save_dir = ctx.tmp / f"runs_{arm}"
        cfg = _mnist_config(ctx, [*_mnist_overrides(ctx, cache, save_dir), f"loss.type={arm}"])
        trainer = Trainer(cfg)
        try:
            report = json.loads((trainer.out / "architecture.json").read_text())
            names[arm] = trainer.out.name
            files[arm] = sorted(p.name for p in trainer.out.iterdir())
        finally:
            trainer.log.close()
        live = json.loads(_require(golden / "saved" / f"mnist_ve_{arm}_s0" / "architecture.json").read_text())
        architecture[arm] = report["architecture_sha256"]
        numeric[arm] = {
            "initial_backbone_sha256": report["initial_backbone_sha256"],
            "trainer_initial_hash_matches_report": trainer.initial_hash == report["initial_backbone_sha256"],
            "matches_live_run": report["initial_backbone_sha256"] == live["initial_backbone_sha256"],
            "matches_expected": report["initial_backbone_sha256"] == MNIST_INITIAL_BACKBONE_SHA256,
        }
        del trainer
        gc.collect()
    hashes = {v["initial_backbone_sha256"] for v in numeric.values()}
    return {
        "exact": {
            "run_names": names,
            "run_files": files,
            "architecture_sha256": architecture,
            "architecture_matches_expected": all(
                v == MNIST_ARCHITECTURE_SHA256 for v in architecture.values()
            ),
            "stats_cache_unchanged": before == _listing(cache),
            # Pairing holds by construction (same seed and RNG order) on any machine.
            "same_initial_hash_across_arms": len(hashes) == 1,
        },
        "numeric": numeric,
    }


# ---------------------------------------------------------------- 4. stored configs


def _attempt(fn):
    try:
        return fn()
    except Exception as error:  # the failure itself is part of the contract
        # Type only: messages may be reworded (e.g. a tuple becoming a list repr).
        return {"error": type(error).__name__}


def _stored_config_paths(saved):
    found = []
    for folder, dirs, files in os.walk(saved):  # never follows symlinks
        dirs.sort()
        if "config.resolved.json" in files:
            found.append(Path(folder) / "config.resolved.json")
    return sorted(found)


@probe(GROUP, "stored_configs", local=True)
def stored_configs(ctx):
    resume_signature = symbol("resume_signature", *CHECKPOINTS)
    validate = symbol("validate", *CONFIG)
    experiment_name = symbol("experiment_name", *CONFIG)
    ldm_validate = symbol("validate", *LDM_CONFIG)
    ldm_name = symbol("experiment_name", *LDM_CONFIG)
    ldm_signature = symbol("signature", *LDM_TRAINING)
    golden = _golden(ctx)
    image, ldm = {}, {}
    for path in _stored_config_paths(golden / "saved"):
        rel = str(path.relative_to(golden))
        raw = json.loads(path.read_text())
        schema = raw.get("schema_version")
        if schema == 1:
            image[rel] = {
                "signature_sha256": _attempt(lambda: json_digest(resume_signature(validate(raw)))),
                "name": _attempt(lambda: experiment_name(validate(raw))),
                "raw_signature_sha256": _attempt(lambda: json_digest(resume_signature(copy.deepcopy(raw)))),
                "raw_name": _attempt(lambda: experiment_name(copy.deepcopy(raw))),
            }
        elif schema == "fourier-ldm-v1":
            ldm[rel] = {
                "signature_sha256": _attempt(lambda: json_digest(ldm_signature(ldm_validate(copy.deepcopy(raw))))),
                "name": _attempt(lambda: ldm_name(ldm_validate(copy.deepcopy(raw)))),
                "raw_signature_sha256": _attempt(lambda: json_digest(ldm_signature(copy.deepcopy(raw)))),
            }
    return {"exact": {"image": image, "ldm": ldm}}


def check_stored_configs(ctx):
    """Compare runs present in both golden and current; return what was skipped.

    New runs appear (and scratch runs may disappear) in the live checkout, so
    only the intersection is compared.  The anchors below must always be there.
    """
    golden = gp.load_golden(GROUP)
    assert STORED_CONFIGS_KEY in golden["probes"], "No golden for stored_configs; rerun record_goldens.py"
    expected = golden["probes"][STORED_CONFIGS_KEY]["exact"]
    actual = gp.run(STORED_CONFIGS_KEY, ctx)["exact"]
    assert sorted(expected) == sorted(actual), "stored_configs sections differ"
    report = {"compared": [], "golden_only": [], "current_only": []}
    for section in sorted(expected):
        e, a = expected[section], actual[section]
        for rel in sorted(set(e) & set(a)):
            gp.compare(e[rel], a[rel], exact=True, path=f"{STORED_CONFIGS_KEY}.{section}[{rel}]")
            report["compared"].append(rel)
        report["golden_only"] += sorted(set(e) - set(a))
        report["current_only"] += sorted(set(a) - set(e))
    anchors = [
        "saved/mnist_ve_score_s0/config.resolved.json",
        "saved/mnist_ve_fourier_gaussian_s0/config.resolved.json",
        "saved/recovered/mnist_ve_scalar_gaussian_s0/config.resolved.json",
        LDM_RUN_CONFIG,
    ]
    missing = [rel for rel in anchors if rel not in report["compared"]]
    assert not missing, f"stored_configs anchors not compared: {missing}"
    return report


# ---------------------------------------------------------------- 5. LDM latent-cache identity


@probe(GROUP, "ldm_cache_identity", local=True)
def ldm_cache_identity(ctx):
    import torch

    ldm_validate = symbol("validate", *LDM_CONFIG)
    override = symbol("override", *LDM_CONFIG)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    prepare_cache = symbol("prepare_cache", *LDM_DATA_MODULE)
    image_splits = symbol("image_splits", *LDM_DATA_MODULE)
    representation = symbol("representation", *LDM_DATA_MODULE)
    settings = symbol("settings", *LDM_DATA_MODULE)
    encoding_identity = symbol("encoding_identity", *LDM_DATA_MODULE)
    cache_format = symbol("CACHE_FORMAT", *LDM_DATA_MODULE)
    digest_json = symbol("digest_json", *DIGEST_JSON)
    golden = _golden(ctx)

    # Everything a loader could write goes to ctx.tmp: a copy of the (small)
    # cache, and a working directory whose relative data paths are symlinks.
    cache = ctx.tmp / "ldm_cache"
    shutil.copytree(_require(golden / LDM_CACHE), cache)
    work = ctx.tmp / "work"
    (work / Path(LDM_DATA).parent).mkdir(parents=True)
    (work / LDM_DATA).symlink_to(_require(golden / LDM_DATA), target_is_directory=True)
    os.chdir(work)  # gp.deterministic() restores the cwd

    first_stage = _require(golden / LDM_FIRST_STAGE)
    raw = json.loads(_require(golden / LDM_RUN_CONFIG).read_text())
    cfg = ldm_validate(override(raw, [f"cache.dir={cache}", f"first_stage.checkpoint={first_stage}", "device=cpu"]))
    spec = load_spec(cfg)
    manifest = json.loads((cache / "manifest.json").read_text())
    stored = manifest["identity"]
    from_manifest = digest_json({k: manifest[k] for k in LDM_CACHE_IDENTITY_KEYS})

    sources = {}
    for split, ds in image_splits(cfg, spec).items():
        sources[split] = {
            "list_sha256": file_sha256(cfg["data"][f"{split}_list"]),
            "file_metadata_sha256": ds.fingerprint,
            "count": len(ds),
        }
    inputs = {
        "format": cache_format,
        "representation": representation(spec),
        "settings": settings(cfg),
        "sources": sources,
        "first_stage_sha256": file_sha256(first_stage),
        "encoding": encoding_identity(manifest["first_stage"]),
    }
    from_config = digest_json(inputs)

    before = _listing(cache)
    if not (cache / "manifest.json").is_file():  # never let prepare_cache encode
        raise FileNotFoundError(cache / "manifest.json")
    prepared = prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda *_: None)
    architecture = json.loads(_require(golden / LDM_ARCHITECTURE).read_text())
    return {
        "exact": {
            "stored_identity": stored,
            "recomputed_from_manifest": from_manifest,
            "recomputed_from_config": from_config,
            "prepare_cache_identity": prepared["identity"],
            "run_architecture_cache_identity": architecture["cache_identity"],
            "identity_equal": len({stored, from_manifest, from_config, prepared["identity"]}) == 1,
            "sections_match_manifest": {k: inputs[k] == manifest[k] for k in LDM_CACHE_IDENTITY_KEYS},
            "section_sha256": {k: digest_json(inputs[k]) for k in LDM_CACHE_IDENTITY_KEYS},
            "cache_unchanged": before == _listing(cache),
        }
    }
