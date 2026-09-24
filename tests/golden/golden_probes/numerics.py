"""Numerical and pairing contracts: construction/RNG order and bit-exact numerics.

Everything runs on synthetic smoke data on the CPU with one thread.  Facts that
hold on every machine (index lists, RNG states of integer generators, config
digests, file listings, identities) go to ``exact``; anything derived from
float arithmetic (losses, weights, samples, PNG/NPZ bytes) goes to
``numeric`` as a digest PLUS a few summary floats, so the check stays
bit-exact on the recording machine and still bites within tolerance elsewhere.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from golden_probes import (
    config_path, entrypoint, file_sha256, json_digest, probe, resolve, symbol, tensor_digest,
)

# New (planned) module paths first, epoch-0 paths last.
CONFIG = ("fourier_score.parse_config", "fourier_score.config")
GATES = ("fourier_score.gates", "fourier_score.method")
REFERENCE = ("fourier_score.model.reference", "fourier_score.method")
MODEL = ("fourier_score.model.model", "fourier_score.model")
LOSS = ("fourier_score.model.loss", "fourier_score.loss")
PROCESS = ("fourier_score.model.process", "fourier_score.diffusion")
SAMPLING = ("fourier_score.model.sampling", "fourier_score.diffusion")
SPECTRAL = ("fourier_score.model.spectral", "fourier_score.spectral")
METRIC = ("fourier_score.model.metric", "fourier_score.evaluation")
DATA = ("fourier_score.data_loader.data_loaders", "fourier_score.data")
STATS_ID = (
    "fourier_score.data_loader.statistics",
    "fourier_score.data_loader.data_loaders",
    "fourier_score.data",
)
ESTIMATE = ("fourier_score.data_loader.statistics", "fourier_score.statistics")
TRAINER = ("fourier_score.trainer.trainer", "fourier_score.training")
LDM_CONFIG = ("fourier_score.ldm.config",)
LDM_DATA = ("fourier_score.ldm.data",)
LDM_FIRST = ("fourier_score.ldm.first_stage",)
DIGEST_JSON = ("fourier_score.provenance", "fourier_score.ldm.data")
FILE_SHA = ("fourier_score.provenance", "fourier_score.ldm.first_stage")

# Literal on purpose (not imported): the gate modes the contract covers.
GATE_MODES = (
    "none",
    "constant",
    "log_sigma",
    "log_sigma_plateau",
    "spectral_cap",
    "linear_sigma",
    "tanh_sigma",
    "linear_log_sigma",
    "bounded_log_sigmoid",
)
GATE_PARAMS = {
    "defaults": {},
    "custom": {"sigma_switch": 0.5, "sharpness": 2.5, "value": 0.3,
               "sigma_lo": 0.1, "sigma_hi": 1.5, "delta": 0.25},
    "steep": {"sigma_switch": 2.0, "sharpness": 12.0, "value": 0.0,
              "sigma_lo": 0.05, "sigma_hi": 20.0, "delta": 2.0},
}
DDPM = ("process.type=ddpm", "sampling.method=ddpm", "sampling.steps=16")
PRODUCTION = (
    "trainer.warmup=2",
    "data_loader.args.random_flip=true",
    "loss.reduction=half_sum",
    "trainer.microbatch_size=1",
)
TIME_KEY = re.compile(r"seconds|per_second|^eta")


# ---------------------------------------------------------------- helpers


def summary(tensor, head=3):
    """A few float64 aggregates so off-machine tolerance checks still bite."""
    import torch

    t = tensor.detach().to("cpu", torch.float64).flatten()
    if not t.numel():
        return {"numel": 0}
    return {
        "numel": t.numel(),
        "sum": float(t.sum()),
        "abs_sum": float(t.abs().sum()),
        "sq_sum": float(t.square().sum()),
        "min": float(t.min()),
        "max": float(t.max()),
        "head": t[:head].tolist(),
    }


def tensor_fact(tensor, head=3):
    return {"sha": tensor_digest(tensor), **summary(tensor, head)}


def brief(tensor):
    """Compact fact: digest plus two float64 aggregates."""
    import torch

    t = tensor.detach().to("cpu", torch.float64)
    return {"sha": tensor_digest(tensor), "sum": float(t.sum()), "abs_sum": float(t.abs().sum())}


def state_fact(state):
    """Digest of a (nested) tensor dict plus aggregates over its float tensors."""
    import torch

    flat = []

    def walk(value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                flat.append(value.detach().flatten().double())
        elif isinstance(value, dict):
            for k in sorted(value, key=str):
                walk(value[k])
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(state)
    fact = summary(torch.cat(flat), head=0) if flat else {"numel": 0}
    fact.pop("head", None)
    return {"sha": tensor_digest(state), **fact}


def array_digest(arrays):
    """sha256 over named numpy arrays (name, dtype, shape, raw bytes)."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        h.update(f"{name}:{value.dtype}:{value.shape}".encode())
        h.update(value.tobytes())
    return h.hexdigest()


def int_digest(tensor):
    """Digest for integer tensors (RNG states, indices): exact on every machine."""
    return tensor_digest(tensor)


def skeleton(obj):
    """Structure and non-float leaves; floats become the string 'float'."""
    if isinstance(obj, dict):
        return {k: skeleton(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [skeleton(v) for v in obj]
    if isinstance(obj, float):
        return "float"
    return obj


def normalize(obj, ctx):
    """Replace scratch / repository paths by '<TMP>' / '<ROOT>'."""
    pairs = []
    for path, label in ((ctx.tmp, "<TMP>"), (ctx.root, "<ROOT>")):
        for form in {str(path), str(Path(path).resolve())}:
            pairs.append((form, label))
    pairs.sort(key=lambda item: -len(item[0]))

    def walk(value):
        if isinstance(value, dict):
            return {walk(k): walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            for form, label in pairs:
                value = value.replace(form, label)
            return value
        return value

    return walk(obj)


def error_text(fn):
    try:
        fn()
    except Exception as error:  # the message itself is the contract
        return f"{type(error).__name__}: {error}"
    return None


def smoke_cfg(ctx, *changes, save="runs", config="configs/smoke.json"):
    load_config = symbol("load_config", *CONFIG)
    base = [
        f"trainer.save_dir={ctx.tmp / save}",
        f"fourier.cache_dir={ctx.tmp / 'stats'}",
        "device=cpu",
        "backend.cpu_threads=1",
        "trainer.console=quiet",
    ]
    return load_config(str(config_path(config)), base + list(changes))


def run_dir(save_dir):
    runs = sorted(p for p in Path(save_dir).iterdir() if p.is_dir())
    assert len(runs) == 1, runs
    return runs[0]


def tiny_stats(channels=1, size=8):
    """Seeded mean and a decaying, conjugate-symmetric power spectrum."""
    import torch

    conj = symbol("conjugate_symmetrize", *SPECTRAL)
    g = torch.Generator().manual_seed(7)
    f = torch.fft.fftfreq(size, dtype=torch.float64)
    base = 0.02 + 1.0 / (1.0 + 60.0 * (f[:, None].square() + f[None, :].square()))
    jitter = conj(1 + 0.3 * torch.rand(channels, size, size, generator=g, dtype=torch.float64))
    power = (base * jitter).float()
    mean = 0.1 * torch.rand(channels, size, size, generator=g) - 0.05
    return {"mean": mean, "power": power}


def tiny_model(cfg, stats, seed=0):
    """Seeded build plus a fixed perturbation so every layer matters."""
    import torch

    build_model = symbol("build_model", *MODEL)
    torch.manual_seed(seed)
    model = build_model(cfg, stats, "cpu")
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def subprocess_env():
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    return env


def run_script(ctx, script, *args):
    result = subprocess.run(
        entrypoint(ctx, script, *args),
        cwd=ctx.root,
        env=subprocess_env(),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"{script} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def train_smoke(ctx, *changes):
    """Train configs/smoke.json in-process exactly as train.py does."""
    Trainer = symbol("Trainer", *TRAINER)
    cfg = smoke_cfg(ctx, *changes)
    trainer = Trainer(cfg)
    trainer.train()
    return trainer, run_dir(ctx.tmp / "runs")


def strip_record(record):
    return {k: v for k, v in record.items() if not TIME_KEY.search(k)}


# ---------------------------------------------------------------- 1. pairing


@probe("numerics", "pairing")
def pairing(ctx):
    """Trainer(cfg) construction order: init hashes, first batches, RNG states."""
    import torch

    OBJECTIVES = symbol("OBJECTIVES", *GATES)
    exact, numeric = {}, {}
    for process, extra in (("ve", ()), ("ddpm", DDPM)):
        for objective in OBJECTIVES:
            Trainer = symbol("Trainer", *TRAINER)
            save = f"pair_{process}_{objective}"
            cfg = smoke_cfg(ctx, f"loss.type={objective}", *extra, save=save)
            trainer = Trainer(cfg)
            torch_state = torch.get_rng_state()
            run = run_dir(ctx.tmp / save)
            arch = json.loads((run / "architecture.json").read_text())
            env = json.loads((run / "environment.json").read_text())
            full = trainer.bundle.full
            images = torch.stack([full[i] for i in range(len(full))])
            batches, ids = [], []
            for _ in range(3):
                batch = trainer.stream.next_batch()
                trainer.stream.advance()
                batches.append(batch)
                ids.append([
                    int(torch.nonzero((images == x).flatten(1).all(1)).flatten()[0])
                    for x in batch
                ])
            name = f"{process}/{objective}"
            ema = trainer.ema.state_dict()
            groups = trainer.optimizer.state_dict()["param_groups"]
            exact[name] = {
                "run": run.name,
                "architecture_sha256": arch["architecture_sha256"],
                "trainable_parameters": arch["trainable_parameters"],
                "all_parameters": arch["all_parameters"],
                "initial_hash_matches_architecture_json": trainer.initial_hash
                == arch["initial_backbone_sha256"],
                "spectral_transform_resolved": env.get("spectral_transform_resolved"),
                "environment_keys": sorted(env),
                "first_batch_indices": ids,
                "stream_state": trainer.stream.state_dict(),
                "generator_state": int_digest(trainer.generator.get_state()),
                "torch_rng_after_init": int_digest(torch_state),
                "ema_decay": ema["decay"],
                "ema_num_updates": ema["num_updates"],
                "ema_shadow_keys_sha": json_digest(sorted(ema["shadow"])),
                "optimizer_groups": [
                    {"n_params": len(g["params"]), "lr": g["lr"], "betas": list(g["betas"]),
                     "eps": g["eps"], "weight_decay": g["weight_decay"]}
                    for g in groups
                ],
            }
            numeric[name] = {
                "initial_backbone_sha256": arch["initial_backbone_sha256"],
                "backbone": state_fact(dict(trainer.model.backbone.named_parameters())),
                "model_state": state_fact(trainer.model.state_dict()),
                "ema_shadow": state_fact(ema["shadow"]),
                "first_batches": [brief(b) for b in batches],
            }
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 2. trajectory


TRAJECTORY_ARMS = {
    "score_ve": ("loss.type=score",),
    "fourier_gaussian_nr_log_sigma": (
        "loss.type=fourier_gaussian",
        "loss.objective=normalized_residual",
        "fourier.gate.mode=log_sigma",
    ),
    "diffusion_ddpm": ("loss.type=diffusion", *DDPM),
}


@probe("numerics", "trajectory")
def trajectory(ctx):
    """Three optimizer steps + EMA evaluations for smoke and production-like runs."""
    import torch

    Trainer = symbol("Trainer", *TRAINER)
    exact, numeric = {}, {}
    for variant, changes in (("smoke", ()), ("production", PRODUCTION)):
        for arm, arm_changes in TRAJECTORY_ARMS.items():
            save = f"traj_{variant}_{arm}"
            cfg = smoke_cfg(ctx, *changes, *arm_changes, save=save)
            trainer = Trainer(cfg)
            trainer.train()
            run = run_dir(ctx.tmp / save)
            records = [
                strip_record(json.loads(line))
                for line in (run / "metrics.jsonl").read_text().splitlines()
            ]
            ema = trainer.ema.state_dict()
            optimizer = trainer.optimizer.state_dict()
            name = f"{variant}/{arm}"
            exact[name] = {
                "run": run.name,
                "files": sorted(p.name for p in run.iterdir()),
                "records": skeleton(records),
                "stream_state": trainer.stream.state_dict(),
                "generator_state": int_digest(trainer.generator.get_state()),
                "torch_rng_after_train": int_digest(torch.get_rng_state()),
                "ema_num_updates": ema["num_updates"],
                "evaluated_steps": [r["step"] for r in records if r["split"] != "train"],
            }
            numeric[name] = {
                "losses": [r["loss"] for r in records if r["split"] == "train"],
                "records": records,
                "model": state_fact(trainer.model.state_dict()),
                "ema_shadow": state_fact(ema["shadow"]),
                "optimizer_state": state_fact(optimizer["state"]),
            }
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 3. dsm_eval


def eval_images(n=9):
    import torch

    g = torch.Generator().manual_seed(11)
    return [2 * torch.rand(1, 8, 8, generator=g) - 1 for _ in range(n)]


@probe("numerics", "dsm_eval")
def dsm_eval(ctx):
    """evaluate_dsm with noise and radial frequency bins, partial last batch."""
    evaluate_dsm = symbol("evaluate_dsm", *METRIC)
    import torch

    arms = {
        "ve/fourier_gaussian_log_sigma": ("loss.type=fourier_gaussian", "fourier.gate.mode=log_sigma"),
        "ve/score": ("loss.type=score",),
        "ddpm/diffusion": ("loss.type=diffusion", *DDPM),
    }
    evaluation = (
        "evaluation.batch_size=3",
        "evaluation.max_images=7",
        "evaluation.noise_bins=4",
        "evaluation.frequency_bins=3",
    )
    exact, numeric = {}, {}
    data = eval_images()
    for name, changes in arms.items():
        cfg = smoke_cfg(ctx, *changes, *evaluation)
        model = tiny_model(cfg, tiny_stats())
        model.train()
        rng = torch.get_rng_state()
        result = evaluate_dsm(model, cfg, data, torch.device("cpu"))
        exact[name] = {
            "result": skeleton(result),
            "restores_training_mode": model.training,
            "global_rng_untouched": torch.equal(rng, torch.get_rng_state()),
        }
        numeric[name] = result
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 4. samplers


SAMPLER_ARMS = {
    "ve/score/heun": ("loss.type=score", "sampling.method=heun"),
    "ve/score/heun_no_denoise": ("loss.type=score", "sampling.method=heun", "sampling.denoise=false"),
    "ve/score/pc_c0": ("loss.type=score", "sampling.method=pc", "sampling.corrector_steps=0"),
    "ve/score/pc_c1": ("loss.type=score", "sampling.method=pc", "sampling.corrector_steps=1"),
    "ve/score/pc_c2_no_denoise": (
        "loss.type=score", "sampling.method=pc", "sampling.corrector_steps=2", "sampling.denoise=false",
    ),
    "ve/fourier_gaussian/heun": ("loss.type=fourier_gaussian", "sampling.method=heun"),
    "ve/fourier_gaussian/pc_c1": ("loss.type=fourier_gaussian", "sampling.method=pc", "sampling.corrector_steps=1"),
    "ve/fourier_gaussian_spectral_cap/heun": (
        "loss.type=fourier_gaussian", "fourier.gate.mode=spectral_cap", "sampling.method=heun",
    ),
    "ve/scalar_gaussian_bounded/pc_c1": (
        "loss.type=scalar_gaussian", "fourier.gate.mode=bounded_log_sigmoid",
        "fourier.gate.sigma_lo=0.1", "fourier.gate.sigma_hi=1.5", "fourier.gate.sigma_switch=0.5",
        "sampling.method=pc", "sampling.corrector_steps=1",
    ),
    "ddpm/diffusion/ddpm": ("loss.type=diffusion", *DDPM),
    "ddpm/diffusion/ddpm_clip": ("loss.type=diffusion", *DDPM, "sampling.clip_denoised=true"),
    "ddpm/fourier_gaussian/ddpm": ("loss.type=fourier_gaussian", *DDPM),
}


@probe("numerics", "samplers")
def samplers(ctx):
    """sample_batch for every valid (process, sampler) with a fixed generator."""
    import torch

    sample_batch = symbol("sample_batch", *SAMPLING)
    exact, numeric = {}, {}
    stats = tiny_stats()
    for name, changes in SAMPLER_ARMS.items():
        cfg = smoke_cfg(ctx, *changes)
        model = tiny_model(cfg, stats).eval()
        g = torch.Generator().manual_seed(1234)
        x, calls = sample_batch(model, cfg, 3, torch.device("cpu"), g)
        exact[name] = {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "calls": calls,
            "generator_state": int_digest(g.get_state()),
        }
        numeric[name] = tensor_fact(x, head=4)
    # The sampler itself rejects mismatched process/method pairs.
    invalid = {}
    for name, changes, patch in (
        ("ve_with_ddpm_method", ("loss.type=score",), {"method": "ddpm"}),
        ("ddpm_with_heun", ("loss.type=diffusion", *DDPM), {"method": "heun"}),
        ("ddpm_partial_grid", ("loss.type=diffusion", *DDPM), {"steps": 8}),
    ):
        cfg = smoke_cfg(ctx, *changes)
        model = tiny_model(cfg, stats).eval()
        bad = copy.deepcopy(cfg)
        bad["sampling"].update(patch)
        invalid[name] = error_text(
            lambda: sample_batch(model, bad, 2, torch.device("cpu"), torch.Generator().manual_seed(0))
        )
    exact["invalid"] = invalid
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 5. sample_cli


@probe("numerics", "sample_cli")
def sample_cli(ctx):
    """sample.py on a tiny smoke run: shards (with remainder), PNGs, settings.

    5 samples in batches of 2 leave a remainder round.  The smoke sampler
    (heun) is per-sample; the PC corrector variant uses a batch-mean Langevin
    norm, so it also pins that the final round still samples the FULL batch.
    """
    import numpy as np

    _, run = train_smoke(ctx)
    checkpoint = run / "last.pt"
    exact, numeric = {}, {}
    for variant, extra in (
        ("heun", ()),
        ("pc_c1", ("--set", "sampling.method=pc", "--set", "sampling.corrector_steps=1")),
    ):
        out = ctx.tmp / f"samples_{variant}"
        stdout = run_script(
            ctx, "sample.py", "-r", checkpoint, "-o", out,
            "--num-samples", 5, "--batch-size", 2, "--steps", 3, "--device", "cpu", *extra,
        )
        settings = json.loads((out / "settings.json").read_text())
        shards, shard_facts = {}, {}
        for path in sorted(out.glob("samples_*.npz")):
            with np.load(path, allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files}
            shards[path.name] = {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()}
            shard_facts[path.name] = {
                "file_sha256": file_sha256(path),
                "content_sha256": array_digest(arrays),
                "mean": float(np.mean([a.astype(np.float64).mean() for a in arrays.values()])),
            }
        stripped = normalize({
            k: v for k, v in settings.items()
            if k not in ("checkpoint_sha256", "training_wall_seconds", "wall_seconds",
                         "environment", "terminal_out_of_range_fraction")
        }, ctx)
        exact[variant] = {
            "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
            "shards": shards,
            "settings": stripped,
            "settings_keys": sorted(settings),
            "checkpoint_sha256_is_file_hash": settings["checkpoint_sha256"] == file_sha256(checkpoint),
            "stdout": normalize(stdout, ctx).splitlines(),
        }
        numeric[variant] = {
            "shards": shard_facts,
            "png": {p.name: file_sha256(p) for p in sorted((out / "png").glob("*.png"))},
            "preview_png": file_sha256(out / "preview.png"),
            "terminal_out_of_range_fraction": settings["terminal_out_of_range_fraction"],
        }
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 6. evaluate_cli


@probe("numerics", "evaluate_cli")
def evaluate_cli(ctx):
    """evaluate.py dsm on a tiny smoke run's last.pt (EMA)."""
    _, run = train_smoke(ctx)
    output = ctx.tmp / "dsm.json"
    stdout = run_script(
        ctx, "evaluate.py", "dsm", "-r", run / "last.pt", "-o", output, "--device", "cpu",
    )
    result = json.loads(output.read_text())
    stripped = normalize(
        {k: v for k, v in result.items() if k not in ("training_wall_seconds", "environment")}, ctx
    )
    return {
        "exact": {
            "result": skeleton(stripped),
            "keys": sorted(result),
            "stdout_matches_file": json.loads(stdout) == result,
        },
        "numeric": stripped,
    }


# ---------------------------------------------------------------- 7. export_real


@probe("numerics", "export_real")
def export_real(ctx):
    """scripts/export_real.py on the smoke synthetic dataset (train and validation)."""
    exact, numeric = {}, {}
    for split, extra in (("train", ()), ("validation", ("--limit", 3))):
        out = ctx.tmp / f"real_{split}"
        stdout = run_script(
            ctx, "scripts/export_real.py", "-c", config_path("configs/smoke.json"),
            "--device", "cpu", "-o", out, "--split", split, *extra,
        )
        settings = normalize(json.loads((out / "settings.json").read_text()), ctx)
        exact[split] = {
            "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
            "settings": settings,
            "stdout": normalize(stdout, ctx).splitlines(),
        }
        numeric[split] = {p.name: file_sha256(p) for p in sorted((out / "png").glob("*.png"))}
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 8. gates_numeric


def sigma_grid():
    """25 geometric points on [1e-2, 50] plus every configured breakpoint."""
    import math

    lo, hi = math.log(1e-2), math.log(50.0)
    grid = [1e-2] + [math.exp(lo + i * (hi - lo) / 24) for i in range(1, 24)] + [50.0]
    breakpoints = sorted({v for p in GATE_PARAMS.values() for k, v in p.items()
                          if k.startswith("sigma")} | {0.8, 1.0})
    return grid, breakpoints


@probe("numerics", "gates_numeric")
def gates_numeric(ctx):
    """_gate_weights on a sigma grid for every mode; reference target/score outputs."""
    import torch

    FourierGaussian = symbol("FourierGaussian", *REFERENCE)
    stats = tiny_stats(channels=2)
    grid, breakpoints = sigma_grid()
    sigmas = grid + breakpoints
    exact = {"grid_len": len(grid), "breakpoints": breakpoints, "invalid": {}, "weights": {}}
    weights, outputs = {}, {}
    g = torch.Generator().manual_seed(21)
    clean = 2 * torch.rand(4, 2, 8, 8, generator=g) - 1
    noise = torch.randn(4, 2, 8, 8, generator=g)
    raw = torch.randn(4, 2, 8, 8, generator=g)
    y = torch.randn(4, 2, 8, 8, generator=g)
    fwd_sigma = torch.tensor([0.02, 0.5, 1.0, 20.0])
    ddpm_alpha = torch.tensor([0.999, 0.9, 0.6, 0.1])
    for mode in GATE_MODES:
        for label, params in GATE_PARAMS.items():
            gate = {"mode": mode, **params}
            key = f"{mode}/{label}"
            problem = error_text(lambda: FourierGaussian(stats, "fft", gate=gate))
            if problem is not None:
                exact["invalid"][key] = problem
                continue
            refs = {
                cov: FourierGaussian(stats, "fft", covariance=cov, gate=gate)
                for cov in ("fourier", "scalar")
            }
            for dtype in (torch.float32, torch.float64):
                s = torch.tensor(sigmas, dtype=dtype)
                (gf, cf), (gs, cs) = (refs[c]._gate_weights(s) for c in ("fourier", "scalar"))
                same = gf.shape == gs.shape and torch.equal(gf, gs) and torch.equal(cf, cs)
                cell = f"{key}/{str(dtype).removeprefix('torch.')}"
                exact["weights"][cell] = {
                    "shape": {"fourier": list(gf.shape), "scalar": list(gs.shape)},
                    "dtype": str(gf.dtype),
                    "gate_value_is_g": torch.equal(refs["fourier"].gate_value(s), gf)
                    and torch.equal(refs["scalar"].gate_value(s), gs),
                    "scalar_equals_fourier": same,
                }
                fact = {"g": brief(gf), "complement": brief(cf)}
                if dtype == torch.float32:
                    fact["g_at_breakpoints"] = gf.flatten(1).double()[len(grid):, 0].tolist()
                    fact["complement_at_breakpoints"] = cf.flatten(1).double()[len(grid):, 0].tolist()
                if not same:
                    fact.update(scalar_g=brief(gs), scalar_complement=brief(cs))
                weights[cell] = fact
            for cov, ref in refs.items():
                name = f"{key}/{cov}"
                outputs[f"{name}/normalized_target"] = brief(ref.normalized_target(clean, noise, fwd_sigma))
                outputs[f"{name}/scaled_score_ve"] = brief(ref.scaled_score(raw, y, torch.ones(4), fwd_sigma))
                if mode == "none":
                    outputs[f"{name}/scaled_score_ddpm"] = brief(
                        ref.scaled_score(raw, y, ddpm_alpha, (1 - ddpm_alpha.square()).sqrt())
                    )
            if label == "custom" or mode in ("none", "spectral_cap"):
                mm = FourierGaussian(stats, "matmul", gate=gate)
                outputs[f"{key}/fourier/matmul/normalized_target"] = brief(
                    mm.normalized_target(clean, noise, fwd_sigma)
                )
                outputs[f"{key}/fourier/matmul/scaled_score_ve"] = brief(
                    mm.scaled_score(raw, y, torch.ones(4), fwd_sigma)
                )
    return {"exact": exact, "numeric": {"weights": weights, "outputs": outputs}}


# ---------------------------------------------------------------- 9. losses


LOSS_ARMS = {
    "ve/score/dsm": ("loss.type=score",),
    "ve/diffusion/dsm": ("loss.type=diffusion",),
    "ve/scalar_gaussian/dsm": ("loss.type=scalar_gaussian",),
    "ve/fourier_gaussian/dsm": ("loss.type=fourier_gaussian",),
    "ve/scalar_gaussian/normalized_residual": ("loss.type=scalar_gaussian", "loss.objective=normalized_residual"),
    "ve/fourier_gaussian/normalized_residual": ("loss.type=fourier_gaussian", "loss.objective=normalized_residual"),
    "ve/fourier_gaussian_log_sigma/normalized_residual": (
        "loss.type=fourier_gaussian", "loss.objective=normalized_residual", "fourier.gate.mode=log_sigma",
    ),
    "ve/fourier_gaussian_spectral_cap/dsm": ("loss.type=fourier_gaussian", "fourier.gate.mode=spectral_cap"),
    "ddpm/score/dsm": ("loss.type=score", *DDPM),
    "ddpm/diffusion/dsm": ("loss.type=diffusion", *DDPM),
    "ddpm/scalar_gaussian/dsm": ("loss.type=scalar_gaussian", *DDPM),
    "ddpm/fourier_gaussian/dsm": ("loss.type=fourier_gaussian", *DDPM),
}


@probe("numerics", "losses")
def losses(ctx):
    """training_loss for each parameterization x objective x reduction on fixed tensors."""
    import torch

    training_loss = symbol("training_loss", *LOSS)
    stats = tiny_stats()
    g = torch.Generator().manual_seed(31)
    clean = 2 * torch.rand(4, 1, 8, 8, generator=g) - 1
    noise = torch.randn(4, 1, 8, 8, generator=g)
    numeric = {}
    for name, changes in LOSS_ARMS.items():
        cfg = smoke_cfg(ctx, *changes)
        model = tiny_model(cfg, stats)
        process = model.process
        coordinate = (
            torch.tensor([1e-5, 0.3, 0.7, 1.0]) if process.kind == "ve" else torch.tensor([0, 5, 10, 15])
        )
        level = process.level(coordinate)
        for reduction in ("mean", "half_sum"):
            model.zero_grad(set_to_none=True)
            loss = training_loss(
                model, clean, level, noise, reduction, objective=cfg["loss"]["objective"]
            )
            loss.backward()
            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            numeric[f"{name}/{reduction}"] = {
                "loss": float(loss.detach()),
                "grad": state_fact(grads),
            }
    cfg = smoke_cfg(ctx, "loss.type=score")
    model = tiny_model(cfg, stats)
    level = model.process.level(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    invalid = {
        "normalized_residual_score": error_text(
            lambda: training_loss(model, clean, level, noise, objective="normalized_residual")
        ),
        "unknown_objective": error_text(
            lambda: training_loss(model, clean, level, noise, objective="l1")
        ),
    }
    return {"exact": {"arms": sorted(numeric), "invalid": invalid}, "numeric": numeric}


# ---------------------------------------------------------------- 10. stats_identity


def jsonable_metadata(meta):
    import torch

    return {
        k: {"tensor": str(v.dtype), "values": v.tolist()} if torch.is_tensor(v) else v
        for k, v in meta.items()
    }


@probe("numerics", "stats_identity")
def stats_identity_probe(ctx):
    """Smoke metadata, stats identity, dataset fingerprints, stats values, cache names."""
    import torch

    build_data = symbol("build_data", *DATA)
    stats_identity = symbol("stats_identity", *STATS_ID)
    prepare_stats = symbol("prepare_stats", *STATS_ID)
    estimate_stats = symbol("estimate_stats", *ESTIMATE)
    exact, numeric = {}, {}
    for name, changes in (
        ("smoke", ()),
        ("flip", ("data_loader.args.random_flip=true",)),
        ("test_split", ("data_loader.args.validation_size=0",)),
    ):
        cfg = smoke_cfg(ctx, *changes)
        bundle = build_data(cfg)
        meta = bundle.metadata
        identity = stats_identity(meta)
        stats = prepare_stats(cfg, bundle)
        again = prepare_stats(cfg, bundle)
        exact[name] = {
            "metadata": jsonable_metadata(meta),
            "stats_identity": identity,
            "full_fingerprint": bundle.full.fingerprint(),
            "validation_fingerprint": bundle.validation.fingerprint()
            if hasattr(bundle.validation, "fingerprint") else None,
            "sizes": [len(bundle.full), len(bundle.train), len(bundle.validation)],
            "stats_keys": sorted(stats),
            "stats_identity_field": stats["identity"],
            "n_effective": stats["n_effective"],
            "floor": stats["floor"],
            "cache_reload_equal": tensor_digest(again) == tensor_digest(stats),
        }
        numeric[name] = {
            "mean": tensor_fact(stats["mean"]),
            "power": tensor_fact(stats["power"]),
            "floored_fraction": stats["floored_fraction"],
            "first_train_image": tensor_fact(bundle.train[0]),
        }
    exact["cache_files"] = sorted(p.name for p in (ctx.tmp / "stats").iterdir())
    g = torch.Generator().manual_seed(41)
    batches = [torch.randn(3, 2, 6, 6, generator=g) for _ in range(3)]
    variances = [torch.rand(3, 2, 6, 6, generator=g) for _ in range(3)]
    direct = estimate_stats(iter(batches), 1e-3)
    posterior = estimate_stats(iter(zip(batches, variances)), 1e-3, posterior_variance=True)
    for name, value in (("estimate", direct), ("estimate_posterior", posterior)):
        exact[name] = {"keys": sorted(value), "n_effective": value["n_effective"], "floor": value["floor"]}
        numeric[name] = {
            "mean": tensor_fact(value["mean"]),
            "power": tensor_fact(value["power"]),
            "floored_fraction": value["floored_fraction"],
        }
    return {"exact": exact, "numeric": numeric}


# ---------------------------------------------------------------- 11. ldm_identity


def latent_experiment(ctx):
    """tests/test_ldm.py::latent_experiment with AutoencoderKL, fixed file mtimes."""
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    defaults = symbol("defaults", *LDM_CONFIG)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    FrozenFirstStage = symbol("FrozenFirstStage", *LDM_FIRST)
    tmp_path = ctx.tmp
    cfg = defaults("ffhq")
    source = yaml.safe_load((ctx.root / "configs/ldm/upstream/ffhq.yaml").read_text())
    p = source["model"]["params"]
    kl = True
    p.update(image_size=4, channels=2, timesteps=10, scale_by_std=kl)
    p["unet_config"]["params"] = {
        "image_size": 4,
        "in_channels": 2,
        "out_channels": 2,
        "model_channels": 32,
        "attention_resolutions": [1, 2],
        "num_res_blocks": 1,
        "channel_mult": [1, 2],
        "num_heads": 4,
    }
    p["first_stage_config"] = {
        "target": "ldm.models.autoencoder.AutoencoderKL",
        "params": {
            "embed_dim": 2,
            "ddconfig": {
                "double_z": kl,
                "z_channels": 2,
                "resolution": 8,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 32,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            "lossconfig": {"target": "torch.nn.Identity"},
        },
    }
    source["data"]["params"]["train"]["params"]["size"] = 8
    spec_path = tmp_path / "native.yaml"
    spec_path.write_text(yaml.safe_dump(source))
    cfg.update(upstream_config=str(spec_path), device="cpu", name="tiny_{parameterization}_s{seed}")
    cfg["sampling"].update(steps=5, num_samples=2, batch_size=2, decode_batch_size=1, eta=0.0)
    cfg["evaluation"].update(batch_size=2, max_images=2, noise_bins=2, frequency_bins=2)
    cfg["training"].update(
        iterations=2, batch_size=2, microbatch_size=1, save_dir=str(tmp_path / "runs"),
        save_every=1, snapshot_every=0, eval_every=1, log_every=1, console="quiet", lr=1e-3,
    )
    cfg["backend"]["cpu_threads"] = 2
    cfg["cache"].update(dir=str(tmp_path / "cache"), batch_size=2)
    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(17)
    for i in range(6):
        path = root / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)).save(path)
        # ImageList fingerprints (name, size, mtime_ns); pin mtimes for determinism.
        os.utime(path, ns=(1_700_000_000_000_000_000 + i, 1_700_000_000_000_000_000 + i))
    for split, ids in (("train", range(4)), ("validation", range(4, 6))):
        listing = tmp_path / f"{split}.txt"
        listing.write_text("".join(f"{i}.png\n" for i in ids))
        cfg["data"][split + "_list"] = str(listing)
    cfg["data"].update(root=str(root), random_flip=True)
    spec = load_spec(cfg)
    torch.manual_seed(123)
    stage = FrozenFirstStage(spec)
    checkpoint = tmp_path / "public.ckpt"
    state = {"first_stage_model." + k: v for k, v in stage.state_dict().items()}
    if kl:
        state["scale_factor"] = torch.tensor(0.7)
    torch.save({"state_dict": state}, checkpoint)
    cfg["first_stage"]["checkpoint"] = str(checkpoint)
    return cfg, spec


@probe("numerics", "ldm_identity")
def ldm_identity(ctx):
    """Latent-cache identity inputs and their digest composition (paths normalized)."""
    import torch

    cfg, spec = latent_experiment(ctx)
    ldm_data = resolve(*LDM_DATA)
    digest_json = symbol("digest_json", *DIGEST_JSON)
    file_sha = symbol("file_sha256", *FILE_SHA)
    load_first_stage = symbol("load_first_stage", *LDM_FIRST)
    load_checkpoint = symbol("load_checkpoint", "fourier_score.utils")
    manifest = ldm_data.prepare_cache(cfg, spec, torch.device("cpu"), progress=lambda _: None)
    reopened = ldm_data.open_cache(cfg, spec)
    splits = ldm_data.image_splits(cfg, spec)
    _, first = load_first_stage(cfg["first_stage"]["checkpoint"], spec)
    representation = ldm_data.representation(spec)
    settings = ldm_data.settings(cfg)
    inputs = {
        "format": ldm_data.CACHE_FORMAT,
        "representation": representation,
        "settings": settings,
        "sources": {
            key: {
                "list_sha256": file_sha(cfg["data"][f"{key}_list"]),
                "file_metadata_sha256": ds.fingerprint,
                "count": len(ds),
            }
            for key, ds in splits.items()
        },
        "first_stage_sha256": file_sha(cfg["first_stage"]["checkpoint"]),
        "encoding": ldm_data.encoding_identity(first),
    }
    normalized = normalize(inputs, ctx)
    # Machine-dependent leaves (PNG byte sizes, float weight init) are masked
    # in the exact form and kept in the numeric one.
    masked = copy.deepcopy(normalized)
    masked["first_stage_sha256"] = "<float-derived>"
    for item in masked["sources"].values():
        item["file_metadata_sha256"] = "<size/mtime-derived>"
    stats = load_checkpoint(Path(cfg["cache"]["dir"]) / "stats.pt")
    return {
        "exact": {
            "identity_is_digest_of_inputs": digest_json(inputs) == manifest["identity"],
            "reopened_identity_equal": reopened["identity"] == manifest["identity"],
            "manifest_inputs_equal": {
                k: manifest[k] == inputs[k] for k in inputs
            },
            "manifest_keys": sorted(manifest),
            "arrays": manifest["arrays"],
            "files": sorted(manifest["files"]),
            "first_stage": normalize(
                {k: v for k, v in manifest["first_stage"].items() if k != "checkpoint_sha256"}, ctx
            ),
            "representation": representation,
            "representation_digest": digest_json(representation),
            "settings": normalize(settings, ctx),
            "settings_digest": digest_json(normalize(settings, ctx)),
            "masked_inputs": masked,
            "masked_identity": digest_json(masked),
            "digest_json_probe": digest_json({"b": [1, 2.5, None], "a": "x"}),
            "stats": {
                k: stats[k] for k in ("n_effective", "floor", "source_split", "posterior_moments")
            },
        },
        "numeric": {
            "normalized_identity": digest_json(normalized),
            "first_stage_sha256": inputs["first_stage_sha256"],
            "file_metadata_sha256": {
                k: v["file_metadata_sha256"] for k, v in inputs["sources"].items()
            },
            "stats_mean": tensor_fact(stats["mean"]),
            "stats_power": tensor_fact(stats["power"]),
            "floored_fraction": stats["floored_fraction"],
        },
    }


# ---------------------------------------------------------------- 12. spectral


@probe("numerics", "spectral")
def spectral(ctx):
    """conjugate_symmetrize, SpectralFilter backends/gradients and FFT band helpers."""
    import torch

    conj = symbol("conjugate_symmetrize", *SPECTRAL)
    SpectralFilter = symbol("SpectralFilter", *SPECTRAL)
    bands = symbol("radial_frequency_bands", *METRIC)
    band_sums = symbol("frequency_band_sums", *METRIC)
    g = torch.Generator().manual_seed(51)
    exact, numeric = {}, {}
    for name, shape, dtype in (
        ("even_f32", (2, 3, 8, 8), torch.float32),
        ("odd_f32", (1, 5, 7), torch.float32),
        ("even_f64", (3, 6, 10), torch.float64),
    ):
        x = torch.rand(shape, generator=g, dtype=dtype)
        numeric[f"conjugate_symmetrize/{name}"] = tensor_fact(conj(x))
    for name, (h, w) in (("8x8", (8, 8)), ("6x10", (6, 10)), ("5x7", (5, 7))):
        x = torch.randn(2, 3, h, w, generator=g)
        multipliers = {
            "full": conj(0.1 + torch.rand(2, 3, h, w, generator=g)),
            "per_channel": 0.5 + torch.rand(1, 3, 1, 1, generator=g),
        }
        for backend in ("auto", "fft", "matmul", "cpu"):
            filt = SpectralFilter(h, w, backend)
            exact[f"resolved/{name}/{backend}"] = filt.resolved_backend(torch.device("cpu"))
            for label, multiplier in multipliers.items():
                inp = x.clone().requires_grad_(True)
                out = filt(inp, multiplier)
                out.square().sum().backward()
                numeric[f"filter/{name}/{backend}/{label}"] = {
                    "out": brief(out),
                    "grad": brief(inp.grad),
                }
    exact["filter_shape_error"] = error_text(
        lambda: SpectralFilter(8, 8, "fft")(torch.zeros(1, 1, 6, 8), torch.ones(1))
    )
    exact["filter_backend_error"] = error_text(lambda: SpectralFilter(8, 8, "gpu"))
    for name, (h, w, n) in (("8x8x3", (8, 8, 3)), ("7x9x4", (7, 9, 4))):
        ids, counts, edges = bands(h, w, n)
        residual = torch.randn(3, 2, h, w, generator=g)
        exact[f"bands/{name}"] = {"ids": ids.tolist(), "counts": counts.tolist()}
        numeric[f"bands/{name}"] = {
            "edges": edges,
            "sums": tensor_fact(band_sums(residual, ids, n)),
        }
    return {"exact": exact, "numeric": numeric}
