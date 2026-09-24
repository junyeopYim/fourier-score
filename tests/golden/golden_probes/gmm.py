"""GMM contracts: arm strings, configs, oracle banks, short training, selection.

Protects the E1.4 ``gate_arm`` consolidation (every legacy factory must keep
its exact ``json.dumps(asdict(arm), sort_keys=True)`` string, int vs float
included) and the A3 move of the ``scripts/run_gmm_*.py`` orchestration into
``experiments/``: the runner arm lists and presets are rebuilt here from the
epoch-0 runner logic and cross-checked against the archived
``assets/gmm_*`` protocols, which are recorded verbatim as ground truth.

Everything runs on a tiny 4x4 configuration so the whole group stays fast.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, fields, replace
import hashlib
import json

from golden_probes import probe, shapes, symbol, tensor_digest

GMM = ("fourier_score.gmm",)
# E1.4 moves select_gate into the experiments pipeline.
PIPELINE = ("experiments.gmm.pipeline", "fourier_score.gmm")
COVARIANCES = ("scalar", "fourier")
SHAPED_MODES = ("linear_sigma", "tanh_sigma")
LOG_MODES = ("linear_log_sigma", "bounded_log_sigmoid")
# Fixed stand-in for the runner provenance (python/torch/git/source hashes).
PROVENANCE = {"golden": "gmm-contract-v1", "python": "fixed", "source_sha256": "fixed", "torch": "fixed"}
# Train-bank version shared by every runner and each runner's test-bank version.
TRAIN_BANK = "gmm-gated-v1"
TEST_BANKS = {
    "comparison": "gmm-gated-v1",
    "plateau": "gmm-plateau-v1",
    "spectral_gate": "gmm-spectral-cap-v1",
    "gate_shapes": "gmm-linear-tanh-v1",
    "log_gates": "gmm-log-gates-v1",
}
BANK_VERSIONS = ("gmm-oracle-v1", *dict.fromkeys(TEST_BANKS.values()))
# Archived assets/<name>/protocol.json -> the runner that wrote it.
ARCHIVED_PROTOCOLS = {
    "gmm_plateau_comparison": "plateau",
    "gmm_spectral_comparison": "spectral_gate",
    "gmm_gate_shape_comparison": "gate_shapes",
    "gmm_log_gate_comparison": "log_gates",
}


def gmm(name):
    return symbol(name, *GMM)


def arm_json(arm):
    return json.dumps(asdict(arm), sort_keys=True)


def summary(tensor, n=4):
    """Digest plus a few floats so cross-machine tolerance checks still bite."""
    value = tensor.detach().cpu()
    flat = value.double().flatten()
    return {
        "digest": tensor_digest(value),
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "first": flat[:n].tolist(),
    }


def error_name(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the exception type is the fact
        return type(error).__name__
    return None


def dataclass_fields(cls):
    return [[f.name, "<required>" if f.default is MISSING else f.default] for f in fields(cls)]


def without_time(state):
    return {k: v for k, v in state.items() if k != "optimizer_seconds"}


def roundtrip(obj):
    return json.loads(json.dumps(obj))


def tiny_config(**overrides):
    values = dict(preset="golden", run_tag="golden", image_size=4, width=16, depth=2, time_features=8,
                  batch_size=8, steps=20, eval_every=10, n_noise_levels=3, val_per_noise=12,
                  test_per_noise=20, eval_batch_size=8, spectrum_lambdas=(0.0, 1.0), seeds=(42,),
                  distributions=("gmm",), cpu_threads=1, device="cpu")
    values.update(overrides)
    return gmm("GMMConfig")(**values)


# ------------------------------------------------------------ runner logic


def runner_arms():
    """Arm tuples each epoch-0 ``scripts/run_gmm_*.py`` main() builds with default flags."""
    baseline = gmm("BASELINE_ARMS")
    gated, plateau, spectral = gmm("gated_arm"), gmm("plateau_arm"), gmm("spectral_cap_arm")
    shaped, log_gate = gmm("shaped_gate_arm"), gmm("log_gate_arm")
    gated_15 = tuple(gated(cov, 1.5, 4.0) for cov in COVARIANCES)
    shared_controls = (*baseline, *gated_15, *(plateau(cov) for cov in COVARIANCES),
                       *(spectral(cov) for cov in COVARIANCES))
    shaped_new = tuple(shaped(cov, mode, 1.5, 4.0) for mode in SHAPED_MODES for cov in COVARIANCES)
    log_new = tuple(log_gate(cov, mode, sigma_lo=0.1, sigma_hi=3.0 if mode == "linear_log_sigma" else 1.5,
                             sigma_switch=0.75, sharpness=2.0)
                    for mode in LOG_MODES for cov in COVARIANCES)
    log_controls = (*shared_controls, *(shaped(cov, mode) for mode in SHAPED_MODES for cov in COVARIANCES))
    return {
        "comparison": {
            "arms": (*baseline, *(gated(cov, switch, 4.0) for switch in (0.5, 1.0, 1.5) for cov in COVARIANCES)),
            # final_arms after the archived selection of sigma_switch=1.5
            "final_arms": (*baseline, *gated_15),
        },
        "plateau": {
            "arms": (*baseline, *gated_15, *(plateau(cov, 1.5, 4.0, 0.8, 1.0) for cov in COVARIANCES)),
        },
        "spectral_gate": {
            "arms": (*baseline, *gated_15, *(plateau(cov, 1.5, 4.0, 0.8, 1.0) for cov in COVARIANCES),
                     *(spectral(cov, 1.5, 4.0, 1.0, 2.0, 0.5) for cov in COVARIANCES)),
        },
        "gate_shapes": {"arms": (*shared_controls, *shaped_new), "new_arms": shaped_new},
        "log_gates": {"arms": (*log_controls, *log_new), "new_arms": log_new},
    }


def runner_config(arms, preset, run_tag, bank_version=TRAIN_BANK):
    """The GMMConfig every runner main() builds (identical presets in all five)."""
    cfg = gmm("GMMConfig")(preset=preset, steps=5000, eval_every=500, width=192, seeds=(42, 43, 44),
                           spectrum_lambdas=(0.0, 0.5, 1.0), distributions=("gmm",), n_noise_levels=9,
                           val_per_noise=512, test_per_noise=2048, cpu_threads=1, device="cpu",
                           bank_version=bank_version)
    if preset == "smoke":
        cfg = replace(cfg, image_size=4, width=32, depth=2, steps=12, eval_every=6, batch_size=32,
                      n_noise_levels=3, val_per_noise=32, test_per_noise=64, spectrum_lambdas=(0.0, 1.0))
    return replace(cfg, methods=tuple(arm.name for arm in arms), run_tag=run_tag)


def runner_transition_sigmas():
    """Default ``transition_sigmas`` of the four fixed-protocol runners (sigma range [0.1, 3])."""
    import numpy as np

    lo, hi = 0.1, 3.0
    plateau = sorted(set([max(lo, 0.8 * 0.8), max(lo, 0.8 * 0.95), *np.linspace(0.8, 1.0, 11).tolist(),
                          min(hi, 1.0 * 1.05), min(hi, 1.0 * 1.2)]))
    spectral = sorted(set([max(lo, 1.0 * 0.6), max(lo, 1.0 * 0.8), *np.geomspace(1.0, 2.0, 13).tolist(),
                           min(hi, 2.0 * 1.2)]))
    shapes_ = sorted({s for s in [*np.linspace(0.5, 2.5, 17).tolist(), 0.8, 1.0, 1.5,
                                  1.5 * (1 - 2 / 4.0), 1.5 * (1 + 2 / 4.0)] if lo <= s <= hi})
    log = sorted({s for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                              1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 0.1, 0.75, 1.5, 3.0] if lo <= s <= hi})
    return {"plateau": plateau, "spectral_gate": spectral, "gate_shapes": shapes_, "log_gates": log}


# ------------------------------------------------------------ 1. arm strings


@probe("gmm", "arm_strings")
def arm_strings(ctx):
    arm_cls = gmm("GMMArm")
    baseline = gmm("BASELINE_ARMS")
    gated, plateau, spectral = gmm("gated_arm"), gmm("plateau_arm"), gmm("spectral_cap_arm")
    shaped, log_gate = gmm("shaped_gate_arm"), gmm("log_gate_arm")

    defaults, params, int_params, errors = {}, {}, {}, {}
    for cov in COVARIANCES:
        defaults[f"gated_arm({cov})"] = gated(cov)
        defaults[f"plateau_arm({cov})"] = plateau(cov)
        defaults[f"spectral_cap_arm({cov})"] = spectral(cov)
        for mode in SHAPED_MODES:
            defaults[f"shaped_gate_arm({cov},{mode})"] = shaped(cov, mode)
        for mode in LOG_MODES:
            defaults[f"log_gate_arm({cov},{mode})"] = log_gate(cov, mode)

        params[f"gated_arm({cov},0.5)"] = gated(cov, 0.5)
        params[f"gated_arm({cov},1.5,4.0)"] = gated(cov, 1.5, 4.0)
        params[f"gated_arm({cov},2.25,8.0)"] = gated(cov, 2.25, 8.0)
        params[f"plateau_arm({cov},1.0,2.0,0.5,2.0)"] = plateau(cov, 1.0, 2.0, 0.5, 2.0)
        params[f"plateau_arm({cov},sigma_lo=0.25,sigma_hi=0.75)"] = plateau(cov, sigma_lo=0.25, sigma_hi=0.75)
        params[f"spectral_cap_arm({cov},1.2,3.0,0.7,1.8,0.25)"] = spectral(cov, 1.2, 3.0, 0.7, 1.8, 0.25)
        params[f"spectral_cap_arm({cov},delta=1.0)"] = spectral(cov, delta=1.0)
        for mode in SHAPED_MODES:
            params[f"shaped_gate_arm({cov},{mode},1.0,2.0)"] = shaped(cov, mode, 1.0, 2.0)
        params[f"log_gate_arm({cov},linear_log_sigma,lo=0.2,hi=2.0)"] = log_gate(
            cov, "linear_log_sigma", sigma_lo=0.2, sigma_hi=2.0)
        params[f"log_gate_arm({cov},bounded_log_sigmoid,lo=0.2,hi=2.0,s=1.0,p=3.0)"] = log_gate(
            cov, "bounded_log_sigmoid", sigma_lo=0.2, sigma_hi=2.0, sigma_switch=1.0, sharpness=3.0)
        params[f"log_gate_arm({cov},linear_log_sigma,s=1.5,p=4.0)"] = log_gate(
            cov, "linear_log_sigma", sigma_switch=1.5, sharpness=4.0)

        # Python ints pass through unchanged into asdict (and hence signatures).
        int_params[f"gated_arm({cov},1,4)"] = gated(cov, 1, 4)
        int_params[f"plateau_arm({cov},2,4,1,2)"] = plateau(cov, 2, 4, 1, 2)
        int_params[f"spectral_cap_arm({cov},2,4,1,3,1)"] = spectral(cov, 2, 4, 1, 3, 1)
        int_params[f"shaped_gate_arm({cov},tanh_sigma,1,2)"] = shaped(cov, "tanh_sigma", 1, 2)
        int_params[f"log_gate_arm({cov},linear_log_sigma,lo=1,hi=3)"] = log_gate(
            cov, "linear_log_sigma", sigma_lo=1, sigma_hi=3)

    errors["gated_arm(bad)"] = error_name(gated, "diagonal")
    errors["plateau_arm(bad)"] = error_name(plateau, "diagonal")
    errors["spectral_cap_arm(bad)"] = error_name(spectral, "diagonal")
    errors["shaped_gate_arm(fourier,log_sigma)"] = error_name(shaped, "fourier", "log_sigma")
    errors["shaped_gate_arm(bad,linear_sigma)"] = error_name(shaped, "diagonal", "linear_sigma")
    errors["log_gate_arm(fourier,tanh_sigma)"] = error_name(log_gate, "fourier", "tanh_sigma")
    errors["log_gate_arm(bounded,switch_outside)"] = error_name(
        log_gate, "fourier", "bounded_log_sigmoid", sigma_switch=2.0)
    errors["log_gate_arm(bounded,sharpness_1)"] = error_name(
        log_gate, "fourier", "bounded_log_sigmoid", sharpness=1.0)
    errors["plateau_arm(lo>=hi)"] = error_name(plateau, "fourier", sigma_lo=1.0, sigma_hi=1.0)

    notebook = gmm("NOTEBOOK_ARMS")
    runners = {f"{runner}.{name}": [arm_json(arm) for arm in arms]
               for runner, lists in runner_arms().items() for name, arms in lists.items()}

    # Archived ground truth, re-serialized from the original JSON text.
    persisted = {}
    archive = {name: f"assets/{name}/protocol.json" for name in ARCHIVED_PROTOCOLS}
    archive["gmm_gated_comparison.summary"] = "assets/gmm_gated_comparison/summary.json"
    archive["log_gate_design.design"] = "assets/log_gate_design/design.json"
    for name, path in archive.items():
        text = (ctx.root / path).read_text()
        persisted[name] = [json.dumps(arm, sort_keys=True) for arm in json.loads(text)["arms"]]

    # Do the rebuilt runner lists reproduce the archive (on the keys it recorded)?
    rebuilt = runner_arms()
    sources = {name: rebuilt[runner]["arms"] for name, runner in ARCHIVED_PROTOCOLS.items()}
    sources.update({
        "gmm_gated_comparison.summary": rebuilt["comparison"]["final_arms"],
        "log_gate_design.design": tuple(a for a in rebuilt["log_gates"]["new_arms"]
                                        if a.parameterization == "fourier_gaussian"),
    })
    matches = {}
    for name, arms in sources.items():
        archived = [json.loads(s) for s in persisted[name]]
        ours = [asdict(a) for a in arms]
        matches[name] = {
            "exact": ours == archived,
            "on_archived_keys": len(ours) == len(archived) and all(
                {k: v for k, v in o.items() if k in a} == a for o, a in zip(ours, archived)),
        }

    return {"exact": {
        "GMMArm_fields": dataclass_fields(arm_cls),
        "BASELINE_ARMS": [arm_json(arm) for arm in baseline],
        "NOTEBOOK_ARMS": {"order": list(notebook), "arms": {k: arm_json(v) for k, v in notebook.items()}},
        "factory_defaults": {k: arm_json(v) for k, v in defaults.items()},
        "factory_params": {k: arm_json(v) for k, v in params.items()},
        "factory_int_params": {k: arm_json(v) for k, v in int_params.items()},
        "factory_errors": errors,
        "runners": runners,
        "persisted": persisted,
        "runner_reproduces_persisted": matches,
    }}


# ------------------------------------------------------------ 2. config facts


@probe("gmm", "config_facts")
def config_facts(ctx):
    cfg_cls = gmm("GMMConfig")
    process_config = gmm("process_config")
    family_cls = gmm("MatchedMomentFamily")
    bank_seed = gmm("bank_seed")

    default = cfg_cls(preset="golden")
    runner_configs, test_banks = {}, {}
    for runner, lists in runner_arms().items():
        for preset in ("smoke", "experiment"):
            cfg = runner_config(lists["arms"], preset, f"gmm_{runner}_golden")
            runner_configs[f"{runner}.{preset}"] = json.dumps(asdict(cfg), sort_keys=True)
            if runner != "comparison":
                test_cfg = replace(cfg, bank_version=TEST_BANKS[runner])
                transition = replace(test_cfg, bank_version=test_cfg.bank_version + "-transition",
                                     test_per_noise=min(1024, test_cfg.test_per_noise))
                test_banks[f"{runner}.{preset}"] = {
                    "test": json.dumps(asdict(test_cfg), sort_keys=True),
                    "transition": json.dumps(asdict(transition), sort_keys=True),
                }

    # Archived configs vs the rebuilt runner config with the archived run_tag.
    persisted, transition_matches = {}, {}
    transitions = runner_transition_sigmas()
    rebuilt = runner_arms()
    for name, runner in ARCHIVED_PROTOCOLS.items():
        protocol = json.loads((ctx.root / "assets" / name / "protocol.json").read_text())
        ours = roundtrip(asdict(runner_config(rebuilt[runner]["arms"], "experiment",
                                              protocol["config"]["run_tag"])))
        persisted[name] = {
            "config": json.dumps(protocol["config"], sort_keys=True),
            "test_bank_version": protocol["test_bank_version"],
            "transition_sigmas": json.dumps(protocol["transition_sigmas"]),
            "runner_config_matches": ours == protocol["config"],
            "runner_test_bank_matches": TEST_BANKS[runner] == protocol["test_bank_version"],
        }
        transition_matches[name] = transitions[runner] == protocol["transition_sigmas"]
    summary_path = ctx.root / "assets/gmm_gated_comparison/summary.json"
    gated_summary = json.loads(summary_path.read_text())
    ours = roundtrip(asdict(runner_config(rebuilt["comparison"]["arms"], "experiment",
                                          gated_summary["config"]["run_tag"])))
    persisted["gmm_gated_comparison.summary"] = {
        "config": json.dumps(gated_summary["config"], sort_keys=True),
        "runner_config_matches": ours == gated_summary["config"],
    }

    tiny = tiny_config()
    case_ids = {}
    for distribution in ("gaussian", "gmm"):
        for lam in (0, 0.0, 0.25, 0.5, 1, 1.0, 1 / 3, 0.1, 1e-3):
            case_ids[f"{distribution}|{lam!r}"] = family_cls(tiny, distribution, lam).case_id

    seeds = {}
    for version in (*BANK_VERSIONS, *(v + "-transition" for v in TEST_BANKS.values() if v != TRAIN_BANK)):
        cfg = replace(tiny, bank_version=version)
        for distribution, lam in (("gmm", 0.0), ("gmm", 0.5), ("gmm", 1.0), ("gaussian", 0.0), ("gaussian", 1.0)):
            family = family_cls(cfg, distribution, lam)
            for split in ("validation", "test"):
                seeds[f"{version}|{family.case_id}|{split}"] = bank_seed(family, split)

    return {
        "exact": {
            "GMMConfig_fields": dataclass_fields(cfg_cls),
            "default_asdict": json.dumps(asdict(default), sort_keys=True),
            "process_config": {
                "default": process_config(default),
                "custom": process_config(replace(default, sigma_min=0.05, sigma_max=5.0)),
                "tiny": process_config(tiny),
            },
            "runner_configs": runner_configs,
            "runner_test_configs": test_banks,
            "persisted": persisted,
            "case_ids": case_ids,
            "bank_seeds": seeds,
        },
        "numeric": {"runner_transition_sigmas": transitions,
                    "runner_transition_sigmas_match_persisted": transition_matches},
    }


# ------------------------------------------------------------ 3. family and banks


def bank_facts(bank):
    import torch

    keys = ("y", "noise", "oracle_scaled", "reference_scaled", "scalar_reference_scaled")
    exact = {
        "split": bank.split,
        "n_observations": bank.n_observations,
        "n_entries": len(bank.entries),
        "entry_keys": sorted(bank.entries[0]),
        "noise_bins": [e["noise_bin"] for e in bank.entries],
        "shapes": {k: [list(e[k].shape) for e in bank.entries] for k in keys},
        "dtypes": {k: str(bank.entries[0][k].dtype) for k in keys},
    }
    numeric = {
        "fingerprint": bank.fingerprint,
        "sigma": [e["sigma"] for e in bank.entries],
        "coordinate": [e["coordinate"] for e in bank.entries],
        "entries_digest": tensor_digest(bank.entries),
        "tensors": {k: summary(torch.cat([e[k] for e in bank.entries])) for k in keys},
    }
    return exact, numeric


@probe("gmm", "family_bank")
def family_bank(ctx):
    import torch

    family_cls = gmm("MatchedMomentFamily")
    make_power, make_bank, frequency_bands = gmm("make_power"), gmm("make_bank"), gmm("frequency_bands")
    cfg = tiny_config()
    exact, numeric = {}, {}

    g = torch.Generator().manual_seed(11)
    y = torch.randn(3, 1, cfg.image_size, cfg.image_size, generator=g, dtype=torch.float64)
    alpha = torch.tensor([1.0, 0.9, 0.5], dtype=torch.float64)
    sigma = torch.tensor([0.1, 0.7, 3.0], dtype=torch.float64)

    for distribution, lam in (("gmm", 0.0), ("gmm", 1.0), ("gaussian", 1.0)):
        family = family_cls(cfg, distribution, lam)
        key = family.case_id
        mean, covariance = family.analytic_moments()
        log_density, score = family.log_prob_and_score(y, alpha, sigma)
        exact[key] = {
            "case_id": family.case_id, "distribution": family.distribution, "lam": family.lam,
            "size": family.size, "d": family.d, "rho": family.rho, "n_components": family.n_components,
            "stats": shapes(family.stats),
        }
        numeric[key] = {
            "within_fraction": family.within_fraction,
            "make_power": summary(make_power(cfg.image_size, lam, cfg)),
            "power": summary(family.power),
            "stats": {k: summary(v) for k, v in family.stats.items()},
            "means": summary(family.means),
            "covariance": summary(family.covariance),
            "eigenvalues": summary(family.eigenvalues),
            "analytic_mean": summary(mean),
            "analytic_covariance": summary(covariance),
            "sample_cpu": summary(family.sample_cpu(6, torch.Generator().manual_seed(7))),
            "sample_cpu_float64": summary(family.sample_cpu(6, torch.Generator().manual_seed(7),
                                                            dtype=torch.float64)),
            "log_density": summary(log_density),
            "score": summary(score),
            "gaussian_score": summary(family.gaussian_score(y, alpha, sigma)),
            "gaussian_score_scalar": summary(family.gaussian_score(y, alpha, sigma, scalar=True)),
        }

    banks_exact, banks_numeric = {}, {}
    for lam in (0.0, 1.0):
        family = family_cls(cfg, "gmm", lam)
        for split in ("validation", "test"):
            e, n = bank_facts(make_bank(family, split))
            banks_exact[f"{family.case_id}|{split}"] = e
            banks_numeric[f"{family.case_id}|{split}"] = n
    # The transition diagnostic: separate RNG namespace and explicit sigma grid.
    transition = family_cls(replace(cfg, bank_version=cfg.bank_version + "-transition",
                                    test_per_noise=min(1024, cfg.test_per_noise)), "gmm", 1.0)
    e, n = bank_facts(make_bank(transition, "test", sigmas=[0.2, 0.5, 1.0, 2.0]))
    banks_exact["transition|test"] = e
    banks_numeric["transition|test"] = n

    family = family_cls(cfg, "gmm", 1.0)
    errors = {
        "split_train": error_name(make_bank, family, "train"),
        "sigmas_below_range": error_name(make_bank, family, "test", sigmas=[0.05, 1.0]),
        "sigmas_above_range": error_name(make_bank, family, "test", sigmas=[1.0, 4.0]),
        "sigmas_decreasing": error_name(make_bank, family, "test", sigmas=[1.0, 0.5]),
        "sigmas_empty": error_name(make_bank, family, "test", sigmas=[]),
        "family_bad_distribution": error_name(family_cls, cfg, "uniform", 1.0),
    }

    bands = {}
    for name, band_cfg in (("tiny", cfg), ("default", gmm("GMMConfig")(preset="golden")),
                           ("bins3", tiny_config(frequency_bins=3))):
        masks, counts = frequency_bands(band_cfg)
        bands[name] = {"masks": [m.int().tolist() for m in masks], "counts": counts}

    return {
        "exact": {"families": exact, "banks": banks_exact, "errors": errors, "frequency_bands": bands},
        "numeric": {"families": numeric, "banks": banks_numeric},
    }


# ------------------------------------------------------------ 4. short training


def train_arms():
    return (gmm("BASELINE_ARMS")[0], gmm("gated_arm")("scalar", 1.5, 4.0),
            gmm("log_gate_arm")("fourier", "bounded_log_sigmoid"))


def train_facts(root, folder, state, cfg, family, validation, arm, seed):
    import torch

    checkpoint = folder / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metrics = json.loads((folder / "metrics.json").read_text())
    description = dict(config=asdict(cfg), arm=asdict(arm), seed=seed, family=family.case_id,
                       validation_sha256=validation.fingerprint, provenance=PROVENANCE)
    signature = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    groups = payload["optimizer"]["param_groups"]
    ema_backbone = [v for k, v in payload["ema"].items() if k.startswith("backbone.")]
    exact = {
        "files": sorted(str(p.relative_to(root)) for p in folder.rglob("*") if p.is_file()),
        "checkpoint_keys": sorted(payload),
        "checkpoint_config": roundtrip(payload["config"]),
        "checkpoint_provenance": payload["provenance"],
        "state_keys": sorted(state),
        "metrics_keys": sorted(metrics),
        "description": {k: v for k, v in roundtrip(description).items() if k != "validation_sha256"},
        "signature_matches_description": metrics["signature"] == signature == state["signature"],
        "payload_state_matches_metrics": roundtrip(without_time(payload["state"])) == without_time(metrics),
        "returned_state_matches_metrics": roundtrip(without_time(state)) == without_time(metrics),
        "identity": {k: metrics[k] for k in ("distribution", "spectrum_lambda", "method", "arm", "seed",
                                             "step", "completed")},
        "validation_steps": [m["step"] for m in metrics["validation"]],
        "validation_splits": [m["split"] for m in metrics["validation"]],
        "ema_keys": shapes(payload["ema"]),
        "model_keys": shapes(payload["model"]),
        "optimizer": [{k: g[k] for k in ("lr", "betas", "eps", "weight_decay")} | {"n_params": len(g["params"])}
                      for g in groups],
    }
    numeric = {
        "metrics": without_time(metrics),
        "ema": tensor_digest(payload["ema"]),
        "ema_backbone": summary(torch.cat([v.flatten() for v in ema_backbone])),
        "model": tensor_digest(payload["model"]),
        "optimizer_state": tensor_digest(payload["optimizer"]["state"]),
        "data_rng": tensor_digest(payload["data_rng"]),
    }
    return exact, numeric


@probe("gmm", "train_arm_short")
def train_arm_short(ctx):
    family_cls, make_bank, train_arm = gmm("MatchedMomentFamily"), gmm("make_bank"), gmm("train_arm")
    cfg = tiny_config()
    family = family_cls(cfg, "gmm", 1.0)
    validation = make_bank(family, "validation")
    root = ctx.tmp / "runs"
    exact, numeric, states = {}, {}, []
    for arm in train_arms():
        state = train_arm(family, arm, 42, validation, root, PROVENANCE)
        states.append(state)
        folder = next(p.parent for p in root.rglob("checkpoint.pt") if p.parent.name.startswith(arm.name + "_seed"))
        e, n = train_facts(root, folder, state, cfg, family, validation, arm, 42)
        rerun = train_arm(family, arm, 42, validation, root, PROVENANCE)
        e["rerun_returns_saved_state"] = roundtrip(without_time(rerun)) == roundtrip(without_time(state))
        exact[arm.name] = e
        numeric[arm.name] = n

    # Paired arms share initialization, training stream and final data RNG.
    pairing = {key: len({s[key] for s in states}) == 1
               for key in ("initial_backbone_sha256", "training_stream_first_batch_sha256",
                           "final_data_rng_sha256")}
    pairing["final_ema_differs"] = len({s["final_ema_backbone_sha256"] for s in states}) == len(states)

    # Interrupted + resumed training reproduces the uninterrupted run.
    arm = train_arms()[1]
    paused = train_arm(family, arm, 42, validation, ctx.tmp / "split", PROVENANCE, stop_after=7)
    resumed = train_arm(family, arm, 42, validation, ctx.tmp / "split", PROVENANCE)
    full = states[1]
    resume = {
        "paused_completed": paused["completed"],
        "paused_steps": [m["step"] for m in paused["validation"]],
        "resumed_steps": [m["step"] for m in resumed["validation"]],
        "same_final_ema": resumed["final_ema_backbone_sha256"] == full["final_ema_backbone_sha256"],
        "same_final_rng": resumed["final_data_rng_sha256"] == full["final_data_rng_sha256"],
        "same_final_validation": resumed["validation"][-1] == full["validation"][-1],
        "mismatch_error": error_name(train_arm, family, arm, 42, validation, ctx.tmp / "split",
                                     {"golden": "different"}),
        "gpu_device_error": error_name(train_arm, family_cls(replace(cfg, device="cuda"), "gmm", 1.0), arm,
                                       42, validation, ctx.tmp / "gpu", PROVENANCE),
    }
    return {"exact": {"arms": exact, "pairing": pairing, "resume": resume},
            "numeric": {"arms": numeric, "paused_validation": paused["validation"]}}


# ------------------------------------------------------------ 5. evaluation


def metric_shape(metric):
    return {
        "keys": sorted(metric),
        "split": metric["split"],
        "n_observations": metric["n_observations"],
        "per_noise_keys": [sorted(row) for row in metric["per_noise"]],
        "n_frequency": len(metric["score_error_by_frequency"]),
        "none_fields": sorted(k for k, v in metric.items() if v is None),
    }


@probe("gmm", "evaluate_model")
def evaluate_model_probe(ctx):
    import torch

    family_cls, make_bank, make_model = gmm("MatchedMomentFamily"), gmm("make_bank"), gmm("make_model")
    evaluate_model, train_arm = gmm("evaluate_model"), gmm("train_arm")
    cfg = tiny_config()
    family = family_cls(cfg, "gmm", 1.0)
    validation, test = make_bank(family, "validation"), make_bank(family, "test")
    transition = make_bank(family_cls(replace(cfg, bank_version=cfg.bank_version + "-transition"),
                                      "gmm", 1.0), "test", sigmas=[0.2, 0.5, 1.0, 2.0])
    arm = gmm("gated_arm")("fourier", 1.5, 4.0)
    train_arm(family, arm, 42, validation, ctx.tmp / "runs", PROVENANCE)
    checkpoint = next((ctx.tmp / "runs").rglob("checkpoint.pt"))
    trained = make_model(family, arm, 42)
    trained.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["ema"], strict=True)

    flat = family_cls(cfg, "gmm", 0.0)
    flat_validation = make_bank(flat, "validation")
    metrics = {
        "trained|test": evaluate_model(trained, family, test),
        "trained|transition": evaluate_model(trained, family, transition),
        "initial_score|test": evaluate_model(make_model(family, gmm("BASELINE_ARMS")[0], 42), family, test),
        "initial_fourier_normalized|test": evaluate_model(make_model(family, gmm("BASELINE_ARMS")[4], 42),
                                                          family, test),
        "reference_fourier|test": evaluate_model(None, family, test),
        "reference_scalar|test": evaluate_model(None, family, test, reference="scalar"),
        "reference_fourier|lambda0_validation": evaluate_model(None, flat, flat_validation),
        "reference_scalar|lambda0_validation": evaluate_model(None, flat, flat_validation, reference="scalar"),
    }
    return {"exact": {k: metric_shape(v) for k, v in metrics.items()},
            "numeric": metrics}


# ------------------------------------------------------------ 6. gate selection


def selection_results(switches=(0.5, 1.0, 1.5), lambdas=(0.0, 1.0), seeds=(42, 43), error=None):
    """Hand-built results in the format train_arm returns (dyadic errors: exact means)."""
    gated, plateau = gmm("gated_arm"), gmm("plateau_arm")
    error = error or (lambda switch, cov, lam, seed: 0.5 + 0.25 * (1.5 - switch) + 0.125 * lam
                      + (0.0625 if cov == "fourier" else 0.0) + 0.03125 * (seed - 42))
    results = []
    for lam in lambdas:
        for seed in seeds:
            # Non-log_sigma arms (baselines, plateaus) are ignored by selection.
            for arm in (*gmm("BASELINE_ARMS"), plateau("fourier")):
                results.append(dict(arm=asdict(arm), completed=True, step=8, spectrum_lambda=lam, seed=seed,
                                    method=arm.name,
                                    validation=[dict(split="validation", step=8, score_error=0.0)]))
            for switch in switches:
                for cov in COVARIANCES:
                    arm = gated(cov, switch, 4.0)
                    results.append(dict(arm=asdict(arm), completed=True, step=8, spectrum_lambda=lam, seed=seed,
                                        method=arm.name,
                                        validation=[dict(split="validation", step=4, score_error=9.0),
                                                    dict(split="validation", step=8,
                                                         score_error=error(switch, cov, lam, seed))]))
    return results


@probe("gmm", "select_gate")
def select_gate_probe(ctx):
    import copy

    select_gate = symbol("select_gate", *PIPELINE)

    def outcome(results, switches):
        try:
            return {"selection": select_gate(results, switches)}
        except Exception as error:  # noqa: BLE001 - the error is the fact
            return {"error": type(error).__name__, "message": str(error)}

    base = selection_results()
    cases = {
        "default": outcome(base, [0.5, 1.0, 1.5]),
        "switch_order": outcome(base, [1.5, 0.5, 1.0]),
        "tie_prefers_smaller_switch": outcome(
            selection_results(error=lambda s, c, l, seed: 0.5 if s in (1.0, 1.5) else 0.75), [1.5, 1.0, 0.5]),
        "two_switches": outcome(selection_results(switches=(0.5, 1.5)), [0.5, 1.5]),
    }
    incomplete = copy.deepcopy(base)
    incomplete[0]["completed"] = False
    cases["incomplete"] = outcome(incomplete, [0.5, 1.0, 1.5])
    cases["missing_switch"] = outcome(base, [0.5, 1.0, 1.5, 2.0])
    cases["extra_switch"] = outcome(base, [0.5, 1.0])
    unpaired = [r for r in base if not (r["arm"]["gate_mode"] == "log_sigma" and r["arm"]["sigma_switch"] == 1.0
                                        and r["seed"] == 43 and r["spectrum_lambda"] == 1.0)]
    cases["unpaired"] = outcome(unpaired, [0.5, 1.0, 1.5])
    gated_rows = [i for i, r in enumerate(base) if r["arm"]["gate_mode"] == "log_sigma"]
    cases["duplicate"] = outcome(base + [copy.deepcopy(base[gated_rows[0]])], [0.5, 1.0, 1.5])
    stale = copy.deepcopy(base)
    stale[gated_rows[0]]["validation"][-1]["step"] = 4
    cases["not_final_step"] = outcome(stale, [0.5, 1.0, 1.5])
    wrong_split = copy.deepcopy(base)
    wrong_split[gated_rows[0]]["validation"][-1]["split"] = "test"
    cases["test_split"] = outcome(wrong_split, [0.5, 1.0, 1.5])
    cases["no_gated_runs"] = outcome([r for r in base if r["arm"]["gate_mode"] != "log_sigma"], [])
    return {"exact": cases}


# ------------------------------------------------------------ 7. toy model


def patterned(state, prefix="backbone."):
    """Deterministic non-trivial weights (the last layer is zero at init)."""
    import torch

    out = {}
    for i, (key, value) in enumerate(state.items()):
        if key.startswith(prefix):
            n = value.numel()
            value = (torch.sin(torch.arange(n, dtype=torch.float64) * 0.37 + i) * 0.3).reshape(value.shape)
            out[key] = value.to(state[key].dtype)
        else:
            out[key] = value
    return out


@probe("gmm", "toy_model")
def toy_model(ctx):
    import torch

    family_cls, make_model = gmm("MatchedMomentFamily"), gmm("make_model")
    toy_cls, mlp_cls, arm_cls = gmm("ToyScoreModel"), gmm("SmallJointMLP"), gmm("GMMArm")
    cfg = tiny_config()
    family = family_cls(cfg, "gmm", 1.0)
    g = torch.Generator().manual_seed(123)
    y = torch.randn(3, 1, cfg.image_size, cfg.image_size, generator=g)
    coordinate = torch.tensor([0.05, 0.5, 0.95])

    arms = list(gmm("BASELINE_ARMS"))
    for cov in COVARIANCES:
        arms += [gmm("gated_arm")(cov), gmm("plateau_arm")(cov), gmm("spectral_cap_arm")(cov),
                 *(gmm("shaped_gate_arm")(cov, mode) for mode in SHAPED_MODES),
                 *(gmm("log_gate_arm")(cov, mode) for mode in LOG_MODES)]

    exact, numeric = {}, {}
    for arm in arms:
        model = make_model(family, arm, 42)
        level = model.process.level(coordinate)
        with torch.no_grad():
            initial = {"forward": summary(model(y, level)), "scaled_score": summary(model.scaled_score(y, level))}
            model.load_state_dict(patterned(model.state_dict()), strict=True)
            forward = {"forward": summary(model(y, level)), "scaled_score": summary(model.scaled_score(y, level))}
        exact[arm.name] = {
            "state_dict": shapes(model.state_dict()),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "n_trainable_backbone": sum(p.numel() for p in model.backbone.parameters()),
            "method": model.method,
            "loss_objective": model.loss_objective,
            "embedding": model.embedding,
            "has_reference": model.reference is not None,
        }
        numeric[arm.name] = {
            "state": tensor_digest(make_model(family, arm, 42).state_dict()),
            "level_sigma": summary(level.sigma),
            "initial": initial,
            "patterned": forward,
        }

    # Same seed, different arms: identical backbone initialization.
    inits = {tensor_digest(make_model(family, arm, 42).backbone.state_dict()) for arm in arms}
    other_seed = tensor_digest(make_model(family, arms[0], 43).backbone.state_dict())

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        mlp = mlp_cls(cfg)
    mlp_init = tensor_digest(mlp.state_dict())
    sigma = torch.tensor([0.1, 0.7, 3.0])
    with torch.no_grad():
        mlp_initial = summary(mlp(y, sigma))
        mlp.load_state_dict(patterned(mlp.state_dict(), prefix=""), strict=True)
        mlp_forward = summary(mlp(y, sigma))

    by_name = toy_cls(cfg, family.stats, "scalar_gated")
    return {
        "exact": {
            "arms": exact,
            "same_seed_same_backbone": len(inits) == 1,
            "other_seed_differs": other_seed not in inits,
            "notebook_method_by_name": {"method": by_name.method, "arm": arm_json(by_name.arm)},
            "invalid_arm_error": error_name(toy_cls, cfg, family.stats,
                                            arm_cls("bad", "score", "normalized_residual")),
            "unknown_parameterization_error": error_name(toy_cls, cfg, family.stats, arm_cls("bad", "diffusion")),
            "mlp_state_dict": shapes(mlp.state_dict()),
            "default_mlp_state_dict": shapes(mlp_cls(gmm("GMMConfig")(preset="golden")).state_dict()),
        },
        "numeric": {
            "arms": numeric,
            "mlp_init": mlp_init,
            "mlp_initial_forward": mlp_initial,
            "mlp_patterned_forward": mlp_forward,
        },
    }
