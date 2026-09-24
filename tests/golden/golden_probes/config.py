"""Config contracts: canonical configs, run names, resume signatures, gates.

These protect the planned ``parse_config.py`` / ``gates.py`` rewrite (E1):

* every persisted format / version string,
* every image and LDM config file resolved, named and signed,
* the gate grid (all modes x defaults, non-default and legacy forms),
* verbatim config error strings (image, gate, LDM),
* ``--set`` override semantics,
* the ``train.py`` / ``ldm.py`` dry-run CLIs.

All values are exact: configs, names and signatures are pure dict/str logic.
Absolute paths are replaced by ``<ROOT>``, ``<TMP>`` and ``<PYTHON>``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
import re
import subprocess
import sys

from golden_probes import file_sha256, json_digest, probe, symbol

CONFIG = ("fourier_score.config",)
GATES = ("fourier_score.gates", "fourier_score.method")
CHECKPOINTS = ("fourier_score.trainer.checkpoints", "fourier_score.checkpoints")
LDM_CONFIG = ("fourier_score.ldm.config",)
LDM_TRAINING = ("fourier_score.ldm.training",)

IMAGE_CONFIGS = (
    "config.json",
    "configs/base.json",
    "configs/celeba64_folder.json",
    "configs/cifar10.json",
    "configs/cifar10_1m3.json",
    "configs/cifar10_950k.json",
    "configs/cifar10_ablation.json",
    "configs/cifar10_ddpm.json",
    "configs/cifar10_full.json",
    "configs/cifar10_paper950k.json",
    "configs/ffhq256_folder.json",
    "configs/mnist.json",
    "configs/mnist_ddpm.json",
    "configs/smoke.json",
    "configs/score_sde/cifar10_ncsnpp_continuous.json",
    "configs/score_sde/cifar10_ncsnpp_deep_continuous.json",
    "configs/score_sde/ffhq_256_ncsnpp_continuous.json",
)
LDM_CONFIGS = (
    "configs/ldm/celebahq.json",
    "configs/ldm/ffhq.json",
    "configs/ldm/lsun_bedrooms.json",
    "configs/ldm/lsun_churches.json",
    "configs/ldm/lsun_churches_l2.json",
)
# Literal on purpose: the accepted set must not be read back from the code.
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


# ---------------------------------------------------------------- helpers


def _scrub(value, ctx):
    """Replace machine paths inside every string of a nested value."""
    needles = {}
    for exe in (sys.executable, os.path.realpath(sys.executable)):
        needles[exe] = "<PYTHON>"
    for tmp in (str(ctx.tmp), os.path.realpath(ctx.tmp)):
        needles[tmp] = "<TMP>"
    for root in (str(ctx.root), os.path.realpath(ctx.root)):
        needles[root] = "<ROOT>"
    ordered = sorted(needles.items(), key=lambda item: -len(item[0]))

    def walk(v):
        if isinstance(v, str):
            for needle, token in ordered:
                v = v.replace(needle, token)
            return v
        if isinstance(v, dict):
            return {walk(k): walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        return v

    return walk(value)


def _error(exc, ctx):
    return {"error": type(exc).__name__, "message": _scrub(str(exc), ctx)}


def _attempt(ctx, fn, *args):
    """``(True, value)`` or ``(False, {"error", "message"})``."""
    try:
        return True, fn(*args)
    except Exception as exc:  # noqa: BLE001 - the exception is the contract
        return False, _error(exc, ctx)


def _raises(ctx, fn, *args):
    ok, value = _attempt(ctx, fn, *args)
    return {"no_error": True} if ok else value


def _key(rel):
    return rel.removesuffix(".json").replace("/", ".")


def _typed(value):
    return {"value": value, "type": type(value).__name__}


def _get(cfg, dotted):
    for part in dotted.split("."):
        cfg = cfg[part]
    return cfg


def _write_json(folder, files, ctx):
    folder.mkdir(parents=True, exist_ok=True)
    for name, obj in files.items():
        text = obj if isinstance(obj, str) else json.dumps(obj)
        (folder / name).write_text(text.replace("<ROOT>", str(ctx.root)))
    return folder


# ---------------------------------------------------------------- persisted strings


def _bank_literals(root):
    pattern = re.compile(r"""["'](gmm-[a-z0-9]+(?:-[a-z0-9]+)*-v\d+)["']""")
    found = set()
    for top in ("fourier_score", "scripts", "experiments"):
        base = root / top
        if base.is_dir():
            for path in base.rglob("*.py"):
                found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return sorted(found)


@probe("config", "persisted_format_strings")
def persisted_format_strings(ctx):
    os.chdir(ctx.root)
    fmt = symbol("FORMAT", "fourier_score.trainer.checkpoints", "fourier_score.checkpoints")
    load_config = symbol("load_config", *CONFIG)
    ldm_load_config = symbol("load_config", *LDM_CONFIG)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    build_data = symbol(
        "build_data", "fourier_score.data_loader.data_loaders", "fourier_score.data"
    )
    gmm_config = symbol("GMMConfig", "fourier_score.gmm")
    base = json.loads((ctx.root / "configs/base.json").read_text(encoding="utf-8"))
    image_cfg = load_config("configs/smoke.json")
    ldm_cfg = ldm_load_config("configs/ldm/ffhq.json")
    return {
        "exact": {
            "image_checkpoint_format": fmt,
            "ldm_training_format": symbol("FORMAT", *LDM_TRAINING),
            "ldm_cache_format": symbol("CACHE_FORMAT", "fourier_score.ldm.data"),
            "image_schema_version_base_json": base["schema_version"],
            "image_schema_version_resolved": image_cfg["schema_version"],
            "ldm_schema_version_resolved": ldm_cfg["schema_version"],
            "stats_metadata_schema_version": build_data(image_cfg).metadata["schema_version"],
            "gmm_default_bank_version": gmm_config(preset="smoke").bank_version,
            "gmm_bank_version_literals": _bank_literals(ctx.root),
            "ldm_spec_upstream_revision": load_spec(ldm_cfg)["upstream_revision"],
            "download_ldm_format": symbol("FORMAT", "scripts.download_ldm"),
            "download_ldm_upstream_revision": symbol("UPSTREAM_REVISION", "scripts.download_ldm"),
            "download_ldm_data_format": symbol("FORMAT", "scripts.download_ldm_data"),
            "download_score_sde_format": symbol("FORMAT", "scripts.download_score_sde"),
            "download_score_sde_revision": symbol("REVISION", "scripts.download_score_sde"),
        }
    }


@probe("config", "smoke_resolved")
def smoke_resolved(ctx):
    load_config = symbol("load_config", "fourier_score.config")
    cfg = load_config(str(ctx.root / "configs/smoke.json"))
    return {"exact": {"sha256": json_digest(cfg), "name": cfg["name"]}}


# ---------------------------------------------------------------- config files


@probe("config", "config_inventory")
def config_inventory(ctx):
    """Every config file and its bytes (configs/ is frozen by the plan)."""
    files = [ctx.root / "config.json", *sorted((ctx.root / "configs").rglob("*"))]
    digests = {
        p.relative_to(ctx.root).as_posix(): file_sha256(p) for p in files if p.is_file()
    }
    image = sorted(
        k for k in digests
        if k == "config.json"
        or re.fullmatch(r"configs/[^/]+\.json", k)
        or re.fullmatch(r"configs/score_sde/[^/]+\.json", k)
    )
    ldm = sorted(k for k in digests if re.fullmatch(r"configs/ldm/[^/]+\.json", k))
    return {
        "exact": {
            "files_sha256": digests,
            "image_configs": image,
            "ldm_configs": ldm,
            "image_configs_all_probed": image == sorted(IMAGE_CONFIGS),
            "ldm_configs_all_probed": ldm == sorted(LDM_CONFIGS),
        }
    }


def _register_image_config(rel):
    @probe("config", "image_config." + _key(rel))
    def image_config(ctx):
        os.chdir(ctx.root)
        load_config = symbol("load_config", *CONFIG)
        validate = symbol("validate", *CONFIG)
        experiment_name = symbol("experiment_name", *CONFIG)
        resume_signature = symbol("resume_signature", *CHECKPOINTS)
        cfg = load_config(rel)
        name = experiment_name(cfg)
        signature = resume_signature(cfg)
        revalidated = validate(json.loads(json.dumps(cfg)))
        cfg, signature, name = _scrub((cfg, signature, name), ctx)
        return {
            "exact": {
                "resolved": cfg,
                "resolved_sha256": json_digest(cfg),
                "experiment_name": name,
                "resume_signature": signature,
                "resume_signature_sha256": json_digest(signature),
                "revalidated_is_identical": _scrub(revalidated, ctx) == cfg,
            }
        }

    return image_config


def _register_ldm_config(rel):
    @probe("config", "ldm_config." + _key(rel))
    def ldm_config(ctx):
        os.chdir(ctx.root)
        load_config = symbol("load_config", *LDM_CONFIG)
        validate = symbol("validate", *LDM_CONFIG)
        experiment_name = symbol("experiment_name", *LDM_CONFIG)
        load_spec = symbol("load_spec", *LDM_CONFIG)
        signature = symbol("signature", *LDM_TRAINING)
        cfg = load_config(rel)
        name = experiment_name(cfg)
        sig = signature(cfg)
        spec = load_spec(cfg)
        revalidated = validate(json.loads(json.dumps(cfg)))
        cfg, sig, name, spec = _scrub((cfg, sig, name, spec), ctx)
        return {
            "exact": {
                "resolved": cfg,
                "resolved_sha256": json_digest(cfg),
                "experiment_name": name,
                "signature": sig,
                "signature_sha256": json_digest(sig),
                "revalidated_is_identical": _scrub(revalidated, ctx) == cfg,
                "spec": {
                    "sha256": json_digest(spec),
                    "keys": sorted(spec),
                    "custom_upstream_config": spec["custom_upstream_config"],
                    "upstream_revision": spec["upstream_revision"],
                    "upstream_config_sha256": spec["upstream_config_sha256"],
                    "model": spec["model"],
                    "first_stage_kind": spec["first_stage"]["kind"],
                    "image_size": spec["image_size"],
                    "latent_shape": spec["latent_shape"],
                    "timesteps": spec["timesteps"],
                    "linear_start": spec["linear_start"],
                    "linear_end": spec["linear_end"],
                    "scale_by_std": spec["scale_by_std"],
                    "scale_factor": spec["scale_factor"],
                    "upstream_loss": spec["upstream_loss"],
                    "loss_type": spec["loss_type"],
                    "paper_training": spec["paper_training"],
                    "unet_sha256": json_digest(spec["unet"]),
                    "first_stage_params_sha256": json_digest(spec["first_stage"]["params"]),
                    "scheduler": spec["scheduler"],
                },
            }
        }

    return ldm_config


for _rel in IMAGE_CONFIGS:
    _register_image_config(_rel)
for _rel in LDM_CONFIGS:
    _register_ldm_config(_rel)


@probe("config", "ldm_custom_upstream_config")
def ldm_custom_upstream_config(ctx):
    """An explicit upstream_config path is labeled custom and hashed by bytes."""
    os.chdir(ctx.root)
    load_config = symbol("load_config", *LDM_CONFIG)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    source = ctx.root / "configs/ldm/upstream/ffhq.yaml"
    copied = ctx.tmp / "ffhq_copy.yaml"
    copied.write_bytes(source.read_bytes())
    stock = load_spec(load_config("configs/ldm/ffhq.json"))
    custom = load_spec(load_config("configs/ldm/ffhq.json", [f"upstream_config={copied}"]))
    differing = sorted(k for k in stock if stock[k] != custom[k])
    return {
        "exact": {
            "stock_custom": stock["custom_upstream_config"],
            "copy_custom": custom["custom_upstream_config"],
            "copy_same_sha256": custom["upstream_config_sha256"] == stock["upstream_config_sha256"],
            "differing_spec_keys": differing,
        }
    }


# ---------------------------------------------------------------- gates


@probe("config", "gate_modes_accepted")
def gate_modes_accepted(ctx):
    validate_gate = symbol("validate_gate", *GATES)
    candidates = (
        *GATE_MODES, "", "None", "NONE", "sigmoid", "log", "linear", "tanh", "plateau",
        "cap", "spectral", "log_linear", "bounded_sigmoid", "linear_log", "log_sigma ",
    )
    # Parameters valid for every mode (bounded needs lo < switch < hi).
    params = {"sigma_lo": 0.5, "sigma_hi": 2.0, "sigma_switch": 1.0, "sharpness": 4.0}
    accepted, rejected, errors = [], [], set()
    for mode in candidates:
        ok, value = _attempt(ctx, validate_gate, {"mode": mode, **params})
        if ok:
            accepted.append(mode)
        else:
            rejected.append(mode)
            errors.add(json.dumps(value, sort_keys=True))
    return {
        "exact": {
            "accepted_sorted": sorted(accepted),
            "rejected_sorted": sorted(rejected),
            "rejection_errors": [json.loads(e) for e in sorted(errors)],
        }
    }


def _gate_cases(mode):
    """Gate inputs per mode.

    Every mode that prints a number in its run name has at least one case
    whose printed values are NOT dyadic (0.1, 0.3, 0.9, 1.1, 1/3).  Dyadic
    values (0.25, 0.75, 1, 2, 4) print identically under ``.17g``, ``.6g``,
    ``repr`` and ``g``, so they cannot pin the precision of the run-name
    formatter; e.g. ``0.3 -> s0p29999999999999999`` and ``1.1 ->
    p1p1000000000000001`` under the epoch-0 ``.17g`` rule.
    """
    m = mode
    ignored = {"sigma_lo": 0.1, "sigma_hi": 3.0, "delta": 0.25}
    non_dyadic = {"sigma_switch": 0.3, "sharpness": 1.1}
    cases = {
        "none": [
            None,
            {"mode": m},
            {"mode": m, "sigma_switch": 2.0, "sharpness": 8.0, "value": 0.5, **ignored},
        ],
        "constant": [
            {"mode": m},
            {"mode": m, "value": 0.5},
            {"mode": m, "value": 0},
            {"mode": m, "value": 0.25, "sigma_switch": 2.0, "sharpness": 8.0, **ignored},
            {"mode": m, "value": 0.1},
            {"mode": m, "value": 1 / 3},
        ],
        "log_sigma": [
            {"mode": m},
            {"mode": m, "sigma_switch": 1.5},
            {"mode": m, "sigma_switch": 0.5, "sharpness": 8.0},
            {"mode": m, "sigma_switch": 2, "sharpness": 1e20},
            {"mode": m, "sigma_switch": 1e-05, "sharpness": 0.3},
            {"mode": m, "sigma_switch": 1.5, "value": 0.5, **ignored},
            {"mode": m, **non_dyadic},
        ],
        "log_sigma_plateau": [
            {"mode": m},
            {"mode": m, "sigma_switch": 1.5},
            {"mode": m, "sigma_switch": 1.5, "sigma_hi": 1.1},
            {"mode": m, "sigma_switch": 2, "sigma_lo": 1, "sigma_hi": 3, "sharpness": 2.0,
             "delta": 0.25},
            {"mode": m, "sigma_switch": 0.9, "sigma_lo": 0.3, "sigma_hi": 1.1, "sharpness": 1.1},
        ],
        "spectral_cap": [
            {"mode": m},
            {"mode": m, "sigma_switch": 1.5, "sigma_lo": 1.0, "sigma_hi": 2.0},
            {"mode": m, "sigma_switch": 1.5, "sigma_lo": 1.0, "sigma_hi": 2.0, "delta": 0.25},
            {"mode": m, "sigma_switch": 1.5, "sigma_lo": 1.0, "sigma_hi": 2.5, "delta": 0.001},
            {"mode": m, "delta": 0.1},
            {"mode": m, "sigma_switch": 0.9, "sigma_lo": 0.3, "sigma_hi": 1.1, "sharpness": 1.1,
             "delta": 0.1},
        ],
        "linear_sigma": [
            {"mode": m},
            {"mode": m, "sigma_switch": 1.5},
            {"mode": m, "sigma_switch": 1.5, "sharpness": 8.0},
            {"mode": m, "sigma_switch": 1.5, **ignored},
            {"mode": m, **non_dyadic},
        ],
        "linear_log_sigma": [
            {"mode": m},
            {"mode": m, "sigma_lo": 0.1, "sigma_hi": 3.0},
            {"mode": m, "sigma_lo": 0.1, "sigma_hi": 1.5, "sigma_switch": 0.75, "sharpness": 2.0},
            {"mode": m, "sigma_lo": 0.2, "sigma_hi": 3.0, "delta": 0.25},
            {"mode": m, "sigma_lo": 0.3, "sigma_hi": 1.1, **non_dyadic},
        ],
        "bounded_log_sigmoid": [
            {"mode": m},
            {"mode": m, "sigma_lo": 0.1, "sigma_hi": 1.5, "sigma_switch": 0.75, "sharpness": 2.0},
            {"mode": m, "sigma_lo": 0.1, "sigma_hi": 3.0, "sigma_switch": 0.75, "sharpness": 4.0},
            {"mode": m, "sigma_lo": 0.2, "sigma_hi": 1.5, "sigma_switch": 0.75, "sharpness": 2.0},
            {"mode": m, "sigma_lo": 0.5, "sigma_hi": 2.0, "sigma_switch": 1.0, "sharpness": 1.0},
            {"mode": m, "sigma_lo": 0.1, "sigma_hi": 3.0, **non_dyadic},
        ],
    }
    cases["tanh_sigma"] = cases["linear_sigma"]
    return cases[mode]


# Keys that older v1 checkpoints lack; stored configs keep their original form.
# no_delta / no_bounds run for every mode (signature pops must tolerate
# missing keys); no_gate only for "none" and no_objective only for DSM,
# the only settings under which those stored forms exist.
LEGACY_FORMS = {
    "no_delta": (("fourier", "gate", "delta"),),
    "no_bounds": (("fourier", "gate", "sigma_lo"), ("fourier", "gate", "sigma_hi"),
                  ("fourier", "gate", "delta")),
    "no_gate": (("fourier", "gate"),),
    "no_objective": (("loss", "objective"),),
}


def _legacy_forms(mode, objective):
    for form, paths in LEGACY_FORMS.items():
        if form == "no_gate" and mode != "none":
            continue
        if form == "no_objective" and objective != "dsm":
            continue
        yield form, paths


def _without(cfg, paths):
    out = copy.deepcopy(cfg)
    for path in paths:
        parent = out
        for key in path[:-1]:
            parent = parent.get(key, {}) if isinstance(parent, dict) else {}
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    return out


def _gate_grid(ctx, mode):
    os.chdir(ctx.root)
    validate_gate = symbol("validate_gate", *GATES)
    gate_suffix = symbol("gate_suffix", *GATES)
    load_config = symbol("load_config", *CONFIG)
    validate = symbol("validate", *CONFIG)
    experiment_name = symbol("experiment_name", *CONFIG)
    resume_signature = symbol("resume_signature", *CHECKPOINTS)
    smoke = load_config("configs/smoke.json", ["loss.type=fourier_gaussian"])
    cases = []
    for gate in _gate_cases(mode):
        case = {"input": "<absent>" if gate is None else gate}
        ok, value = _attempt(ctx, validate_gate, copy.deepcopy(gate))
        case["validate_gate"] = value
        if not ok:
            cases.append(case)
            continue
        case["gate_suffix"] = gate_suffix(copy.deepcopy(gate))
        for objective in ("dsm", "normalized_residual"):
            raw = copy.deepcopy(smoke)
            raw["loss"]["objective"] = objective
            if gate is None:
                del raw["fourier"]["gate"]
            else:
                raw["fourier"]["gate"] = copy.deepcopy(gate)
            ok, cfg = _attempt(ctx, validate, copy.deepcopy(raw))
            if not ok:
                case[objective] = cfg
                continue
            signature = resume_signature(cfg)
            name = experiment_name(cfg)
            legacy = {}
            for form, paths in _legacy_forms(mode, objective):
                old = _without(cfg, paths)
                if old == cfg:
                    continue
                ok_sig, old_sig = _attempt(ctx, resume_signature, copy.deepcopy(old))
                ok_name, old_name = _attempt(ctx, experiment_name, copy.deepcopy(old))
                ok_val, revalidated = _attempt(ctx, validate, copy.deepcopy(old))
                legacy[form] = {
                    "signature_equal": old_sig == signature if ok_sig else old_sig,
                    "name_equal": old_name == name if ok_name else old_name,
                    "revalidates_equal": revalidated == cfg if ok_val else revalidated,
                }
            case[objective] = {
                "experiment_name": name,
                "cfg_sha256": json_digest(cfg),
                "signature_sha256": json_digest(signature),
                "signature_gate": signature["fourier"].get("gate", "<absent>"),
                "legacy": legacy,
            }
        cases.append(case)
    return {"exact": {"cases": cases}}


def _register_gate_grid(mode):
    @probe("config", "gate_grid." + mode)
    def gate_grid(ctx):
        return _gate_grid(ctx, mode)

    return gate_grid


for _mode in GATE_MODES:
    _register_gate_grid(_mode)


@probe("config", "gate_errors")
def gate_errors(ctx):
    validate_gate = symbol("validate_gate", *GATES)
    gate_suffix = symbol("gate_suffix", *GATES)
    nan, inf = float("nan"), float("inf")
    cases = {
        "not_a_dict": "log_sigma",
        "list": ["log_sigma"],
        "unknown_key": {"mode": "log_sigma", "width": 1.0},
        "unknown_mode": {"mode": "sigmoid"},
        "mode_not_string": {"mode": 1},
        "nan_switch": {"mode": "log_sigma", "sigma_switch": nan},
        "inf_sharpness": {"mode": "log_sigma", "sharpness": inf},
        "string_value": {"mode": "constant", "value": "0.5"},
        "bool_value": {"mode": "constant", "value": True},
        "none_delta": {"mode": "spectral_cap", "delta": None},
        "zero_switch": {"mode": "log_sigma", "sigma_switch": 0},
        "negative_sharpness": {"mode": "tanh_sigma", "sharpness": -1.0},
        "value_above_one": {"mode": "constant", "value": 1.5},
        "value_below_zero": {"mode": "constant", "value": -0.1},
        "bounds_inverted": {"mode": "log_sigma_plateau", "sigma_lo": 1.0, "sigma_hi": 0.5},
        "bounds_equal": {"mode": "linear_log_sigma", "sigma_lo": 1.0, "sigma_hi": 1.0},
        "zero_lo": {"mode": "spectral_cap", "sigma_lo": 0.0},
        "zero_delta": {"mode": "spectral_cap", "delta": 0},
        "bounded_default_switch": {"mode": "bounded_log_sigmoid"},
        "bounded_switch_at_hi": {"mode": "bounded_log_sigmoid", "sigma_lo": 0.1,
                                 "sigma_hi": 1.5, "sigma_switch": 1.5},
        "bounded_sharpness_one": {"mode": "bounded_log_sigmoid", "sigma_lo": 0.1,
                                  "sigma_hi": 1.5, "sigma_switch": 0.75, "sharpness": 1.0},
        "invalid_params_on_disabled_gate": {"mode": "none", "value": 2.0},
    }
    out = {}
    for name, gate in cases.items():
        out[name] = {
            "validate_gate": _raises(ctx, validate_gate, copy.deepcopy(gate)),
            "gate_suffix": _raises(ctx, gate_suffix, copy.deepcopy(gate)),
        }
    return {"exact": out}


@probe("config", "mnist_remaining_arms")
def mnist_remaining_arms(ctx):
    """Run names and resume signatures of the 19 arms the live MNIST study resumes."""
    os.chdir(ctx.root)
    arms = symbol("ARMS", "scripts.run_mnist_remaining")
    run_config = symbol("run_config", "scripts.run_mnist_remaining")
    load_config = symbol("load_config", *CONFIG)
    experiment_name = symbol("experiment_name", *CONFIG)
    resume_signature = symbol("resume_signature", *CHECKPOINTS)
    base = load_config("configs/mnist.json")
    out = []
    for arm in arms:
        cfg = run_config(base, arm, 0, "saved/mnist_remaining_100k", 100000, "cuda")
        signature = resume_signature(cfg)
        out.append({
            "arm": arm.name,
            "experiment_name": experiment_name(cfg),
            "cfg_sha256": json_digest(cfg),
            "signature_sha256": json_digest(signature),
            "signature_gate": signature["fourier"].get("gate", "<absent>"),
            "signature_loss": signature["loss"],
        })
    return {"exact": {"arms": out}}


# ---------------------------------------------------------------- config errors


SMOKE_OVERRIDE_ERRORS = {
    "unknown_top_key": ["nope=1"],
    "unknown_nested_key": ["trainer.nope=1"],
    "key_below_scalar": ["seed.x=1"],
    "missing_equals": ["seed"],
    "float_for_int": ["seed=1.5"],
    "bool_for_int": ["seed=true"],
    "bare_string_for_int": ["trainer.iterations=abc"],
    "int_for_string": ["name=123"],
    "null_for_int": ["seed=null"],
    "scalar_for_object": ["trainer=5"],
    "non_finite_inf": ["process.sigma_max=Infinity"],
    "non_finite_nan": ["optimizer.args.lr=NaN"],
    "invalid_parameterization": ["parameterization=eps"],
    "invalid_loss_type": ["loss.type=epsilon"],
    "invalid_objective": ["loss.objective=score_matching"],
    "invalid_reduction": ["loss.reduction=sum"],
    "invalid_process": ["process.type=vp"],
    "invalid_dataset": ["data_loader.args.dataset=imagenet"],
    "gate_invalid_mode": ["fourier.gate.mode=sigmoid"],
    "gate_unknown_key": ['fourier.gate={"mode":"log_sigma","width":1}'],
    "gate_nonpositive_sharpness": ["fourier.gate.mode=log_sigma", "fourier.gate.sharpness=0"],
    "gate_non_finite": ["fourier.gate.mode=log_sigma", "fourier.gate.sigma_switch=NaN"],
    "gate_bounded_defaults": ["fourier.gate.mode=bounded_log_sigmoid"],
    "gate_requires_gaussian": ["parameterization=score", "fourier.gate.mode=log_sigma"],
    "gate_requires_ve": ["process.type=ddpm", "sampling.method=ddpm", "sampling.steps=16",
                         "fourier.gate.mode=log_sigma"],
    "normalized_residual_requires_gaussian": ["parameterization=diffusion",
                                              "loss.objective=normalized_residual"],
    "normalized_residual_requires_ve": ["process.type=ddpm", "sampling.method=ddpm",
                                        "sampling.steps=16", "loss.objective=normalized_residual"],
    "sampler_process_mismatch": ["process.type=ddpm"],
    "ve_rejects_ddpm_sampler": ["sampling.method=ddpm"],
    "ddpm_steps_mismatch": ["process.type=ddpm", "sampling.method=ddpm"],
    "clip_denoised_on_ve": ["sampling.clip_denoised=true"],
    "name_unknown_field": ["name=run_{model}"],
    "name_format_spec": ["name=run_{seed:03d}_{parameterization}"],
    "name_partial_template": ["name=run_s{seed}"],
    "name_not_slug": ["name=bad name"],
    "name_empty": ["name="],
    "invalid_device": ["device=tpu"],
    "schema_version": ["schema_version=2"],
    "negative_seed": ["seed=-1"],
    "nf_not_multiple_of_4": ["arch.args.nf=6"],
    "attn_resolutions": ["arch.args.attn_resolutions=[3]"],
    "fir_kernel": ["arch.args.fir_kernel=[0,0]"],
    "mnist_shape": ["data_loader.args.dataset=mnist"],
    "microbatch_too_large": ["trainer.microbatch_size=3"],
    "adam_betas": ["optimizer.args.betas=[0.9]"],
    "noise_schedule": ["process.sigma_min=3.0"],
    "trainer_interval": ["trainer.save_every=0"],
    "power_floor": ["fourier.power_floor=0"],
    "evaluation_bins": ["evaluation.frequency_bins=-1"],
    # One trigger per remaining raise in validate() (and per clause of a
    # multi-clause condition), so every verbatim message and every clause
    # is pinned; each entry was checked to reach its intended raise.
    "choice_console": ["trainer.console=rich"],
    "choice_arch_type": ["arch.type=DDPM"],
    "choice_data_loader_type": ["data_loader.type=FolderLoader"],
    "choice_optimizer_type": ["optimizer.type=SGD"],
    "choice_precision": ["backend.precision=bf16"],
    "choice_spectral_transform": ["backend.spectral_transform=dct"],
    "choice_resblock_type": ["arch.args.resblock_type=resnet"],
    "choice_embedding_type": ["arch.args.embedding_type=sinusoidal"],
    "choice_progressive": ["arch.args.progressive=input_skip"],
    "choice_progressive_input": ["arch.args.progressive_input=output_skip"],
    "choice_progressive_combine": ["arch.args.progressive_combine=add"],
    "choice_nonlinearity": ["arch.args.nonlinearity=gelu"],
    "negative_split_seed": ["data_loader.args.split_seed=-1"],
    "nf_zero": ["arch.args.nf=0"],
    "ch_mult_empty": ["arch.args.ch_mult=[]"],
    "ch_mult_zero": ["arch.args.ch_mult=[1,0]"],
    "ch_mult_float": ["arch.args.ch_mult=[1,2.0]"],
    "num_res_blocks_zero": ["arch.args.num_res_blocks=0"],
    "dropout_one": ["arch.args.dropout=1.0"],
    "dropout_negative": ["arch.args.dropout=-0.1"],
    "init_scale_negative": ["arch.args.init_scale=-1.0"],
    "fourier_scale_zero": ["arch.args.fourier_scale=0"],
    "fir_kernel_empty": ["arch.args.fir_kernel=[]"],
    "fir_kernel_negative": ["arch.args.fir_kernel=[1,-1,3]"],
    "image_size_too_small": ["data_loader.args.image_size=2"],
    "image_size_not_divisible": ["data_loader.args.image_size=9"],
    "channels_two": ["data_loader.args.channels=2"],
    "mnist_channels": ["data_loader.args.dataset=mnist", "data_loader.args.image_size=32",
                       "arch.args.attn_resolutions=[16]", "data_loader.args.channels=3"],
    "cifar10_channels": ["data_loader.args.dataset=cifar10"],
    "cifar10_size": ["data_loader.args.dataset=cifar10", "data_loader.args.channels=3"],
    "data_batch_size_zero": ["data_loader.args.batch_size=0"],
    "num_workers_negative": ["data_loader.args.num_workers=-1"],
    "validation_size_negative": ["data_loader.args.validation_size=-1"],
    "synthetic_size_small": ["data_loader.args.synthetic_size=3"],
    "crop_size_zero": ["data_loader.args.crop_size=0"],
    "crop_size_float": ["data_loader.args.crop_size=4.0"],
    "sigma_min_zero": ["process.sigma_min=0"],
    "num_scales_one": ["process.num_scales=1"],
    "t_min_zero": ["process.t_min=0"],
    "t_min_one": ["process.t_min=1"],
    "beta_start_zero": ["process.beta_start=0"],
    "beta_end_one": ["process.beta_end=1"],
    "betas_inverted": ["process.beta_start=0.03"],
    "sampling_steps_zero": ["sampling.steps=0"],
    "sampling_batch_size_zero": ["sampling.batch_size=0"],
    "sampling_num_samples_zero": ["sampling.num_samples=0"],
    "sampling_steps_one": ["sampling.steps=1"],
    "corrector_steps_negative": ["sampling.corrector_steps=-1"],
    "snr_zero": ["sampling.snr=0"],
    "trainer_iterations_zero": ["trainer.iterations=0"],
    "trainer_snapshot_every_zero": ["trainer.snapshot_every=0"],
    "trainer_log_every_zero": ["trainer.log_every=0"],
    "trainer_eval_every_zero": ["trainer.eval_every=0"],
    "progress_every_seconds_zero": ["trainer.progress_every_seconds=0"],
    "warmup_negative": ["trainer.warmup=-1"],
    "ema_decay_one": ["trainer.ema_decay=1.0"],
    "ema_decay_zero": ["trainer.ema_decay=0"],
    "grad_clip_zero": ["trainer.grad_clip=0"],
    "microbatch_zero": ["trainer.microbatch_size=0"],
    "adam_lr_zero": ["optimizer.args.lr=0"],
    "adam_eps_zero": ["optimizer.args.eps=0"],
    "adam_negative_weight_decay": ["optimizer.args.weight_decay=-1"],
    "adam_beta_one": ["optimizer.args.betas=[0.9,1.0]"],
    "stats_batch_size_zero": ["fourier.stats_batch_size=0"],
    "cpu_threads_zero": ["backend.cpu_threads=0"],
    "evaluation_batch_size_zero": ["evaluation.batch_size=0"],
    "evaluation_max_images_zero": ["evaluation.max_images=0"],
    "evaluation_noise_bins_zero": ["evaluation.noise_bins=0"],
    "name_one_template_field": ["name=run_{parameterization}"],
}

# The accepted side of the same bounds (closed ends, smallest legal values):
# pins that a clause is not tightened (``< 1`` -> ``<= 1``) by a rewrite.
SMOKE_OVERRIDE_ACCEPTED = {
    "nf_four": ["arch.args.nf=4"],
    "image_size_four": ["data_loader.args.image_size=4"],
    "channels_three": ["data_loader.args.channels=3"],
    "crop_size_one": ["data_loader.args.crop_size=1"],
    "validation_size_zero": ["data_loader.args.validation_size=0"],
    "synthetic_size_four": ["data_loader.args.synthetic_size=4"],
    "seeds_zero": ["seed=0", "data_loader.args.split_seed=0"],
    "dropout_below_one": ["arch.args.dropout=0.999"],
    "fir_kernel_with_zero": ["arch.args.fir_kernel=[0,1]"],
    "betas_equal": ["process.beta_start=0.02"],
    "num_scales_two": ["process.num_scales=2", "sampling.steps=2"],
    "sampling_steps_two": ["sampling.steps=2"],
    "sampling_ones": ["sampling.batch_size=1", "sampling.num_samples=1"],
    "trainer_ones": ["trainer.iterations=1", "trainer.snapshot_every=1", "trainer.log_every=1",
                     "trainer.eval_every=1"],
    "microbatch_equals_batch": ["trainer.microbatch_size=2"],
    "adam_zero_betas": ["optimizer.args.betas=[0,0]"],
    "fourier_backend_ones": ["fourier.stats_batch_size=1", "backend.cpu_threads=1"],
    "evaluation_ones": ["evaluation.batch_size=1", "evaluation.max_images=1",
                        "evaluation.noise_bins=1"],
    "device_cuda_index": ["device=cuda:1"],
}

SMOKE = "<ROOT>/configs/smoke.json"
FILE_ERRORS = {
    "extends_cycle": {"a.json": {"extends": "b.json"}, "b.json": {"extends": "a.json"}},
    "extends_self": {"a.json": {"extends": "a.json"}},
    "extends_not_string": {"a.json": {"extends": ["base.json"]}},
    "extends_missing_parent": {"a.json": {"extends": "missing.json"}},
    "not_an_object": {"a.json": [1, 2]},
    "invalid_json": {"a.json": "not json"},
    "unknown_key_in_file": {"a.json": {"extends": SMOKE, "bogus": 1}},
    "unknown_nested_key_in_file": {"a.json": {"extends": SMOKE, "trainer": {"bogus": 1}}},
    "scalar_for_object_in_file": {"a.json": {"extends": SMOKE, "trainer": 5}},
    "parameterization_disagrees": {
        "a.json": {"extends": SMOKE, "parameterization": "score", "loss": {"type": "diffusion"}}
    },
    "loss_not_object_with_parameterization": {"a.json": {"parameterization": "score", "loss": 5}},
    "invalid_parameterization_in_file": {"a.json": {"extends": SMOKE, "parameterization": "eps"}},
}


@probe("config", "image_config_errors")
def image_config_errors(ctx):
    os.chdir(ctx.root)
    load_config = symbol("load_config", *CONFIG)
    overrides = {
        name: _raises(ctx, load_config, "configs/smoke.json", entries)
        for name, entries in SMOKE_OVERRIDE_ERRORS.items()
    }
    accepted = {
        name: _raises(ctx, load_config, "configs/smoke.json", entries)
        for name, entries in SMOKE_OVERRIDE_ACCEPTED.items()
    }
    files = {}
    for name, spec in FILE_ERRORS.items():
        folder = _write_json(ctx.tmp / name, spec, ctx)
        files[name] = _raises(ctx, load_config, str(folder / "a.json"))
    return {"exact": {"overrides": overrides, "accepted": accepted, "files": files}}


LDM_OVERRIDE_ERRORS = {
    "unknown_key": ["nope=1"],
    "unknown_nested_key": ["training.nope=1"],
    "key_below_scalar": ["seed.x=1"],
    "missing_equals": ["seed"],
    "protocol": ["protocol=ema"],
    "parameterization": ["parameterization=score"],
    "schema_version": ["schema_version=fourier-ldm-v2"],
    "unknown_model": ["model=imagenet"],
    "bare_string_for_float": ["training.lr=abc"],
    "non_finite": ["training.lr=Infinity"],
    "bool_for_int": ["seed=true"],
    "int_for_bool": ["data.random_flip=1"],
    "preprocessing": ["data.preprocessing=bilinear"],
    "console": ["training.console=rich"],
    "precision": ["backend.precision=fp16"],
    "spectral_transform": ["backend.spectral_transform=dct"],
    "nonpositive_batch": ["training.batch_size=0"],
    "negative_seed": ["seed=-1"],
    "adam": ["training.lr=0"],
    "adam_betas": ["training.betas=[0.9]"],
    "ema_decay": ["training.ema_decay=1.0"],
    "grad_clip": ["training.grad_clip=-1"],
    "sampler": ["sampling.method=plms"],
    "eta": ["sampling.eta=2"],
    "empty_path": ["data.root="],
    "upstream_config_type": ["upstream_config=5"],
    "name_template": ["name={foo}"],
    "name_slash": ["name=a/b"],
    "name_dotdot": ["name=.."],
}
LDM_SPEC_ERRORS = {
    "ddim_steps_not_divisor": ["sampling.steps=333"],
    "ddim_steps_all": ["sampling.steps=1000"],
    "ddpm_steps": ["sampling.method=ddpm"],
}
LDM_FILE_ERRORS = {
    "unknown_model": {"a.json": {"model": "imagenet"}},
    "unknown_key_in_file": {"a.json": {"model": "ffhq", "bogus": 1}},
    "extends_cycle": {"a.json": {"extends": "b.json", "model": "ffhq"},
                      "b.json": {"extends": "a.json"}},
    "wrong_schema_in_file": {"a.json": {"model": "ffhq", "schema_version": 1}},
}


@probe("config", "ldm_config_errors")
def ldm_config_errors(ctx):
    os.chdir(ctx.root)
    load_config = symbol("load_config", *LDM_CONFIG)
    load_spec = symbol("load_spec", *LDM_CONFIG)
    base = "configs/ldm/ffhq.json"
    overrides = {
        name: _raises(ctx, load_config, base, entries)
        for name, entries in LDM_OVERRIDE_ERRORS.items()
    }
    spec = {
        name: _raises(ctx, lambda e: load_spec(load_config(base, e)), entries)
        for name, entries in LDM_SPEC_ERRORS.items()
    }
    files = {}
    for name, files_spec in LDM_FILE_ERRORS.items():
        folder = _write_json(ctx.tmp / name, files_spec, ctx)
        files[name] = _raises(ctx, load_config, str(folder / "a.json"))
    source = (ctx.root / "configs/ldm/upstream/ffhq.yaml").read_text(encoding="utf-8")

    def edited(name, old, new):
        assert source.count(old) == 1, (name, old)
        path = ctx.tmp / f"{name}.yaml"
        path.write_text(source.replace(old, new, 1), encoding="utf-8")
        return path

    # Text edits (not a YAML round trip) so the probe does not depend on the
    # installed PyYAML's dump format.  One case per load_spec() raise.
    top, unet, first = "    linear_start: 0.0015\n", "        num_head_channels: 32\n", \
        "        n_embed: 8192\n"
    yaml_cases = {
        "not_latent_diffusion": ("ldm.models.diffusion.ddpm.LatentDiffusion",
                                 "ldm.models.diffusion.ddpm.DDPM"),
        "unsupported_denoiser": ("ldm.modules.diffusionmodules.openaimodel.UNetModel",
                                 "ldm.modules.diffusionmodules.openaimodel.EncoderUNetModel"),
        "conditional_stage": ("cond_stage_config: __is_unconditional__",
                              "cond_stage_config: __is_first_stage__"),
        "unsupported_first_stage": ("ldm.models.autoencoder.VQModelInterface",
                                    "ldm.models.autoencoder.IdentityFirstStage"),
        "x0_parameterization": (top, top + "    parameterization: x0\n"),
        "learned_logvar": (top, top + "    learn_logvar: true\n"),
        "cosine_schedule": (top, top + "    beta_schedule: cosine\n"),
        "remapped_codebook": (first, first + "        remap: data/index_remap.npy\n"),
        "native_loss": (top, top + "    loss_type: huber\n"),
        "spatial_transformer": (unet, unet + "        use_spatial_transformer: true\n"),
        "class_conditional_unet": (unet, unet + "        num_classes: 10\n"),
        "lr_schedule": (top, top + "    scheduler_config:\n"
                        "      target: ldm.lr_scheduler.LambdaWarmUpCosineScheduler\n"
                        "      params: {}\n"),
    }
    upstream = {}
    for name, (old, new) in yaml_cases.items():
        path = edited(name, old, new)
        upstream[name] = _raises(
            ctx, lambda p: load_spec(load_config(base, [f"upstream_config={p}"])), path
        )
    # The accepted side: explicit defaults, the l1 loss, the supported
    # scheduler, and a non-native upstream loss under the l2 protocol.
    accepted_cases = {
        "explicit_defaults": (top, top + "    parameterization: eps\n    learn_logvar: false\n"
                              "    beta_schedule: linear\n", []),
        "l1_loss": (top, top + "    loss_type: l1\n", []),
        "huber_under_l2_protocol": (top, top + "    loss_type: huber\n", ["protocol=l2"]),
        "lambda_linear_scheduler": (top, top + "    scheduler_config:\n"
                                    "      target: ldm.lr_scheduler.LambdaLinearScheduler\n"
                                    "      params: {warm_up_steps: [10], cycle_lengths: [100],"
                                    " f_start: [0.5], f_max: [1.0], f_min: [1.0]}\n", []),
    }
    upstream_accepted = {}
    for name, (old, new, extra) in accepted_cases.items():
        path = edited("accepted_" + name, old, new)
        ok, value = _attempt(
            ctx, lambda p: load_spec(load_config(base, [f"upstream_config={p}", *extra])), path
        )
        upstream_accepted[name] = {
            "loss_type": value["loss_type"],
            "upstream_loss": value["upstream_loss"],
            "scheduler": value["scheduler"],
            "custom_upstream_config": value["custom_upstream_config"],
        } if ok else value
    return {"exact": {"overrides": overrides, "spec": spec, "files": files, "upstream": upstream,
                      "upstream_accepted": upstream_accepted}}


# ---------------------------------------------------------------- overrides


OVERRIDE_SCENARIOS = {
    "json_scalars": ["seed=7", "process.sigma_min=0.02", "data_loader.args.random_flip=true",
                     "trainer.microbatch_size=1", "trainer.microbatch_size=null"],
    "int_for_float": ["process.sigma_max=3", "optimizer.args.lr=1", "sampling.snr=1"],
    "float_spelling": ["process.sigma_max=3.0", "optimizer.args.lr=2e-4", "process.t_min=1E-5"],
    "whitespace_json": ["seed= 5", "sampling.snr=0.2 "],
    "bare_strings": ["device=cpu", "trainer.save_dir=saved/a=b", "fourier.cache_dir=data/stats x"],
    "quoted_string": ['trainer.save_dir="saved/q"', 'device="cpu"'],
    "template_name": ["name=run_{parameterization}_s{seed}", "seed=3"],
    "literal_name": ["name=my-run_1"],
    "alias": ["parameterization=scalar_gaussian"],
    "alias_after_type": ["loss.type=score", "parameterization=fourier_gaussian"],
    "type_after_alias": ["parameterization=score", "loss.type=diffusion"],
    "objective": ["loss.objective=normalized_residual"],
    "nested_object_partial": ['fourier.gate={"mode":"log_sigma","sigma_switch":1.5}'],
    "nested_leaf": ["fourier.gate.mode=spectral_cap", "fourier.gate.sigma_lo=1",
                    "fourier.gate.sigma_hi=2.0", "fourier.gate.sigma_switch=1.5"],
    "list_values": ["optimizer.args.betas=[0.5,0.9]", "arch.args.fir_kernel=[1,2,1]"],
    "last_wins": ["seed=1", "seed=2", "device=cpu", "device=auto"],
    "download_flag": ["data_loader.args.download=true"],
    "ddpm_switch": ["process.type=ddpm", "sampling.method=ddpm", "sampling.steps=16",
                    "loss.type=diffusion", "sampling.clip_denoised=true"],
}


@probe("config", "override_semantics")
def override_semantics(ctx):
    os.chdir(ctx.root)
    load_config = symbol("load_config", *CONFIG)
    experiment_name = symbol("experiment_name", *CONFIG)
    resume_signature = symbol("resume_signature", *CHECKPOINTS)
    base = load_config("configs/smoke.json")
    out = {}
    for name, entries in OVERRIDE_SCENARIOS.items():
        cfg = load_config("configs/smoke.json", entries)
        keys = []
        for entry in entries:
            key = entry.split("=", 1)[0]
            key = "loss.type" if key == "parameterization" else key
            if key not in keys:
                keys.append(key)
        flat_base, flat_cfg = _flat(base), _flat(cfg)
        changed = sorted(
            k for k in set(flat_base) | set(flat_cfg)
            if flat_base.get(k, "<absent>") != flat_cfg.get(k, "<absent>")
        )
        out[name] = {
            "cfg_sha256": json_digest(cfg),
            "experiment_name": experiment_name(cfg),
            "signature_sha256": json_digest(resume_signature(cfg)),
            "values": {key: _typed(_get(cfg, key)) for key in keys},
            "changed_keys": changed,
        }
    return {"exact": {"base_sha256": json_digest(base), "scenarios": out}}


@probe("config", "legacy_v1_forms")
def legacy_v1_forms(ctx):
    """Stored v1 configs predate optional keys; validate() must fill them."""
    os.chdir(ctx.root)
    load_config = symbol("load_config", *CONFIG)
    validate = symbol("validate", *CONFIG)
    experiment_name = symbol("experiment_name", *CONFIG)
    resume_signature = symbol("resume_signature", *CHECKPOINTS)
    cfg = load_config("configs/smoke.json")
    removals = {
        "no_objective": [("loss", "objective")],
        "no_gate": [("fourier", "gate")],
        "no_console": [("trainer", "console"), ("trainer", "progress_every_seconds")],
        "no_frequency_bins": [("evaluation", "frequency_bins")],
        "no_crop_size": [("data_loader", "args", "crop_size")],
        "pre_gate_v1": [("loss", "objective"), ("fourier", "gate"), ("trainer", "console"),
                        ("trainer", "progress_every_seconds"),
                        ("evaluation", "frequency_bins")],
        "no_iterations": [("trainer", "iterations")],
        "no_arch_nf": [("arch", "args", "nf")],
    }
    signature = resume_signature(cfg)
    out = {}
    for name, paths in removals.items():
        old = _without(cfg, paths)
        ok_sig, old_sig = _attempt(ctx, resume_signature, copy.deepcopy(old))
        ok_val, revalidated = _attempt(ctx, validate, copy.deepcopy(old))
        ok_name, name_old = _attempt(ctx, experiment_name, copy.deepcopy(old))
        out[name] = {
            "signature_equal": old_sig == signature if ok_sig else old_sig,
            "revalidates_equal": revalidated == cfg if ok_val else revalidated,
            "revalidated_sha256": json_digest(revalidated) if ok_val else None,
            "name": name_old,
        }
    return {"exact": {"signature_sha256": json_digest(signature), "forms": out}}


# ---------------------------------------------------------------- CLI dry runs


def _run_cli(ctx, script, args):
    """Run an entrypoint from the tree under test; parse stdout JSON."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, str(ctx.root / script), *args],
        cwd=ctx.root, env=env, capture_output=True, text=True, timeout=600,
    )
    out = {"returncode": proc.returncode}
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out["stdout"] = _scrub(proc.stdout, ctx)
        else:
            out["stdout_json"] = _scrub(parsed, ctx)
            out["stdout_is_indent2_json"] = proc.stdout in (
                json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
                json.dumps(parsed, indent=2) + "\n",
            )
    else:
        lines = [line for line in proc.stderr.splitlines() if line.strip()]
        out["stderr_last_line"] = _scrub(lines[-1] if lines else "", ctx)
    return out


def _flat(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else k))
        return out
    return {prefix: obj}


@probe("config", "cli_dry_run")
def cli_dry_run(ctx):
    smoke_args = ["-c", "configs/smoke.json"]
    tmp = ctx.tmp
    jobs = {
        "train_smoke": ("train.py", [*smoke_args, "--dry-run"]),
        "train_plan_command": ("train.py", [*smoke_args, "--parameterization", "score", "--download",
                                            "--dry-run"]),
        "train_flags_override_set": ("train.py", [
            *smoke_args, "--set", "device=mps", "--set", "loss.type=diffusion",
            "--device", "cpu", "--parameterization", "score", "--set", "name=x", "--dry-run",
        ]),
        "train_default_config": ("train.py", ["--dry-run"]),
        "train_config_and_resume": ("train.py", [*smoke_args, "-r", "missing.pt", "--dry-run"]),
        "ldm_train_dry_run": ("ldm.py", ["train", "-c", "configs/ldm/lsun_churches_l2.json",
                                         "--parameterization", "fourier_gaussian", "--dry-run"]),
        "ldm_compare_dry_run": ("ldm.py", [
            "compare", "-c", "configs/ldm/lsun_churches_l2.json", "--seeds", "7", "--dry-run",
            "--set", f"cache.dir={tmp}/cache", "--set", f"training.save_dir={tmp}/runs",
        ]),
        "ldm_config_and_resume": ("ldm.py", ["train", "-c", "configs/ldm/ffhq.json",
                                             "-r", "missing.pt", "--dry-run"]),
    }
    names = sorted(jobs)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(zip(names, pool.map(lambda n: _run_cli(ctx, *jobs[n]), names)))
    # The full smoke config is image_config.configs.smoke; record the rest as
    # digests plus the leaves that differ from the plain smoke dry run.
    os.chdir(ctx.root)
    load_config = symbol("load_config", *CONFIG)
    in_process = {
        "train_smoke": load_config("configs/smoke.json"),
        "train_default_config": load_config("config.json"),
    }
    # A failed run keeps its returncode/stderr line so the diff says why.
    reference = _flat(results["train_smoke"].get("stdout_json", {}))
    for name in ("train_smoke", "train_flags_override_set", "train_default_config"):
        result = results[name]
        if "stdout_json" not in result:
            continue
        parsed = result.pop("stdout_json")
        flat = _flat(parsed)
        result["stdout_sha256"] = json_digest(parsed)
        if name in in_process:
            result["equals_in_process_load_config"] = parsed == in_process[name]
        if name != "train_smoke":
            result["differs_from_smoke"] = {
                k: flat.get(k, "<absent>") for k in sorted(set(flat) | set(reference))
                if flat.get(k, "<absent>") != reference.get(k, "<absent>")
            }
    ldm = results["ldm_train_dry_run"].get("stdout_json", {})
    if "spec" in ldm:
        ldm["spec"] = {"sha256": json_digest(ldm["spec"]), "keys": sorted(ldm["spec"])}
    results["ldm_compare_dry_run"]["no_side_effects"] = not any(
        (tmp / d).exists() for d in ("cache", "runs")
    )
    return {"exact": results}
