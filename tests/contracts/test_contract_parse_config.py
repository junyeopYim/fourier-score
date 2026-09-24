"""Contract: ``fourier_score.parse_config`` carries both config dialects unchanged.

The pixel and LDM loaders share one reader, override parser, type walk and
``CustomArgs`` flag expansion. Their epoch-0 modules (``git show`` at tag
``pre-template-refactor``; skipped without git or the tag) are the oracle for
seeded random overrides, stored-config forms and inheritance chains: resolved
configs (key order included), exception types and messages must all match.

Deliberate differences, pinned below instead: a malformed LDM file (not an
object, non-string ``extends``) now gets the pixel reader's ValueError where
epoch 0 crashed or ignored a falsy ``extends``; an empty ``extends`` fails as in
the pixel reader (it names the file's own folder) where epoch 0 ignored it;
``ldm.py --device ""`` is applied like any given flag (epoch 0 dropped empty
values); and ``ldm.py`` refuses ``-c ""`` with ``-r`` like ``train.py`` does
(epoch 0 ignored an empty ``-c`` there).
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

import pytest

import golden_probes as gp
from fourier_score.parse_config import (
    CustomArgs, add_options, check_types, cli_overrides, from_args
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = ("fourier_score.config",)
LDM_CONFIG = ("fourier_score.ldm.config",)
SMOKE = str(REPO_ROOT / "configs/smoke.json")
BASE = str(REPO_ROOT / "configs/base.json")
FFHQ = str(REPO_ROOT / "configs/ldm/ffhq.json")
CHURCHES = str(REPO_ROOT / "configs/ldm/lsun_churches.json")
BASE_CONFIG = {"image": SMOKE, "ldm": FFHQ}
RAW = (
    "0", "1", "-1", "2", "3", "4", "1.5", "0.5", "-0.1", "1e-5", "1e400", "-1e400", "NaN",
    "Infinity", "-Infinity", "true", "false", "null", '"x"', "abc", "", " 7 ", "[]", "[1]",
    "[1,2.0]", "[0.9,0.999]", "{}", '{"mode":"log_sigma"}', '{"x":1}', "1" + "0" * 400,
    "cpu", "cuda:0", "auto", "ddpm", "ddim", "heun", "pc", "ve", "score", "diffusion",
    "epsilon", "fourier_gaussian", "scalar_gaussian", "mean", "half_sum", "normalized_residual",
    "log_sigma", "none", "synthetic", "mnist", "l2", "upstream", "ffhq", "celebahq", "quiet",
    "run_{parameterization}_s{seed}", "{model}_{seed}", "a/b",
)
UNKNOWN = (
    "nope", "trainer.nope", "training.nope", "seed.x", "fourier.gate.width", "parameterization"
)


def tag_source(path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    result = subprocess.run(
        ["git", "show", f"{gp.EPOCH0_TAG}:{path}"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode:
        pytest.skip(f"{gp.EPOCH0_TAG}:{path} unavailable: {result.stderr.strip()}")
    return result.stdout


def exec_source(path, edits, **namespace):
    source = tag_source(path)
    for old, new in edits.items():
        assert source.count(old) == 1, (path, old)
        source = source.replace(old, new)
    module = types.ModuleType(path)
    module.__dict__.update(namespace)
    exec(compile(source, f"<{gp.EPOCH0_TAG}:{path}>", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def epoch0():
    # gates and provenance are proven identical to their epoch-0 sources.
    config = exec_source("fourier_score/config.py", {
        "from fourier_score.method import": "from fourier_score.gates import",
        "ROOT = Path(__file__).resolve().parents[1]": "from fourier_score.provenance import ROOT",
    })
    ldm = exec_source("fourier_score/ldm/config.py", {
        "from fourier_score.config import deep_merge\n": "",
        "from fourier_score.method import": "from fourier_score.gates import",
        "from fourier_score.utils import ROOT": "from fourier_score.provenance import ROOT",
        "from .dataset_sources import": "from fourier_score.ldm.dataset_sources import",
    }, deep_merge=config.deep_merge)
    return types.SimpleNamespace(image=config, ldm=ldm)


@pytest.fixture(scope="module")
def current():
    image = types.SimpleNamespace(**{
        name: gp.symbol(name, *CONFIG) for name in ("load_config", "validate", "apply_overrides")
    })
    ldm = types.SimpleNamespace(**{
        name: gp.symbol(name, *LDM_CONFIG) for name in ("load_config", "validate", "override")
    })
    return types.SimpleNamespace(image=image, ldm=ldm)


def outcome(fn, *args):
    try:
        return "ok", json.dumps(fn(*args))
    except Exception as error:  # noqa: BLE001 - the exception is the contract
        return type(error).__name__, str(error)


def tree(cfg, prefix=""):
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        yield path, value
        if isinstance(value, dict):
            yield from tree(value, path)


def candidates(value):
    out = list(RAW)
    if type(value) is bool:
        out.append(json.dumps(not value))
    elif type(value) is int:
        out += [str(value + 1), str(value - 1), str(2 * value), str(-value)]
    elif type(value) is float:
        out += [repr(value * f) for f in (0.5, 2.0, -1.0, 10.0)]
    elif isinstance(value, (str, list)):
        out += [value if isinstance(value, str) else json.dumps(value), json.dumps(value[:-1])]
    return out


def random_entries(rng, keyed):
    entries = []
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.03:
            entries.append(rng.choice(("seed", "", "trainer.iterations")))
        else:
            key, value = rng.choice(keyed)
            entries.append(f"{key}={rng.choice(candidates(value))}")
    return entries


def without(cfg, rng):
    out = json.loads(json.dumps(cfg))
    for _ in range(rng.randint(1, 3)):
        parent = out
        while True:
            key = rng.choice(sorted(parent))
            if isinstance(parent[key], dict) and parent[key] and rng.random() < 0.7:
                parent = parent[key]
            else:
                del parent[key]
                break
    return out


def assert_diverse(results):
    ok = sum(kind == "ok" for kind, _ in results)
    messages = {text for kind, text in results if kind != "ok"}
    assert ok >= 50 and len(messages) >= 30, (ok, len(messages))


@pytest.mark.parametrize("pipeline", sorted(BASE_CONFIG))
def test_overrides_match_epoch0(epoch0, current, pipeline):
    old, new, path = getattr(epoch0, pipeline), getattr(current, pipeline), BASE_CONFIG[pipeline]
    keyed = list(tree(old.load_config(path))) + [(key, None) for key in UNKNOWN]
    rng = random.Random(1303)
    results = []
    for _ in range(3000):
        entries = random_entries(rng, keyed)
        expected = outcome(old.load_config, path, entries)
        assert outcome(new.load_config, path, entries) == expected, entries
        results.append(expected)
    assert_diverse(results)


@pytest.mark.parametrize("pipeline", sorted(BASE_CONFIG))
def test_stored_forms_match_epoch0(epoch0, current, pipeline):
    old, new, path = getattr(epoch0, pipeline), getattr(current, pipeline), BASE_CONFIG[pipeline]
    overrides = old.apply_overrides if pipeline == "image" else old.override
    cfg = old.load_config(path)
    keyed = list(tree(cfg))
    rng = random.Random(1304)
    results = []
    for _ in range(1500):
        stored = without(cfg, rng) if rng.random() < 0.5 else cfg
        try:
            stored = overrides(stored, random_entries(rng, keyed)[:1])
        except ValueError:
            pass
        expected = outcome(old.validate, json.loads(json.dumps(stored)))
        assert outcome(new.validate, json.loads(json.dumps(stored))) == expected, stored
        results.append(expected)
    assert_diverse(results)
    new_overrides = new.apply_overrides if pipeline == "image" else new.override
    for entries in (["seed"], ["nope=1"], ["seed.x=1"], ["parameterization=score"], ["seed=2"]):
        assert outcome(new_overrides, cfg, entries) == outcome(overrides, cfg, entries)


def write_chain(folder, files):
    for name, body in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(body if isinstance(body, str) else json.dumps(body))
    return str(folder / "a.json")


# Parents are spelled relative to the declaring file; sub/b.json sits one level down.
PARENTS = {
    "a.json": ("sub/b.json", "c.json", "a.json", "missing.json"),
    "sub/b.json": ("../c.json", "../a.json", "b.json", "../sub/b.json"),
    "c.json": ("sub/b.json", "a.json", "c.json"),
}


@pytest.mark.parametrize("pipeline", ["image", "ldm"])
def test_inheritance_chains_match_epoch0(epoch0, current, pipeline, tmp_path):
    old, new = getattr(epoch0, pipeline), getattr(current, pipeline)
    if pipeline == "image":
        roots = (SMOKE, BASE, None)
        bodies = ({}, {"seed": 3}, {"parameterization": "score"}, {"loss": {"type": "diffusion"}},
                  {"parameterization": "score", "loss": {"type": "diffusion"}},
                  {"parameterization": "score", "loss": 5}, {"bogus": 1}, {"trainer": 5},
                  {"trainer": {"iterations": 7}}, {"name": "x"})
        broken = ([1, 2], '"text"', "not json", {"extends": 5}, {"extends": ["c.json"]},
                  {"extends": False}, {"extends": ""})
    else:
        roots = (FFHQ, CHURCHES, None)
        bodies = ({}, {"model": "celebahq"}, {"model": "imagenet"}, {"protocol": "l2"},
                  {"seed": 3}, {"bogus": 1}, {"training": 5}, {"training": {"lr": 1}},
                  {"schema_version": 1}, {"parameterization": "fourier_gaussian"})
        broken = ("not json",)
    rng = random.Random(1305)
    results = []
    for case in range(600):
        files = {}
        for name, parents in PARENTS.items():
            if rng.random() < 0.05:
                files[name] = rng.choice(broken)
                continue
            body = dict(rng.choice(bodies))
            parent = rng.choice(parents + roots)
            if parent is not None:
                body["extends"] = parent
            elif rng.random() < 0.2:
                body["extends"] = None
            files[name] = body
        path = write_chain(tmp_path / str(case), files)
        expected = outcome(old.load_config, path)
        assert outcome(new.load_config, path) == expected, files
        results.append(expected)
    ok = sum(kind == "ok" for kind, _ in results)
    assert ok >= 50 and len({kind for kind, _ in results}) >= 3, results


@pytest.mark.parametrize("body, message", [
    ([1, 2], "Configuration must be a JSON object"),
    ({"model": "ffhq", "extends": 5}, "extends must be one JSON filename"),
    ({"model": "ffhq", "extends": False}, "extends must be one JSON filename"),
])
def test_ldm_reader_shares_the_pixel_file_checks(tmp_path, body, message):
    load_config = gp.symbol("load_config", *LDM_CONFIG)
    path = write_chain(tmp_path, {"a.json": body})
    with pytest.raises(ValueError) as error:
        load_config(path)
    assert str(error.value) == message


def test_empty_ldm_extends_fails_like_the_pixel_reader(tmp_path):
    ldm = gp.symbol("load_config", *LDM_CONFIG)
    image = gp.symbol("load_config", *CONFIG)
    ldm_path = write_chain(tmp_path / "ldm", {"a.json": {"model": "ffhq", "extends": ""}})
    image_path = write_chain(tmp_path / "image", {"a.json": {"seed": 1, "extends": ""}})
    assert outcome(ldm, ldm_path)[0] == outcome(image, image_path)[0] == "IsADirectoryError"


def test_extends_resolves_relative_to_the_declaring_file(tmp_path):
    load_config = gp.symbol("load_config", *CONFIG)
    write_chain(tmp_path, {
        "a.json": {"extends": "one/b.json", "seed": 5},
        "one/b.json": {"extends": "../two/c.json", "trainer": {"iterations": 9}},
        "two/c.json": {"extends": SMOKE, "parameterization": "score"},
    })
    cfg = load_config(str(tmp_path / "a.json"))
    assert (cfg["seed"], cfg["trainer"]["iterations"], cfg["loss"]["type"]) == (5, 9, "score")
    assert cfg["sampling"] == load_config(SMOKE)["sampling"]


def test_check_types_keeps_each_finite_policy():
    reference = {"a": 1.0, "b": None, "c": {"d": 2}}

    def pixel(cfg):
        return outcome(check_types, reference, cfg)

    def ldm(cfg):
        invalid = "Invalid type/value: .{path}"
        return outcome(lambda c: check_types(reference, c, invalid, nonfinite=None), cfg)

    good = {"a": 1, "b": "anything", "c": {"d": 3}}
    assert pixel(good) == ldm(good) == ("ok", "null")
    nan_float = {**good, "a": float("nan")}
    assert pixel(nan_float) == ("ValueError", "a must be finite")
    assert ldm(nan_float) == ("ValueError", "Invalid type/value: .a")
    nan_free = {**good, "b": float("inf")}
    assert pixel(nan_free) == ("ValueError", "b must be finite")
    assert ldm(nan_free) == ("ok", "null")
    nested = {**good, "c": {"d": True}}
    assert pixel(nested) == ("ValueError", "c.d: invalid type bool")
    assert ldm(nested) == ("ValueError", "Invalid type/value: .c.d")


# -------------------------------------------------------------------- CLI flags


def epoch0_train_changes(args):
    """train.py at the tag, verbatim."""
    changes = list(args.set)
    if args.device is not None:
        changes.append("device=" + args.device)
    if args.parameterization is not None:
        changes.append("parameterization=" + args.parameterization)
    if args.download:
        changes.append("data_loader.args.download=true")
    return changes


def epoch0_ldm_changes(args):
    """fourier_score/ldm/cli.py main() at the tag, verbatim."""
    changes = list(args.set)
    if args.device:
        changes.append("device=" + args.device)
    if getattr(args, "parameterization", None):
        changes.append("parameterization=" + args.parameterization)
    if args.command == "sample":
        for key in ("num_samples", "batch_size", "steps"):
            if getattr(args, key) is not None:
                changes.append(f"sampling.{key}={getattr(args, key)}")
    return changes


@pytest.mark.parametrize("fn, path", [
    (epoch0_train_changes, "train.py"),
    (epoch0_ldm_changes, "fourier_score/ldm/cli.py"),
])
def test_epoch0_flag_logic_is_the_tag_source(fn, path):
    body = inspect.getsource(fn).splitlines()[2:-1]  # without def, docstring, return
    assert "\n".join(body) + "\n" in tag_source(path)


def flag_grid(groups, *required):
    """Shuffled argv: one alternative from about half of the groups, plus ``required``."""
    rng = random.Random(1306)
    for _ in range(200):
        units = [rng.choice(group) for group in groups if rng.random() < 0.5]
        units += required
        rng.shuffle(units)
        yield [token for unit in units for token in unit]


def test_train_flags_expand_like_epoch0():
    add_config_args = gp.symbol("add_config_args", *CONFIG)
    options = (*gp.symbol("CLI_OPTIONS", *CONFIG), gp.symbol("DOWNLOAD", *CONFIG))
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("-r", "--resume")
    parser.add_argument("--dry-run", action="store_true")
    add_options(parser, options[-1:])
    groups = [
        [["--set", "seed=1"], ["--set", "device=mps"], ["--set", "data_loader.args.download=false"]],
        [["--set", "loss.type=diffusion"], ["--set", "parameterization=score"]],
        [["--device", "cpu"], ["--device", ""], ["--device", "cuda:1"]],
        [["--parameterization", "score"], ["--parameterization", "fourier_gaussian"]],
        [["--download"]], [["--dry-run"]], [["-r", "x.pt"]], [["-c", "configs/smoke.json"]],
    ]
    lengths = set()
    for argv in flag_grid(groups):
        args = parser.parse_args(argv)
        assert cli_overrides(args, options) == epoch0_train_changes(args), argv
        lengths.add(len(epoch0_train_changes(args)))
    assert lengths == {0, 1, 2, 3, 4, 5}


def ldm_entries(monkeypatch, argv):
    """(parsed args, --set entries) of ``ldm.py *argv``, stopping right after parsing."""
    from fourier_score.ldm import cli

    class Parsed(Exception):
        pass

    def spy(args, options):
        raise Parsed(args, cli_overrides(args, options))

    monkeypatch.setattr("fourier_score.parse_config.cli_overrides", spy)
    with pytest.raises(Parsed) as parsed:
        cli.main(argv)
    return parsed.value.args


LDM_FLAGS = {
    "train": ([[["--parameterization", "fourier_gaussian"], ["--parameterization", "epsilon"]],
               [["--dry-run"]], [["-r", "x.pt"]]], ()),
    "sample": ([[["--num-samples", "0"], ["--num-samples", "7"]], [["--batch-size", "2"]],
                [["--steps", "10"]], [["--weights", "raw"]]], (["-o", "out"], ["--pretrained"])),
    "evaluate": ([], (["-r", "x.pt"], ["-o", "out"])),
    "prepare": ([[["--dry-run"]]], ()),
    "compare": ([[["--seeds", "7"]], [["--dry-run"]]], ()),
    "reconstruct": ([[["--split", "train"]]], (["-o", "out"],)),
}


@pytest.mark.parametrize("command", sorted(LDM_FLAGS))
def test_ldm_flags_expand_like_epoch0(monkeypatch, command):
    groups, required = LDM_FLAGS[command]
    common = [[["--set", "seed=1"], ["--set", "sampling.steps=100"]], [["--set", "training.lr=1"]],
              [["--device", "cpu"], ["--device", "cuda:1"]], [["-c", "configs/ldm/ffhq.json"]]]
    for argv in flag_grid(common + groups, *required):
        args, entries = ldm_entries(monkeypatch, [command, *argv])
        assert entries == epoch0_ldm_changes(args), argv


def test_empty_ldm_device_is_applied_like_any_flag(monkeypatch):
    _, entries = ldm_entries(monkeypatch, ["train", "--device", "", "--set", "seed=1"])
    assert entries == ["seed=1", "device="]


def test_from_args_loads_the_config_or_the_resume_checkpoint(capsys):
    options = [CustomArgs(["--seed"], "seed", type=int)]
    parser = add_options(argparse.ArgumentParser(prog="entry"), options)
    parser.add_argument("-c", "--config")
    parser.add_argument("-r", "--resume")
    parser.add_argument("--set", action="append", default=[])

    def run(*argv):
        return from_args(
            parser, parser.parse_args(argv), options,
            lambda path, changes: ("load", path, changes), "default.json",
            lambda path, changes: (("resume", path, changes), "checkpoint"), "no -c with -r",
        )

    assert run("--set", "a=1", "--seed", "2") == (("load", "default.json", ["a=1", "seed=2"]), None)
    assert run("-c", "x.json") == (("load", "x.json", []), None)
    assert run("-c", "") == (("load", "default.json", []), None)
    assert run("-r", "x.pt", "--seed", "3") == (("resume", "x.pt", ["seed=3"]), "checkpoint")
    for argv in (["-c", "x.json", "-r", "x.pt"], ["-c", "", "-r", "x.pt"]):
        with pytest.raises(SystemExit) as error:
            run(*argv)
        assert error.value.code == 2
        assert capsys.readouterr().err.splitlines()[-1] == "entry: error: no -c with -r"


def test_ldm_compare_forwards_set_then_device():
    from fourier_score.ldm.cli import comparison_commands

    load_config = gp.symbol("load_config", *LDM_CONFIG)
    args = types.SimpleNamespace(
        set=["training.iterations=5", "device=cpu"], device="cuda", config=None,
        seeds=[7], parameterizations=["epsilon"],
    )
    (job,) = comparison_commands(args, load_config(FFHQ, args.set))
    sets = job["argv"][job["argv"].index("--set"):][1::2]
    assert sets[:4] == ["training.iterations=5", "device=cpu", "device=cuda", "seed=7"]


def test_custom_args_become_argparse_options():
    options = [
        CustomArgs(["-n", "--num-samples"], "sampling.num_samples", type=int),
        CustomArgs(["--flag"], "a.flag", action="store_true", help="h"),
        CustomArgs(["--kind"], "a.kind", choices=("x", "y")),
    ]
    parser = add_options(argparse.ArgumentParser(), options)
    parser.set_defaults(set=["a=1"])
    args = parser.parse_args(["-n", "0", "--flag", "--kind", "y", "--num-samples", "3"])
    assert cli_overrides(args, options) == [
        "a=1", "sampling.num_samples=3", "a.flag=true", "a.kind=y"
    ]
    assert cli_overrides(parser.parse_args(["-n", "0"]), options) == [
        "a=1", "sampling.num_samples=0"
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["--kind", "z"])


def test_parse_config_imports_without_torch():
    code = "import fourier_score.parse_config, sys; assert 'torch' not in sys.modules"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
