"""Fetch original Score-SDE PyTorch references into pretrained/score_sde/."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_ldm import file_info, sha256_argument
from scripts.download_ldm_data import acquire

REVISION = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
REPOSITORY = "https://github.com/yang-song/score_sde_pytorch"
RAW = f"https://raw.githubusercontent.com/yang-song/score_sde_pytorch/{REVISION}"
FORMAT = "score-sde-download-v1"
# File IDs were resolved from the folders linked by the pinned official README.
MODELS = {
    "cifar10_ncsnpp_continuous": {
        "folder_id": "1b0gy_LLgO_DaQBgoWXwlVnL_rcAUgREh",
        "file_id": "1JInV8bPGy18QiIzZcS1iECGHCuXL6_Nz",
        "filename": "checkpoint_24.pth",
        "defaults": "default_cifar10_configs.py",
        "description": "CIFAR-10 32, continuous VE, NCSN++ (4 residual blocks)",
    },
    "cifar10_ncsnpp_deep_continuous": {
        "folder_id": "11s6A_xM7qiztdj8AHQWqaIAUSC3I7uX2",
        "file_id": "1yS8QZb_6tCeZkY7DK4_RI-Crc6LQLILN",
        "filename": "checkpoint_12.pth",
        "defaults": "default_cifar10_configs.py",
        "description": "CIFAR-10 32, continuous VE, NCSN++ deep (8 residual blocks)",
    },
    "ffhq_256_ncsnpp_continuous": {
        "folder_id": "1KG72ZKUCUa8dDcA03hOf1BsnK8kBcdPD",
        "file_id": "1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy",
        "filename": "checkpoint_48.pth",
        "defaults": "default_lsun_configs.py",
        "description": "FFHQ 256, continuous VE, NCSN++",
    },
}


def download_plan(model, output_dir="pretrained/score_sde"):
    item = MODELS[model]
    paths = (f"configs/ve/{model}.py", "configs/" + item["defaults"], "LICENSE")
    return {
        "model": model,
        "description": item["description"],
        "destination": str(Path(output_dir) / model),
        "checkpoint_name": item["filename"],
        "checkpoint_url": "https://drive.google.com/uc?id=" + item["file_id"],
        "official_folder": "https://drive.google.com/drive/folders/" + item["folder_id"],
        "upstream_revision": REVISION,
        "model_zoo": f"{REPOSITORY}/blob/{REVISION}/README.md#pretrained-checkpoints",
        "config_sources": {"upstream/" + path: f"{RAW}/{path}" for path in paths},
    }


def verify_bundle(destination, plan, expected_sha256=None):
    destination = Path(destination)
    try:
        record = json.loads((destination / "download.json").read_text())
        if record["format"] != FORMAT or record["plan"] != plan:
            raise ValueError("Model/source manifest mismatch")
        names = {plan["checkpoint_name"], *plan["config_sources"]}
        if set(record["files"]) != names:
            raise ValueError("Incomplete file manifest")
        for name in names:
            if file_info(destination / name) != record["files"][name]:
                raise ValueError(f"Downloaded file changed: {name}")
        if expected_sha256 and record["files"][plan["checkpoint_name"]]["sha256"] != expected_sha256:
            raise ValueError("Checkpoint SHA-256 does not match --sha256")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot reuse Score-SDE bundle {destination}: {error}") from error
    return record


def download_model(model, output_dir="pretrained/score_sde", expected_sha256=None, timeout=30):
    plan = download_plan(model, output_dir)
    destination = Path(plan["destination"])
    marker = destination / "download.json"
    if marker.exists():
        verify_bundle(destination, plan, expected_sha256)
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for relative, url in plan["config_sources"].items():
        path = acquire(url, destination / relative, timeout)
        files[relative] = file_info(path)
    print(f"Downloading {model}/{plan['checkpoint_name']} …", file=sys.stderr, flush=True)
    path = acquire(plan["checkpoint_url"], destination / plan["checkpoint_name"], timeout, sha256=expected_sha256)
    files[plan["checkpoint_name"]] = file_info(path)
    record = {
        "format": FORMAT,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "files": files,
        "expected_checkpoint_sha256": expected_sha256,
    }
    partial = destination / "download.json.part"
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.rename(marker)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", choices=MODELS)
    selection.add_argument("--all", action="store_true", help="Download the three supported references")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained/score_sde"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sha256", type=sha256_argument, help="Trusted checkpoint digest, for a single --model")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or (args.sha256 and not args.model):
        parser.error("--timeout must be positive; --sha256 requires one --model")
    if args.list:
        for name, item in MODELS.items():
            print(f"{name}: {item['description']}")
        return
    selected = list(MODELS) if args.all else [args.model]
    if args.dry_run:
        print(json.dumps([download_plan(name, args.output_dir) for name in selected], indent=2))
        return
    for name in selected:
        destination = download_model(name, args.output_dir, args.sha256, args.timeout)
        print(f"Ready: {destination / MODELS[name]['filename']}", flush=True)


if __name__ == "__main__":
    main()
