"""Download an official unconditional CompVis LDM checkpoint and matching config.

Uses only Python's standard library. Does not load checkpoints or install LDM.
The official ZIPs contain model.ckpt; their configs live in the upstream repo.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


UPSTREAM_REVISION = "a506df5756472e2ebaf9078affdde2c4f1502cd4"
UPSTREAM_REPOSITORY = "https://github.com/CompVis/latent-diffusion"
MODEL_ZOO = f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_REVISION}/README.md#pretrained-ldms"
MODELS = {
    "ffhq": ("ffhq", "ffhq256", "FFHQ 256, LDM-VQ-4, latent 3x64x64"),
    "celebahq": ("celeba", "celeba256", "CelebA-HQ 256, LDM-VQ-4, latent 3x64x64"),
    "lsun_churches": (
        "lsun_churches",
        "lsun_churches256",
        "LSUN-Churches 256, LDM-KL-8, latent 4x32x32",
    ),
    "lsun_bedrooms": (
        "lsun_bedrooms",
        "lsun_beds256",
        "LSUN-Bedrooms 256, LDM-VQ-4, latent 3x64x64",
    ),
}
FORMAT = "compvis-ldm-download-v1"
CHUNK_BYTES = 8 * 1024 * 1024


def download_plan(model, output_dir):
    archive, upstream_directory, description = MODELS[model]
    return {
        "model": model,
        "description": description,
        "destination": str(Path(output_dir) / model),
        "archive_url": f"https://ommer-lab.com/files/latent-diffusion/{archive}.zip",
        "config_url": f"https://raw.githubusercontent.com/CompVis/latent-diffusion/{UPSTREAM_REVISION}/models/ldm/{upstream_directory}/config.yaml",
        "upstream_revision": UPSTREAM_REVISION,
    }


def file_info(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def fetch(url, destination, timeout, max_bytes=None):
    """Stream into a temporary workspace; incomplete downloads are never published."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "fourier-score-ldm-download"}
    )
    digest = hashlib.sha256()
    count = 0
    last_report = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        expected = int(length) if length is not None else None
        if max_bytes is not None and expected is not None and expected > max_bytes:
            raise ValueError(f"Response too large: {url}")
        with destination.open("wb") as output:
            while chunk := response.read(CHUNK_BYTES):
                count += len(chunk)
                if max_bytes is not None and count > max_bytes:
                    raise ValueError(f"Response too large: {url}")
                output.write(chunk)
                digest.update(chunk)
                if time.monotonic() - last_report >= 5:
                    total = f" / {expected / 2**20:.0f}" if expected is not None else ""
                    print(
                        f"Downloading: {count / 2**20:.0f}{total} MiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_report = time.monotonic()
        if expected is not None and count != expected:
            raise ValueError(
                f"Incomplete download: received {count} of {expected} bytes"
            )
        if count == 0:
            raise ValueError(f"Empty download: {url}")
        return {
            "url": url,
            "resolved_url": response.url,
            "bytes": count,
            "sha256": digest.hexdigest(),
            "etag": response.headers.get("ETag"),
        }


def extract_checkpoint(archive, destination):
    """Extract only one regular model.ckpt into a fixed local filename."""
    with zipfile.ZipFile(archive) as source:
        candidates = [
            item
            for item in source.infolist()
            if PurePosixPath(item.filename).name == "model.ckpt" and not item.is_dir()
        ]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one model.ckpt in the official archive")
        item = candidates[0]
        path = PurePosixPath(item.filename)
        kind = stat.S_IFMT(item.external_attr >> 16)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item.filename
            or kind not in (0, stat.S_IFREG)
        ):
            raise ValueError("Checkpoint ZIP member must be a regular relative file")
        if item.file_size == 0:
            raise ValueError("Empty checkpoint in archive")
        # ZipFile checks CRC while streaming. Never unpickle or execute the file.
        with source.open(item) as reader, destination.open("wb") as output:
            shutil.copyfileobj(reader, output, length=CHUNK_BYTES)
    return file_info(destination)


def verify_existing(destination, plan, expected_sha256):
    """Re-use a completed download only if both local files still match its manifest."""
    try:
        manifest = json.loads((destination / "download.json").read_text())
        if manifest["format"] != FORMAT or manifest["model"] != plan["model"]:
            raise ValueError("Download manifest does not match the selected model")
        if manifest["upstream_revision"] != UPSTREAM_REVISION:
            raise ValueError("Download uses a different upstream config revision")
        if (
            manifest["archive"]["url"] != plan["archive_url"]
            or manifest["config"]["url"] != plan["config_url"]
        ):
            raise ValueError("Download source does not match the selected model")
        if (
            expected_sha256 is not None
            and manifest["archive"]["sha256"] != expected_sha256
        ):
            raise ValueError("Archive SHA-256 does not match --sha256")
        for name in ("model.ckpt", "config.yaml"):
            if file_info(destination / name) != manifest["files"][name]:
                raise ValueError(f"Downloaded file changed: {name}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot reuse {destination}: {error}. Use a new --output-dir or repair that directory."
        ) from error


def download_model(
    model, output_dir="pretrained/ldm", expected_sha256=None, timeout=30
):
    plan = download_plan(model, output_dir)
    destination = Path(plan["destination"])
    if destination.exists():
        verify_existing(destination, plan, expected_sha256)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{model}-", dir=destination.parent
    ) as temporary:
        work = Path(temporary)
        bundle = work / "bundle"
        bundle.mkdir()
        # Fetch the small, commit-pinned config before spending bandwidth on weights.
        config = fetch(
            plan["config_url"], bundle / "config.yaml", timeout, max_bytes=1024 * 1024
        )
        if (
            "ldm.models.diffusion.ddpm.LatentDiffusion"
            not in (bundle / "config.yaml").read_text()
        ):
            raise ValueError(
                "Downloaded config is not a CompVis LatentDiffusion config"
            )
        archive_path = work / "model.zip"
        archive = fetch(plan["archive_url"], archive_path, timeout)
        if expected_sha256 is not None and archive["sha256"] != expected_sha256:
            raise ValueError("Archive SHA-256 does not match --sha256")
        checkpoint = extract_checkpoint(archive_path, bundle / "model.ckpt")
        manifest = {
            "format": FORMAT,
            "model": model,
            "description": plan["description"],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "model_zoo": MODEL_ZOO,
            "upstream_revision": UPSTREAM_REVISION,
            "archive": archive,
            "config": config,
            "expected_archive_sha256": expected_sha256,
            "files": {
                "model.ckpt": checkpoint,
                "config.yaml": file_info(bundle / "config.yaml"),
            },
        }
        (bundle / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Publish only after download, extraction and checks all succeed.
        if destination.exists():
            raise FileExistsError(
                f"Destination appeared during download: {destination}"
            )
        bundle.rename(destination)
    return destination


def sha256_argument(value):
    value = value.lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("SHA-256 must be 64 hexadecimal characters")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--list", action="store_true", help="List models without network access"
    )
    selection.add_argument("--model", choices=MODELS)
    parser.add_argument("--output-dir", type=Path, default=Path("pretrained/ldm"))
    parser.add_argument("--with-data", action="store_true", help="Also prepare the dataset and official splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--data-source", type=Path, help="Existing original image folder/ZIP or LSUN LMDB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show sources and destination without downloading",
    )
    parser.add_argument(
        "--sha256",
        type=sha256_argument,
        help="Optional trusted SHA-256 of the upstream ZIP",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Network timeout in seconds"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.data_source and not args.with_data:
        parser.error("--data-source requires --with-data")
    if args.list:
        for name, (_, _, description) in MODELS.items():
            print(f"{name:16} {description}")
        return
    if args.dry_run:
        plan = download_plan(args.model, args.output_dir)
        if args.with_data:
            from scripts.download_ldm_data import dataset_plan

            plan["dataset"] = dataset_plan(args.model, args.data_dir, args.data_source)
        print(json.dumps(plan, indent=2))
        return
    if args.with_data and args.model == "celebahq" and args.data_source is None:
        parser.error("CelebA-HQ needs --data-source with the original imgHQXXXXX.npy folder/ZIP")
    if args.data_source and not args.data_source.exists():
        parser.error(f"Dataset source does not exist: {args.data_source}")
    destination = download_model(args.model, args.output_dir, args.sha256, args.timeout)
    print(f"Ready: {destination / 'model.ckpt'}")
    print(f"Config: {destination / 'config.yaml'}")
    print(f"Download record: {destination / 'download.json'}")
    if args.with_data:
        from scripts.download_ldm_data import prepare_dataset

        result = prepare_dataset(args.model, args.data_dir, source=args.data_source, timeout=args.timeout)
        print(f"Dataset ready: {result['plan']['destination']}")


if __name__ == "__main__":
    main()
