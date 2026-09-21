"""Download/import original LDM datasets and install the exact CompVis splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import urllib.request
from urllib.parse import urlparse
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fourier_score.ldm.dataset_sources import (
    DATASETS,
    FACE_LIST_URL,
    FFHQ_LICENSE_URL,
    FFHQ_METADATA_MD5,
    FFHQ_METADATA_URL,
    LSUN_LIST_SHA256,
    LSUN_LIST_URL,
)
from scripts.download_ldm import CHUNK_BYTES, file_info, sha256_argument

FORMAT = "compvis-ldm-dataset-v1"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            result.update(chunk)
    return result.hexdigest()


def acquire(url, path, timeout=30, sha256=None, md5=None):
    """Resume partial HTTP/Drive downloads; only publish verified complete files."""
    path = Path(path)
    receipt = path.with_name(path.name + ".download.json")
    if path.exists():
        if not receipt.exists() or json.loads(receipt.read_text())["url"] != url:
            raise ValueError(f"Unmanaged download or changed URL: {path}")
        expected = json.loads(receipt.read_text())["file"]
        if file_info(path) != expected:
            raise ValueError(f"Corrupt download: {path}")
        if sha256 and expected["sha256"] != sha256:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if md5 and digest(path, "md5") != md5:
            raise ValueError(f"MD5 mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # URL-specific temporary names prevent resuming bytes from a different source.
    partial = path.with_name(
        path.name + "." + hashlib.sha256(url.encode()).hexdigest()[:12] + ".part"
    )
    if urlparse(url).hostname in ("drive.google.com", "drive.usercontent.google.com"):
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError(
                "Google Drive downloads need: uv sync --locked --extra datasets"
            ) from error
        if not gdown.download(url, str(partial), quiet=True, resume=True, fuzzy=True):
            raise RuntimeError(
                f"Google Drive download failed: {url}. Retry after any publisher quota clears."
            )
    else:
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "fourier-score-ldm-data"}
        if start:
            headers["Range"] = f"bytes={start}-"
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as response:
            append = response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(
                f"bytes {start}-"
            ):
                raise ValueError("Server returned an unexpected download range")
            received = 0
            last_report = time.monotonic()
            with partial.open("ab" if append else "wb") as stream:
                while chunk := response.read(CHUNK_BYTES):
                    stream.write(chunk)
                    received += len(chunk)
                    if time.monotonic() - last_report >= 5:
                        print(
                            f"Downloading {path.name}: {(received + (start if append else 0)) / 2**20:.0f} MiB",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_report = time.monotonic()
            length = response.headers.get("Content-Length")
            if length is not None and received != int(length):
                raise ValueError(f"Incomplete download: {path}; retry to resume")
    if not partial.stat().st_size:
        raise ValueError(f"Empty download: {url}")
    info = file_info(partial)
    if (sha256 and info["sha256"] != sha256) or (md5 and digest(partial, "md5") != md5):
        partial.unlink()
        raise ValueError(f"Download checksum mismatch: {url}")
    partial.rename(path)
    receipt.write_text(json.dumps({"url": url, "file": info}, indent=2) + "\n")
    return path


def install_bytes(path, payload):
    path = Path(path)
    if not payload:
        raise ValueError(f"Empty source file: {path}")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing file differs; refusing to overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.rename(path)


def install_splits(model, data_dir, timeout=30):
    spec = DATASETS[model]
    data_dir = Path(data_dir)
    cache = data_dir / ".downloads/ldm/splits"
    names = []
    for relative, count, expected in spec["splits"]:
        path = data_dir / relative
        if path.exists():
            payload = path.read_bytes()
        elif model.startswith("lsun_"):
            archive = acquire(
                LSUN_LIST_URL, cache / "lsun.zip", timeout, sha256=LSUN_LIST_SHA256
            )
            with zipfile.ZipFile(archive) as source:
                payload = source.read(relative)
        else:
            source = acquire(
                FACE_LIST_URL + relative, cache / relative, timeout, sha256=expected
            )
            payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(
                f"Official split checksum mismatch: {path}; keep custom splits at another path"
            )
        entries = payload.decode().splitlines()
        if len(entries) != count or any(
            not n or PurePosixPath(n).name != n or "\\" in n for n in entries
        ):
            raise ValueError(f"Invalid official split: {path}")
        names.extend(entries)
        install_bytes(path, payload)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate or overlapping dataset splits")
    return names


def dataset_plan(model, data_dir="data", source=None, url=None):
    spec = DATASETS[model]
    return {
        "model": model,
        "destination": str(Path(data_dir) / spec["root"]),
        "train_list": str(Path(data_dir) / spec["splits"][0][0]),
        "validation_list": str(Path(data_dir) / spec["splits"][1][0]),
        "counts": {"train": spec["splits"][0][1], "validation": spec["splits"][1][1]},
        "source": str(Path(source).resolve()) if source else (url or spec["url"]),
        "reference": spec["reference"],
        "requires_source": model == "celebahq" and not (source or url),
    }


def import_images(source, destination, names):
    """Flatten original names without resizing, renumbering or changing splits."""
    source = Path(source)
    if source.is_dir():
        wanted = set(names)
        indexed = {}
        for path in source.rglob("*"):
            if path.is_file() and path.name in wanted:
                if path.name in indexed:
                    raise ValueError(f"Ambiguous image basename: {path.name}")
                indexed[path.name] = path
        missing = wanted - indexed.keys()
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required images, e.g. {min(missing)}"
            )
        for i, name in enumerate(names):
            origin, target = indexed[name], destination / name
            if not origin.stat().st_size:
                raise ValueError(f"Empty source image: {origin}")
            if target.exists():
                if not os.path.samefile(origin, target) and file_info(
                    origin
                ) != file_info(target):
                    raise ValueError(f"Existing image differs: {target}")
            else:
                try:
                    os.link(origin, target)
                except OSError:
                    partial = target.with_name(target.name + ".part")
                    shutil.copyfile(origin, partial)
                    partial.rename(target)
            progress(i, len(names))
        return
    with zipfile.ZipFile(source) as archive:
        wanted = set(names)
        indexed = {}
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.name not in wanted or item.is_dir():
                continue
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Dataset ZIP contains a non-regular image path")
            if path.name in indexed:
                raise ValueError(f"Ambiguous image basename: {path.name}")
            indexed[path.name] = item
        missing = wanted - indexed.keys()
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required images, e.g. {min(missing)}"
            )
        for i, name in enumerate(names):
            install_bytes(destination / name, archive.read(indexed[name]))
            progress(i, len(names))


def export_lsun(source, destination, names):
    try:
        import lmdb
    except ImportError as error:
        raise RuntimeError(
            "LSUN LMDB export needs: uv sync --locked --extra datasets"
        ) from error
    source = Path(source)
    if source.is_file():
        database = source.with_name(source.name + ".lmdb")
        database.mkdir(exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            members = [
                item
                for item in archive.infolist()
                if PurePosixPath(item.filename).name == "data.mdb" and not item.is_dir()
            ]
            if len(members) != 1:
                raise ValueError("Expected one data.mdb in the LSUN archive")
            target = database / "data.mdb"
            if not target.exists():
                partial = database / "data.mdb.part"
                with archive.open(members[0]) as reader, partial.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, CHUNK_BYTES)
                partial.rename(target)
            if target.stat().st_size != members[0].file_size:
                raise ValueError(f"Incomplete LSUN database: {target}")
    else:
        database = source
    env = lmdb.open(
        str(database), readonly=True, lock=False, readahead=False, max_readers=1
    )
    try:
        with env.begin() as transaction:
            for i, name in enumerate(names):
                payload = transaction.get(Path(name).stem.encode("ascii"))
                if payload is None:
                    raise ValueError(
                        f"Official split key missing from LSUN training LMDB: {name}"
                    )
                install_bytes(destination / name, payload)
                progress(i, len(names))
    finally:
        env.close()


def download_ffhq(destination, cache, names, workers, timeout):
    import gdown  # fail before downloading metadata if the optional extra is missing

    del gdown
    metadata = acquire(
        FFHQ_METADATA_URL,
        cache / "ffhq-dataset-v2.json",
        timeout,
        md5=FFHQ_METADATA_MD5,
    )
    acquire(FFHQ_LICENSE_URL, cache / "LICENSE.txt", timeout)
    entries = json.loads(metadata.read_text())

    def download(name):
        item = entries[str(int(Path(name).stem))]["image"]
        if PurePosixPath(item["file_path"]).name != name:
            raise ValueError(f"FFHQ metadata filename mismatch: {name}")
        target = destination / name
        if target.exists():
            if (
                target.stat().st_size != item["file_size"]
                or digest(target, "md5") != item["file_md5"]
            ):
                raise ValueError(f"FFHQ image checksum mismatch: {target}")
            return
        staged = acquire(
            item["file_url"], cache / "images" / name, timeout, md5=item["file_md5"]
        )
        if staged.stat().st_size != item["file_size"]:
            raise ValueError(f"FFHQ image size mismatch: {name}")
        staged.rename(target)
        staged.with_name(staged.name + ".download.json").unlink()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(download, names)):
            progress(i, len(names))


def progress(index, total):
    if (index + 1) % 1000 == 0 or index + 1 == total:
        print(f"Images: {index + 1}/{total}", file=sys.stderr, flush=True)


def prepare_dataset(
    model,
    data_dir="data",
    *,
    source=None,
    url=None,
    sha256=None,
    splits_only=False,
    workers=8,
    timeout=30,
):
    if source and url:
        raise ValueError("Choose --source or --url, not both")
    if workers < 1 or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    if sha256 and (
        (source and Path(source).is_dir())
        or (not source and not url and model == "ffhq")
    ):
        raise ValueError(
            "--sha256 applies to a ZIP source, not an image directory or FFHQ metadata"
        )
    plan = dataset_plan(model, data_dir, source, url)
    if plan["requires_source"] and not splits_only:
        raise ValueError(
            "CelebA-HQ requires the original imgHQXXXXX.npy files: use --source DIR/ZIP or --url ZIP_URL. See "
            + plan["reference"]
        )
    names = install_splits(model, data_dir, timeout)
    if splits_only:
        return {**plan, "splits_ready": True, "images_ready": False}
    destination = Path(plan["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "download.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("format") != FORMAT or record.get("plan") != plan:
            raise ValueError(f"Dataset source/layout changed: {marker}")
        if sha256 and record.get("archive_sha256") != sha256:
            raise ValueError("Archive SHA-256 does not match --sha256")
        if all(
            (destination / name).is_file() and (destination / name).stat().st_size
            for name in names
        ):
            return record
    cache = Path(data_dir) / ".downloads/ldm" / model
    archive_hash = None
    if source:
        origin = Path(source).resolve()
        if not origin.exists():
            raise FileNotFoundError(origin)
        if origin.is_file():
            archive_hash = digest(origin)
            if sha256 and archive_hash != sha256:
                raise ValueError("Archive SHA-256 mismatch")
    elif url or model.startswith("lsun_"):
        origin = acquire(plan["source"], cache / "images.zip", timeout, sha256=sha256)
        receipt = origin.with_name(origin.name + ".download.json")
        archive_hash = json.loads(receipt.read_text())["file"]["sha256"]
    else:
        origin = None
    if origin is None:
        download_ffhq(destination, cache, names, workers, timeout)
    else:
        lmdb_source = origin.is_dir() and (origin / "data.mdb").is_file()
        if model.startswith("lsun_") and origin.is_file():
            with zipfile.ZipFile(origin) as archive:
                lmdb_source = any(
                    PurePosixPath(p).name == "data.mdb" for p in archive.namelist()
                )
        if lmdb_source:
            export_lsun(origin, destination, names)
        else:
            import_images(origin, destination, names)
    record = {
        "format": FORMAT,
        "plan": plan,
        "archive_sha256": archive_hash,
        "images_ready": True,
        "splits_ready": True,
        "image_count": len(names),
        "reuse_check": "Official split checksums and required nonempty image files",
    }
    install_bytes(marker, (json.dumps(record, indent=2) + "\n").encode())
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source", type=Path, help="Existing image folder, ZIP, or LSUN training LMDB"
    )
    source.add_argument(
        "--url", help="Download a ZIP containing the original image files or LSUN LMDB"
    )
    parser.add_argument(
        "--sha256", type=sha256_argument, help="Trusted SHA-256 of the image/LMDB ZIP"
    )
    parser.add_argument("--splits-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent FFHQ image downloads"
    )
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.dry_run:
        result = dataset_plan(args.model, args.data_dir, args.source, args.url)
    else:
        result = prepare_dataset(
            args.model,
            args.data_dir,
            source=args.source,
            url=args.url,
            sha256=args.sha256,
            splits_only=args.splits_only,
            workers=args.workers,
            timeout=args.timeout,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
