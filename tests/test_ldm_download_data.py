"""Small real HTTP/ZIP/LMDB workflows; no public dataset downloads in pytest."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts import download_ldm_data as download


@contextmanager
def serve(files):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-")[6:-1])
            calls.append((self.path, start))
            data = files[self.path]
            self.send_response(206 if start else 200)
            if start:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}"
                )
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def splits(monkeypatch, data_dir, model, train, validation):
    spec = dict(download.DATASETS[model])
    entries = []
    for original, names in zip(spec["splits"], (train, validation)):
        relative = original[0]
        payload = ("\n".join(names) + "\n").encode()
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append((relative, len(names), hashlib.sha256(payload).hexdigest()))
    spec["splits"] = entries
    monkeypatch.setitem(download.DATASETS, model, spec)
    return spec


def test_http_resume_checksum_and_offline_reuse(tmp_path):
    payload = b"dataset bytes" * 100
    target = tmp_path / "archive.zip"
    with serve({"/archive.zip": payload}) as (base, calls):
        url = base + "/archive.zip"
        partial = target.with_name(
            target.name + "." + hashlib.sha256(url.encode()).hexdigest()[:12] + ".part"
        )
        partial.write_bytes(payload[:37])
        download.acquire(url, target, sha256=hashlib.sha256(payload).hexdigest())
        assert target.read_bytes() == payload and calls == [("/archive.zip", 37)]
    assert download.acquire(url, target) == target
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        download.acquire(url, target)


def test_lsun_zip_preserves_original_keys_and_compvis_split(tmp_path, monkeypatch):
    lmdb = pytest.importorskip("lmdb")
    data_dir = tmp_path / "data"
    spec = splits(
        monkeypatch, data_dir, "lsun_churches", ["aa.webp", "bb.webp"], ["cc.webp"]
    )
    database = tmp_path / "train_lmdb"
    env = lmdb.open(str(database), map_size=1024 * 1024)
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), (80, 20, 0)).save(payload, format="WEBP", lossless=True)
    with env.begin(write=True) as txn:
        for key in (b"aa", b"bb", b"cc"):
            txn.put(key, payload.getvalue())
    env.close()
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(database / "data.mdb", "church_outdoor_train_lmdb/data.mdb")
    with serve({"/images.zip": bundle.getvalue()}) as (base, _):
        record = download.prepare_dataset(
            "lsun_churches", data_dir, url=base + "/images.zip"
        )
    assert record["image_count"] == 3
    root = data_dir / spec["root"]
    assert all(
        (root / name).read_bytes() == payload.getvalue()
        for name in ("aa.webp", "bb.webp", "cc.webp")
    )
    # Reuse with the HTTP server stopped; no LSUN publisher validation set was used.
    assert (
        download.prepare_dataset("lsun_churches", data_dir, url=base + "/images.zip")
        == record
    )
    (data_dir / spec["splits"][0][0]).write_text("custom.webp\n")
    with pytest.raises(ValueError, match="split checksum mismatch"):
        download.prepare_dataset("lsun_churches", data_dir, url=base + "/images.zip")


def test_ffhq_download_flattens_names_and_checks_publisher_md5(tmp_path, monkeypatch):
    pytest.importorskip("gdown")
    data_dir = tmp_path / "data"
    splits(monkeypatch, data_dir, "ffhq", ["00000.png", "00002.png"], ["00001.png"])
    payload = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 50, 80)).save(payload, format="PNG")
    files = {"/image.png": payload.getvalue(), "/LICENSE.txt": b"fixture license"}
    with serve(files) as (base, _):
        entries = {
            str(i): {
                "image": {
                    "file_url": base + "/image.png",
                    "file_path": f"images1024x1024/00000/{i:05}.png",
                    "file_size": len(payload.getvalue()),
                    "file_md5": hashlib.md5(payload.getvalue()).hexdigest(),
                }
            }
            for i in range(3)
        }
        files["/metadata.json"] = json.dumps(entries).encode()
        monkeypatch.setattr(download, "FFHQ_METADATA_URL", base + "/metadata.json")
        monkeypatch.setattr(
            download,
            "FFHQ_METADATA_MD5",
            hashlib.md5(files["/metadata.json"]).hexdigest(),
        )
        monkeypatch.setattr(download, "FFHQ_LICENSE_URL", base + "/LICENSE.txt")
        result = download.prepare_dataset("ffhq", data_dir, workers=2)
    assert result["image_count"] == 3
    assert (data_dir / "ffhq/00002.png").read_bytes() == payload.getvalue()
    assert (data_dir / "ffhqtrain.txt").read_text() == "00000.png\n00002.png\n"


def test_celebahq_archive_import_and_missing_file_never_marks_ready(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "data"
    splits(monkeypatch, data_dir, "celebahq", ["imgHQ00002.npy"], ["imgHQ00000.npy"])
    payload = io.BytesIO()
    np.save(payload, np.zeros((1, 3, 8, 8), dtype=np.uint8))
    source = tmp_path / "celebahq.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("original/imgHQ00002.npy", payload.getvalue())
    with pytest.raises(ValueError, match="Missing 1"):
        download.prepare_dataset("celebahq", data_dir, source=source)
    assert not (data_dir / "celebahq/download.json").exists()
    with zipfile.ZipFile(source, "a") as archive:
        archive.writestr("original/imgHQ00000.npy", payload.getvalue())
    result = download.prepare_dataset("celebahq", data_dir, source=source)
    assert result["image_count"] == 2
    np.testing.assert_array_equal(
        np.load(data_dir / "celebahq/imgHQ00002.npy"),
        np.zeros((1, 3, 8, 8), dtype=np.uint8),
    )
