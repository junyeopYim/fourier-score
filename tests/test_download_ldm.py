"""Exercise the real streaming downloader against a tiny local HTTP server."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import pytest

from scripts import download_ldm


CONFIG = b"model:\n  target: ldm.models.diffusion.ddpm.LatentDiffusion\n"
CHECKPOINT = b"test checkpoint bytes; downloading must not try to unpickle this"


def archive_bytes(name="model.ckpt", payload=CHECKPOINT):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


@contextmanager
def serve(archive, config=CONFIG, extra_length=0):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            data = config if self.path == "/config.yaml" else archive
            self.send_response(200)
            self.send_header(
                "Content-Length",
                str(len(data) + (extra_length if self.path != "/config.yaml" else 0)),
            )
            self.end_headers()
            self.wfile.write(data)

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


def local_sources(monkeypatch, base):
    original = download_ldm.download_plan

    def plan(model, output_dir):
        result = original(model, output_dir)
        result.update(
            archive_url=f"{base}/weights.zip", config_url=f"{base}/config.yaml"
        )
        return result

    monkeypatch.setattr(download_ldm, "download_plan", plan)


def test_download_bundle_and_offline_verified_reuse(tmp_path, monkeypatch):
    payload = archive_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with serve(payload) as (base, calls):
        local_sources(monkeypatch, base)
        destination = download_ldm.download_model("ffhq", tmp_path, digest)
        assert (destination / "model.ckpt").read_bytes() == CHECKPOINT
        assert (destination / "config.yaml").read_bytes() == CONFIG
        manifest = json.loads((destination / "download.json").read_text())
        assert manifest["archive"]["sha256"] == digest
        assert manifest["expected_archive_sha256"] == digest
        assert (
            manifest["files"]["model.ckpt"]["sha256"]
            == hashlib.sha256(CHECKPOINT).hexdigest()
        )
        assert calls == ["/config.yaml", "/weights.zip"]
    # Server is stopped: completed downloads must be usable without the network.
    assert download_ldm.download_model("ffhq", tmp_path, digest) == destination
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ffhq"]
    (destination / "model.ckpt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Downloaded file changed"):
        download_ldm.download_model("ffhq", tmp_path)
    assert (destination / "model.ckpt").read_bytes() == b"corrupted"


@pytest.mark.parametrize(
    "failure",
    [
        "checksum",
        "truncated",
        "not_zip",
        "missing",
        "duplicate",
        "traversal",
        "empty",
        "bad_config",
        "crc",
    ],
)
def test_failed_download_never_publishes_a_partial_bundle(
    tmp_path, monkeypatch, failure
):
    payload = archive_bytes()
    config = CONFIG
    expected = None
    extra_length = 0
    if failure == "checksum":
        expected = "0" * 64
    elif failure == "truncated":
        extra_length = 100
    elif failure == "not_zip":
        payload = b"<html>not a model archive</html>"
    elif failure == "missing":
        payload = archive_bytes("readme.txt")
    elif failure == "duplicate":
        buffer = io.BytesIO(payload)
        with zipfile.ZipFile(buffer, "a") as archive:
            archive.writestr("another/model.ckpt", CHECKPOINT)
        payload = buffer.getvalue()
    elif failure == "traversal":
        payload = archive_bytes("../model.ckpt")
    elif failure == "empty":
        payload = archive_bytes(payload=b"")
    elif failure == "bad_config":
        config = b"<html>not a model config</html>"
    elif failure == "crc":
        position = payload.index(CHECKPOINT)
        payload = (
            payload[:position]
            + bytes([payload[position] ^ 1])
            + payload[position + 1 :]
        )
    with serve(payload, config, extra_length) as (base, calls):
        local_sources(monkeypatch, base)
        with pytest.raises((ValueError, zipfile.BadZipFile)):
            download_ldm.download_model("ffhq", tmp_path / "downloads", expected)
        if failure == "bad_config":
            assert calls == ["/config.yaml"]
    assert not list((tmp_path / "downloads").iterdir())
    assert not (tmp_path / "model.ckpt").exists()


def test_existing_unmanaged_directory_is_never_overwritten(tmp_path):
    destination = tmp_path / "ffhq"
    destination.mkdir()
    (destination / "model.ckpt").write_bytes(b"user-owned checkpoint")
    with pytest.raises(ValueError, match="Cannot reuse"):
        download_ldm.download_model("ffhq", tmp_path)
    assert (destination / "model.ckpt").read_bytes() == b"user-owned checkpoint"


def test_dry_run_and_list_need_no_third_party_packages_or_files(tmp_path):
    script = Path("scripts/download_ldm.py").resolve()
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--model", "ffhq", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert (
        plan["archive_url"] == "https://ommer-lab.com/files/latent-diffusion/ffhq.zip"
    )
    assert download_ldm.UPSTREAM_REVISION in plan["config_url"]
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(model in result.stdout for model in download_ldm.MODELS)
    assert not list(tmp_path.iterdir())
