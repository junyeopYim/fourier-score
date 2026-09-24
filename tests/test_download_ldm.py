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
def serve_bundle(archive, config=CONFIG, extra_length=0):
    """Two-route LDM bundle server: ``/config.yaml`` or the archive, no Range.

    Unlike the ``http_server`` fixture it answers every other path with the
    archive and can overstate its Content-Length (``extra_length``) to
    simulate a truncated transfer; ``calls`` lists request paths only.
    """
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
    with serve_bundle(payload) as (base, calls):
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


@pytest.mark.parametrize("failure", ["checksum", "truncated"])
def test_failed_download_never_publishes_a_partial_bundle(tmp_path, monkeypatch, failure):
    payload = archive_bytes()
    expected = "0" * 64 if failure == "checksum" else None
    with serve_bundle(payload, extra_length=100 if failure == "truncated" else 0) as (base, _):
        local_sources(monkeypatch, base)
        with pytest.raises(ValueError):
            download_ldm.download_model("ffhq", tmp_path / "downloads", expected)
    assert not list((tmp_path / "downloads").iterdir())


def test_dry_run_and_list_need_no_third_party_packages_or_files(tmp_path):
    script = Path("scripts/download_ldm.py").resolve()
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--model", "ffhq", "--with-data", "--dry-run"],
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
    assert plan["dataset"]["destination"] == "data/ffhq"
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--list"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(model in result.stdout for model in download_ldm.MODELS)
    assert not list(tmp_path.iterdir())
