from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading

import pytest
import torch
from fourier_score.config import load_config
from fourier_score.data_loader.statistics import placeholder_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
# Golden contract probes (tests/golden/golden_probes) are shared with the recorder.
sys.path.insert(0, str(REPO_ROOT / "tests" / "golden"))


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)

@pytest.fixture
def cfg(tmp_path):
    return load_config('configs/smoke.json',[f'trainer.save_dir={tmp_path}/runs',f'fourier.cache_dir={tmp_path}/stats','backend.cpu_threads=2','device=cpu'])

@pytest.fixture
def stats():
    return placeholder_stats((1, 8, 8))


@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def golden_ctx(tmp_path, monkeypatch):
    """Context for golden contract probes; local-only probes need FOURIER_GOLDEN_ROOT."""
    import golden_probes

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    golden_root = os.environ.get("FOURIER_GOLDEN_ROOT")
    return golden_probes.Context(
        root=REPO_ROOT,
        tmp=tmp_path,
        golden_root=Path(golden_root).resolve() if golden_root else None,
    )


@contextmanager
def serve_files(files):
    """Serve ``{url_path: bytes}`` from 127.0.0.1 with ``Range: bytes=N-`` resume.

    Yields ``(base_url, calls)`` where ``calls`` lists ``(path, range_start)``
    per GET.  ``files`` is read per request, so entries added while the server
    runs are served too.  The server is stopped when the block exits.
    """
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


@pytest.fixture
def http_server():
    """Local HTTP file server factory: ``with http_server(files) as (base, calls):``."""
    return serve_files
