import json

import numpy as np
import torch

from scripts.compare_runs import main


def write_run(root, *, weight=1.0, wall=1.0, source="a" * 64, extra_rng=False):
    root.mkdir()
    rng = {"torch": torch.arange(4, dtype=torch.uint8), **({"cuda": [torch.zeros(2)]} if extra_rng else {})}
    torch.save(
        {"model": {"w": torch.tensor([weight, float("nan")])}, "step": 3, "rng": rng,
         "training_wall_seconds": wall, "source_sha256": source, "environment": {"host": source}},
        root / "last.pt",
    )
    rows = [{"step": 1, "loss": 0.5, "steps_per_second": wall, "eta_train_seconds": wall}]
    (root / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "settings.json").write_text(json.dumps({"seed": 0, "eta": 0.0, "provenance": {"git_revision": source}}))
    np.savez(root / "samples.npz", x=np.arange(3))
    (root / "tensorboard").mkdir()
    (root / "tensorboard" / f"events.out.tfevents.{wall}").write_bytes(b"x")


def test_identical_up_to_provenance_and_wall_clock(tmp_path, capsys):
    write_run(tmp_path / "a")
    write_run(tmp_path / "b", wall=2.5, source="b" * 64)
    assert main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert "4 files compared, 0 differ" in capsys.readouterr().out
    assert main([str(tmp_path / "a" / "last.pt"), str(tmp_path / "b" / "last.pt")]) == 0


def test_reports_tensor_key_and_file_differences(tmp_path, capsys):
    write_run(tmp_path / "a")
    write_run(tmp_path / "b", weight=float(np.nextafter(np.float32(1), np.float32(2))), extra_rng=True)
    (tmp_path / "b" / "extra.png").write_bytes(b"png")
    assert main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    out = capsys.readouterr().out
    assert ".model.w: tensor torch.float32[2] differs in 1 elements" in out
    assert ".rng: keys only in A [], only in B ['cuda']" in out
    assert "DIFF extra.png" in out and "only in B" in out


def test_ignore_key_option(tmp_path):
    write_run(tmp_path / "a")
    write_run(tmp_path / "b")
    settings = tmp_path / "b" / "settings.json"
    settings.write_text(json.dumps({**json.loads(settings.read_text()), "seed": 1}))
    assert main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    assert main([str(tmp_path / "a"), str(tmp_path / "b"), "--ignore-key", "seed"]) == 0
