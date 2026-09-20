"""Small terminal summaries; structured metrics stay in metrics.jsonl."""

import os
import json
from pathlib import Path
import shutil
import sys
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_metrics(record, compact=False):
    step = record["step"]
    total = record.get("total_steps", step)
    if record["split"] == "train":
        text = (
            f"[train] {step}/{total} ({100 * step / max(total, 1):.1f}%)"
            f" | avg={record['loss_avg']:.5g}"
            f" | {record['steps_per_second']:.2f} it/s"
            f" | ETA(train)={duration(record['eta_train_seconds'])}"
        )
        if compact:
            return text
        text += (
            f" | loss={record['loss']:.5g} pixel(avg)={record['loss_pixel_mean_avg']:.5g}"
            f" | lr={record['lr']:.2e} grad={record['grad_norm_before_clip']:.4g}"
        )
        if "cuda_allocated_gib" in record:
            text += (
                f" | CUDA alloc/reserved={record['cuda_allocated_gib']:.2f}/"
                f"{record['cuda_reserved_gib']:.2f} GiB"
            )
        return text
    return (
        f"[{record['split']}] step={step}/{total} {record['weights']}"
        f" | DSM={record['dsm_pixel_mean']:.6f} +/- {record['standard_error']:.6f}"
        f" | {record['n_images']} images | {record['eval_wall_seconds']:.1f}s"
    )


class ConsoleProgress:
    def __init__(self, mode="human", interval=5.0, stream=None):
        self.mode = mode
        self.interval = interval
        self.stream = sys.stderr if stream is None else stream
        self.tty = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self._last_update = None
        self._live = False

    def due(self):
        return self.mode == "human" and (
            self._last_update is None
            or time.perf_counter() - self._last_update >= self.interval
        )

    def clear(self):
        if self._live:
            self.stream.write("\r\x1b[2K")
            self.stream.flush()
            self._live = False

    def message(self, text):
        if self.mode != "human":
            return
        self.clear()
        print(text, file=self.stream, flush=True)

    def progress(self, text, force=False):
        if self.mode != "human" or not (force or self.due()):
            return
        self.clear()
        if self.tty:
            # Leave one column free so a live row never wraps onto the next line.
            width = max(1, shutil.get_terminal_size((120, 24)).columns - 1)
            self.stream.write("\r" + text[:width])
            self.stream.flush()
            self._live = True
        else:
            print(text, file=self.stream, flush=True)
        self._last_update = time.perf_counter()

    def metrics(self, record):
        self.message(format_metrics(record))
        self._last_update = time.perf_counter()

    def close(self):
        self.clear()


class ExperimentLogger:
    def __init__(self, path, tensorboard=False, console=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.console = console if console is not None else ConsoleProgress(mode="json")
        self.file = (path / "metrics.jsonl").open("a", encoding="utf-8")
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(path / "tensorboard"))

    def write(self, record):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.file.write(line + "\n")
        self.file.flush()
        if self.console.mode == "json":
            print(line, flush=True)
        elif self.console.mode == "human":
            self.console.metrics(record)
        if self.tb is not None:
            for k, v in record.items():
                if type(v) in (int, float) and k != "step":
                    self.tb.add_scalar(
                        record.get("split", "train") + "/" + k, v, record.get("step", 0)
                    )

    def close(self):
        self.console.close()
        self.file.close()
        if self.tb is not None:
            self.tb.close()
