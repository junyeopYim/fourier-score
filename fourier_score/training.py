"""One step-based trainer shared by all output parameterizations."""

import time
from pathlib import Path
from fourier_score.ema import EMA
from fourier_score.checkpoints import FORMAT, resume_signature
import torch
from fourier_score.data import BatchStream
from fourier_score.data import build_data, prepare_stats
from fourier_score.model import build_model, architecture_report, weight_hash
from fourier_score.loss import training_loss
from fourier_score.evaluation import evaluate_dsm
from fourier_score.utils import (
    configure_runtime,
    seed_all,
    atomic_save,
    capture_rng,
    restore_rng,
    environment,
    json_write,
    source_hash,
)
from fourier_score.logging import ExperimentLogger
from fourier_score.logging import ConsoleProgress, duration, format_metrics
from fourier_score.config import experiment_name


class Trainer:
    def __init__(self, cfg, checkpoint=None):
        session_started = time.perf_counter()
        self.console = ConsoleProgress(
            cfg["trainer"]["console"], cfg["trainer"]["progress_every_seconds"]
        )
        self.console.message(
            f"[setup] {experiment_name(cfg)} | configuring device={cfg['device']}"
        )
        device = configure_runtime(cfg)
        self.console.message(
            f"[setup] loading {cfg['data_loader']['args']['dataset']} data | device={device}"
        )
        self.bundle = build_data(cfg)
        # All arms use the same split/cache identity. Computing statistics does
        # not leak validation data and happens before the model RNG is seeded.
        self.console.message("[setup] loading/computing train-only statistics")
        stats = prepare_stats(
            cfg, self.bundle, checkpoint["stats"] if checkpoint else None
        )
        seed_all(cfg["seed"], device)
        self.console.message("[setup] building model / optimizer / EMA")
        model = build_model(cfg, stats, device)
        a = cfg["data_loader"]["args"]
        stream = BatchStream(
            self.bundle.train,
            a["batch_size"],
            cfg["seed"] + 1000,
            a["num_workers"],
            device.type == "cuda",
        )
        self.cfg = cfg
        self.model = model
        self.stats = stats
        self.stream = stream
        self.device = device
        self.step = 0
        self._session_started = session_started
        self._previous_wall_seconds = (
            checkpoint.get("training_wall_seconds", 0.0) if checkpoint else 0.0
        )
        self.out = Path(cfg["trainer"]["save_dir"]) / experiment_name(cfg)
        if checkpoint is None and self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"Run exists: {self.out}; use --resume or a new name")
        self.out.mkdir(parents=True, exist_ok=True)
        o = dict(cfg["optimizer"]["args"])
        o["betas"] = tuple(o["betas"])
        self.optimizer = torch.optim.Adam(model.parameters(), **o)
        self.ema = EMA(model, cfg["trainer"]["ema_decay"])
        self.generator = torch.Generator().manual_seed(cfg["seed"] + 2000)
        self.env = environment(device, cfg)
        self.initial_hash = weight_hash(model.backbone)
        if checkpoint is not None:
            if (
                checkpoint.get("format") != FORMAT
                or checkpoint.get("kind") != "training"
            ):
                raise ValueError("Only v1 training checkpoints can resume")
            if checkpoint["signature"] != resume_signature(cfg):
                raise ValueError(
                    "Resume config mismatch: model/loss/process/data/optimizer/backend must match"
                )
            if checkpoint["source_sha256"] != source_hash():
                raise ValueError(
                    "Source differs from checkpoint; resume with the original source revision"
                )
            old = checkpoint["environment"]
            if old["device"] != str(device) or old["torch"] != str(torch.__version__):
                raise ValueError(
                    "Resume requires same device type/index and PyTorch version; start a new explicit fine-tuning experiment instead"
                )
            model.load_state_dict(checkpoint["model"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.ema.load_state_dict(checkpoint["ema"], model)
            self.stream.load_state_dict(checkpoint["stream"])
            self.generator.set_state(checkpoint["generator"].cpu())
            self.step = checkpoint["step"]
            self.initial_hash = checkpoint["initial_backbone_sha256"]
            restore_rng(checkpoint["rng"], device)
        report = architecture_report(model.backbone)
        report["initial_backbone_sha256"] = self.initial_hash
        if model.reference is not None:
            self.env["spectral_transform_resolved"] = model.reference.resolved_backend(
                device
            )
        json_write(cfg, self.out / "config.resolved.json")
        json_write(self.env, self.out / "environment.json")
        json_write(report, self.out / "architecture.json")
        self.log = ExperimentLogger(
            self.out, cfg["trainer"]["tensorboard"], self.console
        )
        self.console.message(
            f"[run] {experiment_name(cfg)} | {self.env['device_name']} | "
            f"{cfg['loss']['type']} | step={self.step}/{cfg['trainer']['iterations']}"
        )
        self.console.message(
            f"[config] batch={a['batch_size']} microbatch={cfg['trainer']['microbatch_size'] or a['batch_size']}"
            f" | loss={cfg['loss']['reduction']} | log_every={cfg['trainer']['log_every']}"
            f" eval_every={cfg['trainer']['eval_every']}"
        )
        self.console.message(f"[metrics] {self.out / 'metrics.jsonl'}")
        if cfg["trainer"]["tensorboard"]:
            self.console.message(f"[tensorboard] {self.out / 'tensorboard'}")

    def train_step(self, clean):
        self.model.train()
        if self.cfg["data_loader"]["args"]["random_flip"]:
            flip = torch.rand(len(clean), generator=self.generator) < 0.5
            clean = torch.where(flip[:, None, None, None], clean.flip(-1), clean)
        lev = self.model.process.sample(len(clean), self.device, self.generator)
        noise = torch.randn(clean.shape, generator=self.generator).to(self.device)
        clean = clean.to(self.device)
        micro = self.cfg["trainer"]["microbatch_size"] or len(clean)
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, len(clean), micro):
            end = min(start + micro, len(clean))
            loss = training_loss(
                self.model,
                clean[start:end],
                lev.slice(start, end),
                noise[start:end],
                self.cfg["loss"]["reduction"],
            )
            loss = loss * ((end - start) / len(clean))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {self.step}")
            loss.backward()
            total += float(loss.detach())
        grad_norm = self.optimize()
        return total, grad_norm

    def evaluate(self):
        begin = time.perf_counter()
        split = self.bundle.metadata["eval_split"]
        n = min(len(self.bundle.validation), self.cfg["evaluation"]["max_images"])
        self.console.message(f"[eval] step={self.step} | {split} EMA | 0/{n} images")

        def progress(done, total):
            self.console.progress(
                f"[eval] step={self.step} | {split} EMA | {done}/{total} images"
                f" ({100 * done / total:.1f}%) | elapsed={duration(time.perf_counter() - begin)}"
            )

        with self.ema.average_parameters(self.model):
            result = evaluate_dsm(
                self.model,
                self.cfg,
                self.bundle.validation,
                self.device,
                progress=progress,
            )
        wall_seconds = self.training_wall_seconds()
        result.update(
            step=self.step,
            total_steps=self.cfg["trainer"]["iterations"],
            split=split,
            weights="EMA",
            eval_wall_seconds=time.perf_counter() - begin,
            training_wall_seconds=wall_seconds,
        )
        self.log.write(result)
        return result

    def save(self, snapshot=False):
        paths = str(self.out / "last.pt")
        if snapshot:
            paths += f", {self.out / f'ema_{self.step:09d}.pt'}"
        self.console.message(f"[save] step={self.step} | writing {paths}")
        begin = time.perf_counter()
        self._save_checkpoint(snapshot)
        self.console.message(
            f"[saved] step={self.step} | {time.perf_counter() - begin:.1f}s"
        )

    def train(self):
        t = self.cfg["trainer"]
        begin = time.perf_counter()
        if self.step >= t["iterations"]:
            self.log.close()
            raise ValueError("iterations must exceed the checkpoint step")
        session_steps = 0
        train_seconds = 0.0
        window_loss = 0.0
        window_images = 0
        window_steps = 0
        window_seconds = 0.0
        pixels = (
            self.cfg["data_loader"]["args"]["channels"]
            * self.cfg["data_loader"]["args"]["image_size"] ** 2
        )
        pixel_scale = 2 / pixels if self.cfg["loss"]["reduction"] == "half_sum" else 1.0
        try:
            if self.step == 0:
                self.evaluate()
            self.console.message(
                f"[train] step={self.step}/{t['iterations']} | starting optimizer updates"
            )
            while self.step < t["iterations"]:
                step_begin = time.perf_counter()
                clean = self.stream.next_batch()
                loss, norm = self.train_step(clean)
                seconds = time.perf_counter() - step_begin
                session_steps += 1
                train_seconds += seconds
                window_steps += 1
                window_seconds += seconds
                window_loss += loss * len(clean)
                window_images += len(clean)
                final = self.step == t["iterations"]
                log_due = final or self.step % t["log_every"] == 0
                if log_due or session_steps == 1 or self.console.due():
                    record = {
                        "step": self.step,
                        "total_steps": t["iterations"],
                        "split": "train",
                        "loss": loss,
                        "loss_avg": window_loss / window_images,
                        "window_steps": window_steps,
                        "window_images": window_images,
                        "loss_pixel_mean": loss * pixel_scale,
                        "loss_pixel_mean_avg": window_loss
                        / window_images
                        * pixel_scale,
                        "reduction": self.cfg["loss"]["reduction"],
                        "grad_norm_before_clip": norm,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "steps_per_second": window_steps / max(window_seconds, 1e-12),
                        "images_per_second": window_images / max(window_seconds, 1e-12),
                        "eta_train_seconds": (t["iterations"] - self.step)
                        * train_seconds
                        / session_steps,
                    }
                    if self.device.type == "cuda":
                        record.update(
                            cuda_allocated_gib=torch.cuda.memory_allocated(self.device)
                            / 2**30,
                            cuda_reserved_gib=torch.cuda.memory_reserved(self.device)
                            / 2**30,
                        )
                    if log_due:
                        record.update(
                            session_wall_seconds=time.perf_counter() - begin,
                            training_wall_seconds=self.training_wall_seconds(),
                        )
                        self.log.write(record)
                        window_loss = 0.0
                        window_images = 0
                        window_steps = 0
                        window_seconds = 0.0
                    else:
                        self.console.progress(
                            format_metrics(record, compact=True),
                            force=session_steps == 1,
                        )
                if final or self.step % t["eval_every"] == 0:
                    self.evaluate()
                snapshot = final or self.step % t["snapshot_every"] == 0
                if snapshot or self.step % t["save_every"] == 0:
                    self.save(snapshot)
            self.console.message(
                f"[done] step={self.step}/{t['iterations']}"
                f" | session={duration(time.perf_counter() - begin)} | checkpoint={self.out / 'last.pt'}"
            )
        except (Exception, KeyboardInterrupt):
            self.console.message(f"[stopped] step={self.step}/{t['iterations']}")
            raise
        finally:
            self.log.close()

    def training_wall_seconds(self):
        """Cumulative active-session time, including setup, evaluation and prior saves.

        Excludes downtime between sessions and the write of this checkpoint.
        Synchronize only at reporting/checkpoint boundaries, not every update.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return self._previous_wall_seconds + time.perf_counter() - self._session_started

    def optimize(self):
        warmup = self.cfg["trainer"]["warmup"]
        lr = self.cfg["optimizer"]["args"]["lr"] * (
            min(self.step / warmup, 1.0) if warmup else 1.0
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg["trainer"]["grad_clip"],
            error_if_nonfinite=True,
            foreach=False,
        )
        self.optimizer.step()
        self.step += 1
        self.ema.update(self.model)
        self.stream.advance()
        return float(norm)

    def _save_checkpoint(self, snapshot=False):
        elapsed = self.training_wall_seconds()
        # The process keeps using its loaded code even if the checkout changes.
        # Reuse the startup fingerprint: rereading files here can fail after a
        # refactor or falsely attribute old in-memory code to a new revision.
        run_source_sha256 = self.env["source_sha256"]
        state = {
            "format": FORMAT,
            "kind": "training",
            "config": self.cfg,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "stats": self.stats,
            "stream": self.stream.state_dict(),
            "generator": self.generator.get_state(),
            "rng": capture_rng(self.device),
            "signature": resume_signature(self.cfg),
            "source_sha256": run_source_sha256,
            "environment": self.env,
            "initial_backbone_sha256": self.initial_hash,
            "training_wall_seconds": elapsed,
        }
        atomic_save(state, self.out / "last.pt")
        if snapshot:
            with self.ema.average_parameters(self.model):
                atomic_save(
                    {
                        "format": FORMAT,
                        "kind": "ema",
                        "config": self.cfg,
                        "step": self.step,
                        "model": self.model.state_dict(),
                        "stats": self.stats,
                        "source_sha256": run_source_sha256,
                        "environment": self.env,
                        "training_wall_seconds": elapsed,
                    },
                    self.out / f"ema_{self.step:09d}.pt",
                )
