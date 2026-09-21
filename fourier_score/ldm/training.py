"""Matched-budget latent-only training, native AdamW/LitEma, exact cursor resume."""

import copy
import json
import time
import warnings
from pathlib import Path

import torch

from fourier_score.data import BatchStream
from fourier_score.logging import ConsoleProgress
from fourier_score.model import weight_hash
from fourier_score.utils import (
    atomic_save,
    capture_rng,
    configure_runtime,
    environment,
    json_write,
    load_checkpoint,
    restore_rng,
    seed_all,
    source_hash,
)

from .config import experiment_name, lr_multiplier, override, validate
from .data import LatentDataset, draw_latents, open_cache
from .model import Denoiser, ema_scope
from .upstream.ema import LitEma

FORMAT = "fourier-ldm-training-v1"


def signature(cfg):
    result = copy.deepcopy(cfg)
    for k in ("name", "device", "sampling", "evaluation"):
        result.pop(k)
    for k in (
        "iterations",
        "save_dir",
        "save_every",
        "snapshot_every",
        "eval_every",
        "log_every",
        "console",
    ):
        result["training"].pop(k)
    result["cache"].pop("dir")
    result["cache"].pop("num_workers")
    result["cache"].pop("batch_size")
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Trainer:
    def __init__(self, cfg, spec, checkpoint=None):
        self.started = time.perf_counter()
        self.cfg, self.spec = cfg, spec
        self.console = ConsoleProgress(cfg["training"]["console"])
        self.out = Path(cfg["training"]["save_dir"]) / experiment_name(cfg)
        if checkpoint is None and self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(
                f"Run exists: {self.out}; resume or choose a new name"
            )
        self.device = configure_runtime(cfg)
        self.console.message(
            "[setup] verifying latent cache and training-only statistics"
        )
        self.cache = open_cache(cfg, spec)
        self.stats = load_checkpoint(Path(cfg["cache"]["dir"]) / "stats.pt")
        seed_all(cfg["seed"], self.device)
        self.model = Denoiser(
            spec,
            cfg["parameterization"],
            self.stats,
            cfg["backend"]["spectral_transform"],
            cfg["training"]["gradient_checkpointing"],
        ).to(self.device)
        self.initial_hash = weight_hash(self.model.diffusion_model)
        t = cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t["lr"],
            betas=tuple(t["betas"]),
            eps=t["eps"],
            weight_decay=t["weight_decay"],
            foreach=False,
        )
        self.ema = LitEma(self.model, decay=t["ema_decay"]).to(self.device)
        self.stream = BatchStream(
            LatentDataset(cfg["cache"]["dir"], "train", self.cache),
            t["batch_size"],
            cfg["seed"] + 1000,
            cfg["cache"]["num_workers"],
        )
        self.generator = torch.Generator().manual_seed(cfg["seed"] + 2000)
        self.env = environment(self.device, cfg)
        self.step, self.previous_wall, self.optimizer_seconds = 0, 0.0, 0.0
        if checkpoint is not None:
            if (
                checkpoint.get("format") != FORMAT
                or checkpoint.get("kind") != "training"
            ):
                raise ValueError("Only LDM training checkpoints can resume")
            if checkpoint["signature"] != signature(cfg) or checkpoint["spec"] != spec:
                raise ValueError(
                    "Resume model/process/loss/optimizer configuration mismatch"
                )
            if checkpoint["cache"]["identity"] != self.cache["identity"]:
                raise ValueError("Resume latent cache mismatch")
            if checkpoint["source_sha256"] != source_hash():
                raise ValueError("Training resume requires the same source revision")
            if any(
                checkpoint["environment"][k] != self.env[k] for k in ("torch", "device")
            ):
                raise ValueError("Resume requires the same PyTorch version and device")
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.ema.load_state_dict(checkpoint["ema"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.stream.load_state_dict(checkpoint["stream"])
            self.generator.set_state(checkpoint["generator"])
            self.step = checkpoint["step"]
            self.initial_hash = checkpoint["initial_unet_sha256"]
            self.previous_wall = checkpoint["training_wall_seconds"]
            self.optimizer_seconds = checkpoint["optimizer_wall_seconds"]
            restore_rng(checkpoint["rng"], self.device)
        if self.step > t["iterations"]:
            raise ValueError("Requested update limit is behind checkpoint step")
        self.out.mkdir(parents=True, exist_ok=True)
        json_write(cfg, self.out / "config.resolved.json")
        json_write(spec, self.out / "ldm_spec.json")
        json_write(self.env, self.out / "environment.json")
        json_write(
            {
                "initial_unet_sha256": self.initial_hash,
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "first_stage": self.cache["first_stage"],
                "cache_identity": self.cache["identity"],
            },
            self.out / "architecture.json",
        )
        self.console.message(
            f"[run] {experiment_name(cfg)} | {spec['loss_type']} | {self.device} | batch={t['batch_size']} microbatch={t['microbatch_size']}"
        )

    def train_step(self, batch):
        self.model.train()
        clean = draw_latents(batch, self.cache["first_stage"], self.generator)
        t = torch.randint(
            self.spec["timesteps"], (len(clean),), generator=self.generator
        ).to(self.device)
        noise = torch.randn(clean.shape, generator=self.generator).to(self.device)
        clean = clean.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        micro = self.cfg["training"]["microbatch_size"]
        for start in range(0, len(clean), micro):
            end = min(start + micro, len(clean))
            loss = self.model.loss(clean[start:end], t[start:end], noise[start:end]) * (
                (end - start) / len(clean)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite LDM loss at update {self.step}")
            loss.backward()
            total += float(loss.detach())
        grad = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg["training"]["grad_clip"] or float("inf"),
            error_if_nonfinite=True,
            foreach=False,
        )
        lr = self.cfg["training"]["lr"] * lr_multiplier(self.spec, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.ema(self.model)
        self.step += 1
        self.stream.advance()
        return {"loss": total, "lr": lr, "grad_norm": float(grad), "images": len(clean)}

    def wall_seconds(self):
        return self.previous_wall + time.perf_counter() - self.started

    def save(self, snapshot=False):
        self.console.message(f"[save] update={self.step}: {self.out / 'last.pt'}")
        common = {
            "format": FORMAT,
            "config": self.cfg,
            "spec": self.spec,
            "step": self.step,
            "stats": self.stats,
            "cache": self.cache,
            "source_sha256": source_hash(),
            "environment": self.env,
            "initial_unet_sha256": self.initial_hash,
            "training_wall_seconds": self.wall_seconds(),
            "optimizer_wall_seconds": self.optimizer_seconds,
        }
        atomic_save(
            {
                **common,
                "kind": "training",
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ema": self.ema.state_dict(),
                "signature": signature(self.cfg),
                "stream": self.stream.state_dict(),
                "generator": self.generator.get_state(),
                "rng": capture_rng(self.device),
            },
            self.out / "last.pt",
        )
        if snapshot:
            state = dict(self.model.state_dict())
            buffers = dict(self.ema.named_buffers())
            for name, ema_name in self.ema.m_name2s_name.items():
                state[name] = buffers[ema_name]
            atomic_save(
                {**common, "kind": "ema", "model": state},
                self.out / f"ema_{self.step:09d}.pt",
            )

    def train(self):
        from .evaluation import evaluate_latents

        options = self.cfg["training"]
        window, window_time, window_steps = 0.0, 0.0, 0

        def record(stream, row):
            row.update(
                step=self.step,
                training_wall_seconds=self.wall_seconds(),
                optimizer_wall_seconds=self.optimizer_seconds,
            )
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if options["console"] == "json":
                print(json.dumps(row), flush=True)

        with (self.out / "metrics.jsonl").open("a") as stream:
            try:
                while self.step < options["iterations"]:
                    synchronize(self.device)
                    begin = time.perf_counter()
                    values = self.train_step(self.stream.next_batch())
                    synchronize(self.device)
                    elapsed = time.perf_counter() - begin
                    self.optimizer_seconds += elapsed
                    window += values["loss"]
                    window_time += elapsed
                    window_steps += 1
                    self.console.progress(
                        f"[train] {self.step}/{options['iterations']} | loss={values['loss']:.5g} | {window_steps / window_time:.3f} update/s"
                    )
                    if (
                        self.step == 1
                        or self.step % options["log_every"] == 0
                        or self.step == options["iterations"]
                    ):
                        record(
                            stream,
                            {
                                "split": "train",
                                **values,
                                "loss_avg": window / window_steps,
                                "steps_per_second": window_steps / window_time,
                                "window_steps": window_steps,
                            },
                        )
                        window, window_time, window_steps = 0.0, 0.0, 0
                    if options["eval_every"] and self.step % options["eval_every"] == 0:
                        with ema_scope(self.model, self.ema):
                            metrics = evaluate_latents(
                                self.model, self.cfg, self.cache, self.device
                            )
                        record(
                            stream, {"split": "validation", "weights": "EMA", **metrics}
                        )
                        self.console.message(
                            f"[eval] {self.step}: epsilon_mse={metrics['epsilon_mse']:.6f}"
                        )
                    snapshot = (
                        options["snapshot_every"]
                        and self.step % options["snapshot_every"] == 0
                    )
                    if (
                        self.step % options["save_every"] == 0
                        or snapshot
                        or self.step == options["iterations"]
                    ):
                        self.save(
                            snapshot=bool(snapshot)
                            or self.step == options["iterations"]
                        )
            finally:
                # Never save a partial update after an interrupt/OOM.
                self.console.close()
        return self.out / "last.pt"


def load_trained(path, changes=(), device=None):
    state = load_checkpoint(path)
    if state.get("format") != FORMAT:
        raise ValueError("Expected an LDM experiment checkpoint")
    for change in changes:
        key = change.split("=", 1)[0]
        if (
            key != "device"
            and not key.startswith(("sampling.", "evaluation.", "backend."))
            and key not in ("cache.dir", "first_stage.checkpoint")
        ):
            raise ValueError(f"Inference cannot change trained configuration: {key}")
    cfg = validate(
        override(state["config"], [*changes, *([f"device={device}"] if device else [])])
    )
    dev = configure_runtime(cfg)
    model = Denoiser(
        state["spec"],
        cfg["parameterization"],
        state["stats"],
        cfg["backend"]["spectral_transform"],
    ).to(dev)
    model.load_state_dict(state["model"], strict=True)
    if state["kind"] == "training":
        ema = LitEma(model, decay=cfg["training"]["ema_decay"]).to(dev)
        ema.load_state_dict(state["ema"], strict=True)
        ema.copy_to(model)
    elif state["kind"] != "ema":
        raise ValueError("Unknown LDM checkpoint kind")
    if state["source_sha256"] != source_hash():
        warnings.warn(
            "Inference source differs from the training checkpoint", stacklevel=2
        )
    return model.eval(), cfg, dev, state
