"""Time real latent optimizer updates and project each native LDM training budget."""

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fourier_score.ldm.config import (
    PAPER_TRAINING,
    PARAMETERIZATIONS,
    load_config,
    load_spec,
)
from fourier_score.ldm.data import prepare_cache
from fourier_score.ldm.training import Trainer, synchronize
from fourier_score.utils import ROOT, configure_runtime, json_write, source_hash

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def benchmark(cfg, spec, output, steps, warmup):
    cfg = copy.deepcopy(cfg)
    cfg["name"] = cfg["parameterization"]
    cfg["training"].update(
        iterations=steps, save_dir=str(output / "runs"), console="quiet"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer = Trainer(cfg, spec)
    batch_size = cfg["training"]["batch_size"]
    count = trainer.cache["sources"]["train"]["count"]
    if count < batch_size or (count % batch_size and steps > count // batch_size):
        raise ValueError(
            "Not enough full batches: use a larger subset or a multiple of the effective batch"
        )
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    records, last_report = [], 0.0
    print(
        f"[benchmark] {cfg['model']} / {cfg['parameterization']} | batch={batch_size}, microbatch={cfg['training']['microbatch_size']}",
        flush=True,
    )
    with (trainer.out / "timings.jsonl").open("x") as stream:
        for _ in range(steps):
            synchronize(trainer.device)
            started = time.perf_counter()
            values = trainer.train_step(trainer.stream.next_batch())
            synchronize(trainer.device)
            seconds = time.perf_counter() - started
            if values["images"] != batch_size:
                raise ValueError(
                    "Short optimizer batch would invalidate the timing comparison"
                )
            row = {
                "step": trainer.step,
                "seconds": seconds,
                "warmup": trainer.step <= warmup,
                **values,
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            records.append(row)
            now = time.perf_counter()
            if trainer.step == 1 or trainer.step % 10 == 0 or now - last_report >= 30:
                average = statistics.mean(
                    r["seconds"]
                    for r in records[max(warmup, len(records) - 10) :] or records
                )
                print(
                    f"[benchmark] {cfg['model']} / {cfg['parameterization']} | {trainer.step}/{steps} | {average:.3f} s/update | remaining {(steps - trainer.step) * average / 60:.1f} min",
                    flush=True,
                )
                last_report = now
    measured = [r["seconds"] for r in records[warmup:]]
    average = statistics.mean(measured)
    paper_batch, _, paper_steps = PAPER_TRAINING[cfg["model"]]
    result = {
        "model": cfg["model"],
        "parameterization": cfg["parameterization"],
        "optimizer_updates": steps,
        "warmup_updates_excluded": warmup,
        "effective_batch": batch_size,
        "microbatch": cfg["training"]["microbatch_size"],
        "paper_effective_batch": paper_batch,
        "paper_optimizer_updates": paper_steps,
        "paper_batch_matched": batch_size == paper_batch,
        "optimizer_seconds_all_updates": sum(r["seconds"] for r in records),
        "steady_mean_seconds_per_update": average,
        "steady_median_seconds_per_update": statistics.median(measured),
        "steady_stdev_seconds_per_update": statistics.stdev(measured)
        if len(measured) > 1
        else 0.0,
        "steady_images_per_second": batch_size / average,
        "projected_training_seconds": average * paper_steps
        if batch_size == paper_batch
        else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device)
        if trainer.device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device)
        if trainer.device.type == "cuda"
        else None,
        "initial_unet_sha256": trainer.initial_hash,
        "first_stage": trainer.cache["first_stage"],
        "cache_identity": trainer.cache["identity"],
        "training_images": count,
        "cache_preparation_seconds": trainer.cache["preparation_wall_seconds"],
        "config": cfg,
        "spec": spec,
        "environment": trainer.env,
        "source_sha256": source_hash(),
        "benchmark_script_sha256": SCRIPT_SHA256,
        "timing_scope": "Synchronized optimizer updates including cached data access, forward/backward, AdamW and LitEma. No evaluation, checkpoint writes, downloads or encoding included. Finite repeated subset; not a convergence experiment.",
        "checkpoint_saved": False,
    }
    json_write(result, trainer.out / "benchmark.json")
    trainer.console.close()
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--parameterizations",
        nargs="+",
        choices=PARAMETERIZATIONS,
        default=[
            "epsilon",
            "scalar_gaussian",
            "fourier_gaussian",
        ],
    )
    args = parser.parse_args(argv)
    if not 0 <= args.warmup < args.steps:
        parser.error("Require 0 <= warmup < steps")
    if len(set(args.parameterizations)) != len(args.parameterizations):
        parser.error("Parameterizations must be unique")
    if args.output.exists():
        parser.error("Choose a new output directory for a new benchmark")
    cfg = load_config(args.config, [*args.overrides, "device=" + args.device])
    spec = load_spec(cfg)
    # Queue benchmark jobs so encoding or another arm cannot share the measured GPU.
    lock_path = ROOT / "saved/.ldm_benchmark.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open("a")
    print(f"[benchmark] waiting for exclusive benchmark slot: {args.config}", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    if args.prepare:
        prepare_cache(cfg, spec, configure_runtime(cfg))
    args.output.mkdir(parents=True)
    results = []
    for arm in args.parameterizations:
        cfg["parameterization"] = arm
        result = benchmark(cfg, spec, args.output, args.steps, args.warmup)
        if (
            results
            and result["initial_unet_sha256"] != results[0]["initial_unet_sha256"]
        ):
            raise ValueError("Paired backbone initialization changed across arms")
        results.append(result)
        json_write(
            {
                "complete": len(results) == len(args.parameterizations),
                "results": results,
            },
            args.output / "summary.json",
        )
    print(args.output / "summary.json", flush=True)
    lock.close()


if __name__ == "__main__":
    main()
