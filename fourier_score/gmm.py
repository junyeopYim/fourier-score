"""Shared matched-moment GMM, backbone and exact-score evaluation.

Extracted from the companion notebook so command-line comparisons and the
notebook use the same populations, initializations, noise and oracle metrics.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fourier_score.diffusion import NoiseLevel, NoiseProcess
from fourier_score.loss import training_loss
from fourier_score.method import FourierGaussian, GAUSSIAN_OBJECTIVES, gate_suffix, validate_gate
from fourier_score.spectral import conjugate_symmetrize
from fourier_score.utils import atomic_save, json_write

@dataclass(frozen=True)
class GMMConfig:
    preset: str
    run_tag: str = "v2"
    image_size: int = 8
    rho: float = 0.85
    spectrum_lambdas: tuple[float, ...] = (0.0, 1.0)
    spectrum_knee: float = 0.15
    spectrum_exponent: float = 1.5
    geometry_seed: int = 31415
    distributions: tuple[str, ...] = ("gaussian", "gmm")
    methods: tuple[str, ...] = (
        "score", "scalar_gaussian", "fourier_gaussian"
    )
    seeds: tuple[int, ...] = (42,)
    steps: int = 1500
    eval_every: int = 250
    batch_size: int = 128
    width: int = 128
    depth: int = 3
    time_features: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    sigma_min: float = 0.10
    sigma_max: float = 3.0
    n_noise_levels: int = 7
    val_per_noise: int = 256
    test_per_noise: int = 1024
    eval_batch_size: int = 128
    frequency_bins: int = 4
    cpu_threads: int = 4
    device: str = "auto"
    bank_version: str = "gmm-oracle-v1"



@dataclass(frozen=True)
class GMMArm:
    name: str
    parameterization: str
    objective: str = "dsm"
    gate_mode: str = "none"
    sigma_switch: float = 1.0
    sharpness: float = 4.0

    @property
    def gate(self):
        return validate_gate(dict(mode=self.gate_mode, sigma_switch=self.sigma_switch,
                                  sharpness=self.sharpness))


BASELINE_ARMS = (
    GMMArm("score", "score"),
    GMMArm("scalar_gaussian", "scalar_gaussian"),
    GMMArm("fourier_gaussian", "fourier_gaussian"),
    GMMArm("scalar_normalized", "scalar_gaussian", "normalized_residual"),
    GMMArm("fourier_normalized", "fourier_gaussian", "normalized_residual"),
)


def gated_arm(covariance, sigma_switch=1.0, sharpness=4.0):
    if covariance not in ("scalar", "fourier"):
        raise ValueError(covariance)
    gate = dict(mode="log_sigma", sigma_switch=sigma_switch, sharpness=sharpness)
    return GMMArm(covariance + gate_suffix(gate), covariance + "_gaussian",
                  "normalized_residual", "log_sigma", sigma_switch, sharpness)


NOTEBOOK_ARMS = {arm.name: arm for arm in BASELINE_ARMS}
NOTEBOOK_ARMS.update({f"{covariance}_gated": gated_arm(covariance)
                     for covariance in ("scalar", "fourier")})


def process_config(cfg):
    return dict(type="ve", sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max,
                num_scales=1000, t_min=0.0, beta_start=1e-4, beta_end=0.02)


def fft_filter(x: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
    """Orthonormal full-FFT real filter; valid for conjugate-symmetric multipliers."""
    return torch.fft.ifft2(torch.fft.fft2(x, norm="ortho") * multiplier, norm="ortho").real


def make_power(size: int, lam: float, cfg: GMMConfig) -> torch.Tensor:
    f = torch.fft.fftfreq(size, dtype=torch.float64)
    radius = torch.sqrt(f[:, None].square() + f[None, :].square())
    structured = (1.0 + (radius / cfg.spectrum_knee).square()).pow(-cfg.spectrum_exponent)
    structured = conjugate_symmetrize(structured)
    structured = structured / structured.mean()
    power = (1.0 - lam) * torch.ones_like(structured) + lam * structured
    assert float(power.min()) > 1e-4, "Population power must stay above the adapter floor."
    return power


class MatchedMomentFamily:
    """Small real Gaussian / equal-weight common-covariance GMM, with exact moments.

    Sampling: dense real square root built from the orthonormal Fourier operator.
    Oracle: exact joint score evaluated in real eigencoordinates.
    """
    def __init__(self, cfg: GMMConfig, distribution: str, lam: float):
        self.cfg = cfg
        self.distribution = distribution
        self.lam = float(lam)
        self.size = cfg.image_size
        self.d = self.size ** 2
        self.rho = 0.0 if distribution == "gaussian" else cfg.rho
        self.power = make_power(self.size, self.lam, cfg)
        eye_images = torch.eye(self.d, dtype=torch.float64).reshape(self.d, self.size, self.size)
        # Each row is the filtered basis vector; the filter matrix is real symmetric.
        self.root = fft_filter(eye_images, self.power.sqrt()).reshape(self.d, self.d)
        self.covariance = self.root.T @ self.root
        self.within_fraction = 1.0 - self.rho ** 2
        if distribution == "gaussian":
            self.means = torch.zeros(1, self.d, dtype=torch.float64)
        elif distribution == "gmm":
            generator = torch.Generator().manual_seed(cfg.geometry_seed)
            matrix = torch.randn(self.d, self.d, generator=generator, dtype=torch.float64)
            q, r = torch.linalg.qr(matrix)
            # Fix the conventional QR signs for a reproducible geometry.
            q = q * torch.where(torch.diag(r) < 0, -1.0, 1.0)[None, :]
            centers = self.rho * math.sqrt(self.d) * torch.cat((q, -q), dim=0)
            self.means = centers @ self.root
        else:
            raise ValueError(distribution)
        self.n_components = len(self.means)
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(self.covariance)
        if self.eigenvalues.min() <= 0:
            raise ValueError("Covariance must be positive definite.")
        self.means_eigen = self.means @ self.eigenvectors
        self.stats = {"mean": torch.zeros(1, self.size, self.size), "power": self.power[None].float()}
        self._means32 = self.means.float()
        self._root32 = self.root.float()
        self.case_id = f"{distribution}_lambda{self.lam:g}".replace(".", "p")

    def sample_cpu(self, n: int, generator: torch.Generator, dtype=torch.float32) -> torch.Tensor:
        # One CPU generator isolates training data from network/evaluation randomness.
        index = torch.randint(self.n_components, (n,), generator=generator)
        z = torch.randn(n, self.d, generator=generator, dtype=dtype)
        means = self._means32 if dtype == torch.float32 else self.means.to(dtype)
        root = self._root32 if dtype == torch.float32 else self.root.to(dtype)
        x = means[index] + math.sqrt(self.within_fraction) * (z @ root)
        return x.reshape(n, 1, self.size, self.size)

    def analytic_moments(self):
        mean = self.means.mean(0)
        centered = self.means - mean
        covariance = centered.T @ centered / self.n_components + self.within_fraction * self.covariance
        return mean, covariance

    def log_prob_and_score(self, y: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor):
        """Exact JOINT noisy density/score for each (y, alpha, sigma).

        y: [B,1,H,H]; alpha/sigma: [B]. This function is differentiable,
        but is used only to build validation/test banks and test the oracle.
        All mixture responsibilities condition on the ENTIRE y.
        """
        if y.ndim != 4 or y.shape[1:] != (1, self.size, self.size):
            raise ValueError("Expected [B,1,H,H].")
        if alpha.shape != (len(y),) or sigma.shape != (len(y),):
            raise ValueError("alpha and sigma must have shape [B].")
        if bool((sigma <= 0).any()):
            raise ValueError("sigma must be positive.")
        u = self.eigenvectors.to(y)
        eigenvalues = self.eigenvalues.to(y)
        means_eigen = self.means_eigen.to(y)
        y_eigen = y.flatten(1) @ u
        variance = alpha[:, None].square() * self.within_fraction * eigenvalues[None] + sigma[:, None].square()
        delta = y_eigen[:, None, :] - alpha[:, None, None] * means_eigen[None]
        log_components = -0.5 * (
            (delta.square() / variance[:, None, :]).sum(-1)
            + variance.log().sum(-1)[:, None] + self.d * math.log(2.0 * math.pi)
        ) - math.log(self.n_components)
        log_density = torch.logsumexp(log_components, dim=1)
        responsibility = torch.softmax(log_components, dim=1)
        score_eigen = -(responsibility[:, :, None] * delta / variance[:, None, :]).sum(1)
        score = (score_eigen @ u.T).reshape_as(y)
        return log_density, score

    def gaussian_score(self, y, alpha, sigma, scalar=False):
        power = self.power.to(y)
        if scalar:
            power = power.mean().expand_as(power)
        denom = alpha[:, None, None, None].square() * power + sigma[:, None, None, None].square()
        return -fft_filter(y, denom.reciprocal())


class SmallJointMLP(nn.Module):
    def __init__(self, cfg: GMMConfig):
        super().__init__()
        self.cfg = cfg
        frequencies = 2.0 ** torch.arange(cfg.time_features // 2, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        layers = []
        n_in = cfg.image_size ** 2 + cfg.time_features
        for _ in range(cfg.depth):
            layers.extend([nn.Linear(n_in, cfg.width), nn.SiLU()])
            n_in = cfg.width
        layers.append(nn.Linear(n_in, cfg.image_size ** 2))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y, sigma):
        normalized_time = (sigma.log() - math.log(self.cfg.sigma_min)) / math.log(self.cfg.sigma_max / self.cfg.sigma_min)
        phase = math.pi * normalized_time[:, None] * self.frequencies[None]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=1)
        return self.net(torch.cat([y.flatten(1), embedding], dim=1)).reshape_as(y)


class ToyScoreModel(nn.Module):
    def __init__(self, cfg, stats, method):
        super().__init__()
        arm = NOTEBOOK_ARMS[method] if isinstance(method, str) else method
        self.arm = arm
        self.method = arm.name
        self.embedding = "fourier"
        self.loss_objective = arm.objective
        self.process = NoiseProcess(process_config(cfg))
        self.backbone = SmallJointMLP(cfg)
        self.reference = None
        if arm.parameterization in GAUSSIAN_OBJECTIVES:
            self.reference = FourierGaussian(stats, backend="fft", gate=arm.gate,
                                             **GAUSSIAN_OBJECTIVES[arm.parameterization])
        elif arm.parameterization != "score" or arm.objective != "dsm" or arm.gate_mode != "none":
            raise ValueError(arm)

    def scaled_score(self, y, level):
        raw = self.backbone(y, level.sigma)
        if self.reference is None:
            return raw
        return self.reference.scaled_score(raw, y, level.alpha, level.sigma)

    def forward(self, y, level):
        return self.scaled_score(y, level) / level.sigma[:, None, None, None]


def make_model(family, arm, seed, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ToyScoreModel(family.cfg, family.stats, arm)
    return model.to(device)


def tensor_state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class OracleBank:
    split: str
    entries: list[dict]
    fingerprint: str
    n_observations: int


def bank_seed(family, split: str) -> int:
    payload = f"{family.cfg.bank_version}:{family.case_id}:{split}:independent-evaluation"
    return 1000000 + int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


@torch.no_grad()
def make_bank(family: MatchedMomentFamily, split: str) -> OracleBank:
    CFG = family.cfg
    PROCESS = NoiseProcess(process_config(CFG))
    if split not in ("validation", "test"):
        raise ValueError("Bank split must be validation or test")
    n = CFG.val_per_noise if split == "validation" else CFG.test_per_noise
    generator = torch.Generator().manual_seed(bank_seed(family, split))
    entries = []
    digest = hashlib.sha256()
    # Midpoint quadrature in log sigma.
    t_grid = (torch.arange(CFG.n_noise_levels, dtype=torch.float32) + 0.5) / CFG.n_noise_levels
    for noise_bin, t in enumerate(t_grid):
        coordinate = torch.full((n,), float(t))
        level = PROCESS.level(coordinate)
        clean = family.sample_cpu(n, generator)
        noise = torch.randn(clean.shape, generator=generator)
        y = PROCESS.perturb(clean, level, noise)  # input is exactly the FP32 value seen by the model
        oracle_parts, reference_parts, scalar_parts = [], [], []
        for lo in range(0, n, CFG.eval_batch_size):
            hi = min(lo + CFG.eval_batch_size, n)
            yy = y[lo:hi].double()
            aa, ss = level.alpha[lo:hi].double(), level.sigma[lo:hi].double()
            _, exact = family.log_prob_and_score(yy, aa, ss)
            ref = family.gaussian_score(yy, aa, ss)
            scalar = family.gaussian_score(yy, aa, ss, scalar=True)
            oracle_parts.append(ss[:, None, None, None] * exact)
            reference_parts.append(ss[:, None, None, None] * ref)
            scalar_parts.append(ss[:, None, None, None] * scalar)
        entry = {
            "noise_bin": noise_bin, "sigma": float(level.sigma[0]), "coordinate": float(t),
            "y": y, "noise": noise, "oracle_scaled": torch.cat(oracle_parts),
            "reference_scaled": torch.cat(reference_parts), "scalar_reference_scaled": torch.cat(scalar_parts),
        }
        for key in ("y", "noise", "oracle_scaled", "reference_scaled", "scalar_reference_scaled"):
            if not torch.isfinite(entry[key]).all():
                raise FloatingPointError(f"Nonfinite {key} in {family.case_id} / {split}")
            digest.update(entry[key].contiguous().numpy().tobytes())
        digest.update(np.asarray([entry["sigma"], entry["coordinate"]], dtype=np.float64).tobytes())
        entries.append(entry)
    return OracleBank(split, entries, digest.hexdigest(), n * len(entries))


def frequency_bands(cfg):
    freq = torch.fft.fftfreq(cfg.image_size, dtype=torch.float64)
    radius = torch.sqrt(freq[:, None].square() + freq[None, :].square())
    ids = (radius / math.sqrt(0.5) * cfg.frequency_bins).long().clamp_max(cfg.frequency_bins - 1)
    masks = [ids == j for j in range(cfg.frequency_bins)]
    return masks, [int(mask.sum()) for mask in masks]


@torch.no_grad()
def evaluate_model(model: ToyScoreModel | None, family, bank: OracleBank, reference="fourier") -> dict:
    """All reported numerical errors are accumulated on CPU float64.

    model=None evaluates the analytic Gaussian reference.
    """
    CFG = family.cfg
    DEVICE = next(model.parameters()).device if model is not None else torch.device("cpu")
    BAND_MASKS, BAND_COUNTS = frequency_bands(CFG)
    if model is not None:
        model.eval()
    rows = []
    total_error = total_ref = total_dsm = total_dot = total_rhat = 0.0
    total_n = 0
    total_band = np.zeros(CFG.frequency_bins, dtype=np.float64)
    for entry in bank.entries:
        predictions = []
        y_all = entry["y"]
        if model is None:
            key = "reference_scaled" if reference == "fourier" else "scalar_reference_scaled"
            predicted = entry[key]
        else:
            for lo in range(0, len(y_all), CFG.eval_batch_size):
                y = y_all[lo:lo + CFG.eval_batch_size].to(DEVICE)
                # Reuse the EXACT stored sigma. Recomputing exp(log_sigma) on CUDA
                # can differ from the CPU bank by an ulp and alter the oracle comparison.
                level = NoiseLevel(
                    alpha=torch.ones(len(y), device=DEVICE),
                    sigma=torch.full((len(y),), entry["sigma"], device=DEVICE),
                    coordinate=torch.full((len(y),), entry["coordinate"], device=DEVICE),
                )
                predictions.append(model.scaled_score(y, level).detach().cpu().double())
            predicted = torch.cat(predictions)
        true = entry["oracle_scaled"]
        error = predicted - true
        rstar = true - entry["reference_scaled"]
        rhat = predicted - entry["reference_scaled"]
        mse = error.square().flatten(1).mean(1)
        reference_error = rstar.square().flatten(1).mean(1)
        noisy_dsm = (predicted + entry["noise"].double()).square().flatten(1).mean(1)
        dot = (rhat * rstar).flatten(1).mean(1)
        rhat_energy = rhat.square().flatten(1).mean(1)
        energies = torch.fft.fft2(error, norm="ortho").abs().square().mean(dim=(0, 1))
        band_values = [float(energies[mask].mean()) if count else None for mask, count in zip(BAND_MASKS, BAND_COUNTS)]
        n = len(mse)
        err_value, ref_value = float(mse.mean()), float(reference_error.mean())
        row = {
            "noise_bin": entry["noise_bin"], "sigma": entry["sigma"], "n": n,
            "score_error": err_value,
            "unweighted_score_error": err_value / entry["sigma"] ** 2,
            "reference_error": ref_value,
            "relative_to_gaussian": err_value / ref_value if ref_value > 1e-10 else None,
            "dsm_pixel_mean": float(noisy_dsm.mean()),
            **{f"frequency_band_{i}": v for i, v in enumerate(band_values)},
        }
        # Check weighting of full-FFT band diagnostics against canonical pixel mean.
        reconstructed = sum((v or 0.0) * c for v, c in zip(band_values, BAND_COUNTS)) / family.d
        if not math.isclose(reconstructed, err_value, rel_tol=2e-8, abs_tol=2e-11):
            raise AssertionError("Frequency-band means do not reproduce pixel score error.")
        rows.append(row)
        total_n += n
        total_error += float(mse.sum())
        total_ref += float(reference_error.sum())
        total_dsm += float(noisy_dsm.sum())
        total_dot += float(dot.sum())
        total_rhat += float(rhat_energy.sum())
        total_band += n * np.asarray([0.0 if v is None else v for v in band_values])
    if not np.isfinite([total_error, total_ref, total_dsm, total_dot, total_rhat]).all():
        raise FloatingPointError("Nonfinite evaluation metric.")
    denominator = math.sqrt(max(total_ref * total_rhat, 0.0))
    return {
        "split": bank.split, "n_observations": total_n, "bank_sha256": bank.fingerprint,
        "score_error": total_error / total_n,
        "reference_error": total_ref / total_n,
        "relative_to_gaussian": total_error / total_ref if total_ref / total_n > 1e-10 else None,
        "residual_cosine": total_dot / denominator if denominator > 1e-10 else None,
        "dsm_pixel_mean": total_dsm / total_n,
        "score_error_by_frequency": [float(total_band[i] / total_n) if BAND_COUNTS[i] else None for i in range(CFG.frequency_bins)],
        "per_noise": rows,
    }


def train_arm(family, arm, seed, validation, output, provenance, *, stop_after=None):
    """Paired online training; never access a test bank during model selection.

    CPU FP32 matches the original comparison. Save the complete optimizer/RNG
    state and final EMA so testing can happen only after selecting a gate.
    """
    cfg = family.cfg
    if cfg.device != "cpu":
        raise ValueError("The controlled GMM comparison currently uses CPU")
    folder = Path(output) / family.case_id / f"{arm.name}_seed{seed}"
    checkpoint = folder / "checkpoint.pt"
    description = dict(config=asdict(cfg), arm=asdict(arm), seed=seed,
                       family=family.case_id, validation_sha256=validation.fingerprint,
                       provenance=provenance)
    signature = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    model = make_model(family, arm, seed)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=cfg.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(100000 + seed)
    state = dict(signature=signature, distribution=family.distribution,
                 spectrum_lambda=family.lam, method=arm.name, arm=asdict(arm), seed=seed,
                 step=0, completed=False, validation=[], optimizer_seconds=0.0,
                 gradient_norm_sum=0.0, gradient_clip_count=0,
                 initial_backbone_sha256=tensor_state_hash(model.backbone.state_dict()),
                 training_stream_first_batch_sha256=None)

    def save():
        state["final_data_rng_sha256"] = tensor_state_hash({"rng": generator.get_state()})
        atomic_save(dict(state=state, config=asdict(cfg), provenance=provenance,
                         model=model.state_dict(), ema=ema.state_dict(),
                         optimizer=optimizer.state_dict(), data_rng=generator.get_state()), checkpoint)
        json_write(state, folder / "metrics.json")

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["state"]
        if state["signature"] != signature:
            raise ValueError(f"GMM configuration/source/bank mismatch at {folder}")
        if state["completed"]:
            return state
        model.load_state_dict(payload["model"], strict=True)
        ema.load_state_dict(payload["ema"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["data_rng"])
    else:
        metric = evaluate_model(ema, family, validation)
        metric.update(step=0, last_training_loss=None)
        state["validation"].append(metric)
        save()

    start = time.perf_counter()
    model.train()
    for step in range(state["step"] + 1, cfg.steps + 1):
        clean = family.sample_cpu(cfg.batch_size, generator)
        level = model.process.sample(cfg.batch_size, "cpu", generator)
        noise = torch.randn(clean.shape, generator=generator)
        if step == 1:
            digest = hashlib.sha256()
            for tensor in (clean, level.coordinate, noise):
                digest.update(tensor.contiguous().numpy().tobytes())
            state["training_stream_first_batch_sha256"] = digest.hexdigest()
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, clean, level, noise, reduction="mean", objective=arm.objective)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), cfg.grad_clip,
                                             error_if_nonfinite=True)
        state["gradient_norm_sum"] += float(norm)
        state["gradient_clip_count"] += int(float(norm) > cfg.grad_clip)
        optimizer.step()
        with torch.no_grad():
            for pe, p in zip(ema.backbone.parameters(), model.backbone.parameters()):
                pe.lerp_(p, 1.0 - cfg.ema_decay)
        if step % cfg.eval_every == 0 or step == cfg.steps or step == stop_after:
            state["optimizer_seconds"] += time.perf_counter() - start
            state["step"] = step
            metric = evaluate_model(ema, family, validation)
            metric.update(step=step, last_training_loss=float(loss.detach()))
            state["validation"].append(metric)
            state["completed"] = step == cfg.steps
            if state["completed"]:
                state["final_ema_backbone_sha256"] = tensor_state_hash(ema.backbone.state_dict())
            save()
            if step == stop_after and step < cfg.steps:
                return state
            start = time.perf_counter()
    print(f"{family.case_id} {arm.name} seed={seed} steps={cfg.steps} "
          f"validation={state['validation'][-1]['score_error']:.6f}", flush=True)
    return state


def select_gate(results, switches):
    """Choose ONE gate for both covariances using final-step validation only.

    Average equally over both covariances, all planned spectra and all seeds.
    All candidates must cover the same complete set of training conditions.
    """
    groups = {}
    conditions = set()
    for result in results:
        arm = result["arm"]
        if not result["completed"]:
            raise ValueError("Model selection requires completed runs")
        if arm["gate_mode"] != "log_sigma":
            continue
        key = (result["spectrum_lambda"], result["seed"], arm["parameterization"])
        conditions.add(key)
        values = groups.setdefault(arm["sigma_switch"], {})
        if key in values:
            raise ValueError("Duplicate gate validation condition")
        metric = result["validation"][-1]
        if metric["split"] != "validation" or metric["step"] != result["step"]:
            raise ValueError("Gate selection requires final-step validation")
        values[key] = metric["score_error"]
    if set(groups) != set(switches) or not conditions:
        raise ValueError("Missing gate candidates")
    rows = []
    for switch in switches:
        if set(groups[switch]) != conditions:
            raise ValueError("Unpaired gate validation conditions")
        rows.append(dict(sigma_switch=switch, n_runs=len(conditions),
                         validation_score_error=float(np.mean(list(groups[switch].values())))))
    chosen = min(rows, key=lambda row: (row["validation_score_error"], row["sigma_switch"]))
    return dict(sigma_switch=chosen["sigma_switch"], candidates=rows,
                criterion="Mean final EMA validation scaled-score MSE across both covariances, spectra and seeds")
