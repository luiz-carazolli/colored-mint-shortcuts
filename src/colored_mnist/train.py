"""Deterministic training, validation, and epoch-boundary checkpoints."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset

from colored_mnist.matrix import (
    RunSpec,
    create_dataloader,
    get_training_dataset,
    get_validation_datasets,
)
from colored_mnist.models import build_model

EPOCHS = 10
BATCH_SIZE = 256
MEMORIZATION_MAX_UPDATES = 100
ADAM = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingSetup:
    step1_dir: Path
    raw_dir: Path
    output_dir: Path
    device: str = "cpu"
    threads: int = 1


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_atomic(path: Path, data: Any, *, checkpoint: bool = False) -> None:
    """Replace a complete artifact so an interruption preserves the previous copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if checkpoint:
                torch.save(data, stream)
            else:
                stream.write(data.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    write_atomic(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(path, buffer.getvalue())


def configure_runtime(setup: TrainingSetup) -> dict[str, Any]:
    """Configure determinism before any accelerator work and report the environment."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(setup.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    cuda = setup.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return {
        "device": setup.device,
        "hardware": torch.cuda.get_device_name() if cuda else (
            platform.processor() or platform.machine()
        ),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "torch_threads": setup.threads,
    }


def recipe() -> dict[str, Any]:
    files = ("models.py", "train.py", "matrix.py", "dataset.py",
             "assignments.py", "config.py", "palette.py", "splits.py")
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dtype": "float32",
        "loss": "cross_entropy",
        "optimizer": {"name": "Adam", **ADAM, "foreach": False, "fused": False},
        "source_sha256": {
            name: file_digest(Path(__file__).with_name(name)) for name in files
        },
    }


def data_provenance(setup: TrainingSetup, dataset_seed: int) -> dict[str, str]:
    paths = (
        "palette.json", "manifests/splits.csv",
        f"manifests/train_seed_{dataset_seed}.csv",
    )
    hashes = {name: file_digest(setup.step1_dir / name) for name in paths}
    val_columns = ("original_id", "source_split", "source_index", "label",
                   "aligned_color_id", "independent_color_id")
    with (setup.step1_dir / "manifests/heldout_views.csv").open(newline="") as stream:
        hashes["validation_views"] = json_digest([
            [row[column] for column in val_columns]
            for row in csv.DictReader(stream) if row["split"] == "val"
        ])
    raw = setup.raw_dir / "MNIST" / "raw"
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"):
        hashes[f"MNIST/raw/{name}"] = file_digest(raw / name)
    return hashes


def make_order_generator(train_seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(train_seed)


def capture_rng(order: torch.Generator, device: torch.device) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), position, has_gauss, cached_gaussian),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "data_order": order.get_state(),
    }


def restore_rng(state: dict[str, Any], order: torch.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position,
                         has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    order.set_state(state["data_order"])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(2, device=device, dtype=torch.float32)
    max_grad_norm = torch.zeros((), device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if optimizer is not None:
                loss.backward()
                grad_norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(param.grad) for param in model.parameters()
                ]))
                max_grad_norm = torch.maximum(max_grad_norm, grad_norm)
                optimizer.step()
            totals[0] += loss.detach() * targets.numel()
            totals[1] += (logits.argmax(dim=1) == targets).sum()
            count += targets.numel()
    if count == 0:
        raise ValueError("Cannot score an empty dataset")
    loss, accuracy = (totals / count).tolist()
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise RuntimeError("Non-finite epoch metrics")
    metrics = {"loss": loss, "accuracy": accuracy}
    if training:
        peak_norm = float(max_grad_norm)
        if not math.isfinite(peak_norm) or peak_norm == 0:
            raise RuntimeError("Non-finite or identically zero training gradients")
        metrics["max_grad_norm"] = peak_norm
    return metrics


def checkpoint_path(setup: TrainingSetup, run_id: str) -> Path:
    return setup.output_dir / "checkpoints" / f"{run_id}.pt"


def read_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def export_history(setup: TrainingSetup, run_id: str, history: list[dict]) -> None:
    base = setup.output_dir / "metrics" / run_id
    write_json(base.with_name(run_id + ".json"), history)
    write_csv(base.with_name(run_id + ".csv"), history)


def run_training(
    spec: RunSpec, setup: TrainingSetup, run_id: str, *, resume: bool,
) -> None:
    environment = configure_runtime(setup)
    device = torch.device(setup.device)
    configuration = {"run_id": run_id, "condition": asdict(spec), "recipe": recipe()}
    config_hash = json_digest(configuration)
    provenance = data_provenance(setup, spec.dataset_seed)
    path = checkpoint_path(setup, run_id)
    if path.exists() != resume:
        raise ValueError(f"{run_id}: expected {'an existing' if resume else 'no'} checkpoint")

    order = make_order_generator(spec.train_seed)
    random.seed(spec.train_seed)
    np.random.seed(spec.train_seed)
    torch.manual_seed(spec.train_seed)
    model = build_model(spec.train_seed).to(device)
    initial_parameters = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), **ADAM, foreach=False, fused=False)
    history: list[dict] = []
    completed = global_step = 0
    started_at = timestamp()

    if resume:
        saved = read_checkpoint(path)
        if (saved["version"], saved["config_sha256"], saved["data_sha256"],
                saved["environment"]) != (CHECKPOINT_VERSION, config_hash, provenance, environment):
            raise ValueError(f"{run_id}: checkpoint configuration, data, or environment changed")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = saved["history"]
        completed = saved["completed_epoch"]
        global_step = saved["global_step"]
        started_at = saved["started_at"]
        export_history(setup, run_id, history)
        if completed == EPOCHS:
            return

    train_data = get_training_dataset(spec, setup.step1_dir, setup.raw_dir)
    val_data = get_validation_datasets(setup.step1_dir, setup.raw_dir)
    if len(train_data) != spec.num_train_samples or any(len(ds) != 6000 for ds in val_data.values()):
        raise ValueError(f"{run_id}: dataset counts differ from the experiment")
    train_loader = create_dataloader(train_data, BATCH_SIZE, shuffle=True, generator=order)
    val_loaders = {
        name: create_dataloader(ds, BATCH_SIZE, generator=torch.Generator().manual_seed(0))
        for name, ds in val_data.items()
    }
    if resume:
        restore_rng(saved["rng"], order, device)
    for epoch in range(completed + 1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        global_step += len(train_loader)
        row = {"epoch": epoch, "global_step": global_step}
        epoch_metrics = {"train": train_metrics}
        for view, loader in val_loaders.items():
            epoch_metrics[view] = run_epoch(model, loader, device)
        for name, metrics in epoch_metrics.items():
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        if epoch == EPOCHS:
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all() or torch.equal(param, initial_parameters[name]):
                    raise RuntimeError(f"{run_id}: invalid or unchanged parameter tensor {name}")
        history.append(row)
        write_atomic(path, {
            "version": CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": epoch,
            "global_step": global_step,
            "rng": capture_rng(order, device),
            "config": configuration,
            "config_sha256": config_hash,
            "data_sha256": provenance,
            "environment": environment,
            "history": history,
            "started_at": started_at,
            "updated_at": timestamp(),
        }, checkpoint=True)
        export_history(setup, run_id, history)
        print(f"{run_id} epoch {epoch}/{EPOCHS}: {row}", flush=True)


def memorization_check(spec: RunSpec, setup: TrainingSetup) -> None:
    environment = configure_runtime(setup)
    device = torch.device(setup.device)
    dataset = get_training_dataset(spec, setup.step1_dir, setup.raw_dir)
    order = make_order_generator(spec.train_seed)
    indices = torch.randperm(len(dataset), generator=order)[:64].tolist()
    images, targets = next(iter(create_dataloader(Subset(dataset, indices), 64, generator=order)))
    images, targets = images.to(device), targets.to(device)
    model = build_model(spec.train_seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **ADAM, foreach=False, fused=False)
    with torch.no_grad():
        initial_loss = float(nn.functional.cross_entropy(model(images), targets))
    for step in range(1, MEMORIZATION_MAX_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        if any(param.grad is None or not torch.isfinite(param.grad).all()
               for param in model.parameters()):
            raise RuntimeError("Invalid gradients in the memorization check")
        optimizer.step()
        with torch.no_grad():
            logits = model(images)
            loss_value = float(nn.functional.cross_entropy(logits, targets))
        if loss_value < 0.01:
            break
    passed = loss_value < 0.01
    write_json(setup.output_dir / "memorization_check.json", {
        "passed": passed,
        "max_updates": MEMORIZATION_MAX_UPDATES,
        "updates": step,
        "initial_loss": initial_loss,
        "loss": loss_value,
        "correct": int((logits.argmax(dim=1) == targets).sum()),
        "batch_size": targets.numel(),
        "environment": environment,
        "recipe_sha256": json_digest(recipe()),
        "data_sha256": data_provenance(setup, spec.dataset_seed),
    })
    if not passed:
        raise RuntimeError(f"Memorization check failed after {step} updates: loss={loss_value:.6f}")
    print(f"Memorization check passed after {step} updates: loss={loss_value:.6f}")
