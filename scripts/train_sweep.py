#!/usr/bin/env python3
"""Prepare or run the fixed Colored MNIST training experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colored_mnist.matrix import RunSpec, build_experimental_matrix
from colored_mnist.train import (
    EPOCHS,
    TrainingSetup,
    checkpoint_path,
    configure_runtime,
    data_provenance,
    json_digest,
    memorization_check,
    read_checkpoint,
    recipe,
    run_training,
    write_csv,
    write_json,
)

PILOT_ID = "pilot_p_0.10_dseed_101_tseed_201"


def refresh_registry(setup: TrainingSetup, specs: list[RunSpec], pilot: RunSpec) -> None:
    rows = []
    fixed_recipe = recipe()
    for run_id, spec in [(PILOT_ID, pilot), *((spec.run_id, spec) for spec in specs)]:
        path = checkpoint_path(setup, run_id)
        saved = read_checkpoint(path) if path.exists() else {}
        epoch = saved.get("completed_epoch", 0)
        last = saved["history"][-1] if epoch else {}
        config = {"run_id": run_id, "condition": spec.to_dict(), "recipe": fixed_recipe}
        environment = saved.get("environment", {})
        rows.append({
            **spec.to_dict(),
            "run_id": run_id,
            "role": "pilot" if run_id == PILOT_ID else "sweep",
            "status": "complete" if epoch == EPOCHS else "partial" if epoch else "pending",
            "config_sha256": saved.get("config_sha256", json_digest(config)),
            "completed_epoch": epoch,
            "global_step": saved.get("global_step", 0),
            "started_at": saved.get("started_at"),
            "updated_at": saved.get("updated_at"),
            "device": environment.get("device"),
            "hardware": environment.get("hardware"),
            "checkpoint_path": path.relative_to(setup.output_dir).as_posix() if epoch else None,
            "metrics_path": f"metrics/{run_id}.json" if epoch else None,
            **{
                f"{view}_{metric}": last.get(f"{view}_{metric}") if epoch == EPOCHS else None
                for view in ("train", "val_aligned", "val_independent")
                for metric in ("loss", "accuracy")
            },
        })
    write_json(setup.output_dir / "runs_manifest.json", rows)
    write_csv(setup.output_dir / "runs_manifest.csv", rows)


def require_memorization(setup: TrainingSetup, pilot: RunSpec, environment: dict) -> None:
    path = setup.output_dir / "memorization_check.json"
    if path.exists():
        report = json.loads(path.read_text())
        if (report["passed"] and report["environment"] == environment
                and report["recipe_sha256"] == json_digest(recipe())
                and report["data_sha256"] == data_provenance(setup, pilot.dataset_seed)):
            return
    raise ValueError("Run the memorization check successfully with the current setup first")


def require_pilot(setup: TrainingSetup, pilot: RunSpec, environment: dict) -> None:
    path = checkpoint_path(setup, PILOT_ID)
    if path.exists():
        saved = read_checkpoint(path)
        config = {"run_id": PILOT_ID, "condition": pilot.to_dict(), "recipe": recipe()}
        if (saved["completed_epoch"] == EPOCHS and saved["environment"] == environment
                and saved["config_sha256"] == json_digest(config)
                and saved["data_sha256"] == data_provenance(setup, pilot.dataset_seed)):
            return
    raise ValueError("Complete the pilot with the current setup first")


def require_consistent_sweep(
    setup: TrainingSetup, specs: list[RunSpec], environment: dict,
) -> None:
    fixed_recipe = recipe()
    provenance = {}
    for spec in specs:
        path = checkpoint_path(setup, spec.run_id)
        if not path.exists():
            continue
        saved = read_checkpoint(path)
        config = {"run_id": spec.run_id, "condition": spec.to_dict(), "recipe": fixed_recipe}
        if spec.dataset_seed not in provenance:
            provenance[spec.dataset_seed] = data_provenance(setup, spec.dataset_seed)
        if (saved["environment"] != environment
                or saved["config_sha256"] != json_digest(config)
                or saved["data_sha256"] != provenance[spec.dataset_seed]):
            raise ValueError(f"{spec.run_id}: existing sweep checkpoint uses a different setup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-dir", type=Path, default=PROJECT_ROOT / "artifacts/step_1_dataset")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/step_2_training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="write the registry without optimization")
    commands.add_parser("memorize", help="run the 64-example optimization check")
    commands.add_parser("pilot", help="train the baseline pilot").add_argument("--resume", action="store_true")
    run = commands.add_parser("run", help="train a single sweep condition")
    run.add_argument("run_id")
    run.add_argument("--resume", action="store_true")
    sweep = commands.add_parser("sweep", help="train the full experimental matrix")
    sweep.add_argument("--jobs", type=int, default=1)
    sweep.add_argument("--resume", action="store_true", help="continue saved runs and start missing runs")
    args = parser.parse_args()
    if args.threads < 1 or getattr(args, "jobs", 1) < 1:
        parser.error("--threads and --jobs must be positive")
    if args.device == "cuda" and getattr(args, "jobs", 1) != 1:
        parser.error("CUDA uses one run at a time")
    return args


def execute(args: argparse.Namespace, setup: TrainingSetup) -> None:
    specs = build_experimental_matrix(setup.step1_dir, setup.raw_dir)
    by_id = {spec.run_id: spec for spec in specs}
    pilot = by_id["p_0.10_dseed_101_tseed_201"]
    if args.command == "prepare":
        refresh_registry(setup, specs, pilot)
        print(f"Prepared {len(specs)} sweep entries and one pilot entry")
        return
    if args.command == "memorize":
        memorization_check(pilot, setup)
        return

    environment = configure_runtime(setup)
    require_memorization(setup, pilot, environment)
    try:
        if args.command == "pilot":
            run_training(pilot, setup, PILOT_ID, resume=args.resume)
            return
        require_pilot(setup, pilot, environment)
        require_consistent_sweep(setup, specs, environment)
        if args.command == "run":
            if args.run_id not in by_id:
                raise ValueError(f"Unknown run ID {args.run_id!r}. See runs_manifest.csv")
            run_training(by_id[args.run_id], setup, args.run_id, resume=args.resume)
            return

        existing = {spec.run_id for spec in specs if checkpoint_path(setup, spec.run_id).exists()}
        if existing and not args.resume:
            raise ValueError("Sweep checkpoints already exist. Use sweep --resume to continue")
        if args.jobs == 1:
            for spec in specs:
                run_training(spec, setup, spec.run_id, resume=spec.run_id in existing)
                refresh_registry(setup, specs, pilot)
        else:
            with ProcessPoolExecutor(
                max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(run_training, spec, setup, spec.run_id, resume=spec.run_id in existing)
                    for spec in specs
                ]
                for future in as_completed(futures):
                    future.result()
                    refresh_registry(setup, specs, pilot)
    finally:
        refresh_registry(setup, specs, pilot)


def main() -> None:
    args = parse_args()
    setup = TrainingSetup(args.step1_dir, args.raw_dir, args.output_dir, args.device, args.threads)
    setup.output_dir.mkdir(parents=True, exist_ok=True)
    with (setup.output_dir / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another runner is using this output directory") from None
        execute(args, setup)


if __name__ == "__main__":
    main()
