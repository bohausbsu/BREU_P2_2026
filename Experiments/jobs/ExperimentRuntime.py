"""
Experiment 1 – SPECTRA runtime experiment.

Usage
-----
  # For FFNN
  python ExperimentRuntime.py --model ffnn --dataset data.csv --target-col Placement --out-csv ffnn_time.csv

  # For CNN
  python ExperimentRuntime.py --model cnn --dataset-root PetImages --out-csv cnn_time.csv

  # For AE
  python ExperimentRuntime.py --model ae --dataset-root mnist --out-csv ae_time.csv
"""

import argparse
import csv
import os
import sys
import time

import torch

# Set paths for our scripts so that imports work properly
_HERE = os.path.dirname(os.path.abspath(__file__))
_AT_DIR = os.path.join(_HERE, "..")
_AT_PKG_DIR = os.path.join(_AT_DIR, "AnomalyTransformer")
sys.path.insert(0, _AT_DIR)
sys.path.insert(0, _AT_PKG_DIR)
sys.path.insert(0, _HERE)

from AnomalyTransformer import train as at_train
from AnomalyTransformer.AnomalyAttention import AnomalyTransformer as ATModel
from AnomalyTransformer.dataset import get_dataloader, split_data
from Experiments.jobs.snapshot_trainer_a import (EightKMLP, FourKMLP, SixKMLP,
                                                 TenKMLP, TwelveKMLP)
from Experiments.jobs.snapshot_trainer_a import \
    load_full_dataset as load_full_dataset_a
from Experiments.jobs.snapshot_trainer_a import run as snapshot_run_a
from Experiments.jobs.snapshot_trainer_b import (EightKCNN, FourKCNN, SixKCNN,
                                                 TenKCNN, TwelveKCNN)
from Experiments.jobs.snapshot_trainer_b import \
    load_full_dataset as load_full_dataset_b
from Experiments.jobs.snapshot_trainer_b import run as snapshot_run_b
from Experiments.jobs.snapshot_trainer_c import (EightKAE, FourKAE, SixKAE,
                                                 TenKAE, TwelveKAE)
from Experiments.jobs.snapshot_trainer_c import \
    load_full_dataset as load_full_dataset_x
from Experiments.jobs.snapshot_trainer_c import run as snapshot_run_c

EFF_SIGNAL_COL = 2

FAMILY_MODEL_NAMES = {
    "ffnn": ["FourKMLP", "SixKMLP", "EightKMLP", "TenKMLP", "TwelveKMLP"],
    "cnn": ["FourKCNN", "SixKCNN", "EightKCNN", "TenKCNN", "TwelveKCNN"],
    "ae": ["FourKAE", "SixKAE", "EightKAE", "TenKAE", "TwelveKAE"],
}

FAMILY_SNAPSHOT_RUN = {
    "ffnn": snapshot_run_a,
    "cnn": snapshot_run_b,
    "ae": snapshot_run_c,
}

# This is just here so my LSP doesn't scream at me
_MODEL_CLASS_REGISTRY = {
    "FourKMLP": FourKMLP,
    "SixKMLP": SixKMLP,
    "EightKMLP": EightKMLP,
    "TenKMLP": TenKMLP,
    "TwelveKMLP": TwelveKMLP,
    "FourKCNN": FourKCNN,
    "SixKCNN": SixKCNN,
    "EightKCNN": EightKCNN,
    "TenKCNN": TenKCNN,
    "TwelveKCNN": TwelveKCNN,
    "FourKAE": FourKAE,
    "SixKAE": SixKAE,
    "EightKAE": EightKAE,
    "TenKAE": TenKAE,
    "TwelveKAE": TwelveKAE,
}


def _train_at(miner_snaps, window_size, at_cfg, device):
    """Same function as the Experiment1a.py"""
    if len(miner_snaps) < window_size * 3:
        return None, None, None, None

    train_data, val_data, test_data = split_data(miner_snaps)
    if len(val_data) < window_size or len(test_data) < window_size:
        return None, None, None, None

    norm_mean = train_data.mean(axis=0)
    norm_std = train_data.std(axis=0) + 1e-8

    def norm(x):
        """Same function as the Experiment1a.py"""
        return (x - norm_mean) / norm_std

    train_loader = get_dataloader(norm(train_data), window_size, at_cfg["batch_size"])
    val_loader = get_dataloader(
        norm(val_data), window_size, at_cfg["batch_size"], shuffle=False
    )

    model = ATModel(
        d_input=miner_snaps.shape[1],
        d_model=at_cfg["d_model"],
        n_heads=at_cfg["n_heads"],
        d_ff=at_cfg["d_ff"],
        n_layers=at_cfg["n_layers"],
    ).to(device)

    at_train.train(
        model,
        train_loader,
        val_loader,
        n_epochs=at_cfg["n_epochs"],
        lam=at_cfg["lam"],
        lr=at_cfg["lr"],
        patience=at_cfg["patience"],
        device=device,
    )

    val_scores = at_train.get_window_scores(model, val_loader, device)
    val_timeline = at_train.windows_to_timeline(val_scores, len(val_data))
    threshold = at_train.get_threshold(val_timeline, r=at_cfg["r"])

    return model, threshold, norm_mean, norm_std


def _score_run(
    model, snaps, window_size, threshold, norm_mean, norm_std, at_cfg, device
):
    """Same function as the Experiment1a.py"""
    if len(snaps) < window_size:
        return None

    normalized = (snaps - norm_mean) / norm_std
    loader = get_dataloader(
        normalized, window_size, at_cfg["batch_size"], shuffle=False
    )

    scores = at_train.get_window_scores(model, loader, device)
    timeline = at_train.windows_to_timeline(scores, len(normalized))
    flagged_frac = float((timeline > threshold).float().mean().item())

    return flagged_frac


def resolve_device(device_arg):
    """Same function as the Experiment1a.py"""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")

        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(
    family,
    window_size,
    flag_frac,
    miner_snap_cfg,
    snap_cfg,
    at_cfg,
    n_benign,
    n_malicious,
    train_frac,
    out_csv,
    flip_frac=0.5,
    seed=0,
    benign_seed_base=1000,
    malicious_seed_base=2000,
    eff_signal_ratio=0.5,
    device="auto",
    # ffnn-only
    dataset_path=None,
    target_col=None,
    # cnn/ae shared
    dataset_root=None,
    image_size=32,
    max_samples=None,
):
    # Set device
    device = resolve_device(device)

    # Set experiment variables
    snapshot_run = FAMILY_SNAPSHOT_RUN[family]
    model_names = FAMILY_MODEL_NAMES[family]
    experiment_id = f"rt_{family}"

    print(f"Family: {family}  |  Device: {device}  |  seed: {seed}")

    # Start time
    script_t0 = time.perf_counter()
    t0 = time.perf_counter()

    # Load proper dataset depending on which type of model we're testing
    if family == "ffnn":
        X_all, y_all = load_full_dataset_a(dataset_path, target_col)
    elif family == "cnn":
        X_all, y_all = load_full_dataset_b(
            dataset_root, image_size=image_size, max_samples=max_samples
        )
    else:  # ae
        X_all = load_full_dataset_x(dataset_root, max_samples=max_samples)
        y_all = None

    dataset_load_time = time.perf_counter() - t0

    # Split into data and labels
    n_miner = int(len(X_all) * train_frac)  # Size of the dataset
    X_miner = X_all[:n_miner]
    X_sci = X_all
    y_miner = y_all[:n_miner] if y_all is not None else None
    y_sci = y_all if y_all is not None else None

    print(
        f"Dataset: {len(X_all)} rows  ->  miner {len(X_miner)}, scientist {len(X_sci)}  "
        f"(load took {dataset_load_time:.2f}s)"
    )

    def _base_kwargs(is_malicious, run_seed, X_data, y_data, model_name):
        """Arguments to the run function based on the model."""
        if family == "ffnn":
            return dict(
                data_path=dataset_path,
                target_col=target_col,
                is_malicious=is_malicious,
                seed=run_seed,
                flip_frac=flip_frac,
                X_data=X_data,
                y_data=y_data,
                model_class_str=model_name,
                device=device,
            )

        if family == "cnn":
            return dict(
                data_root=dataset_root,
                is_malicious=is_malicious,
                seed=run_seed,
                flip_frac=flip_frac,
                X_data=X_data,
                y_data=y_data,
                model_class_str=model_name,
                image_size=image_size,
                device=device,
            )

        # Else it's the auto-encoder (ae)
        return dict(
            data_root=dataset_root,
            is_malicious=is_malicious,
            seed=run_seed,
            flip_frac=flip_frac,
            X_data=X_data,
            model_class_str=model_name,
            device=device,
        )

    # Containers
    all_results = []
    grand_miner_time = grand_scientist_time = grand_at_time = 0.0

    # Iterate through the different models
    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        model_t0 = time.perf_counter()
        t0 = time.perf_counter()

        miner_kwargs = _base_kwargs(False, seed, X_miner, y_miner, model_name)
        miner_kwargs.update(
            out_csv=None,
            batch_size=miner_snap_cfg["batch_size"],
            n_epochs=miner_snap_cfg["n_epochs"],
            lr=miner_snap_cfg["lr"],
        )

        # Get snapshots out of the miner
        miner_snaps = snapshot_run(**miner_kwargs)

        miner_time = time.perf_counter() - t0

        print(f"  [Miner] {len(miner_snaps)} snapshots  ({miner_time:.2f}s)")

        # Train the anomaly detector
        t0 = time.perf_counter()
        at_model, threshold, norm_mean, norm_std = _train_at(
            miner_snaps, window_size, at_cfg, device
        )
        at_train_time = time.perf_counter() - t0

        if at_model is None:
            raise RuntimeError(
                f"need >= {window_size * 3} miner snapshots (got {len(miner_snaps)}). "
                "Increase --miner-epochs."
            )

        print(f"  [AT] trained ({at_train_time:.2f}s), threshold={threshold:.4f}")

        # Alex's stuff again
        miner_eff_signal = miner_snaps[:, EFF_SIGNAL_COL]
        eff_signal_mean = float(miner_eff_signal.mean())
        eff_signal_cutoff = eff_signal_mean * eff_signal_ratio

        scientist_time_total = 0.0
        at_score_time_total = 0.0
        run_records = []

        def _scientist_run(is_malicious, run_seed, label, actual):
            """Train scientist models and get the snapshots."""
            nonlocal scientist_time_total, at_score_time_total

            t_train0 = time.perf_counter()
            kwargs = _base_kwargs(is_malicious, run_seed, X_sci, y_sci, model_name)
            kwargs.update(
                out_csv=None,
                batch_size=snap_cfg["batch_size"],
                n_epochs=snap_cfg["n_epochs"],
                lr=snap_cfg["lr"],
            )

            snaps = snapshot_run(**kwargs)
            train_time = time.perf_counter() - t_train0
            scientist_time_total += train_time

            t_score0 = time.perf_counter()
            frac = _score_run(
                at_model,
                snaps,
                window_size,
                threshold,
                norm_mean,
                norm_std,
                at_cfg,
                device,
            )

            score_time = time.perf_counter() - t_score0
            at_score_time_total += score_time

            eff_mean = float(snaps[:, EFF_SIGNAL_COL].mean())
            at_pred = (1 if frac > flag_frac else 0) if frac is not None else -1
            eff_pred = 1 if eff_mean < eff_signal_cutoff else 0
            pred = (
                at_pred
                if at_pred < 0
                else (1 if (at_pred == 1 or eff_pred == 1) else 0)
            )

            run_records.append({"actual": actual, "predicted": pred})

            print(
                f"  Scientist {label:8s}: train={train_time:.2f}s  score={score_time:.2f}s  "
                f"-> {'BAD' if pred == 1 else 'ok'}"
            )

        # Train and collect regular scientist snapshots
        for i in range(n_benign):
            _scientist_run(False, benign_seed_base + i, f"ben_{i}", 0)

        # Train and collect malicious scientist snapshots
        for i in range(n_malicious):
            _scientist_run(True, malicious_seed_base + i, f"mal_{i}", 1)

        # Compute total time
        at_time_total = at_train_time + at_score_time_total

        # Compute metrics
        tp = sum(1 for r in run_records if r["actual"] == 1 and r["predicted"] == 1)
        fp = sum(1 for r in run_records if r["actual"] == 0 and r["predicted"] == 1)
        tn = sum(1 for r in run_records if r["actual"] == 0 and r["predicted"] == 0)
        fn = sum(1 for r in run_records if r["actual"] == 1 and r["predicted"] == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        total_time = miner_time + scientist_time_total + at_time_total
        model_wall_time = time.perf_counter() - model_t0

        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
        print(
            f"  Runtime: miner={miner_time:.2f}s  "
            f"scientist(sum of {n_benign + n_malicious})={scientist_time_total:.2f}s  "
            f"AT(train+score)={at_time_total:.2f}s  total={total_time:.2f}s  "
            f"(wall-clock check: {model_wall_time:.2f}s)"
        )

        grand_miner_time += miner_time
        grand_scientist_time += scientist_time_total
        grand_at_time += at_time_total

        all_results.append(
            {
                "experiment": experiment_id,
                "seed": seed,
                "model_class": model_name,
                "family": family,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "miner_time_sec": round(miner_time, 4),
                "scientist_time_sec": round(scientist_time_total, 4),
                "at_time_sec": round(at_time_total, 4),
                "total_time_sec": round(total_time, 4),
                "model_wall_time_sec": round(model_wall_time, 4),
            }
        )

    # Compute overall totals for metrics and times
    grand_total_time = grand_miner_time + grand_scientist_time + grand_at_time
    grand_tp = sum(r["tp"] for r in all_results)
    grand_fp = sum(r["fp"] for r in all_results)
    grand_tn = sum(r["tn"] for r in all_results)
    grand_fn = sum(r["fn"] for r in all_results)
    grand_precision = (
        grand_tp / (grand_tp + grand_fp) if (grand_tp + grand_fp) > 0 else 0.0
    )

    grand_recall = (
        grand_tp / (grand_tp + grand_fn) if (grand_tp + grand_fn) > 0 else 0.0
    )

    grand_f1 = (
        2 * grand_precision * grand_recall / (grand_precision + grand_recall)
        if (grand_precision + grand_recall) > 0
        else 0.0
    )

    script_total_time = time.perf_counter() - script_t0

    all_results.append(
        {
            "experiment": experiment_id,
            "seed": seed,
            "model_class": "ALL",
            "family": family,
            "tp": grand_tp,
            "fp": grand_fp,
            "tn": grand_tn,
            "fn": grand_fn,
            "precision": round(grand_precision, 4),
            "recall": round(grand_recall, 4),
            "f1": round(grand_f1, 4),
            "miner_time_sec": round(grand_miner_time, 4),
            "scientist_time_sec": round(grand_scientist_time, 4),
            "at_time_sec": round(grand_at_time, 4),
            "total_time_sec": round(grand_total_time, 4),
            "model_wall_time_sec": round(script_total_time, 4),
        }
    )

    print(
        f"\n=== TOTAL RUNTIME (sum across all model sizes) ===\n"
        f"  miner:     {grand_miner_time:10.2f} s\n"
        f"  scientist: {grand_scientist_time:10.2f} s\n"
        f"  AT model:  {grand_at_time:10.2f} s\n"
        f"  TOTAL:     {grand_total_time:10.2f} s\n"
        f"  (dataset load: {dataset_load_time:.2f}s, full script wall-clock: {script_total_time:.2f}s)"
    )

    # Output to files
    if out_csv and all_results:
        fields = [
            "experiment",
            "seed",
            "model_class",
            "family",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1",
            "miner_time_sec",
            "scientist_time_sec",
            "at_time_sec",
            "total_time_sec",
            "model_wall_time_sec",
        ]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nSaved results -> {out_csv}")

    return all_results


def parse_args():
    """Defines the CLI args and parses them."""
    p = argparse.ArgumentParser(
        description="SPECTRA runtime profiling (FFNN, CNN, or AE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True, choices=["ffnn", "cnn", "ae"])
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for numpy/torch. Seeds the miner phase directly and "
        "offsets the benign/malicious scientist seed ranges.",
    )

    p.add_argument(
        "--flip-frac",
        type=float,
        default=0.5,
        help="Fraction of gradient entries sign-flipped for malicious runs "
        "(single value -- this script sweeps model size, not flip_frac)",
    )

    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--flag-frac", type=float, default=0.1)
    p.add_argument("--n-benign", type=int, default=10)
    p.add_argument("--n-malicious", type=int, default=10)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out-csv", default="experimentRuntime_results.csv")
    p.add_argument("--eff-signal-ratio", default=0.5, type=float)

    fg = p.add_argument_group("FFNN dataset (--model ffnn)")
    fg.add_argument("--dataset", help="Path to CSV dataset")
    fg.add_argument("--target-col", help="Target column name")

    dg = p.add_argument_group("CNN/AE dataset (--model cnn or --model ae)")
    dg.add_argument(
        "--dataset-root",
        help="Folder with Cat/ and Dog/ subfolders (CNN) or MNIST cache dir (AE)",
    )

    dg.add_argument("--image-size", type=int, default=32, help="CNN only")
    dg.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on samples loaded into memory (CNN default 1500, AE default 3000)",
    )

    mg = p.add_argument_group("miner snapshot trainer")
    mg.add_argument("--miner-batch-size", type=int, default=64)
    mg.add_argument("--miner-epochs", type=int, default=25)
    mg.add_argument("--miner-lr", type=float, default=1e-3)

    sg = p.add_argument_group("scientist snapshot trainer")
    sg.add_argument("--snap-batch-size", type=int, default=64)
    sg.add_argument("--snap-epochs", type=int, default=25)
    sg.add_argument("--snap-lr", type=float, default=1e-3)

    at = p.add_argument_group("anomaly transformer")
    at.add_argument("--at-d-model", type=int, default=64)
    at.add_argument("--at-n-heads", type=int, default=4)
    at.add_argument("--at-d-ff", type=int, default=128)
    at.add_argument("--at-n-layers", type=int, default=2)
    at.add_argument("--at-epochs", type=int, default=15)
    at.add_argument("--at-patience", type=int, default=5)
    at.add_argument("--at-lam", type=float, default=3.0)
    at.add_argument("--at-lr", type=float, default=1e-4)
    at.add_argument("--at-r", type=float, default=0.01)
    at.add_argument("--at-batch-size", type=int, default=32)

    args = p.parse_args()

    if args.model == "ffnn" and (not args.dataset or not args.target_col):
        p.error("--model ffnn requires --dataset and --target-col")

    if args.model in ("cnn", "ae") and not args.dataset_root:
        p.error(f"--model {args.model} requires --dataset-root")

    if args.max_samples is None:
        args.max_samples = 1500 if args.model == "cnn" else 3000  # unused for ffnn

    return args


if __name__ == "__main__":
    args = parse_args()  # Parse CLI args

    # Define configs
    miner_snap_cfg = {
        "batch_size": args.miner_batch_size,
        "n_epochs": args.miner_epochs,
        "lr": args.miner_lr,
    }

    snap_cfg = {
        "batch_size": args.snap_batch_size,
        "n_epochs": args.snap_epochs,
        "lr": args.snap_lr,
    }

    at_cfg = {
        "d_model": args.at_d_model,
        "n_heads": args.at_n_heads,
        "d_ff": args.at_d_ff,
        "n_layers": args.at_n_layers,
        "n_epochs": args.at_epochs,
        "patience": args.at_patience,
        "lam": args.at_lam,
        "lr": args.at_lr,
        "r": args.at_r,
        "batch_size": args.at_batch_size,
    }

    # Compute seeds
    benign_seed_base = 1000 + args.seed * 100
    malicious_seed_base = 2000 + args.seed * 100

    # Run the experiment
    run_experiment(
        family=args.model,
        window_size=args.window_size,
        flag_frac=args.flag_frac,
        miner_snap_cfg=miner_snap_cfg,
        snap_cfg=snap_cfg,
        at_cfg=at_cfg,
        n_benign=args.n_benign,
        n_malicious=args.n_malicious,
        train_frac=args.train_frac,
        out_csv=args.out_csv,
        flip_frac=args.flip_frac,
        seed=args.seed,
        benign_seed_base=benign_seed_base,
        malicious_seed_base=malicious_seed_base,
        device=args.device,
        dataset_path=os.path.abspath(args.dataset) if args.dataset else None,
        target_col=args.target_col,
        dataset_root=os.path.abspath(args.dataset_root) if args.dataset_root else None,
        image_size=args.image_size,
        max_samples=args.max_samples,
        eff_signal_ratio=args.eff_signal_ratio,
    )
