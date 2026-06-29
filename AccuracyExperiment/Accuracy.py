"""
Usage:
    python Accuracy.py --dataset data.csv --target-col Placement --is-bad 0
    python Accuracy.py --dataset data.csv --target-col Placement --is-bad 1
"""

import sys
import os
import csv
import argparse
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Allow imports from AnomalyTransformer/ and this directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_AT_DIR = os.path.join(_HERE, "..")
_AT_PKG_DIR = os.path.join(_AT_DIR, "AnomalyTransformer")
sys.path.insert(0, _AT_DIR)
sys.path.insert(0, _AT_PKG_DIR)
sys.path.insert(0, _HERE)

from snapshot_trainer import run as snapshot_run
from AnomalyTransformer.AnomalyAttention import AnomalyTransformer as ATModel
from AnomalyTransformer.dataset import load_csv, split_data, normalize_splits, make_loaders
from AnomalyTransformer import train as at_train


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _score_snapshots(snapshot_csv, window_size, device, at_cfg):
    """
    Train the Anomaly Transformer on a snapshot CSV and return the fraction
    of test-set timesteps flagged as anomalous.

    Returns None if the snapshot is too short for the given window_size.
    """
    data, _ = load_csv(snapshot_csv, skip_cols=1, has_header=True)

    if len(data) < window_size * 3:
        return None

    train_data, val_data, test_data = split_data(data)
    if len(val_data) < window_size or len(test_data) < window_size:
        return None

    train_data, val_data, test_data = normalize_splits(train_data, val_data, test_data)
    train_loader, val_loader, test_loader = make_loaders(
        train_data, val_data, test_data, window_size, batch_size=at_cfg["batch_size"]
    )

    model = ATModel(
        d_input=data.shape[1],
        d_model=at_cfg["d_model"],
        n_heads=at_cfg["n_heads"],
        d_ff=at_cfg["d_ff"],
        n_layers=at_cfg["n_layers"],
    ).to(device)

    at_train.train(
        model, train_loader, val_loader,
        n_epochs=at_cfg["n_epochs"],
        lam=at_cfg["lam"],
        lr=at_cfg["lr"],
        patience=at_cfg["patience"],
        device=device,
    )

    val_scores = at_train.get_window_scores(model, val_loader, device)
    val_timeline = at_train.windows_to_timeline(val_scores, len(val_data))
    threshold = at_train.get_threshold(val_timeline, r=at_cfg["r"])

    test_scores = at_train.get_window_scores(model, test_loader, device)
    test_timeline = at_train.windows_to_timeline(test_scores, len(test_data))
    flagged = (test_timeline > threshold).numpy()

    return float(flagged.mean())


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dataset_path, target_col, is_bad, window_sizes, flag_frac, snap_cfg, at_cfg, out_png, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {os.path.basename(dataset_path)}  ({'bad' if is_bad else 'clean'})\n")

    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_csv = os.path.join(tmpdir, "snapshots.csv")
        try:
            snapshot_run(
                data_path=dataset_path,
                target_col=target_col,
                out_csv=snapshot_csv,
                batch_size=snap_cfg["batch_size"],
                n_epochs=snap_cfg["n_epochs"],
                lr=snap_cfg["lr"],
                hidden=snap_cfg["hidden"],
            )
        except Exception as e:
            print(f"snapshot_trainer failed: {e}")
            return []

        for window_size in window_sizes:
            print(f"{'='*50}")
            print(f"Window size: {window_size}")
            print(f"{'='*50}")

            frac = _score_snapshots(snapshot_csv, window_size, device, at_cfg)

            if frac is None:
                print(f"  Skipped — not enough snapshots for window_size={window_size}")
                continue

            pred = int(frac > flag_frac)
            correct = int(pred == is_bad)
            status = "correct" if correct else "WRONG"
            print(f"  Flagged fraction: {frac:.3f}  ->  predicted {'BAD' if pred else 'clean'}  ({status})")

            rows.append({
                "window_size":  window_size,
                "flagged_frac": round(frac, 4),
                "predicted":    pred,
                "actual":       is_bad,
                "correct":      correct,
            })

    if not rows:
        print("No valid results.")
        return rows

    n_correct = sum(r["correct"] for r in rows)
    print(f"\nAccuracy: {n_correct}/{len(rows)} window sizes correct")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["window_size", "flagged_frac", "predicted", "actual", "correct"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {out_csv}")

    if out_png:
        xs    = [r["window_size"]  for r in rows]
        fracs = [r["flagged_frac"] for r in rows]
        colors = ["green" if r["correct"] else "red" for r in rows]

        plt.figure(figsize=(8, 5))
        plt.bar(xs, fracs, color=colors, alpha=0.7, width=min(xs[1] - xs[0] if len(xs) > 1 else 20, 20))
        plt.axhline(flag_frac, linestyle="--", color="black", label=f"threshold ({flag_frac})")
        plt.xlabel("Window Size")
        plt.ylabel("Flagged Fraction")
        actual_label = "bad" if is_bad else "clean"
        plt.title(f"Anomaly Transformer — {os.path.basename(dataset_path)} ({actual_label})")
        plt.xticks(xs)
        plt.ylim(0, max(max(fracs) * 1.2, flag_frac * 1.5))
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(out_png)
        print(f"Saved plot to {out_png}")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SPECTRA accuracy experiment")
    p.add_argument("--dataset",       required=True,
                   help="Path to the dataset CSV")
    p.add_argument("--target-col",    required=True,
                   help="Target column in the dataset")
    p.add_argument("--is-bad",        type=int, default=0, choices=[0, 1],
                   help="Ground truth: 1 if dataset is dirty/bad, 0 if clean (default: 0)")
    p.add_argument("--window-sizes",  type=int, nargs="+", default=[32, 64, 96, 128, 160],
                   metavar="W",       help="Window sizes to sweep (default: 32 64 96 128 160)")
    p.add_argument("--flag-frac",     type=float, default=0.3,
                   help="Flagged fraction threshold to call a run bad (default: 0.3)")
    p.add_argument("--out",           default="accuracy_results.png",
                   help="Output plot path")
    p.add_argument("--out-csv",       default="accuracy_results.csv",
                   help="Output results table path")

    sg = p.add_argument_group("snapshot trainer")
    sg.add_argument("--snap-batch-size", type=int,   default=64)
    sg.add_argument("--snap-epochs",     type=int,   default=10)
    sg.add_argument("--snap-lr",         type=float, default=1e-3)
    sg.add_argument("--snap-hidden",     type=int,   default=64)

    ag = p.add_argument_group("anomaly transformer")
    ag.add_argument("--at-d-model",    type=int,   default=64)
    ag.add_argument("--at-n-heads",    type=int,   default=4)
    ag.add_argument("--at-d-ff",       type=int,   default=128)
    ag.add_argument("--at-n-layers",   type=int,   default=2)
    ag.add_argument("--at-epochs",     type=int,   default=15)
    ag.add_argument("--at-patience",   type=int,   default=5)
    ag.add_argument("--at-lam",        type=float, default=3.0)
    ag.add_argument("--at-lr",         type=float, default=1e-4)
    ag.add_argument("--at-r",          type=float, default=0.02,
                    help="Top-r fraction of val scores used to set threshold (default: 0.02)")
    ag.add_argument("--at-batch-size", type=int,   default=32)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    snap_cfg = {
        "batch_size": args.snap_batch_size,
        "n_epochs":   args.snap_epochs,
        "lr":         args.snap_lr,
        "hidden":     args.snap_hidden,
    }
    at_cfg = {
        "d_model":    args.at_d_model,
        "n_heads":    args.at_n_heads,
        "d_ff":       args.at_d_ff,
        "n_layers":   args.at_n_layers,
        "n_epochs":   args.at_epochs,
        "patience":   args.at_patience,
        "lam":        args.at_lam,
        "lr":         args.at_lr,
        "r":          args.at_r,
        "batch_size": args.at_batch_size,
    }

    run_experiment(
        dataset_path=os.path.abspath(args.dataset),
        target_col=args.target_col,
        is_bad=args.is_bad,
        window_sizes=args.window_sizes,
        flag_frac=args.flag_frac,
        snap_cfg=snap_cfg,
        at_cfg=at_cfg,
        out_png=args.out,
        out_csv=args.out_csv,
    )
