"""
Experiment 1 – SPECTRA accuracy experiment (improved design).

Dataset split
-------------
  30 % → miner training set
  100 % → scientist test set

Miner phase
-----------
  Train a SimpleMLP on the miner set, collecting per-batch weight / gradient
  snapshots.  These snapshots are used to train an Anomaly Transformer (AT)
  that learns what *normal* training dynamics look like.

Scientist phase  (20 independent runs)
--------------------------------------
  10 benign runs   – standard training on the scientist set, unique seed each.
  10 malicious runs – Byzantine gradient attack (random 50 % sign-flip),
                      unique seed each.

Detection
---------
  The trained AT is applied to each scientist run's snapshot sequence.
  If the percentage of the flagged timesteps > --flag-frac, the run is predicted
  malicious.  TP / FP / TN / FN, precision, recall, and F1 are
  computed per input-vector-size.

Sweep
-----
  The full pipeline (miner → AT training → 20 scientist runs → metrics) is
  repeated for each input-vector size in {32, 64, 96, 128, 160}.
  The input-vector size is the number of features extracted from each model
  snapshot (= d_input of the AT).

Usage
-----
  python Experiment1a.py --dataset data.csv --target-col Placement
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AT_DIR = os.path.join(_HERE, "..")
_AT_PKG_DIR = os.path.join(_AT_DIR, "AnomalyTransformer")
sys.path.insert(0, _AT_DIR)
sys.path.insert(0, _AT_PKG_DIR)
sys.path.insert(0, _HERE)

from snapshot_trainer_a import (
    EightKMLP,
    FourKMLP,
    SixKMLP,
    TenKMLP,
    TwelveKMLP,
    load_full_dataset,
)
from snapshot_trainer_a import run as snapshot_run

from AnomalyTransformer import train as at_train
from AnomalyTransformer.AnomalyAttention import AnomalyTransformer as ATModel
from AnomalyTransformer.dataset import get_dataloader, split_data

MODEL_CLASSES = [FourKMLP, SixKMLP, EightKMLP, TenKMLP, TwelveKMLP]
EFF_SIGNAL_COL = 2

# ---------------------------------------------------------------------------
# AT helpers
# ---------------------------------------------------------------------------


def _train_at(miner_snaps, window_size, at_cfg, device):
    """Train AT on miner snapshots.

    Returns (model, threshold, norm_mean, norm_std), or (None,)*4 if the
    snapshot sequence is too short for the given window_size.
    """
    if len(miner_snaps) < window_size * 3:
        return None, None, None, None

    train_data, val_data, test_data = split_data(miner_snaps)
    if len(val_data) < window_size or len(test_data) < window_size:
        return None, None, None, None

    norm_mean = train_data.mean(axis=0)
    norm_std = train_data.std(axis=0) + 1e-8

    def norm(x):
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
    """Return fraction of timesteps flagged by the AT (or None if too short)."""
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def resolve_device(device_arg):
    """Resolve the --device CLI choice ('auto' | 'cpu' | 'cuda') to a torch.device."""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(
    dataset_path,
    target_col,
    window_size,
    flag_frac,
    miner_snap_cfg,
    snap_cfg,
    at_cfg,
    n_benign,
    n_malicious,
    train_frac,
    out_csv,
    out_png,
    flip_frac=0.5,
    device="auto",
    eff_signal_ratio=0.3,
    base_seed=42,
    benign_seed_base=1000,
    malicious_seed_base=2000,
):
    device = resolve_device(device)
    print(f"Device: {device}")

    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    print(f"Seed set to: {base_seed}")

    # ── Dataset split ────────────────────────────────────────────────────────
    X_all, y_all = load_full_dataset(dataset_path, target_col)
    n_miner = int(len(X_all) * train_frac)
    X_miner, y_miner = X_all[:n_miner], y_all[:n_miner]
    X_sci, y_sci = X_all, y_all

    print(
        f"Dataset: {len(X_all)} rows  →  miner {len(X_miner)}, "
        f"scientist {len(X_sci)}"
    )

    all_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for model_class in MODEL_CLASSES:
            model_class_str = model_class.__name__

            # ── Miner phase ──────────────────────────────────────────────────
            print("  [Miner] Training model and collecting snapshots...")
            miner_csv = os.path.join(tmpdir, f"miner_{model_class_str}.csv")
            miner_snaps = snapshot_run(
                data_path=dataset_path,
                target_col=target_col,
                out_csv=miner_csv,
                batch_size=miner_snap_cfg["batch_size"],
                n_epochs=miner_snap_cfg["n_epochs"],
                lr=miner_snap_cfg["lr"],
                is_malicious=False,
                seed=base_seed,
                flip_frac=flip_frac,
                X_data=X_miner,
                y_data=y_miner,
                model_class_str=model_class_str,
            )

            print(
                f"  [Miner] {len(miner_snaps)} snapshots  (shape {miner_snaps.shape})"
            )

            # ── AT training phase ────────────────────────────────────────────
            print("  [AT] Training Anomaly Transformer on miner snapshots...")
            at_model, threshold, norm_mean, norm_std = _train_at(
                miner_snaps, window_size, at_cfg, device
            )
            if at_model is None:
                raise RuntimeError(
                    f"need ≥ {window_size * 3} miner snapshots (got {len(miner_snaps)}). "
                    "Increase --miner-epochs."
                )
            print(f"  [AT] Threshold: {threshold:.4f}")

            miner_eff_signal = miner_snaps[:, EFF_SIGNAL_COL]
            eff_signal_mean = float(miner_eff_signal.mean())
            eff_signal_cutoff = eff_signal_mean * eff_signal_ratio
            print(
                f"  [EffSignal] miner mean={eff_signal_mean:.5f}  "
                f"cutoff={eff_signal_cutoff:.5f} (eff_signal_ratio={eff_signal_ratio})"
            )

            # ── Scientist phase ──────────────────────────────────────────────
            run_records = []

            for i in range(n_benign):
                seed = base_seed + benign_seed_base + i
                sci_csv = os.path.join(tmpdir, f"sci_ben_{i}.csv")
                sci_snaps = snapshot_run(
                    data_path=dataset_path,
                    target_col=target_col,
                    out_csv=sci_csv,
                    batch_size=snap_cfg["batch_size"],
                    n_epochs=snap_cfg["n_epochs"],
                    lr=snap_cfg["lr"],
                    is_malicious=False,
                    flip_frac=flip_frac,
                    seed=seed,
                    X_data=X_sci,
                    y_data=y_sci,
                    device=device,
                    model_class_str=model_class_str,
                )

                frac = _score_run(
                    at_model,
                    sci_snaps,
                    window_size,
                    threshold,
                    norm_mean,
                    norm_std,
                    at_cfg,
                    device,
                )
                eff_mean = float(sci_snaps[:, EFF_SIGNAL_COL].mean())
                at_pred = (1 if frac > flag_frac else 0) if frac is not None else -1
                eff_pred = 1 if eff_mean < eff_signal_cutoff else 0
                pred = (
                    at_pred
                    if at_pred < 0
                    else (1 if (at_pred == 1 or eff_pred == 1) else 0)
                )
                run_records.append({"actual": 0, "predicted": pred, "frac": frac})
                frac_str = f"{frac:.3f}" if frac is not None else "N/A"
                print(
                    f"  Benign   run {i+1:2d}: flagged={frac_str}  → {'BAD' if pred == 1 else 'ok'}"
                )

            for i in range(n_malicious):
                seed = base_seed + malicious_seed_base + i
                sci_csv = os.path.join(tmpdir, f"sci_mal_{i}.csv")
                sci_snaps = snapshot_run(
                    data_path=dataset_path,
                    target_col=target_col,
                    out_csv=sci_csv,
                    batch_size=snap_cfg["batch_size"],
                    n_epochs=snap_cfg["n_epochs"],
                    lr=snap_cfg["lr"],
                    is_malicious=True,
                    flip_frac=flip_frac,
                    seed=seed,
                    X_data=X_sci,
                    y_data=y_sci,
                    device=device,
                    model_class_str=model_class_str,
                )

                frac = _score_run(
                    at_model,
                    sci_snaps,
                    window_size,
                    threshold,
                    norm_mean,
                    norm_std,
                    at_cfg,
                    device,
                )

                eff_mean = float(sci_snaps[:, EFF_SIGNAL_COL].mean())
                at_pred = (1 if frac > flag_frac else 0) if frac is not None else -1
                eff_pred = 1 if eff_mean < eff_signal_cutoff else 0
                pred = (
                    at_pred
                    if at_pred < 0
                    else (1 if (at_pred == 1 or eff_pred == 1) else 0)
                )
                run_records.append({"actual": 1, "predicted": pred, "frac": frac})
                frac_str = f"{frac:.3f}" if frac is not None else "N/A"
                print(
                    f"  Malicious run {i+1:2d}: flagged={frac_str}  → {'BAD' if pred == 1 else 'ok'}"
                )

            # ── Metrics ──────────────────────────────────────────────────────
            valid = [r for r in run_records if r["predicted"] >= 0]
            tp = sum(1 for r in valid if r["actual"] == 1 and r["predicted"] == 1)
            fp = sum(1 for r in valid if r["actual"] == 0 and r["predicted"] == 1)
            tn = sum(1 for r in valid if r["actual"] == 0 and r["predicted"] == 0)
            fn = sum(1 for r in valid if r["actual"] == 1 and r["predicted"] == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            print(f"\n  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
            print(f"  Precision={precision:.3f}  Recall={recall:.3f}  " f"F1={f1:.3f}")

            all_results.append(
                {
                    "model_class": model_class_str,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )

    # ── Outputs ──────────────────────────────────────────────────────────────
    if out_csv and all_results:
        fields = ["model_class", "tp", "fp", "tn", "fn", "precision", "recall", "f1"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved results → {out_csv}")

    if out_png and all_results:
        sizes = ["4k", "6k", "8k", "10k", "12k"]
        f1s = [r["f1"] for r in all_results]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sizes, f1s, "b-o", label="F1 score")
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Score")
        ax.set_title(
            f"Experiment 1: SPECTRA Detection vs FFNN Model Size with {flip_frac * 100.0}% of signs flipped"
        )
        ax.set_xticks(sizes)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_png)
        print(f"Saved plot → {out_png}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="SPECTRA Experiment 0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seed",
        required=True,
        help="Base seed to use for the entire experiment.",
        type=int,
        default=42,
    )
    p.add_argument("--dataset", required=True, help="Path to CSV dataset")
    p.add_argument("--target-col", required=True, help="Target column name")
    p.add_argument(
        "--train-frac", type=float, default=0.3, help="Fraction of data given to miners"
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="AT sliding-window length in timesteps",
    )
    p.add_argument(
        "--flag-frac",
        type=float,
        default=0.1,
        help="Flagged-fraction threshold to classify a run as malicious",
    )
    p.add_argument("--n-benign", type=int, default=10)
    p.add_argument("--n-malicious", type=int, default=10)
    p.add_argument(
        "--benign-seed-base",
        type=int,
        default=1000,
        help="Base seed for benign scientist runs (seed = base + run index)",
    )
    p.add_argument(
        "--malicious-seed-base",
        type=int,
        default=2000,
        help="Base seed for malicious scientist runs (seed = base + run index)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to train on. 'cuda' forces GPU (errors if unavailable), "
        "'auto' uses GPU when available",
    )
    p.add_argument("--out", default="experiment10_results.png")
    p.add_argument("--out-csv", default="experiment10_results.csv")
    p.add_argument("--eff-signal-ratio", default=0.3, type=float)

    mg = p.add_argument_group("miner snapshot trainer")
    mg.add_argument("--miner-batch-size", type=int, default=64)
    mg.add_argument(
        "--miner-epochs",
        type=int,
        default=25,
        help="More epochs → more AT training data",
    )
    mg.add_argument("--miner-lr", type=float, default=1e-3)
    mg.add_argument("--miner-hidden", type=int, default=64)

    sg = p.add_argument_group("scientist snapshot trainer")
    sg.add_argument("--snap-batch-size", type=int, default=64)
    sg.add_argument("--snap-epochs", type=int, default=25)
    sg.add_argument("--snap-lr", type=float, default=1e-3)
    sg.add_argument("--snap-hidden", type=int, default=64)

    ag = p.add_argument_group("anomaly transformer")
    ag.add_argument("--at-d-model", type=int, default=64)
    ag.add_argument("--at-n-heads", type=int, default=4)
    ag.add_argument("--at-d-ff", type=int, default=128)
    ag.add_argument("--at-n-layers", type=int, default=2)
    ag.add_argument("--at-epochs", type=int, default=15)
    ag.add_argument("--at-patience", type=int, default=5)
    ag.add_argument("--at-lam", type=float, default=3.0)
    ag.add_argument("--at-lr", type=float, default=1e-4)
    ag.add_argument("--at-r", type=float, default=0.01)
    ag.add_argument("--at-batch-size", type=int, default=32)

    return p.parse_args()


if __name__ == "__main__":
    print("Started python")
    args = parse_args()

    miner_snap_cfg = {
        "batch_size": args.miner_batch_size,
        "n_epochs": args.miner_epochs,
        "lr": args.miner_lr,
        "hidden": args.miner_hidden,
    }

    snap_cfg = {
        "batch_size": args.snap_batch_size,
        "n_epochs": args.snap_epochs,
        "lr": args.snap_lr,
        "hidden": args.snap_hidden,
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

    for flip_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        csv_filename = Path(args.out_csv)
        csv_name = csv_filename.with_stem(csv_filename.stem + f"_{flip_frac}")
        png_filename = Path(args.out)
        png_name = png_filename.with_stem(png_filename.stem + f"_{flip_frac}")

        print(
            f"Running an Accuracy Experiment with {flip_frac * 100.0}% of signs flipped."
        )

        print(f"\tSaving experiment results in {csv_name} & {png_name}.")

        run_experiment(
            dataset_path=os.path.abspath(args.dataset),
            target_col=args.target_col,
            window_size=args.window_size,
            flag_frac=args.flag_frac,
            miner_snap_cfg=miner_snap_cfg,
            snap_cfg=snap_cfg,
            at_cfg=at_cfg,
            n_benign=args.n_benign,
            n_malicious=args.n_malicious,
            train_frac=args.train_frac,
            out_csv=csv_name,
            out_png=png_name,
            flip_frac=flip_frac,
            device=args.device,
            base_seed=args.seed,
            benign_seed_base=args.benign_seed_base,
            malicious_seed_base=args.malicious_seed_base,
            eff_signal_ratio=args.eff_signal_ratio,
        )
