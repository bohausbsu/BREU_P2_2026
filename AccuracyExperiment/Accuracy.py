"""
SPECTRA Accuracy Experiment — Miner / Scientist design

Dataset split:
  Miner    : first 30% of rows  (train set)
  Scientist: full 100% of rows  (test set)

Miner phase (once):
  Trains a SimpleMLP on its 30% slice, capturing per-batch snapshots.

Scientist phase (once, 20 runs: 10 benign + 10 malicious):
  Generates snapshots for each run on the full dataset.

Evaluation (per window size in [32, 64, 96, 128, 160]):
  Trains the Anomaly Transformer on the miner's snapshots for each window size.
  Scores all 20 scientist runs with that AT.
  Records TP/TN/FP/FN and derived metrics.

Usage:
    python Accuracy.py --dataset data.csv --target-col Placement
"""

import sys
import os
import csv
import argparse
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE   = os.path.dirname(os.path.abspath(__file__))
_AT_DIR = os.path.join(_HERE, "..")
_AT_PKG = os.path.join(_AT_DIR, "AnomalyTransformer")
sys.path.insert(0, _AT_DIR)
sys.path.insert(0, _AT_PKG)
sys.path.insert(0, _HERE)

from snapshot_trainer import run as snapshot_run
from AnomalyTransformer.AnomalyAttention import AnomalyTransformer as ATModel
from AnomalyTransformer.dataset import (
    load_csv, split_data, normalize_splits, make_loaders, get_dataloader,
)
from AnomalyTransformer import train as at_train

WINDOW_SIZES = [32, 64, 96, 128, 160]


# ---------------------------------------------------------------------------
# AT helpers
# ---------------------------------------------------------------------------

def _train_at(snapshot_csv, window_size, device, at_cfg):
    """Train an AT on miner snapshots. Returns (model, threshold, norm_mean, norm_std) or all-None."""
    data, _ = load_csv(snapshot_csv, skip_cols=1, has_header=True)
    if len(data) < window_size * 3:
        return None, None, None, None

    train_data, val_data, test_data = split_data(data)
    if len(val_data) < window_size or len(test_data) < window_size:
        return None, None, None, None

    norm_mean = train_data.mean(axis=0)
    norm_std  = train_data.std(axis=0) + 1e-8
    train_data, val_data, test_data = normalize_splits(train_data, val_data, test_data)
    train_loader, val_loader, _ = make_loaders(
        train_data, val_data, test_data, window_size, at_cfg["batch_size"]
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

    val_scores   = at_train.get_window_scores(model, val_loader, device)
    val_timeline = at_train.windows_to_timeline(val_scores, len(val_data))
    threshold    = at_train.get_threshold(val_timeline, r=at_cfg["r"])
    return model, threshold, norm_mean, norm_std


def _score_with_at(model, threshold, norm_mean, norm_std,
                   snapshot_csv, window_size, device, batch_size):
    """Score a snapshot CSV with a pre-trained AT. Returns flagged_frac or None."""
    data, _ = load_csv(snapshot_csv, skip_cols=1, has_header=True)
    if len(data) < window_size:
        return None

    data_norm = (data - norm_mean) / norm_std
    loader    = get_dataloader(data_norm, window_size, batch_size, shuffle=False)

    scores   = at_train.get_window_scores(model, loader, device)
    timeline = at_train.windows_to_timeline(scores, len(data_norm))
    flagged  = (timeline > threshold).numpy()
    return float(flagged.mean())


def _compute_metrics(tp, tn, fp, fn):
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total             if total             else 0.0
    precision = tp / (tp + fp)                if (tp + fp)         else 0.0
    recall    = tp / (tp + fn)                if (tp + fn)         else 0.0
    f1        = (2 * precision * recall
                 / (precision + recall))      if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dataset_path, target_col, window_sizes, n_runs,
                   flag_frac, snap_cfg, at_cfg, out_csv, out_png):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_benign = n_runs // 2
    n_mal    = n_runs - n_benign

    print(f"Device     : {device}")
    print(f"Dataset    : {os.path.basename(dataset_path)}")
    print(f"Runs       : {n_runs}  ({n_benign} benign + {n_mal} malicious)")
    print(f"Windows    : {window_sizes}\n")

    all_summaries = []   # one dict per window size
    all_run_rows  = []   # one dict per (window_size, run)

    for window_size in window_sizes:
        print("\n" + "=" * 60)
        print(f"WINDOW SIZE = {window_size}")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as tmpdir:

            # -------------------------------------------------------------- #
            # Phase 1 — Miner trains on 30% and produces snapshots            #
            # -------------------------------------------------------------- #
            print("  [Miner] Training on first 30% of dataset ...")
            miner_snap = os.path.join(tmpdir, "miner_snapshots.csv")
            snapshot_run(
                data_path=dataset_path, target_col=target_col,
                out_csv=miner_snap,
                start_frac=0.0, end_frac=0.30,
                is_malicious=False,
                batch_size=snap_cfg["batch_size"],
                n_epochs=snap_cfg["n_epochs"],
                lr=snap_cfg["lr"],
                hidden=snap_cfg["hidden"],
                seed=99,
            )

            # -------------------------------------------------------------- #
            # Phase 2 — Train AT on miner snapshots                           #
            # -------------------------------------------------------------- #
            print(f"  [AT] Training Anomaly Transformer (window={window_size}) ...")
            model, threshold, norm_mean, norm_std = _train_at(
                miner_snap, window_size, device, at_cfg
            )

            if model is None:
                print(f"  SKIP: not enough miner snapshots for window={window_size}")
                continue

            print(f"  [AT] Threshold: {threshold:.4f}")

            # -------------------------------------------------------------- #
            # Phase 3 — Scientist generates 20 snapshot runs on 100%          #
            # -------------------------------------------------------------- #
            print(f"  [Scientist] Generating {n_runs} runs on full dataset ...")
            scientist_runs = []

            for i in range(n_benign):
                path = os.path.join(tmpdir, f"sci_benign_{i}.csv")
                print(f"    Benign run {i+1}/{n_benign} (seed={i}) ...")
                snapshot_run(
                    data_path=dataset_path, target_col=target_col,
                    out_csv=path,
                    start_frac=0.0, end_frac=1.0,
                    is_malicious=False,
                    batch_size=snap_cfg["batch_size"],
                    n_epochs=snap_cfg["n_epochs"],
                    lr=snap_cfg["lr"],
                    hidden=snap_cfg["hidden"],
                    seed=i,
                )
                scientist_runs.append((path, 0))

            for i in range(n_mal):
                path = os.path.join(tmpdir, f"sci_malicious_{i}.csv")
                seed = n_benign + i
                print(f"    Malicious run {i+1}/{n_mal} (seed={seed}) ...")
                snapshot_run(
                    data_path=dataset_path, target_col=target_col,
                    out_csv=path,
                    start_frac=0.0, end_frac=1.0,
                    is_malicious=True,
                    poison_frac=snap_cfg.get("poison_frac", 0.5),
                    grad_noise_scale=snap_cfg.get("grad_noise_scale", 0.5),
                    batch_size=snap_cfg["batch_size"],
                    n_epochs=snap_cfg["n_epochs"],
                    lr=snap_cfg["lr"],
                    hidden=snap_cfg["hidden"],
                    seed=seed,
                )
                scientist_runs.append((path, 1))

            # -------------------------------------------------------------- #
            # Phase 4 — Score each scientist run with the AT                  #
            # -------------------------------------------------------------- #
            print(f"  [Eval] Scoring {len(scientist_runs)} runs ...")
            tp = tn = fp = fn = 0

            for run_i, (snap_path, label) in enumerate(scientist_runs):
                frac = _score_with_at(
                    model, threshold, norm_mean, norm_std,
                    snap_path, window_size, device, at_cfg["batch_size"]
                )
                if frac is None:
                    print(f"    Run {run_i:2d}: skipped (too few snapshots)")
                    continue

                predicted = int(frac > flag_frac)
                correct   = int(predicted == label)

                if   label == 1 and predicted == 1: tp += 1
                elif label == 0 and predicted == 0: tn += 1
                elif label == 0 and predicted == 1: fp += 1
                else:                               fn += 1

                role   = "malicious" if label else "benign   "
                result = "OK   " if correct else "WRONG"
                print(f"    Run {run_i:2d} ({role}): flagged={frac:.3f} "
                      f"-> {'BAD  ' if predicted else 'clean'}  [{result}]")

                all_run_rows.append({
                    "window_size":  window_size,
                    "run_id":       run_i,
                    "is_malicious": label,
                    "flagged_frac": round(frac, 4),
                    "predicted":    predicted,
                    "correct":      correct,
                })

        accuracy, precision, recall, f1 = _compute_metrics(tp, tn, fp, fn)
        print(f"\n  Results (window={window_size}): "
              f"Acc={accuracy:.3f}  Prec={precision:.3f}  "
              f"Rec={recall:.3f}  F1={f1:.3f}")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

        all_summaries.append({
            "window_size": window_size,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy":   round(accuracy,  4),
            "precision":  round(precision, 4),
            "recall":     round(recall,    4),
            "f1":         round(f1,        4),
        })

    # ---------------------------------------------------------------------- #
    # Save outputs                                                             #
    # ---------------------------------------------------------------------- #
    if out_csv and all_run_rows:
        run_fields = ["window_size", "run_id", "is_malicious",
                      "flagged_frac", "predicted", "correct"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=run_fields)
            w.writeheader()
            w.writerows(all_run_rows)

        summary_csv = out_csv.replace(".csv", "_summary.csv")
        sum_fields  = ["window_size", "tp", "tn", "fp", "fn",
                       "accuracy", "precision", "recall", "f1"]
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sum_fields)
            w.writeheader()
            w.writerows(all_summaries)

        print(f"\nSaved detailed results → {out_csv}")
        print(f"Saved summary          → {summary_csv}")

    if out_png and all_summaries:
        _plot(all_summaries, all_run_rows, flag_frac,
              os.path.basename(dataset_path), out_png)
        print(f"Saved plot             → {out_png}")

    return all_summaries


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(summaries, run_rows, flag_frac, dataset_name, out_png):
    ws      = [s["window_size"] for s in summaries]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["steelblue", "darkorange", "green", "crimson"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: metrics vs window size
    ax = axes[0]
    x  = np.arange(len(ws))
    w  = 0.2
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [s[metric] for s in summaries]
        ax.bar(x + i * w, vals, width=w, label=metric.capitalize(), color=color, alpha=0.8)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([str(s) for s in ws])
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Metrics vs Window Size\n{dataset_name}")
    ax.legend()
    ax.grid(True, axis="y")

    # Right: flagged_frac per run, grouped by window size (first window only for clarity)
    # Show a heatmap-style scatter: x=run_id, color by window_size
    ax2 = axes[1]
    cmap = plt.cm.get_cmap("tab10", len(ws))
    for j, window_size in enumerate(ws):
        rows = [r for r in run_rows if r["window_size"] == window_size]
        if not rows:
            continue
        run_ids = [r["run_id"]       for r in rows]
        fracs   = [r["flagged_frac"] for r in rows]
        labels  = [r["is_malicious"] for r in rows]
        markers = ["x" if l else "o" for l in labels]
        for rid, frac, marker in zip(run_ids, fracs, markers):
            ax2.scatter(rid, frac, marker=marker,
                        color=cmap(j), s=60, alpha=0.7,
                        label=f"w={window_size}" if rid == run_ids[0] else "")
    ax2.axhline(flag_frac, color="black", linestyle="--", linewidth=1,
                label=f"flag_frac={flag_frac}")
    ax2.set_xlabel("Run ID")
    ax2.set_ylabel("Flagged Fraction")
    ax2.set_title("Flagged Fraction per Run\n(o=benign  x=malicious)")
    handles, lbls = ax2.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = h
    ax2.legend(seen.values(), seen.keys(), fontsize=8)
    ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SPECTRA accuracy experiment — Miner/Scientist with multi-window evaluation"
    )
    p.add_argument("--dataset",      required=True)
    p.add_argument("--target-col",   required=True)
    p.add_argument("--n-runs",       type=int,   default=20,
                   help="Total scientist runs (half benign, half malicious; default 20)")
    p.add_argument("--window-sizes", type=int,   nargs="+", default=WINDOW_SIZES,
                   help="Window sizes to evaluate (default: 32 64 96 128 160)")
    p.add_argument("--flag-frac",    type=float, default=0.3,
                   help="Flagged-fraction threshold to predict malicious (default 0.3)")
    p.add_argument("--out",          default="accuracy_results.png")
    p.add_argument("--out-csv",      default="accuracy_results.csv")

    sg = p.add_argument_group("snapshot trainer")
    sg.add_argument("--snap-batch-size",    type=int,   default=32)
    sg.add_argument("--snap-epochs",        type=int,   default=10)
    sg.add_argument("--snap-lr",            type=float, default=1e-3)
    sg.add_argument("--snap-hidden",        type=int,   default=64)
    sg.add_argument("--snap-poison-frac",   type=float, default=0.8)
    sg.add_argument("--snap-grad-noise",    type=float, default=1)

    ag = p.add_argument_group("anomaly transformer")
    ag.add_argument("--at-d-model",    type=int,   default=64)
    ag.add_argument("--at-n-heads",    type=int,   default=4)
    ag.add_argument("--at-d-ff",       type=int,   default=128)
    ag.add_argument("--at-n-layers",   type=int,   default=2)
    ag.add_argument("--at-epochs",     type=int,   default=15)
    ag.add_argument("--at-patience",   type=int,   default=5)
    ag.add_argument("--at-lam",        type=float, default=3.0)
    ag.add_argument("--at-lr",         type=float, default=1e-4)
    ag.add_argument("--at-r",          type=float, default=0.1)
    ag.add_argument("--at-batch-size", type=int,   default=32)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    snap_cfg = {
        "batch_size":       args.snap_batch_size,
        "n_epochs":         args.snap_epochs,
        "lr":               args.snap_lr,
        "hidden":           args.snap_hidden,
        "poison_frac":      args.snap_poison_frac,
        "grad_noise_scale": args.snap_grad_noise,
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
        window_sizes=args.window_sizes,
        n_runs=args.n_runs,
        flag_frac=args.flag_frac,
        snap_cfg=snap_cfg,
        at_cfg=at_cfg,
        out_csv=args.out_csv,
        out_png=args.out,
    )
