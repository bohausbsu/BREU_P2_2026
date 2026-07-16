"""
Experiment 1a (memory-instrumented) - SPECTRA memory profiling (FFNN / tabular).

 Each phase below runs in its own isolated subprocess, and the peak RSS
 reported for that phase is measured *relative to that subprocess's own
 baseline RSS taken right after it starts* (i.e. after the fixed cost of
 importing torch/numpy/etc. into the fresh process, before any model- or
 dataset-specific work begins). This isolates the marginal memory the
 phase itself needed, rather than being dominated by process cold-start
 overhead.

 The totals reported are:

  miner_mem      = peak-RSS delta of the subprocess that trains the miner
                   model and collects its snapshots
  scientist_mem  = peak-RSS delta of the single subprocess that runs all
                   20 scientist trainings (10 benign + 10 malicious) back
                   to back
  at_mem         = peak-RSS delta of the subprocess that trains the Anomaly
                   Transformer on the miner snapshots AND scores all 20
                   scientist snapshot sequences against it (training +
                   inference are both "what the AT model uses")
  total_mem      = miner_mem + scientist_mem + at_mem

Usage
-----
  # For FFNN
  python ExperimentSAE.py --model EightKMLP --seed 0 --dataset data.csv --target-col Placement \
      --flag-frac 0.1 --at-r 0.01 --out-csv ffnn_mem.csv

  # For CNN
  python ExperimentSAE.py --model EightKCNN --seed 0 --dataset-root PetImages \
      --flag-frac 0.1 --at-r 0.01 --out-csv cnn_mem.csv

  # For AE
  python ExperimentSAE.py --model EightKAE --seed 0 --data-root mnist \
      --flag-frac 0.1 --at-r 0.01 --out-csv ae_mem.csv
"""

import argparse
import csv
import multiprocessing as mp
import os
import resource
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AT_DIR = os.path.join(_HERE, "..")
_AT_PKG_DIR = os.path.join(_AT_DIR, "AnomalyTransformer")
sys.path.insert(0, _AT_DIR)
sys.path.insert(0, _AT_PKG_DIR)
sys.path.insert(0, _HERE)

from snapshot_trainer_a import (EightKMLP, FourKMLP, SixKMLP, TenKMLP,
                                TwelveKMLP)
from snapshot_trainer_a import load_full_dataset as load_full_dataset_a
from snapshot_trainer_a import run as snapshot_run_a
from snapshot_trainer_b import (EightKCNN, FourKCNN, SixKCNN, TenKCNN,
                                TwelveKCNN)
from snapshot_trainer_b import load_full_dataset as load_full_dataset_b
from snapshot_trainer_b import run as snapshot_run_b
from snapshot_trainer_c import EightKAE, FourKAE, SixKAE, TenKAE, TwelveKAE
from snapshot_trainer_c import load_full_dataset as load_full_dataset_x
from snapshot_trainer_c import run as snapshot_run_c

from AnomalyTransformer import train as at_train
from AnomalyTransformer.AnomalyAttention import AnomalyTransformer as ATModel
from AnomalyTransformer.dataset import get_dataloader, split_data

FFNN_MODELS = {
    "FourKMLP": FourKMLP,
    "SixKMLP": SixKMLP,
    "EightKMLP": EightKMLP,
    "TenKMLP": TenKMLP,
    "TwelveKMLP": TwelveKMLP,
}

CNN_MODELS = {
    "FourKCNN": FourKCNN,
    "SixKCNN": SixKCNN,
    "EightKCNN": EightKCNN,
    "TenKCNN": TenKCNN,
    "TwelveKCNN": TwelveKCNN,
}

AE_MODELS = {
    "FourKAE": FourKAE,
    "SixKAE": SixKAE,
    "EightKAE": EightKAE,
    "TenKAE": TenKAE,
    "TwelveKAE": TwelveKAE,
}

ALL_MODEL_CHOICES = list(FFNN_MODELS) + list(CNN_MODELS) + list(AE_MODELS)

EFF_SIGNAL_COL = 2


def resolve_family(model_name):
    if model_name in FFNN_MODELS:
        return "ffnn"
    if model_name in CNN_MODELS:
        return "cnn"
    if model_name in AE_MODELS:
        return "ae"
    raise ValueError(f"Unknown model: {model_name}")


def _peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _subprocess_entrypoint(target, args, kwargs, result_queue):
    baseline_rss = _peak_rss_bytes()
    try:
        result = target(*args, **kwargs)
        result_queue.put(("ok", result, max(_peak_rss_bytes() - baseline_rss, 0)))
    except Exception:
        result_queue.put(
            ("error", traceback.format_exc(), max(_peak_rss_bytes() - baseline_rss, 0))
        )


def run_isolated(target, *args, **kwargs):
    """Run target(*args, **kwargs) in a subprocess.

    Returns (result, peak_rss_delta), where peak_rss_delta is the
    subprocess's peak RSS *above its own baseline* (RSS measured right
    after the subprocess starts, before `target` runs) -- this excludes
    the fixed cost of importing torch/numpy/etc. into the fresh process.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_entrypoint, args=(target, args, kwargs, result_queue)
    )
    proc.start()

    status, payload, peak_rss_delta = result_queue.get()

    proc.join()

    if status == "error":
        raise RuntimeError(f"Subprocess for {target.__name__} failed:\n{payload}")

    return payload, peak_rss_delta


def _miner_worker_ffnn(**kwargs):
    return snapshot_run_a(out_csv=None, **kwargs)


def _scientist_worker_ffnn(**kwargs):
    return snapshot_run_a(out_csv=None, **kwargs)


def _miner_worker_cnn(**kwargs):
    return snapshot_run_b(out_csv=None, **kwargs)


def _scientist_worker_cnn(**kwargs):
    return snapshot_run_b(out_csv=None, **kwargs)


def _miner_worker_ae(**kwargs):
    return snapshot_run_c(out_csv=None, **kwargs)


def _scientist_worker_ae(**kwargs):
    return snapshot_run_c(out_csv=None, **kwargs)


def _scientist_batch_worker(scientist_worker, run_specs):
    """Run every scientist snapshot training back to back in one subprocess.

    run_specs is a list of (label, actual, kwargs) tuples. Batching all
    N runs into a single subprocess (instead of one subprocess per run)
    avoids paying the fixed process/import cold-start cost N times over.
    """
    payloads = []
    for label, actual, kwargs in run_specs:
        snaps = scientist_worker(**kwargs)
        payloads.append({"label": label, "actual": actual, "snaps": snaps})
    return payloads


def _at_worker(
    miner_snaps, window_size, at_cfg, scientist_payloads, flag_frac, eff_signal_ratio
):
    """Train the AT model on miner_snaps, then score every scientist run."""
    device = torch.device("cpu")
    at_model, threshold, norm_mean, norm_std = _train_at(
        miner_snaps, window_size, at_cfg, device
    )

    if at_model is None:
        raise RuntimeError(
            f"need >= {window_size * 3} miner snapshots (got {len(miner_snaps)}). "
            "Increase --miner-epochs."
        )

    miner_eff_signal = miner_snaps[:, EFF_SIGNAL_COL]
    eff_signal_mean = float(miner_eff_signal.mean())
    eff_signal_cutoff = eff_signal_mean * eff_signal_ratio

    results = []
    for payload in scientist_payloads:
        snaps = payload["snaps"]
        frac = _score_run(
            at_model, snaps, window_size, threshold, norm_mean, norm_std, at_cfg, device
        )

        eff_mean = float(snaps[:, EFF_SIGNAL_COL].mean())
        at_pred = (1 if frac > flag_frac else 0) if frac is not None else -1
        eff_pred = 1 if eff_mean < eff_signal_cutoff else 0
        pred = at_pred if at_pred < 0 else (1 if (at_pred == 1 or eff_pred == 1) else 0)
        results.append(
            {
                "label": payload["label"],
                "actual": payload["actual"],
                "predicted": pred,
                "frac": frac,
            }
        )

    return {"threshold": threshold, "results": results}


def _train_at(miner_snaps, window_size, at_cfg, device):
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
def run_experiment(
    model_name,
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
    # ffnn-only
    dataset_path=None,
    target_col=None,
    # cnn-only
    dataset_root=None,
    image_size=32,
    # ae-only
    data_root=None,
    # cnn/ae shared
    max_samples=None,
):
    experiment_id = f"sae_mem_{model_name}"
    print(f"Model: {model_name}  (family={family})  |  seed: {seed}")
    print("Device: cpu (hardcoded -- memory accounting is RSS-based)")

    if family == "ffnn":
        X_all, y_all = load_full_dataset_a(dataset_path, target_col)
    elif family == "cnn":
        X_all, y_all = load_full_dataset_b(
            dataset_root, image_size=image_size, max_samples=max_samples
        )
    else:  # ae
        X_all = load_full_dataset_x(data_root, max_samples=max_samples)
        y_all = None

    n_miner = int(len(X_all) * train_frac)
    X_miner = X_all[:n_miner]
    X_sci = X_all
    y_miner = y_all[:n_miner] if y_all is not None else None
    y_sci = y_all if y_all is not None else None

    print(
        f"Dataset: {len(X_all)} rows  ->  miner {len(X_miner)}, scientist {len(X_sci)}"
    )

    def _base_kwargs(is_malicious, run_seed, X_data, y_data):
        if family == "ffnn":
            return dict(
                data_path=dataset_path,
                target_col=target_col,
                batch_size=None,
                n_epochs=None,
                lr=None,
                is_malicious=is_malicious,
                seed=run_seed,
                flip_frac=flip_frac,
                X_data=X_data,
                y_data=y_data,
                model_class_str=model_name,
            )
        if family == "cnn":
            return dict(
                data_root=dataset_root,
                batch_size=None,
                n_epochs=None,
                lr=None,
                is_malicious=is_malicious,
                seed=run_seed,
                flip_frac=flip_frac,
                X_data=X_data,
                y_data=y_data,
                model_class_str=model_name,
                image_size=image_size,
            )

        return dict(
            data_root=data_root,
            batch_size=None,
            n_epochs=None,
            lr=None,
            is_malicious=is_malicious,
            seed=run_seed,
            flip_frac=flip_frac,
            X_data=X_data,
            model_class_str=model_name,
        )

    miner_worker = {
        "ffnn": _miner_worker_ffnn,
        "cnn": _miner_worker_cnn,
        "ae": _miner_worker_ae,
    }[family]

    scientist_worker = {
        "ffnn": _scientist_worker_ffnn,
        "cnn": _scientist_worker_cnn,
        "ae": _scientist_worker_ae,
    }[family]

    print("[Miner] Training model and collecting snapshots (isolated)...")
    miner_kwargs = _base_kwargs(False, seed, X_miner, y_miner)
    miner_kwargs.update(
        batch_size=miner_snap_cfg["batch_size"],
        n_epochs=miner_snap_cfg["n_epochs"],
        lr=miner_snap_cfg["lr"],
    )
    miner_snaps, miner_mem = run_isolated(miner_worker, **miner_kwargs)
    print(
        f"[Miner] {len(miner_snaps)} snapshots  peak_rss_delta={miner_mem / 1e6:.1f} MB"
    )

    def _scientist_kwargs(is_malicious, run_seed):
        kwargs = _base_kwargs(is_malicious, run_seed, X_sci, y_sci)
        kwargs.update(
            batch_size=snap_cfg["batch_size"],
            n_epochs=snap_cfg["n_epochs"],
            lr=snap_cfg["lr"],
        )
        return kwargs

    run_specs = [
        (f"ben_{i}", 0, _scientist_kwargs(False, benign_seed_base + i))
        for i in range(n_benign)
    ] + [
        (f"mal_{i}", 1, _scientist_kwargs(True, malicious_seed_base + i))
        for i in range(n_malicious)
    ]

    print(
        f"[Scientist] Running {len(run_specs)} benign+malicious trainings "
        "(single isolated subprocess)..."
    )
    scientist_payloads, scientist_mem_total = run_isolated(
        _scientist_batch_worker, scientist_worker, run_specs
    )
    print(f"[Scientist] peak_rss_delta={scientist_mem_total / 1e6:.1f} MB")

    print("[AT] Training + scoring (isolated)...")
    at_result, at_mem = run_isolated(
        _at_worker,
        miner_snaps=miner_snaps,
        window_size=window_size,
        at_cfg=at_cfg,
        scientist_payloads=scientist_payloads,
        flag_frac=flag_frac,
        eff_signal_ratio=eff_signal_ratio,
    )

    print(
        f"[AT] threshold={at_result['threshold']:.4f}  "
        f"peak_rss_delta={at_mem / 1e6:.1f} MB"
    )

    tp = sum(
        1 for r in at_result["results"] if r["actual"] == 1 and r["predicted"] == 1
    )

    fp = sum(
        1 for r in at_result["results"] if r["actual"] == 0 and r["predicted"] == 1
    )

    tn = sum(
        1 for r in at_result["results"] if r["actual"] == 0 and r["predicted"] == 0
    )

    fn = sum(
        1 for r in at_result["results"] if r["actual"] == 1 and r["predicted"] == 0
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total_mem = miner_mem + scientist_mem_total + at_mem

    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(
        f"Memory (peak-RSS deltas): miner={miner_mem / 1e6:.1f} MB  "
        f"scientist({n_benign + n_malicious} runs, 1 subprocess)="
        f"{scientist_mem_total / 1e6:.1f} MB  "
        f"AT={at_mem / 1e6:.1f} MB  total={total_mem / 1e6:.1f} MB"
    )

    result_row = {
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
        "miner_mem_bytes": miner_mem,
        "miner_mem_mb": round(miner_mem / 1e6, 3),
        "scientist_mem_bytes": scientist_mem_total,
        "scientist_mem_mb": round(scientist_mem_total / 1e6, 3),
        "at_mem_bytes": at_mem,
        "at_mem_mb": round(at_mem / 1e6, 3),
        "total_mem_bytes": total_mem,
        "total_mem_mb": round(total_mem / 1e6, 3),
    }

    if out_csv:
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
            "miner_mem_bytes",
            "miner_mem_mb",
            "scientist_mem_bytes",
            "scientist_mem_mb",
            "at_mem_bytes",
            "at_mem_mb",
            "total_mem_bytes",
            "total_mem_mb",
        ]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(result_row)

        print(f"\nSaved results -> {out_csv}")

    return result_row


def parse_args():
    p = argparse.ArgumentParser(
        description="SPECTRA single-model memory profiling (FFNN, CNN, or AE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True, choices=ALL_MODEL_CHOICES)
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
        "(single value -- this script does not sweep)",
    )

    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--flag-frac", type=float, default=0.1)
    p.add_argument("--n-benign", type=int, default=10)
    p.add_argument("--n-malicious", type=int, default=10)
    p.add_argument("--out-csv", default="experimentSAE_mem_results.csv")
    p.add_argument("--eff-signal-ratio", default=0.5, type=float)

    fg = p.add_argument_group("FFNN dataset (--model *KMLP)")
    fg.add_argument("--dataset", help="Path to CSV dataset")
    fg.add_argument("--target-col", help="Target column name")

    cg = p.add_argument_group("CNN dataset (--model *KCNN)")
    cg.add_argument("--dataset-root", help="Folder with Cat/ and Dog/ subfolders")
    cg.add_argument("--image-size", type=int, default=32)

    ag_ds = p.add_argument_group("AE dataset (--model *KAE)")
    ag_ds.add_argument("--data-root", help="Folder to (down)load MNIST into")

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on samples loaded into memory (CNN default 1500, AE default 3000; ignored for FFNN)",
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

    family = resolve_family(args.model)

    if family == "ffnn" and (not args.dataset or not args.target_col):
        p.error("--model *KMLP requires --dataset and --target-col")
    if family == "cnn" and not args.dataset_root:
        p.error("--model *KCNN requires --dataset-root")
    if family == "ae" and not args.data_root:
        p.error("--model *KAE requires --data-root")

    if args.max_samples is None:
        args.max_samples = 1500 if family == "cnn" else 3000  # unused for ffnn

    args.family = family

    return args


if __name__ == "__main__":
    args = parse_args()

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

    benign_seed_base = 1000 + args.seed * 100
    malicious_seed_base = 2000 + args.seed * 100

    run_experiment(
        model_name=args.model,
        family=args.family,
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
        dataset_path=os.path.abspath(args.dataset) if args.dataset else None,
        target_col=args.target_col,
        dataset_root=os.path.abspath(args.dataset_root) if args.dataset_root else None,
        image_size=args.image_size,
        data_root=os.path.abspath(args.data_root) if args.data_root else None,
        max_samples=args.max_samples,
        eff_signal_ratio=args.eff_signal_ratio,
    )
