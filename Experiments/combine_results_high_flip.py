"""
Same as combine_results.py, but only includes runs with flip-frac >= 0.3
(the stronger-attack half of the 0.1-0.5 sweep). Per-seed filenames are
expected to end in "_<flip_frac>.csv" (e.g. ffnn_acc_3_0.4.csv).

Usage
-----
  python combine_results_high_flip.py --results-dir ffnn_acc --title "SPECTRA FFNN Detection (flip 0.3-0.5)" \
      --out-csv ffnn_acc/ffnn_acc_avg_high_flip.csv --out-png ffnn_acc/ffnn_acc_avg_high_flip.png
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Ordered smallest -> largest; matched against a model_class prefix.
_SIZE_ORDER = ["Four", "Six", "Eight", "Ten", "Twelve"]
_SIZE_LABEL = {"Four": "4k", "Six": "6k", "Eight": "8k", "Ten": "10k", "Twelve": "12k"}

_FLIP_FRAC_RE = re.compile(r"_([0-9]*\.?[0-9]+)\.csv$")


def _size_key(model_class):
    for i, name in enumerate(_SIZE_ORDER):
        if model_class.startswith(name):
            return i
    return len(_SIZE_ORDER)  # unknown sizes sort last


def _size_label(model_class):
    for name in _SIZE_ORDER:
        if model_class.startswith(name):
            return _SIZE_LABEL[name]
    return model_class


def _flip_frac(path):
    m = _FLIP_FRAC_RE.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1))


def combine_and_average(results_dir, pattern, out_csv, out_png, title, min_flip_frac):
    all_paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    paths = [p for p in all_paths if (_flip_frac(p) or 0.0) >= min_flip_frac]
    if not paths:
        raise RuntimeError(
            f"No CSVs with flip-frac >= {min_flip_frac} found matching '{pattern}' in {results_dir}"
        )

    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)

    n_seeds = combined["seed"].nunique() if "seed" in combined.columns else len(paths)
    experiment_id = (
        combined["experiment"].iloc[0] if "experiment" in combined.columns else ""
    )

    grouped = (
        combined.groupby("model_class")[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
    )
    grouped["size_label"] = grouped["model_class"].apply(_size_label)
    grouped["size_key"] = grouped["model_class"].apply(_size_key)
    grouped = grouped.sort_values("size_key").reset_index(drop=True)

    out_df = grouped[["model_class", "size_label", "precision", "recall", "f1"]].copy()
    out_df.insert(0, "n_seeds", n_seeds)
    out_df.insert(0, "experiment", experiment_id)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved averaged results ({n_seeds} seeds, {len(paths)} files) -> {out_csv}")

    labels = grouped["size_label"].tolist()
    precision = grouped["precision"].tolist()
    recall = grouped["recall"].tolist()
    f1 = grouped["f1"].tolist()

    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [i - width / 2 for i in x],
        precision,
        width,
        label="Precision",
        color="tab:blue",
    )
    ax.bar(
        [i + width / 2 for i in x], recall, width, label="Recall", color="tab:orange"
    )
    ax.plot(x, f1, "k-o", label="F1 score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model Parameters")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved averaged plot -> {out_png}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine and average SPECTRA per-seed results (flip-frac >= --min-flip-frac only)"
    )
    p.add_argument(
        "--results-dir", required=True, help="Directory containing per-seed CSVs"
    )
    p.add_argument(
        "--pattern",
        default="*_0.*.csv",
        help="Glob (relative to --results-dir) matching per-seed result CSVs",
    )
    p.add_argument("--out-csv", required=True, help="Path for the averaged CSV")
    p.add_argument("--out-png", required=True, help="Path for the averaged chart")
    p.add_argument(
        "--title", required=True, help="Base chart title (seed count is appended)"
    )
    p.add_argument(
        "--min-flip-frac",
        type=float,
        default=0.3,
        help="Only include per-seed CSVs whose filename flip-frac suffix is >= this value",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine_and_average(
        results_dir=args.results_dir,
        pattern=args.pattern,
        out_csv=args.out_csv,
        out_png=args.out_png,
        title=args.title,
        min_flip_frac=args.min_flip_frac,
    )
