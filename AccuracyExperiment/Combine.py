"""
Usage
-----
  python Combine.py
  python Combine.py --pattern "experiment*_results.csv"
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_results(pattern):
    paths = sorted(glob.glob(os.path.join(_HERE, pattern)))
    if not paths:
        raise SystemExit(f"No files matched pattern: {pattern}")

    frames = []
    for path in paths:
        df = pd.read_csv(path, comment='"')
        df = df.dropna(subset=["input_vec_size"])
        df["source"] = os.path.basename(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description="Average experiment results and plot them.")
    p.add_argument("--pattern", default="experiment*_results.csv",
                    help="Glob pattern (relative to this script) for result CSVs")
    p.add_argument("--out-f1", default="combined_f1.png",
                    help="Output path for the average F1 line chart")
    p.add_argument("--out-precision-recall", default="combined_precision_recall.png",
                    help="Output path for the average precision/recall bar chart")
    p.add_argument("--out-csv", default="combined_averages.csv",
                    help="Output path for the averaged data table")
    args = p.parse_args()

    data = load_results(args.pattern)
    print(f"Loaded {data['source'].nunique()} files, {len(data)} rows total.")

    grouped = (
        data.groupby("input_vec_size")[["tp", "fp", "tn", "fn", "precision", "recall", "f1"]]
        .mean()
        .round(3)
        .sort_index()
    )
    grouped.to_csv(os.path.join(_HERE, args.out_csv))
    print(grouped)

    sizes = grouped.index.to_numpy()

    # ── F1 chart: avg F1 line graph ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, grouped["f1"], "b-o", label="F1 score")
    ax.set_xlabel("Input Vector Size")
    ax.set_ylabel("Score")
    ax.set_title("Average F1 Score vs Input Vector Size")
    ax.set_xticks(sizes)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, args.out_f1))
    print(f"Saved plot -> {args.out_f1}")

    # ── Precision/Recall chart: bar graph ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(sizes))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], grouped["precision"], width, label="Precision")
    ax.bar([xi + width / 2 for xi in x], grouped["recall"], width, label="Recall")
    ax.set_xlabel("Input Vector Size")
    ax.set_ylabel("Score")
    ax.set_title("Average Precision / Recall by Input Vector Size")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, args.out_precision_recall))
    print(f"Saved plot -> {args.out_precision_recall}")


if __name__ == "__main__":
    main()
