"""
Combine the per-seed result CSVs produced by Experiment1a/b/c.py into a
single averaged CSV + chart, broken out by flip_frac (10%/20%/30%/40%/50%
sign-flip rate) and averaged only over the seeds within each flip_frac.

Usage
-----
  python combine_results.py --results-dir ffnn_acc --title "SPECTRA FFNN Detection" \
      --out-csv ffnn_acc/ffnn_acc_avg.csv --out-png ffnn_acc/ffnn_acc_avg.png
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

# Matches "..._<seed>-<flag_frac>-<at_r>-<eff_signal_ratio>_<flip_frac>.csv"
_FNAME_RE = re.compile(r"_(?P<seed>\d+)-[\d.]+-[\d.]+-[\d.]+_(?P<flip>[\d.]+)\.csv$")


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


def combine_and_average(results_dir, pattern, out_csv, out_png, title):
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise RuntimeError(f"No CSVs found matching '{pattern}' in {results_dir}")

    dfs = []
    for p in paths:
        m = _FNAME_RE.search(os.path.basename(p))
        if not m:
            raise RuntimeError(f"Filename does not match expected pattern: {p}")
        df = pd.read_csv(p)
        df["seed"] = int(m.group("seed"))
        df["flip_frac"] = float(m.group("flip"))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    experiment_id = (
        combined["experiment"].iloc[0] if "experiment" in combined.columns else ""
    )

    grouped = (
        combined.groupby(["model_class", "flip_frac"])[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
    )
    n_seeds_per_group = combined.groupby(["model_class", "flip_frac"])[
        "seed"
    ].nunique()
    grouped["n_seeds"] = grouped.apply(
        lambda r: n_seeds_per_group[(r["model_class"], r["flip_frac"])], axis=1
    )
    grouped["size_label"] = grouped["model_class"].apply(_size_label)
    grouped["size_key"] = grouped["model_class"].apply(_size_key)
    grouped = grouped.sort_values(["size_key", "flip_frac"]).reset_index(drop=True)

    out_df = grouped[
        ["model_class", "size_label", "flip_frac", "n_seeds", "precision", "recall", "f1"]
    ].copy()
    out_df.insert(0, "experiment", experiment_id)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved averaged results ({len(paths)} files) -> {out_csv}")

    flip_fracs = sorted(grouped["flip_frac"].unique())
    labels = [l for l in dict.fromkeys(grouped["size_label"])]
    x = list(range(len(labels)))
    width = 0.35

    out_png_base, out_png_ext = os.path.splitext(out_png)

    for flip in flip_fracs:
        sub = grouped[grouped["flip_frac"] == flip].sort_values("size_key")
        precision = sub["precision"].tolist()
        recall = sub["recall"].tolist()
        f1 = sub["f1"].tolist()

        fig, ax = plt.subplots(figsize=(5, 5))
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
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(f"{title} - {int(flip * 100)}% signs flipped")
        ax.grid(axis="y")
        ax.legend(loc="upper left")
        fig.tight_layout()

        flip_out_png = f"{out_png_base}_flip{flip}{out_png_ext}"
        fig.savefig(flip_out_png)
        plt.close(fig)
        print(f"Saved plot ({int(flip * 100)}% flipped) -> {flip_out_png}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine and average SPECTRA per-seed results"
    )
    p.add_argument(
        "--results-dir", required=True, help="Directory containing per-seed CSVs"
    )
    p.add_argument(
        "--pattern",
        default="ffnn_acc_[0-9]*.csv",
        help="Glob (relative to --results-dir) matching per-seed result CSVs",
    )
    p.add_argument("--out-csv", required=True, help="Path for the averaged CSV")
    p.add_argument("--out-png", required=True, help="Path for the averaged chart")
    p.add_argument(
        "--title", required=True, help="Base chart title (seed count is appended)"
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
    )
