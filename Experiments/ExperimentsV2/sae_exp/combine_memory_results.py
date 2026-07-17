"""
Combine the per-size *_merged.csv files (each already averaged over 5 seeds)
into a single CSV + line chart of total_mem_mb vs model size, per model class.

Usage
-----
  python combine_memory_results.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_SIZES = ["4k", "6k", "8k", "10k", "12k"]
_OUT_DIR = "memory"


def combine_and_plot(model_class):
    rows = []
    for size in _SIZES:
        df = pd.read_csv(f"sae_{model_class}_{size}_merged.csv")
        rows.append({"size_label": size, "total_mem_mb": df["total_mem_mb"][0]})

    out_df = pd.DataFrame(rows)

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_csv = os.path.join(_OUT_DIR, f"sae_{model_class}_memory_avg.csv")
    out_png = os.path.join(_OUT_DIR, f"sae_{model_class}_memory_avg.png")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved averaged memory results -> {out_csv}")

    x = list(range(len(_SIZES)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, out_df["total_mem_mb"], "k-o", label="total_mem_mb")
    ax.set_xticks(x)
    ax.set_xticklabels(_SIZES)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Total Memory (MB)")
    ax.set_ylim(0, out_df["total_mem_mb"].max() * 1.1)
    ax.set_title(f"SAE {model_class.upper()} Total Memory Usage")
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved memory plot -> {out_png}")


if __name__ == "__main__":
    for model_class in ["ae", "cnn", "ffnn"]:
        combine_and_plot(model_class)
