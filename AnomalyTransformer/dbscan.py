import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from dataset import load_dataset, plot_clusters


def run_dbscan(df, features=None, eps=0.5, min_samples=5):
    if features is not None:
        df = df[features]
    X = StandardScaler().fit_transform(df)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, X


def detect_anomalies(labels):
    """DBSCAN labels noise points as -1 — those are the anomalies."""
    return labels == -1


def parse_args():
    parser = argparse.ArgumentParser(description="Run DBSCAN clustering on a CSV dataset and flag noise points as anomalies.")
    parser.add_argument("--csv", default="housing/housing/housing.csv", help="Path to the input CSV file.")
    parser.add_argument("--features", nargs=2, default=None, metavar=("VAR1", "VAR2"),
                        help="Cluster on exactly two columns instead of the full feature set.")
    parser.add_argument("--eps", type=float, default=0.5, help="Max distance between points in a neighborhood.")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum points to form a dense region.")
    parser.add_argument("--out", default="dbscan_clusters.png", help="Path to save the cluster plot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_dataset(args.csv)
    if "median_house_value" in df.columns:
        capped = (df["median_house_value"] == 500_000).sum()
        df = df[df["median_house_value"] < 500_000]
        print(f"Dropped {capped} capped rows (median_house_value == 500000)")
    print(f"Loaded {len(df)} rows, {df.shape[1]} features")

    labels, X = run_dbscan(df, features=args.features, eps=args.eps, min_samples=args.min_samples)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")
    print(pd.Series(labels).value_counts().sort_index().rename("cluster_size"))

    flagged = detect_anomalies(labels)
    print(f"Flagged {flagged.sum()} / {len(flagged)} rows as anomalous (noise)")

    plot_clusters(df, labels, flagged, features=args.features, title="DBSCAN clusters", anomaly_label="anomaly (noise)", out_path=args.out)
