import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataset import load_dataset, plot_clusters


def run_kmeans(df, features=None, n_clusters=5, random_state=42):
    if features is not None:
        df = df[features]
    X = StandardScaler().fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans, X


def detect_anomalies(X, labels, kmeans, r=0.01):
    """Flag the top r-fraction of points by distance to their assigned cluster
    centroid — the least 'typical' members of their cluster."""
    distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    threshold = np.quantile(distances, 1 - r)
    return distances > threshold, distances


def parse_args():
    parser = argparse.ArgumentParser(description="Run K-Means clustering on a CSV dataset and flag anomalies by distance to centroid.")
    parser.add_argument("--csv", default="housing/housing/housing.csv", help="Path to the input CSV file.")
    parser.add_argument("--features", nargs=2, default=None, metavar=("VAR1", "VAR2"),
                         help="Cluster on exactly two columns instead of the full feature set.")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--r", type=float, default=0.01, help="Top quantile of centroid distances flagged as anomalous.")
    parser.add_argument("--out", default="kmeans_clusters.png", help="Path to save the cluster plot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_dataset(args.csv)
    print(f"Loaded {len(df)} rows, {df.shape[1]} features")

    labels, kmeans, X = run_kmeans(df, features=args.features, n_clusters=args.n_clusters, random_state=args.random_state)
    print(pd.Series(labels).value_counts().sort_index().rename("cluster_size"))

    flagged, distances = detect_anomalies(X, labels, kmeans, r=args.r)
    print(f"Flagged {flagged.sum()} / {len(flagged)} rows as anomalous")

    plot_clusters(df, labels, flagged, features=args.features, title="K-Means clusters", out_path=args.out)