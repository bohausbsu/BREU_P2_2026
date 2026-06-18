import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size=100):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return torch.from_numpy(window.astype(np.float32, copy=False))
    
def get_dataloader(data, window_size=100, batch_size=32, shuffle=True):
    dataset = SlidingWindowDataset(data, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_csv(path, skip_cols=1, has_header=True):
    """Load a CSV, dropping the first `skip_cols` columns (e.g. a timestamp/index).
    If has_header is False (e.g. raw headerless data files), every row is treated as
    data and columns are auto-named. Returns (data, column_names)."""
    if has_header:
        with open(path) as f:
            header = f.readline().strip().split(",")
        raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
        columns = [c.strip('"') for c in header[skip_cols:]]
    else:
        raw = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        columns = [f"feature_{i}" for i in range(raw.shape[1] - skip_cols)]
    return raw[:, skip_cols:], columns

def split_data(data, train_frac=0.7, val_frac=0.85):
    """Split into train/val/test by index. val_frac is the cumulative fraction marking the end of val."""
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return data[:n_train], data[n_train:n_val], data[n_val:]

def normalize_splits(train_data, val_data, test_data):
    """Z-score normalize all three splits using train-set statistics, to avoid leakage."""
    mean, std = train_data.mean(axis=0), train_data.std(axis=0)
    norm = lambda d: (d - mean) / (std + 1e-8)
    return norm(train_data), norm(val_data), norm(test_data)

def make_loaders(train_data, val_data, test_data, window_size=100, batch_size=32):
    """Build train/val/test dataloaders. Val and test are not shuffled."""
    train_loader = get_dataloader(train_data, window_size, batch_size)
    val_loader = get_dataloader(val_data, window_size, batch_size, shuffle=False)
    test_loader = get_dataloader(test_data, window_size, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def load_dataset(path, drop_cols=None):
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)
    return df

def plot_clusters(df, labels, flagged, features=None, title="Clusters", anomaly_label="anomaly", out_path="clusters.png"):
    if features is not None and len(features) == 2:
        xlabel, ylabel = features
    else:
        cols = df.columns.tolist()
        xlabel, ylabel = cols[0], cols[1]
    x, y = df[xlabel].to_numpy(), df[ylabel].to_numpy()

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    label_to_color = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}
    point_colors = [label_to_color[lbl] for lbl in labels]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=point_colors, s=10, alpha=0.3, linewidths=0)
    if flagged.any():
        plt.scatter(x[flagged], y[flagged], c="red", marker="x", s=40, label=anomaly_label, zorder=3)
        plt.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
