import argparse
import csv
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Define the models. Tried to get them as close to the thousand parameter count in their name
class FourKMLP(nn.Module):
    def __init__(self, in_features=8, hidden=58, out_features=1):
        """Constructor"""
        super().__init__()  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),  # Fully connected feed-forward layer
            nn.ReLU(),  # Recitfied Linear Unit (ReLU) activation function. Full formula for ReLU is y = max(0, x) where `x` is the input.
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        """Computation function"""
        return self.net(x)


class SixKMLP(nn.Module):
    def __init__(self, in_features=8, hidden=72, out_features=1):
        """Constructor"""
        super().__init__()  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)

        self.net = nn.Sequential(
            nn.Linear(
                in_features, hidden
            ),  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)
            nn.ReLU(),  # Recitfied Linear Unit (ReLU) activation function. Full formula for ReLU is y = max(0, x) where `x` is the input.
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        """Computation function"""
        return self.net(x)


class EightKMLP(nn.Module):
    def __init__(self, in_features=8, hidden=84, out_features=1):
        """Constructor"""
        super().__init__()  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)

        self.net = nn.Sequential(
            nn.Linear(
                in_features, hidden
            ),  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)
            nn.ReLU(),  # Recitfied Linear Unit (ReLU) activation function. Full formula for ReLU is y = max(0, x) where `x` is the input.
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        """Computation function"""
        return self.net(x)


class TenKMLP(nn.Module):
    def __init__(self, in_features=8, hidden=95, out_features=1):
        """Constructor"""
        super().__init__()  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)

        self.net = nn.Sequential(
            nn.Linear(
                in_features, hidden
            ),  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)
            nn.ReLU(),  # Recitfied Linear Unit (ReLU) activation function. Full formula for ReLU is y = max(0, x) where `x` is the input.
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        """Computation function"""
        return self.net(x)


class TwelveKMLP(nn.Module):
    def __init__(self, in_features=8, hidden=104, out_features=1):
        """Constructor"""
        super().__init__()  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)

        self.net = nn.Sequential(
            nn.Linear(
                in_features, hidden
            ),  # Initialize the pytorch stuff provided by the base class (`pytorch.nn.Module``)
            nn.ReLU(),  # Recitfied Linear Unit (ReLU) activation function. Full formula for ReLU is y = max(0, x) where `x` is the input.
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        """Computation function"""
        return self.net(x)


def make_loader(X_train, y_train, batch_size=64):
    """Turns a numpy ndarray pair of data and labels into a PyTorch dataloader."""
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).unsqueeze(1)

    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)


def load_dataset(path, target_col, train_frac=0.7):
    """Loads a dataset from memory as a pandas DataFrame."""
    df = pd.read_csv(path)  # Read the CSV file from the input path

    # Iterate over columns
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().str.lower().unique()

        if set(vals).issubset({"yes", "no"}):
            df[col] = df[col].str.lower().map({"yes": 1.0, "no": 0.0})

    df = df.select_dtypes(include="number").dropna()

    # Split data into a training data vector and a training labels vector
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    n_train = int(len(X) * train_frac)
    X_train, y_train = X[:n_train], y[:n_train]

    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train = (X_train - X_mean) / X_std

    return X_train, y_train


def load_full_dataset(path, target_col):
    """
    Return (X, y) for the whole file without any splitting or normalisation.

    *** Same as `load_dataset` but without splitting into the training data/labels vectors
    """
    df = pd.read_csv(path)

    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().str.lower().unique()

        if set(vals).issubset({"yes", "no"}):
            df[col] = df[col].str.lower().map({"yes": 1.0, "no": 0.0})

    df = df.select_dtypes(include="number").dropna()
    y = df[target_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)

    return X, y


def extract_snapshot(model, loss_val, prev_loss=0.0, prev_all_w=None):
    """
    Return a 1-D numpy array of exactly `input_vec_size` features.

    The first 7 slots are phase-invariant dynamics features designed to expose
    Byzantine gradient attacks regardless of where in training a run is.
    Remaining slots are uniformly sampled gradient values so the AT can see
    the raw sign distribution (50% sign-flip makes it near-symmetric).
    """
    # This is all Alex's work. Pretty sure all he's doing here is getting
    #  all the weights and gradients and then using them to compute his 7
    #  statistics that are supposed to represent the entire distribution
    #  of the data that it was trained on.
    weights, grads = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.data.flatten())

            if module.weight.grad is not None:
                grads.append(module.weight.grad.data.flatten())

    all_w = torch.cat(weights)
    all_g = torch.cat(grads) if grads else torch.zeros(len(all_w))

    g_mean = all_g.mean().item()
    g_std = all_g.std().item() + 1e-8

    grad_snr = abs(g_mean) / g_std
    loss_delta = float(loss_val) - float(prev_loss)
    effective_signal = g_mean**2 / (g_mean**2 + g_std**2)
    g_norm = all_g.norm(2).item()
    w_norm = all_w.norm(2).item()
    weight_delta_norm = (
        (all_w.cpu() - prev_all_w).norm(2).item() if prev_all_w is not None else 0.0
    )

    scalars = np.array(
        [
            grad_snr,
            loss_delta,
            effective_signal,
            g_norm,
            weight_delta_norm,
            float(loss_val),
            w_norm,
        ],
        dtype=np.float32,
    )

    frac_pos = (all_g > 0).float().mean().item()
    abs_g = all_g.abs()
    q10, q50, q90 = torch.quantile(
        abs_g, torch.tensor([0.1, 0.5, 0.9], device=all_g.device)
    ).tolist()
    g_z = (all_g - g_mean) / g_std
    skew = (g_z**3).mean().item()
    kurtosis = (g_z**4).mean().item() - 3.0
    layer_norms = [g.norm(2).item() for g in grads] if grads else [0.0] * len(weights)

    dist_stats = np.array(
        [
            frac_pos,
            q10,
            q50,
            q90,
            skew,
            kurtosis,
            *layer_norms,
        ],
        dtype=np.float32,
    )

    return np.concatenate([scalars, dist_stats], axis=0)


def gradient_norm(model):
    """Normalizes the gradients of the input model."""
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2

    return math.sqrt(total)


def run(
    data_path,
    target_col,
    out_csv,
    batch_size=64,
    n_epochs=5,
    lr=1e-3,
    is_malicious=False,
    flip_frac=0.5,
    seed=None,
    X_data=None,
    y_data=None,
    device=None,
    model_class_str="SixKMLP",
):
    """
    Train a SimpleMLP and collect per-batch snapshots.

    Parameters
    ----------
    data_path / target_col : used only when X_data / y_data are not provided.
    is_malicious           : if True, apply a Byzantine gradient attack
                             (random sign-flip of `flip_frac` of each
                             gradient tensor, each batch).
    flip_frac              : fraction of gradient entries to sign-flip when
                             is_malicious is True (default 0.5 = 50 %).
    seed                   : torch + numpy RNG seed for reproducibility.
    X_data / y_data        : pre-split numpy arrays (skip file loading).
    device                 : torch.device to train on (defaults to CPU).

    Returns: A numpy ndarray of shape (n_total_batches, input_vec_size)
    """
    # Set device for training
    device = device or torch.device("cpu")

    # Set the seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Fix up data
    if X_data is not None and y_data is not None:
        X_train = X_data.astype(np.float32)
        y_train = y_data.astype(np.float32)
        X_mean = X_train.mean(0)
        X_std = X_train.std(0) + 1e-8
        X_train = (X_train - X_mean) / X_std
    else:
        X_train, y_train = load_dataset(data_path, target_col)

    # Create a training dataloader from the data
    loader = make_loader(X_train, y_train, batch_size)

    # Define model
    model = (
        FourKMLP(in_features=X_train.shape[1])
        if model_class_str == "FourKMLP"
        else (
            SixKMLP(in_features=X_train.shape[1])
            if model_class_str == "SixKMLP"
            else (
                EightKMLP(in_features=X_train.shape[1])
                if model_class_str == "EightKMLP"
                else (
                    TenKMLP(in_features=X_train.shape[1])
                    if model_class_str == "TenKMLP"
                    else (
                        TwelveKMLP(in_features=X_train.shape[1])
                        if model_class_str == "TwelveKMLP"
                        else None
                    )
                )
            )
        )
    )

    if model is None:
        raise RuntimeError(f"Invalid model class name: {model_class_str}")

    # Put model on the training device
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Containers
    snapshots = []
    prev_loss = 0.0
    prev_all_w = None

    # Train
    for _ in range(n_epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()

            # Byzantine attack: randomly flip sign of `flip_frac` of each gradient tensor
            if is_malicious:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            mask = torch.rand_like(p.grad) < flip_frac
                            p.grad.data[mask] *= -1.0

            optimizer.step()
            snapshots.append(
                extract_snapshot(model, loss.item(), prev_loss, prev_all_w)
            )

            with torch.no_grad():
                prev_all_w = torch.cat(
                    [
                        m.weight.data.flatten()
                        for m in model.modules()
                        if isinstance(m, nn.Linear)
                    ]
                ).cpu()

            prev_loss = loss.item()

    arr = np.array(snapshots, dtype=np.float32)

    if out_csv:
        fieldnames = ["batch_idx"] + [
            f"f{i}"
            for i in range(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        ]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, snap in enumerate(snapshots):
                row = {"batch_idx": idx}
                for j, v in enumerate(snap):
                    row[f"f{j}"] = round(float(v), 6)

                writer.writerow(row)

        print(f"Saved {len(snapshots)} snapshots → {out_csv}")

    return arr


if __name__ == "__main__":
    # Define CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", default="snapshots.csv")
    parser.add_argument("--input-vec-size", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--is-malicious", action="store_true")
    parser.add_argument("--flip-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()  # Parse CLI args

    # Run the experiment
    run(
        data_path=args.dataset,
        target_col=args.target_col,
        out_csv=args.out,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        is_malicious=args.is_malicious,
        flip_frac=args.flip_frac,
        seed=args.seed,
    )
