import argparse
import csv
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden=64, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X_train, y_train, batch_size=64):
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)


def load_dataset(path, target_col, train_frac=0.7):
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().str.lower().unique()
        if set(vals).issubset({"yes", "no"}):
            df[col] = df[col].str.lower().map({"yes": 1.0, "no": 0.0})
    df = df.select_dtypes(include="number").dropna()

    y = df[target_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)

    n_train = int(len(X) * train_frac)
    X_train, y_train = X[:n_train], y[:n_train]

    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-8
    X_train = (X_train - X_mean) / X_std

    return X_train, y_train


def load_full_dataset(path, target_col):
    """Return (X, y) for the whole file without any splitting or normalisation."""
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().str.lower().unique()
        if set(vals).issubset({"yes", "no"}):
            df[col] = df[col].str.lower().map({"yes": 1.0, "no": 0.0})
    df = df.select_dtypes(include="number").dropna()
    y = df[target_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    return X, y


def extract_snapshot(model, loss_val, input_vec_size, prev_loss=0.0, prev_all_w=None):
    """Return a 1-D numpy array of exactly `input_vec_size` features.

    The first 7 slots are phase-invariant dynamics features designed to expose
    Byzantine gradient attacks regardless of where in training a run is.
    Remaining slots are uniformly sampled gradient values so the AT can see
    the raw sign distribution (50% sign-flip makes it near-symmetric).
    """
    weights, grads = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.data.flatten())
            if module.weight.grad is not None:
                grads.append(module.weight.grad.data.flatten())

    all_w = torch.cat(weights)
    all_g = torch.cat(grads) if grads else torch.zeros(len(all_w))

    g_mean = all_g.mean().item()
    g_std  = all_g.std().item() + 1e-8

    grad_snr         = abs(g_mean) / g_std
    loss_delta       = float(loss_val) - float(prev_loss)
    effective_signal = g_mean ** 2 / (g_mean ** 2 + g_std ** 2)
    g_norm           = all_g.norm(2).item()
    w_norm           = all_w.norm(2).item()
    weight_delta_norm = (all_w.cpu() - prev_all_w).norm(2).item() if prev_all_w is not None else 0.0

    scalars = np.array([
        grad_snr,
        loss_delta,
        effective_signal,
        g_norm,
        weight_delta_norm,
        float(loss_val),
        w_norm,
    ], dtype=np.float32)

    n_scalar = len(scalars)  # always 7

    if input_vec_size <= n_scalar:
        
        return scalars[:input_vec_size]
    

    n_sample = input_vec_size - n_scalar
    indices = torch.linspace(0, len(all_g) - 1, n_sample).long()
    sampled = all_g[indices].detach().cpu().numpy().astype(np.float32)
    return np.concatenate([scalars, sampled])


def gradient_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def run(data_path, target_col, out_csv, input_vec_size=7,
        batch_size=64, n_epochs=5, lr=1e-3, hidden=64,
        is_malicious=False, seed=None, X_data=None, y_data=None, device=None):
    """Train a SimpleMLP and collect per-batch snapshots.

    Parameters
    ----------
    data_path / target_col : used only when X_data / y_data are not provided.
    input_vec_size         : number of features in each snapshot row.
    is_malicious           : if True, apply a Byzantine gradient attack
                             (random 50 % sign-flip each batch).
    seed                   : torch + numpy RNG seed for reproducibility.
    X_data / y_data        : pre-split numpy arrays (skip file loading).
    device                 : torch.device (or str) to train on. Defaults to CPU.

    Returns
    -------
    numpy array of shape (n_total_batches, input_vec_size)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if X_data is not None and y_data is not None:
        X_train = X_data.astype(np.float32)
        y_train = y_data.astype(np.float32)
        X_mean = X_train.mean(0)
        X_std = X_train.std(0) + 1e-8
        X_train = (X_train - X_mean) / X_std
    else:
        X_train, y_train = load_dataset(data_path, target_col)

    loader = make_loader(X_train, y_train, batch_size)
    model = SimpleMLP(in_features=X_train.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    snapshots = []
    prev_loss  = 0.0
    prev_all_w = None

    for _ in range(n_epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()

            if is_malicious:
                # Byzantine attack: randomly flip sign of 50 % of each gradient tensor
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            mask = torch.rand_like(p.grad) > 0.5
                            p.grad.data[mask] *= -1.0

            optimizer.step()
            snapshots.append(extract_snapshot(model, loss.item(), input_vec_size,
                                              prev_loss, prev_all_w))

            with torch.no_grad():
                prev_all_w = torch.cat([
                    m.weight.data.flatten()
                    for m in model.modules() if isinstance(m, nn.Linear)
                ]).cpu()
            prev_loss = loss.item()

    arr = np.array(snapshots, dtype=np.float32)

    if out_csv:
        fieldnames = ["batch_idx"] + [f"f{i}" for i in range(input_vec_size)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        required=True)
    parser.add_argument("--target-col",     required=True)
    parser.add_argument("--out",            default="snapshots.csv")
    parser.add_argument("--input-vec-size", type=int,   default=7)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--n-epochs",       type=int,   default=5)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden",         type=int,   default=64)
    parser.add_argument("--is-malicious",   action="store_true")
    parser.add_argument("--seed",           type=int,   default=None)
    args = parser.parse_args()

    run(
        data_path=args.dataset,
        target_col=args.target_col,
        out_csv=args.out,
        input_vec_size=args.input_vec_size,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        hidden=args.hidden,
        is_malicious=args.is_malicious,
        seed=args.seed,
    )
