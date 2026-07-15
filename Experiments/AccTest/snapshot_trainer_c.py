"""
Autoencoder variant of the snapshot trainer, for unsupervised reconstruction
on MNIST.

Same contract as snapshot_trainer.py, minus a target column: the model's
own input is the reconstruction target, so `run()` and `load_*` drop
`target_col` entirely.

  - model classes sized to ~6k / 8k / 10k / 12k parameters
  - `run()` trains while collecting per-batch weight/gradient snapshots,
    with the same optional Byzantine gradient attack
  - `extract_snapshot()` is byte-for-byte the same feature definition as
    the MLP version (single hidden layer -> only Linear weights exist,
    so no generalization was needed here)
"""

import argparse
import csv
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model definitions
#
# Single-hidden-layer (bottleneck) autoencoder:
#   Linear(784, h) -> ReLU -> Linear(h, 784) -> Sigmoid
# Parameter count = 1569h + 784, independent of anything but h.
# ---------------------------------------------------------------------------
class _BaseAE(nn.Module):
    def __init__(self, bottleneck=3, in_features=784):
        super().__init__()
        self.encoder = nn.Linear(in_features, bottleneck)
        self.decoder = nn.Linear(bottleneck, in_features)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        z = self.act(self.encoder(x))
        return self.out_act(self.decoder(z))


class FourKAE(nn.Module):
    """~4k params (bottleneck=3)."""

    def __init__(self, bottleneck=3, in_features=784):
        super().__init__()
        self.encoder_1 = nn.Linear(in_features, bottleneck)

        self.decoder_1 = nn.Linear(bottleneck, in_features)

        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        z_1 = self.act(self.encoder_1(x))

        z_fin = self.out_act(self.decoder_1(z_1))

        return z_fin


class SixKAE(nn.Module):
    """~5.5k params (bottleneck=3)."""

    def __init__(self, bottleneck=3, in_features=784):
        super().__init__()
        self.encoder_1 = nn.Linear(in_features, bottleneck)
        self.encoder_2 = nn.Linear(bottleneck, bottleneck // 2)

        self.decoder_1 = nn.Linear(bottleneck // 2, bottleneck)
        self.decoder_2 = nn.Linear(bottleneck, in_features)

        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        z_1 = self.act(self.encoder_1(x))
        z_2 = self.act(self.encoder_2(z_1))

        z_3 = self.act(self.decoder_1(z_2))
        z_fin = self.out_act(self.decoder_2(z_3))

        return z_fin


class EightKAE(nn.Module):
    """~8.6k params (bottleneck=5)."""

    def __init__(self, bottleneck=3, in_features=784):
        super().__init__()
        self.encoder_1 = nn.Linear(in_features, bottleneck)
        self.encoder_2 = nn.Linear(bottleneck, bottleneck // 2)
        self.encoder_3 = nn.Linear(bottleneck // 2, bottleneck // 4)

        self.decoder_1 = nn.Linear(bottleneck // 4, bottleneck // 2)
        self.decoder_2 = nn.Linear(bottleneck // 2, bottleneck)
        self.decoder_3 = nn.Linear(bottleneck, in_features)

        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        z_1 = self.act(self.encoder_1(x))
        z_2 = self.act(self.encoder_2(z_1))
        z_3 = self.act(self.encoder_3(z_2))

        z_4 = self.act(self.decoder_1(z_3))
        z_5 = self.act(self.decoder_2(z_4))
        z_fin = self.out_act(self.decoder_3(z_5))

        return z_fin


class TenKAE(nn.Module):
    """~10.2k params (bottleneck=6)."""

    def __init__(self, bottleneck=3, in_features=784):
        super().__init__()
        self.encoder_1 = nn.Linear(in_features, bottleneck)
        self.encoder_2 = nn.Linear(bottleneck, bottleneck // 2)
        self.encoder_3 = nn.Linear(bottleneck // 2, bottleneck // 4)
        self.encoder_4 = nn.Linear(bottleneck // 4, bottleneck // 8)

        self.decoder_1 = nn.Linear(bottleneck // 8, bottleneck // 4)
        self.decoder_2 = nn.Linear(bottleneck // 4, bottleneck // 2)
        self.decoder_3 = nn.Linear(bottleneck // 2, bottleneck)
        self.decoder_4 = nn.Linear(bottleneck, in_features)

        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        z_1 = self.act(self.encoder_1(x))
        z_2 = self.act(self.encoder_2(z_1))
        z_3 = self.act(self.encoder_3(z_2))
        z_4 = self.act(self.encoder_4(z_3))

        z_5 = self.act(self.decoder_1(z_4))
        z_6 = self.act(self.decoder_2(z_5))
        z_7 = self.act(self.decoder_3(z_6))
        z_fin = self.out_act(self.decoder_4(z_7))

        return z_fin


class TwelveKAE(nn.Module):
    """~11.8k params (bottleneck=7)."""

    def __init__(self, bottleneck=3, in_features=784):
        super().__init__()
        self.encoder_1 = nn.Linear(in_features, bottleneck)
        self.encoder_2 = nn.Linear(bottleneck, bottleneck // 2)
        self.encoder_3 = nn.Linear(bottleneck // 2, bottleneck // 4)
        self.encoder_4 = nn.Linear(bottleneck // 4, bottleneck // 8)
        self.encoder_5 = nn.Linear(bottleneck // 8, bottleneck // 16)

        self.decoder_1 = nn.Linear(bottleneck // 16, bottleneck // 8)
        self.decoder_2 = nn.Linear(bottleneck // 8, bottleneck // 4)
        self.decoder_3 = nn.Linear(bottleneck // 4, bottleneck // 2)
        self.decoder_4 = nn.Linear(bottleneck // 2, bottleneck)
        self.decoder_5 = nn.Linear(bottleneck, in_features)

        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        z_1 = self.act(self.encoder_1(x))
        z_2 = self.act(self.encoder_2(z_1))
        z_3 = self.act(self.encoder_3(z_2))
        z_4 = self.act(self.encoder_4(z_3))
        z_5 = self.act(self.encoder_4(z_4))

        z_6 = self.act(self.decoder_1(z_5))
        z_7 = self.act(self.decoder_2(z_6))
        z_8 = self.act(self.decoder_3(z_7))
        z_9 = self.act(self.decoder_4(z_8))
        z_fin = self.out_act(self.decoder_5(z_9))

        return z_fin


MODEL_CLASS_MAP = {
    "FourKAE": FourKAE,
    "SixKAE": SixKAE,
    "EightKAE": EightKAE,
    "TenKAE": TenKAE,
    "TwelveKAE": TwelveKAE,
}


# ---------------------------------------------------------------------------
# Data loading -- MNIST via torchvision, flattened to 784-d vectors in [0, 1]
# ---------------------------------------------------------------------------
def _load_mnist_flat(data_root, train=True, max_samples=None, seed=0):
    from torchvision import datasets

    ds = datasets.MNIST(root=data_root, train=train, download=True)
    X = ds.data.numpy().astype(np.float32) / 255.0  # N, 28, 28
    X = X.reshape(len(X), -1)  # N, 784

    if max_samples is not None and max_samples < len(X):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]

    return X


def load_dataset(data_root, train_frac=0.7, max_samples=3000, seed=0):
    """Returns X_train only (no y -- reconstruction target is the input itself)."""
    X = _load_mnist_flat(data_root, train=True, max_samples=max_samples, seed=seed)
    n_train = int(len(X) * train_frac)
    return X[:n_train]


def load_full_dataset(data_root, max_samples=3000, seed=0):
    """Return the full unsplit X (no normalization needed -- pixels already in [0,1])."""
    return _load_mnist_flat(data_root, train=True, max_samples=max_samples, seed=seed)


def make_loader(X_train, batch_size=64):
    X_t = torch.from_numpy(X_train)
    return DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Snapshot extraction -- identical to the MLP version (Linear-only model).
# ---------------------------------------------------------------------------
def extract_snapshot(model, loss_val, prev_loss=0.0, prev_all_w=None):
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
        [frac_pos, q10, q50, q90, skew, kurtosis, *layer_norms],
        dtype=np.float32,
    )

    return np.concatenate([scalars, dist_stats], axis=0)


def gradient_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run(
    data_root,
    out_csv,
    batch_size=64,
    n_epochs=5,
    lr=1e-3,
    is_malicious=False,
    flip_frac=0.5,
    seed=None,
    X_data=None,
    device=None,
    model_class_str="EightKAE",
):
    """Train an autoencoder on MNIST and collect per-batch snapshots.

    Same contract as snapshot_trainer.run(), minus target_col/y_data: the
    reconstruction target for a batch is the batch itself.
    """
    device = device or torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if X_data is not None:
        X_train = X_data.astype(np.float32)
    else:
        X_train = load_dataset(data_root)

    loader = make_loader(X_train, batch_size)

    if model_class_str not in MODEL_CLASS_MAP:
        raise RuntimeError(f"Invalid model class name: {model_class_str}")
    model = MODEL_CLASS_MAP[model_class_str](in_features=X_train.shape[1])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    snapshots = []
    prev_loss = 0.0
    prev_all_w = None

    for _ in range(n_epochs):
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, X_batch)  # reconstruction target == input

            optimizer.zero_grad()
            loss.backward()

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
        print(f"Saved {len(snapshots)} snapshots -> {out_csv}")

    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", required=True, help="Folder to (down)load MNIST into"
    )
    parser.add_argument("--out", default="snapshots_ae.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-class", choices=list(MODEL_CLASS_MAP.keys()), default="EightKAE"
    )
    parser.add_argument("--is-malicious", action="store_true")
    parser.add_argument("--flip-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run(
        data_root=args.data_root,
        out_csv=args.out,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        is_malicious=args.is_malicious,
        flip_frac=args.flip_frac,
        seed=args.seed,
        model_class_str=args.model_class,
    )
