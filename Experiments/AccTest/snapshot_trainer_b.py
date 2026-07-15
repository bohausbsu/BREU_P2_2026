"""
CNN variant of the snapshot trainer, for the Cats vs Dogs binary image
classification task.

This mirrors snapshot_trainer.py exactly in spirit:
  - a handful of model classes sized to ~6k / 8k / 10k / 12k parameters
  - a `run()` function that trains the model while collecting per-batch
    weight/gradient snapshots (with an optional Byzantine gradient attack)
  - an `extract_snapshot()` feature vector identical in *definition* to the
    MLP version, just generalized to pull weights/grads from Conv2d layers
    as well as Linear layers.

Expected dataset layout (torchvision ImageFolder convention)::

    <data_root>/
        Cat/  *.jpg
        Dog/  *.jpg

    (the standard layout for the Kaggle "Dogs vs Cats" / Microsoft
    "PetImages" datasets once corrupt files have been removed)

Because parameter count is driven entirely by the conv channel widths (an
AdaptiveAvgPool2d collapses the spatial dimensions before the classifier
head), image resolution can be changed freely via --image-size without
touching the parameter budget below.
"""

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Pillow is required to load the Cats vs Dogs dataset (`pip install Pillow`)"
    ) from e


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class FourKCNN(nn.Module):
    def __init__(self, width=14):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(2 * w, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SixKCNN(nn.Module):
    def __init__(self, width=14):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 4 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4 * w, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EightKCNN(nn.Module):
    def __init__(self, width=14):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 4 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * w, 8 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8 * w, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TenKCNN(nn.Module):
    def __init__(self, width=14):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 4 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * w, 8 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * w, 16 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16 * w, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TwelveKCNN(nn.Module):
    def __init__(self, width=14):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 4 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * w, 8 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * w, 16 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 * w, 32 * w, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32 * w, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


MODEL_CLASS_MAP = {
    "FourKCNN": FourKCNN,
    "SixKCNN": SixKCNN,
    "EightKCNN": EightKCNN,
    "TenKCNN": TenKCNN,
    "TwelveKCNN": TwelveKCNN,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _list_image_paths(data_root):
    """Walk <data_root>/Cat and <data_root>/Dog (case-insensitive) for images."""
    classes = {}
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if not os.path.isdir(full):
            continue
        low = entry.lower()
        if low.startswith("cat"):
            classes[entry] = 0.0
        elif low.startswith("dog"):
            classes[entry] = 1.0

    if len(classes) != 2:
        raise RuntimeError(
            f"Expected a 'Cat' and a 'Dog' subfolder under {data_root}, "
            f"found: {list(classes.keys())}"
        )

    paths, labels = [], []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for cls_name, label in classes.items():
        cls_dir = os.path.join(data_root, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label)

    return paths, labels


def _load_image_array(path, image_size):
    """Load + resize an image to (3, H, W) float32 in [0, 1]. Returns None on failure."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, C
            return np.transpose(arr, (2, 0, 1))  # C, H, W
    except Exception:
        return None


def _load_all_images(data_root, image_size=32, max_samples=None, seed=0):
    paths, labels = _list_image_paths(data_root)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    labels = [labels[i] for i in order]

    if max_samples is not None:
        paths = paths[:max_samples]
        labels = labels[:max_samples]

    X, y = [], []
    for p, lab in zip(paths, labels):
        arr = _load_image_array(p, image_size)
        if arr is not None:
            X.append(arr)
            y.append(lab)

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def load_dataset(data_root, image_size=32, train_frac=0.7, max_samples=1500, seed=0):
    """Returns normalized (X_train, y_train), matching the MLP trainer's contract."""
    X, y = _load_all_images(data_root, image_size, max_samples, seed)

    n_train = int(len(X) * train_frac)
    X_train, y_train = X[:n_train], y[:n_train]

    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std

    return X_train, y_train


def load_full_dataset(data_root, image_size=32, max_samples=1500, seed=0):
    """Return the full (unnormalized) (X, y), matching the MLP trainer's contract."""
    return _load_all_images(data_root, image_size, max_samples, seed)


def make_loader(X_train, y_train, batch_size=32):
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Snapshot extraction -- identical feature definition to the MLP version,
# just widened to also pick up Conv2d weights.
# ---------------------------------------------------------------------------
def extract_snapshot(model, loss_val, prev_loss=0.0, prev_all_w=None):
    weights, grads = [], []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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
    batch_size=32,
    n_epochs=5,
    lr=1e-3,
    is_malicious=False,
    flip_frac=0.5,
    seed=None,
    X_data=None,
    y_data=None,
    device=None,
    model_class_str="EightKCNN",
    image_size=32,
):
    """Train a small CNN on Cats vs Dogs and collect per-batch snapshots.

    Same contract as snapshot_trainer.run(): pass in pre-split X_data/y_data
    (as returned by load_full_dataset) to skip re-loading from disk on every
    call, which matters a lot here since image loading is comparatively slow.
    """
    device = device or torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if X_data is not None and y_data is not None:
        X_train = X_data.astype(np.float32)
        y_train = y_data.astype(np.float32)
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
    else:
        X_train, y_train = load_dataset(data_root, image_size=image_size)

    loader = make_loader(X_train, y_train, batch_size)

    if model_class_str not in MODEL_CLASS_MAP:
        raise RuntimeError(f"Invalid model class name: {model_class_str}")
    model = MODEL_CLASS_MAP[model_class_str]()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    snapshots = []
    prev_loss = 0.0
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
                        if isinstance(m, (nn.Linear, nn.Conv2d))
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
        "--data-root", required=True, help="Folder with Cat/ and Dog/ subfolders"
    )
    parser.add_argument("--out", default="snapshots_cnn.csv")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-class",
        choices=list(MODEL_CLASS_MAP.keys()),
        default="EightKCNN",
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
        image_size=args.image_size,
    )
