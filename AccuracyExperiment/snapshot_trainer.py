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
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)

def make_loader(X_train, y_train, batch_size=64):
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

def load_dataset(path, target_col, start_frac=0.0, end_frac=1.0):
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().str.lower().unique()
        if set(vals).issubset({"yes", "no"}):
            df[col] = df[col].str.lower().map({"yes": 1.0, "no": 0.0})
    df = df.select_dtypes(include="number").dropna()

    y = df[target_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)

    n = len(X)
    i_start = int(n * start_frac)
    i_end   = int(n * end_frac)
    X_slice = X[i_start:i_end]
    y_slice = y[i_start:i_end]

    X_mean, X_std = X_slice.mean(0), X_slice.std(0) + 1e-8
    X_slice = (X_slice - X_mean) / X_std

    return X_slice, y_slice

def scalar_stats(model):
    weights, biases = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.data.flatten())
            if module.bias is not None:
                biases.append(module.bias.data.flatten())
    all_w = torch.cat(weights)
    all_b = torch.cat(biases)
    return all_w.mean().item(), all_w.std().item(), all_b.mean().item(), all_b.std().item()

def gradient_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)

def snapshot_weights(model):
    return torch.cat([p.data.flatten() for p in model.parameters()]).clone()

def weight_update_norm(prev_snap, model):
    curr = torch.cat([p.data.flatten() for p in model.parameters()])
    return (curr - prev_snap).norm(2).item()

def run(data_path, target_col, out_csv, batch_size=64, n_epochs=5, lr=1e-3, hidden=64,
        start_frac=0.0, end_frac=1.0,
        is_malicious=False, poison_frac=0.5, grad_noise_scale=0.5,
        seed=None):
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    X_train, y_train = load_dataset(data_path, target_col, start_frac, end_frac)

    if is_malicious:
        rng = np.random.default_rng()
        n_poison = int(len(y_train) * poison_frac)
        idx = rng.choice(len(y_train), n_poison, replace=False)
        y_train = y_train.copy()
        # Reflect targets around the midpoint (high→low, low→high)
        y_min, y_max = y_train.min(), y_train.max()
        y_train[idx] = y_max + y_min - y_train[idx]

    loader = make_loader(X_train, y_train, batch_size)

    model = SimpleMLP(in_features=X_train.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rows = []
    global_batch = 0
    prev_snap = snapshot_weights(model)

    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()

            if is_malicious and grad_noise_scale > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        noise_std = grad_noise_scale * p.grad.abs().mean().clamp(min=1e-8)
                        p.grad.add_(torch.randn_like(p.grad) * noise_std)

            grad_norm_val = gradient_norm(model)
            optimizer.step()

            update_norm = weight_update_norm(prev_snap, model)
            prev_snap = snapshot_weights(model)

            mean_w, std_w, mean_b, std_b = scalar_stats(model)
            rows.append({
                "batch_idx":          global_batch,
                "mean_weight":        mean_w,
                "std_weight":         std_w,
                "mean_bias":          mean_b,
                "std_bias":           std_b,
                "gradient_norm":      grad_norm_val,
                "train_loss":         loss.item(),
                "weight_update_norm": update_norm,
            })
            global_batch += 1

    fields = ["batch_idx", "mean_weight", "std_weight", "mean_bias", "std_bias",
              "gradient_norm", "train_loss", "weight_update_norm"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} snapshots to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",          required=True)
    parser.add_argument("--target-col",       required=True)
    parser.add_argument("--out",              default="snapshots.csv")
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--n-epochs",         type=int,   default=5)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--hidden",           type=int,   default=64)
    parser.add_argument("--start-frac",       type=float, default=0.0)
    parser.add_argument("--end-frac",         type=float, default=1.0)
    parser.add_argument("--is-malicious",     action="store_true")
    parser.add_argument("--poison-frac",      type=float, default=0.5)
    parser.add_argument("--grad-noise-scale", type=float, default=0.5)
    parser.add_argument("--seed",             type=int,   default=None)
    args = parser.parse_args()

    run(
        data_path=args.dataset,
        target_col=args.target_col,
        out_csv=args.out,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        hidden=args.hidden,
        start_frac=args.start_frac,
        end_frac=args.end_frac,
        is_malicious=args.is_malicious,
        poison_frac=args.poison_frac,
        grad_noise_scale=args.grad_noise_scale,
        seed=args.seed,
    )
