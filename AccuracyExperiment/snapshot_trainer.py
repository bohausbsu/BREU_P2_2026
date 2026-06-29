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
            
def run(data_path, target_col, out_csv, batch_size=64, n_epochs=5, lr=1e-3, hidden=64):
    X_train, y_train = load_dataset(data_path, target_col)
    loader = make_loader(X_train, y_train, batch_size)

    model = SimpleMLP(in_features=X_train.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rows = []
    global_batch = 0

    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = gradient_norm(model)
            optimizer.step()

            mean_w, std_w, mean_b, std_b = scalar_stats((model))
            rows.append({
                "batch_idx": global_batch,
                "mean_weight": mean_w,
                "std_weight": std_w,
                "mean_bias": mean_b,
                "std_bias": std_b,
                "gradient_norm": grad_norm,
                "train_loss": loss.item(),
            })
            global_batch += 1
    fields = ["batch_idx", "mean_weight", "std_weight", "mean_bias", "std_bias", "gradient_norm", "train_loss"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} snapshots to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to input CSV")
    parser.add_argument("--target-col", required=True, help="Name of the target column")
    parser.add_argument("--out", default="snapshots.csv", help="Output snapshot CSV path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    run(
        data_path=args.dataset,
        target_col=args.target_col,
        out_csv=args.out,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        hidden=args.hidden,
    )

