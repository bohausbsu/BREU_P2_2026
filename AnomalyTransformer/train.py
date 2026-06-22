import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AnomalyAttention import AnomalyTransformer
from dataset import load_csv, split_data, normalize_splits, make_loaders


def train(model, dataloader, val_loader=None, n_epochs=10, lam=3, lr=1e-4, patience=5, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_min_loss = 0
        total_max_loss = 0
        for x in dataloader:
            x = x.to(device)

            x_hat, P_list, S_list = model(x)
            min_loss = AnomalyTransformer.minimize_loss(x, x_hat, P_list, S_list, lam)
            optimizer.zero_grad()
            min_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            x_hat, P_list, S_list = model(x)
            max_loss = AnomalyTransformer.maximize_loss(x, x_hat, P_list, S_list, lam)
            optimizer.zero_grad()
            max_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_min_loss += min_loss.item()
            total_max_loss += max_loss.item()

        n_batches = len(dataloader)
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    x_hat, P_list, S_list = model(x)
                    val_loss += ((x - x_hat) ** 2).mean().item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{n_epochs} - min_loss: {total_min_loss/n_batches:.4f}  max_loss: {total_max_loss/n_batches:.4f}  val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
        else:
            print(f"Epoch {epoch+1}/{n_epochs} - min_loss: {total_min_loss/n_batches:.4f}  max_loss: {total_max_loss/n_batches:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss: {best_val_loss:.4f})")


def get_window_scores(model, dataloader, device="cpu"):
    """Run the model over a dataloader and collect anomaly scores. Returns (num_windows, window_size)."""
    model.eval()
    all_scores = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x_hat, P_list, S_list = model(x)
            scores = AnomalyTransformer.anomaly_score(x, x_hat, P_list, S_list)
            all_scores.append(scores.cpu())
    return torch.cat(all_scores, dim=0)


def get_threshold(window_scores, r=0.01):
    return torch.quantile(window_scores.view(-1), 1 - r)


def windows_to_timeline(window_scores, total_len):
    """Average overlapping sliding-window scores back onto a single per-timestep timeline."""
    num_windows, window_size = window_scores.shape
    index = torch.arange(num_windows).unsqueeze(1) + torch.arange(window_size).unsqueeze(0)
    index = index.reshape(-1)

    timeline = torch.zeros(total_len)
    counts = torch.zeros(total_len)
    timeline.index_put_((index,), window_scores.reshape(-1), accumulate=True)
    counts.index_put_((index,), torch.ones_like(index, dtype=torch.float32), accumulate=True)
    return timeline / counts.clamp(min=1)


def plot_anomalies(raw_data, flagged, feature_names, out_path):
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 6), sharex=True, squeeze=False)
    for i, name in enumerate(feature_names):
        ax = axes[i, 0]
        ax.plot(raw_data[:, i], label=name, color="steelblue")
        ax.scatter(
            np.arange(len(raw_data))[flagged],
            raw_data[flagged, i],
            color="red", s=10, label="flagged anomaly", zorder=3,
        )
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Timestep (test split)")
    fig.suptitle("Anomaly Transformer — flagged timesteps on test set")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an Anomaly Transformer on a CSV time series.")
    parser.add_argument("--csv", default="dataset/ml_training_epoch_weight_bias_5000.csv")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lam", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--r", type=float, default=0.01)
    parser.add_argument("--out", default="anomaly_plot.png")
    parser.add_argument("--model-out", default="anomaly_transformer.pt")
    parser.add_argument("--load-model", default=None, help="Path to a saved .pt file to load instead of training")
    parser.add_argument("--no-labels", action="store_true", help="CSV has no ground-truth label column; skip precision/recall evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    na_vals = [args.na_values] if args.na_values is not None else None
    data, feature_names = load_csv(args.csv, skip_cols=2, has_header=True,
                                   sep=args.sep, decimal=args.decimal, na_values=na_vals)
    if not args.no_labels:
        gt_labels = data[:, -1].astype(int)
        data = data[:, :-1]
        feature_names = feature_names[:-1]
    else:
        gt_labels = None
    print(f"Loaded {args.csv}: {data.shape[0]} rows, features={feature_names}")

    train_data, val_data, test_data = split_data(data, args.train_frac)
    raw_test_data = test_data.copy()
    train_data, val_data, test_data = normalize_splits(train_data, val_data, test_data)
    train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, args.window_size, args.batch_size)

    model = AnomalyTransformer(d_input=data.shape[1], d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff, n_layers=args.n_layers).to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model weights from {args.load_model}")
    else:
        train(model, train_loader, val_loader, n_epochs=args.n_epochs, lam=args.lam, lr=args.lr, patience=args.patience, device=device)
        torch.save(model.state_dict(), args.model_out)
        print(f"Saved model weights to {args.model_out}")

    val_scores = get_window_scores(model, val_loader, device=device)
    val_timeline = windows_to_timeline(val_scores, len(val_data))
    threshold = get_threshold(val_timeline, r=args.r)
    print(f"Threshold (from val set): {threshold:.4f}")

    test_scores = get_window_scores(model, test_loader, device=device)
    timeline_scores = windows_to_timeline(test_scores, len(raw_test_data))
    flagged = (timeline_scores > threshold).numpy()
    print(f"Flagged {flagged.sum()} / {len(flagged)} timesteps as anomalous")

    if gt_labels is not None:
        test_labels = gt_labels[int(len(gt_labels) * 0.85):]
        tp = int((flagged & (test_labels == 1)).sum())
        fp = int((flagged & (test_labels == 0)).sum())
        fn = int((~flagged & (test_labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")

    plot_anomalies(raw_test_data, flagged, feature_names, args.out)