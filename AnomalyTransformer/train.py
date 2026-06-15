import torch
import numpy as np
from AnomalyAttention import AnomalyTransformer
from dataset import get_dataloader

def train(model, dataloader, n_epochs=10, lam=3, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x in dataloader:
            x_hat, P_list, S_list = model(x)
            loss = AnomalyTransformer.minimize_loss(x, x_hat, P_list, S_list, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x_hat, P_list, S_list = model(x)
            loss = AnomalyTransformer.maximize_loss(x, x_hat, P_list, S_list, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} - loss: {total_loss:.4f}")

def get_threshold(model, val_dataloader, r=0.01):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for x in val_dataloader:
            x_hat, P_list, S_list = model(x)
            scores = AnomalyTransformer.anomaly_score(x, x_hat, P_list, S_list)
            all_scores.append(scores)
    all_scores = torch.cat(all_scores, dim=0).view(-1)
    threshold = torch.quantile(all_scores, 1 - r)
    return threshold

def predict(model, test_dataloader, threshold):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for x in test_dataloader:
            x_hat, P_list, S_list = model(x)
            scores = AnomalyTransformer.anomaly_score(x, x_hat, P_list, S_list)
            all_scores.append(scores)
    all_scores = torch.cat(all_scores, dim=0).view(-1)
    predictions = (all_scores > threshold).int()
    return predictions, all_scores

if __name__ == "__main__":
    data = np.load("blank.npy")
    dataloader = get_dataloader(data, window_size=100, batch_size=32)

    model = AnomalyTransformer(d_input=..., d_model=512, n_heads=8, d_ff=2048, n_layers=3)
    train(model, dataloader)