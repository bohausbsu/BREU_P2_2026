import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AnomalyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_sigma = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape

        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        sigma = F.softplus(self.W_sigma(x))
        scale = math.sqrt(self.d_head)
        S = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / scale, dim=-1)

        idx = torch.arange(N, device=x.device, dtype=x.dtype)
        dist_sq = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2

        sigma_t = sigma.permute(0, 2, 1).unsqueeze(-1)
        P = torch.exp(-dist_sq / (2 * sigma_t ** 2 + 1e-8))
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)

        Z = torch.matmul(S, V)
        Z = Z.transpose(1, 2).contiguous().view(B, N, self.d_model)
        Z = self.out_proj(Z)

        return Z, P, S

    @staticmethod
    def association_discrepancy(P_list, S_list):
        """P_list, S_list: lists of (B, H, N, N) tensors, one per layer.
        Returns (B, N) discrepancy scores."""
        eps = 1e-8
        total = None
        for P, S in zip(P_list, S_list):
            P_avg, S_avg = P.mean(dim=1), S.mean(dim=1)
            kl_ps = (P_avg * torch.log(P_avg / (S_avg + eps) + eps)).sum(dim=-1)
            kl_sp = (S_avg * torch.log(S_avg / (P_avg + eps) + eps)).sum(dim=-1)
            total = kl_ps + kl_sp if total is None else total + kl_ps + kl_sp
        return total / len(P_list)


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        Z, P, S = self.attn(x)
        x = self.norm1(x + self.drop(Z))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, P, S


class AnomalyTransformer(nn.Module):
    def __init__(self, d_input, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            AnomalyTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.reconstruction = nn.Linear(d_model, d_input)

    def forward(self, x):
        x = self.embedding(x)
        P_list, S_list = [], []
        for block in self.blocks:
            x, P, S = block(x)
            P_list.append(P)
            S_list.append(S)
        x_hat = self.reconstruction(x)
        return x_hat, P_list, S_list

    @staticmethod
    def minimize_loss(x, x_hat, P_list, S_list, lam):
        disc = AnomalyAttention.association_discrepancy(P_list, [S.detach() for S in S_list]).mean()
        return ((x - x_hat) ** 2).mean() + lam * disc

    @staticmethod
    def maximize_loss(x, x_hat, P_list, S_list, lam):
        disc = AnomalyAttention.association_discrepancy([P.detach() for P in P_list], S_list).mean()
        return ((x - x_hat) ** 2).mean() - lam * disc

    @staticmethod
    def anomaly_score(x, x_hat, P_list, S_list):
        disc = AnomalyAttention.association_discrepancy(P_list, S_list)
        recon_err = ((x - x_hat) ** 2).sum(dim=-1)
        return torch.softmax(-disc, dim=-1) * recon_err