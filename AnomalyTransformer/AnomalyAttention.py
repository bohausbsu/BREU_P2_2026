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
        print(f"Input x shape: (B={B}, N={N}, d_model={self.d_model})")
        print(f"  → {B} sequence(s), {N} moments in time, {self.d_model} numbers per moment\n")

        # Q, K, V
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_head).transpose(1,2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_head).transpose(1,2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_head).transpose(1,2)
        print(f"Q/K/V shape after split into heads: {Q.shape}")
        print(f"  → (batch={B}, n_heads={self.n_heads}, moments={N}, d_head={self.d_head})")
        print(f"  → each of the {self.n_heads} heads is working with a chunk of size {self.d_head}\n")

        # Sigma
        sigma = torch.sigmoid(self.W_sigma(x))
        print(f"Sigma shape: {sigma.shape}")
        print(f"  → one sigma value per moment per head: ({B}, {N}, {self.n_heads})")
        print(f"  → sigma values (first moment, all heads): {sigma[0, 0].detach().numpy()}")
        print(f"  → controls how wide each head's bell curve is\n")

        # Series Association S
        scale = math.sqrt(self.d_head)
        S = F.softmax(torch.matmul(Q, K.transpose(-1,-2)) / scale, dim=-1)
        print(f"S (series association) shape: {S.shape}")
        print(f"  → (batch={B}, n_heads={self.n_heads}, moments={N}, moments={N})")
        print(f"  → each row is a moment's attention spread across all {N} moments, learned from data")
        print(f"  → S[0, head=0] attention grid (rows=from, cols=to):\n{S[0, 0].detach().numpy().round(3)}\n")

        # Prior Association P
        idx = torch.arange(N, device=x.device).float()
        dist_sq = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
        print(f"Distance squared grid shape: {dist_sq.shape}")
        print(f"  → raw time distances between every pair of moments:")
        print(f"  {dist_sq.numpy()}\n")

        sigma_t = sigma.permute(0, 2, 1).unsqueeze(-1)
        P = torch.exp(-1 * dist_sq / (2 * sigma_t ** 2 + 1e-8))
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        print(f"P (prior association) shape: {P.shape}")
        print(f"  → (batch={B}, n_heads={self.n_heads}, moments={N}, moments={N})")
        print(f"  → each row is a moment's bell curve spread across all {N} moments")
        print(f"  → P[0, head=0] prior grid (rows=from, cols=to):\n{P[0, 0].detach().numpy().round(3)}\n")

        # Reconstruct Z
        Z = torch.matmul(S, V)
        print(f"Z after attention weighted blend shape: {Z.shape}")
        print(f"  → each moment is now a mix of values, weighted by what S said to pay attention to\n")

        Z = Z.transpose(1,2).contiguous().view(B, N, self.d_model)
        print(f"Z after heads stitched back together shape: {Z.shape}")
        print(f"  → heads concatenated, but not yet synthesized\n")

        Z = self.out_proj(Z)
        print(f"Z after out_proj shape: {Z.shape}")
        print(f"  → every output now influenced by every input across all heads")
        print(f"  → this is the final unified representation\n")

        return Z, P, S
    def association_discrepancy(P_list, S_list):
        """
        P_list, S_list: lists of (B, H, N, N) tensors, on per layer.
        Returns (B, N) discrepancy scores.
        """
        eps = 1e-8
        total = None
        for P, S in zip(P_list, S_list):
            P_avg = P.mean(dim=1)
            S_avg = S.mean(dim=1)

            kl_ps = (P_avg * torch.log(P_avg / (S_avg + eps) + eps)).sum(dim=-1)
            kl_sp = (S_avg * torch.log(S_avg / (P_avg + eps) + eps)).sum(dim=-1)
            layer_disc = kl_ps + kl_sp
            total = layer_disc if total is None else total + layer_disc
        return total / len(P_list)


# --- run it ---
B = 1        # one sequence
N = 5        # five moments in time
d_model = 8  # each moment described by 8 numbers
n_heads = 2  # two specialist heads

torch.manual_seed(0)  # makes random numbers repeatable so you get the same output every run
x = torch.randn(B, N, d_model)
print("=" * 55)
print(f"Running AnomalyAttention with a made-up input tensor")
print(f"  B={B} batch, N={N} moments, d_model={d_model}, n_heads={n_heads}")
print("=" * 55 + "\n")

model = AnomalyAttention(d_model=d_model, n_heads=n_heads)
Z, P, S = model(x)

print("=" * 55)
print("Final outputs:")
print(f"  Z shape: {Z.shape}  ← the processed sequence")
print(f"  S shape: {S.shape}  ← learned attention pattern")
print(f"  P shape: {P.shape}  ← prior attention pattern")
print("=" * 55)