import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AnomalyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        # d_model is the total feature size for each time step.
        # n_heads splits that feature space into smaller chunks so multiple
        # attention patterns can be learned in parallel.
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Learns one sigma per time step per head.
        # Sigma controls how wide or narrow the Gaussian prior should be.
        self.W_sigma = nn.Linear(d_model, n_heads)

        # Combines all heads back into one final d_model-sized representation.
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (B, N, d_model)
        # B = batch size
        # N = number of time steps
        # d_model = number of features at each time step
        B, N, _ = x.shape

        print(f"Input x shape: {x.shape}")
        print(f"  B={B}: number of sequences")
        print(f"  N={N}: number of time steps per sequence")
        print(f"  d_model={self.d_model}: features per time step\n")

        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        print(f"Q, K, V shape: {Q.shape}")
        print("  Shape meaning: (batch, head, time step, features per head)")
        print(f"  Each of the {self.n_heads} heads sees {self.d_head} features.\n")

        sigma = torch.sigmoid(self.W_sigma(x))

        print(f"Sigma shape: {sigma.shape}")
        print("  Shape meaning: (batch, time step, head)")
        print("  Small sigma  -> attention prior stays close to the current time step")
        print("  Large sigma  -> attention prior spreads farther across time")
        print(f"  Example sigma values for first time step: {sigma[0, 0].detach().cpu().numpy()}\n")

        scale = math.sqrt(self.d_head)
        S = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / scale, dim=-1)

        print(f"S shape: {S.shape}")
        print("  S is the learned attention pattern.")
        print("  Rows: current time step")
        print("  Columns: time step being attended to")
        print("  Each row sums to 1.")
        print(f"  S[0, head=0]:\n{S[0, 0].detach().cpu().numpy().round(3)}\n")

        idx = torch.arange(N, device=x.device).float()
        dist_sq = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2

        print(f"Distance grid shape: {dist_sq.shape}")
        print("  This measures how far apart every pair of time steps is.")
        print(f"  dist_sq:\n{dist_sq.detach().cpu().numpy()}\n")

        sigma_t = sigma.permute(0, 2, 1).unsqueeze(-1)

        P = torch.exp(-dist_sq / (2 * sigma_t ** 2 + 1e-8))
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)

        print(f"P shape: {P.shape}")
        print("  P is the prior attention pattern based only on time distance.")
        print("  Rows: current time step")
        print("  Columns: time step being attended to")
        print("  Each row is a Gaussian-like curve over nearby time steps.")
        print(f"  P[0, head=0]:\n{P[0, 0].detach().cpu().numpy().round(3)}\n")

        Z = torch.matmul(S, V)

        print(f"Z shape after S @ V: {Z.shape}")
        print("  Each time step now contains information gathered from other time steps.")
        print("  The weights came from the learned attention matrix S.\n")

        Z = Z.transpose(1, 2).contiguous().view(B, N, self.d_model)

        print(f"Z shape after combining heads: {Z.shape}")
        print("  The separate head outputs are now concatenated back together.\n")

        Z = self.out_proj(Z)

        print(f"Final Z shape: {Z.shape}")
        print("  Final meaning: one processed feature vector per time step.\n")

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