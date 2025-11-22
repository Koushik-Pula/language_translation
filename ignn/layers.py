import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DropEdge:
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate

    def __call__(self, adj):
        if self.drop_rate <= 0:
            return adj

        mask = (torch.rand_like(adj) > self.drop_rate).float()
        return adj * mask


class IGNNLayer(nn.Module):
    def __init__(self, dim, heads=4, dropedge=0.1):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.dk = dim // heads

        # Multi-head projections
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # Output projection
        self.W_o = nn.Linear(dim, dim)

        # DropEdge
        self.dropedge = DropEdge(dropedge)

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        N = x.size(0)

        # Residual connection
        residual = x

        # DropEdge
        adj = self.dropedge(adj)

        # Linear projections
        Q = self.W_q(x).view(N, self.heads, self.dk)
        K = self.W_k(x).view(N, self.heads, self.dk)
        V = self.W_v(x).view(N, self.heads, self.dk)

        # Attention scores (masked by adjacency)
        attn_scores = torch.einsum("ihd,jhd->ijh", Q, K) / (self.dk ** 0.5)
        attn_scores = attn_scores * adj.unsqueeze(-1)  # mask neighbors

        attn_weights = F.softmax(attn_scores, dim=1)  # softmax over neighbors

        # Weighted sum
        out = torch.einsum("ijh,jhd->ihd", attn_weights, V)

        # Merge heads
        out = out.reshape(N, self.dim)

        # Output projection
        out = self.W_o(out)

        # Residual + LayerNorm
        out = self.norm(out + residual)

        return out
