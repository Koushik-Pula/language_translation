import torch
import torch.nn as nn
from ignn.layers import IGNNLayer


class IGNNEncoder(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=4, heads=4):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            IGNNLayer(dim, heads=heads, dropedge=0.1)
            for _ in range(layers)
        ])

        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, token_ids, adj):
        x = self.emb(token_ids)

        for layer in self.layers:
            x = layer(x, adj)

        x = x.transpose(0, 1)
        pooled = self.pool(x).squeeze(-1)

        return pooled
