import torch
import torch.nn as nn

class MemoryAdapter(nn.Module):
    def __init__(self, mem_len=20):
        super().__init__()
        self.mem_len = mem_len

    def forward(self, emb):
        """
        emb: (256,) IGNN output
        → (1, mem_len, 256)
        """
        emb = emb.unsqueeze(0).unsqueeze(1)   # (1, 1, 256)
        emb = emb.repeat(1, self.mem_len, 1)  # (1, mem_len, 256)
        return emb
