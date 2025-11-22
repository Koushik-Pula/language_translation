import torch
import torch.nn as nn
import math

# -----------------------------------------------------
# Positional Encoding
# -----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


# -----------------------------------------------------
# TRANSFORMER DECODER
# -----------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_tokens, memory):
        """
        tgt_tokens: (1, T)
        memory: (1, L, 256) IGNN memory
        """

        tgt_embed = self.embedding(tgt_tokens)
        tgt_embed = self.pos(tgt_embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_embed.size(1)
        ).to(tgt_embed.device)

        out = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits
