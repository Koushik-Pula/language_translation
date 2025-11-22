import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

from ignn.encoder import IGNNEncoder
from transformer_decoder import TransformerDecoder
from encoder_memory import MemoryAdapter

# ======================================================
# CONFIG
# ======================================================

TRAIN_EN = "../datasets_translation/cleaned/train.en"
TRAIN_TE = "../datasets_translation/cleaned/train.te"
VALID_EN = "../datasets_translation/cleaned/valid.en"

GRAPH_DIR = "graphs_100k/train"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

MAX_NODES = 60
MAX_LEN = 60
EPOCHS = 3
LR = 1e-4
PRINT_EVERY = 200

EMBED_DIM = 256
MEM_LEN = 20

# ======================================================
# LOAD TOKENIZERS
# ======================================================

sp_te = spm.SentencePieceProcessor()
sp_te.load("tokenizer/spm_te.model")

sp_en = spm.SentencePieceProcessor()
sp_en.load("tokenizer/spm_en.model")

TE_UNK = sp_te.unk_id()
EN_VOCAB = sp_en.get_piece_size()

# ======================================================
# LOAD IGNN ENCODER
# ======================================================

encoder = IGNNEncoder(
    vocab_size=sp_te.get_piece_size(),
    dim=EMBED_DIM,
    layers=4,
    heads=4
).to(DEVICE)

encoder.load_state_dict(torch.load("models/ignn_encoder_epoch5.pt"))
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False


# ======================================================
# DECODER + MEMORY ADAPTER
# ======================================================

decoder = TransformerDecoder(
    vocab_size=EN_VOCAB,
    embed_dim=EMBED_DIM,
    num_layers=4,
    num_heads=4
).to(DEVICE)

memory_adapter = MemoryAdapter(mem_len=MEM_LEN).to(DEVICE)

optimizer = optim.Adam(decoder.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# ======================================================
# UTIL FUNCTIONS
# ======================================================

def load_graph(idx):
    path = os.path.join(GRAPH_DIR, f"{idx}.pkl")
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        g = pickle.load(f)

    tokens = g["tokens"][:MAX_NODES]
    edges = g["edges"]

    tok_ids = []
    for t in tokens:
        ids = sp_te.encode(t, out_type=int)
        tok_ids.append(ids[0] if ids else TE_UNK)

    N = len(tok_ids)
    if N == 0:
        return None

    adj = torch.zeros((N, N))
    for p, c in edges:
        if p < N and c < N:
            adj[p][c] = adj[c][p] = 1

    return (
        torch.tensor(tok_ids, device=DEVICE),
        adj.to(DEVICE)
    )


def encode_en(text):
    ids = sp_en.encode(text, out_type=int)
    return [sp_en.bos_id()] + ids + [sp_en.eos_id()]


# ======================================================
# TRAINING LOOP
# ======================================================

def train():
    with open(TRAIN_EN, "r", encoding="utf-8") as f:
        en_lines = f.readlines()

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        for i in range(len(en_lines)):

            graph = load_graph(i)
            if graph is None:
                continue

            x, adj = graph

            with torch.no_grad():
                emb = encoder(x, adj)
                memory = memory_adapter(emb)

            tgt_ids = encode_en(en_lines[i])
            tgt_ids = tgt_ids[:MAX_LEN]
            tgt = torch.tensor(tgt_ids, device=DEVICE).unsqueeze(0)

            dec_in = tgt[:, :-1]
            dec_tar = tgt[:, 1:].reshape(-1)

            logits = decoder(dec_in, memory)
            logits = logits.reshape(-1, EN_VOCAB)

            loss = criterion(logits, dec_tar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % PRINT_EVERY == 0:
                print(f"Step {i} | Loss = {loss.item():.4f}")

        torch.save(decoder.state_dict(), f"models/decoder_epoch{epoch+1}.pt")
        print(f"Saved: decoder_epoch{epoch+1}.pt")


if __name__ == "__main__":
    train()
