import torch
import sentencepiece as spm
import stanza
import argparse

from ignn.encoder import IGNNEncoder
from transformer_decoder import TransformerDecoder
from encoder_memory import MemoryAdapter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

MAX_NODES = 60
MEM_LEN = 20
EMBED_DIM = 256
MAX_LEN = 60   # max English output length


# ------------------------------------------------------
# Load SentencePiece models
# ------------------------------------------------------
sp_te = spm.SentencePieceProcessor()
sp_te.load("tokenizer/spm_te.model")

sp_en = spm.SentencePieceProcessor()
sp_en.load("tokenizer/spm_en.model")

EN_BOS = sp_en.bos_id()
EN_EOS = sp_en.eos_id()
EN_UNK = sp_en.unk_id()
EN_VOCAB = sp_en.get_piece_size()


# ------------------------------------------------------
# Load IGNN encoder
# ------------------------------------------------------
encoder = IGNNEncoder(
    vocab_size=sp_te.get_piece_size(),
    dim=EMBED_DIM,
    layers=4,
    heads=4
).to(DEVICE)

encoder.load_state_dict(torch.load("models/ignn_encoder_epoch5.pt", map_location=DEVICE))
encoder.eval()


# ------------------------------------------------------
# Load transformer decoder
# ------------------------------------------------------
decoder = TransformerDecoder(
    vocab_size=EN_VOCAB,
    embed_dim=EMBED_DIM,
    num_layers=4,
    num_heads=4
).to(DEVICE)

# Change this if your latest decoder epoch is different
decoder.load_state_dict(torch.load("models/decoder_epoch3.pt", map_location=DEVICE))
decoder.eval()


# ------------------------------------------------------
# Load Memory Adapter
# ------------------------------------------------------
memory_adapter = MemoryAdapter(mem_len=MEM_LEN).to(DEVICE)


# ------------------------------------------------------
# Initialize STANZA for Telugu
# ------------------------------------------------------
print("Loading Stanza Telugu...")
nlp = stanza.Pipeline("te", processors="tokenize,pos,lemma,depparse")


# ------------------------------------------------------
# Build graph from a Telugu sentence
# ------------------------------------------------------
def build_graph(sentence):
    doc = nlp(sentence)
    parsed = doc.sentences[0]

    tokens = [w.text for w in parsed.words]

    edges = []
    for w in parsed.words:
        if w.head > 0:
            edges.append((w.head - 1, w.id - 1))

    # encode Telugu tokens to IDs
    token_ids = []
    for t in tokens:
        ids = sp_te.encode(t, out_type=int)
        token_ids.append(ids[0] if ids else sp_te.unk_id())

    # build adjacency matrix
    N = min(len(token_ids), MAX_NODES)
    token_ids = token_ids[:N]

    adj = torch.zeros((N, N))
    for parent, child in edges:
        if parent < N and child < N:
            adj[parent][child] = 1
            adj[child][parent] = 1

    x = torch.tensor(token_ids, device=DEVICE)
    adj = adj.to(DEVICE)

    return x, adj


# ------------------------------------------------------
# Auto-regressive decoding loop
# ------------------------------------------------------
def generate_english(memory):
    generated = [EN_BOS]

    for _ in range(MAX_LEN):
        inp = torch.tensor([generated], device=DEVICE)
        logits = decoder(inp, memory)
        next_token = torch.argmax(logits[0, -1]).item()

        if next_token == EN_EOS:
            break

        generated.append(next_token)

    return sp_en.decode(generated[1:])  # skip BOS


# ------------------------------------------------------
# Full translation pipeline
# ------------------------------------------------------
def translate(text):
    x, adj = build_graph(text)

    with torch.no_grad():
        emb = encoder(x, adj)
        memory = memory_adapter(emb)

        english = generate_english(memory)

    return english


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    if args.text is not None:
        telugu = args.text
    else:
        # Read from input.txt
        with open("input.txt", "r", encoding="utf-8") as f:
            telugu = f.read().strip()

    result = translate(telugu)

    print("\n--------------------------------------")
    print("TELUGU:", telugu)
    print("ENGLISH:", result)
    print("--------------------------------------\n")
