import json 

with open("vocab/stoi.json", "r", encoding="utf-8") as f:
    stoi = json.load(f)

with open("vocab/itos.json", "r", encoding="utf-8") as f:
    itos = json.load(f)

itos = {int(k): v for k, v in itos.items()}

vocab_size = len(stoi)
batch_size = 100
block_size = 80
embed_dim = 512
num_heads = 8
FFN_depth = 2048
encoder_layers = 6
epochs = 1000
lr = 3e-4