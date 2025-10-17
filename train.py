import torch
import torch.nn.functional as F
import torch.optim as optim
from model_fluidgpt import GPT
from input_loader import data, stoi, itos

encode = lambda seq: [stoi[s] for s in seq]
decode = lambda ids: [itos[i] for i in ids]
Y = torch.tensor(data[1:], dtype=torch.long)
X = torch.tensor(data[:-1], dtype=torch.long)
spl_idxs = [i+1 for i, x in enumerate(X) if x == 0]

split_ratio = 0.9
split_idx = int(len(spl_idxs) * split_ratio)
X_train, X_val = X[:spl_idxs[split_idx]], X[spl_idxs[split_idx]:]
Y_train, Y_val = Y[:spl_idxs[split_idx]], Y[spl_idxs[split_idx]:]

# ===== Hyperparams =====
vocab_size = len(stoi)
batch_size = 100
block_size = 80
embed_dim = 512
num_heads = 8
FFN_depth = 2048
encoder_layers = 6
epochs = 1000
lr = 3e-4

def get_batch(X_source, Y_source, block_size, batch_size, device):
    Xb, Yb = [], []
    while len(Xb) < batch_size:
        rand_idx = torch.randint(0, len(spl_idxs)-1, ()).item()
        start_idx = spl_idxs[rand_idx]
        end_idx = spl_idxs[rand_idx+1] - 2
        if end_idx - start_idx < block_size:
            continue
        
        x_seq = X_source[start_idx:start_idx+block_size]
        y_seq = Y_source[start_idx:start_idx+block_size]
        if len(x_seq) == block_size and len(y_seq) == block_size:
            Xb.append(x_seq)
            Yb.append(y_seq)
    return {
        torch.stack(Xb).to(device),
        torch.stack(Yb).to(device)}

def train_model():
    model = GPT(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim,
                num_heads=num_heads, FFN_depth=FFN_depth, encoder_layers=encoder_layers)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        x_batch, y_batch = get_batch(X_train, Y_train, block_size, batch_size, device)
        logits = model(x_batch)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(X_val, Y_val, block_size, batch_size, device)
                val_logits = model(x_val)
                val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), y_val.view(-1))

            print(f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f})")

    save_or_not = input('Save model? - y/n ')
    if save_or_not.lower() == 'y':
        torch.save(model.state_dict(), "trained_model.pt")
    else:
        pass

if __name__ == "__main__":
    train_model()
