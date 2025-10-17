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

# ===== Hyperparams =====
vocab_size = len(stoi)
batch_size = 16
block_size = 16
embed_dim = 32
num_heads = 4
FFN_depth = 32
encoder_layers = 10
epochs = 100
lr = 3e-4

def get_batch(X_source, Y_source, block_size, batch_size, device):
    count = 0
    Xb, Yb = [], []
    while count != batch_size:
        rand_idx = torch.randint(0, len(spl_idxs)-1, ()).item()
        start_idx = spl_idxs[rand_idx]
        end_idx = spl_idxs[rand_idx+1] - 2
        if end_idx - start_idx < block_size:
            pass
        else:
            Xb.append(X_source[start_idx:start_idx+block_size])
            Yb.append(Y_source[start_idx:start_idx+block_size])
            count += 1
    return {
        torch.stack(Xb).to(device),
        torch.stack(Yb).to(device)}

def train_model():
    model = GPT(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim,
                num_heads=num_heads, FFN_depth=FFN_depth, encoder_layers=encoder_layers)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    

    for epoch in range(epochs):
        # Training step
        model.train()
        x_batch, y_batch = get_batch(X, Y, block_size, batch_size, device)
        logits = model(x_batch)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(X_val, Y_val, block_size, batch_size, device)
                val_logits = model(x_val)
                val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), y_val.view(-1))

                test_logits = model(test_input)
                probs = F.softmax(test_logits[:, -1, :], dim=-1)
                pred_token = torch.argmax(probs, dim=-1).item()
                pred_prob = probs[0, pred_token].item()
                pred_symbol = "H" if pred_token == 0 else "T"

            print(f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Pred: {pred_symbol} ({pred_prob:.3f})")

    torch.save(model.state_dict(), "trained_model.pt")

if __name__ == "__main__":
    train_model()
