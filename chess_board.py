import torch, chess, torch.nn.functional as F
from model_fluidgpt import GPT
from hyperparams import stoi, itos, block_size,vocab_size,embed_dim,num_heads,FFN_depth,encoder_layers

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(vocab_size, block_size, embed_dim, num_heads, FFN_depth, encoder_layers)
model.load_state_dict(torch.load("trained_model.pt", map_location=device))
model.to(device).eval()
print("Model loaded and ready.")

def generate(model, prompt, max_new_tokens=1, temperature=1.0, top_k=None):
    tokens = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -block_size:]
        logits = model(tokens_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        if top_k is not None:
            values, indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(1, indices, values)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat((tokens, next_token), dim=1)
    return next_token.item()

board = chess.Board()
game_tokens = []

while board.outcome() is None:
    print(board)
    move_san = input("Your move (SAN): ")
    try:
        move = board.parse_san(move_san)
        board.push(move)
        game_tokens.append(stoi[move.uci()])
    except Exception as e:
        print("Invalid move.")
        continue

    # GPT's move
    for _ in range(10):
        token_id = generate(model, prompt=game_tokens)
        uci = itos[token_id]
        move = chess.Move.from_uci(uci)
        san_move = board.san(move)
        if move in board.legal_moves:
            board.push(move)
            print(f"Model plays: {san_move}")
            game_tokens.append(token_id)
            break
    else:
        print("Model could not find a legal move.")
        break
