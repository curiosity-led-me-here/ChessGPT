import chess
import json
import torch

INPUT_JSON = "all_moves.json"
NEW_GAME_TOKEN = "<|NEW_GAME|>"
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def san_to_uci(game_moves):
    board = chess.Board()
    uci_moves = []
    for move_san in game_moves:
        try:
            move = board.parse_san(move_san)
            uci_moves.append(move.uci())
            board.push(move)
        except Exception:
            continue
    return uci_moves


def tokenize_games(games_2d):
    tokenized = []
    for game in games_2d:
        uci_game = san_to_uci(game)
        tokenized.extend(uci_game)
        tokenized.append(NEW_GAME_TOKEN)
    return tokenized

def numerical_tokenization(x):
    vocab = list(sorted(set(x)))
    stoi = {s: i for i, s in enumerate(vocab)} 
    itos = {i: s for i, s in enumerate(vocab)}
    encode = lambda seq: [stoi[s] for s in seq]
    output = encode(x)
    return output, stoi, itos

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    games_2d = json.load(f)

print(f"Loaded {len(games_2d)} games.")
tokenized_moves = tokenize_games(games_2d)
data, stoi, itos = numerical_tokenization(tokenized_moves)    
    
'''
with open("stoi.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)

with open("itos.json", "w", encoding="utf-8") as f:
    json.dump(itos, f, ensure_ascii=False, indent=2)
'''