import chess
import chess.pgn
import json
import torch

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
INPUT_JSON = "all_moves.json"      # 2D array of games and SAN moves
OUTPUT_FILE = "uci_tokenized_moves.json"
NEW_GAME_TOKEN = "<|NEW_GAME|>"
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────
# CORE FUNCTION
# ─────────────────────────────
def san_to_uci(game_moves):
    """
    Convert a list of SAN moves (['e4','e5','Nf3',...])
    into UCI moves (['e2e4','e7e5','g1f3',...])
    using python-chess for legality tracking.
    """
    board = chess.Board()
    uci_moves = []
    for move_san in game_moves:
        try:
            move = board.parse_san(move_san)
            uci_moves.append(move.uci())
            board.push(move)
        except Exception:
            # Skip illegal / malformed moves gracefully
            continue
    return uci_moves


def tokenize_games(games_2d):
    """
    Flatten multiple games into a single list of UCI moves
    with <|NEW_GAME|> separators.
    """
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

# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":
    # Load your 2D array of moves
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        games_2d = json.load(f)

    print(f"Loaded {len(games_2d)} games.")

    # Tokenize all games into UCI + separator
    tokenized_moves = tokenize_games(games_2d)

    print(f"Total tokens (including separators): {len(tokenized_moves)}")
    print("Example:", tokenized_moves[:30])

    data, stoi, itos = numerical_tokenization(tokenized_moves)
    
    
