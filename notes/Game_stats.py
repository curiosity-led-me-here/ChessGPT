import os
import chess.pgn
import pandas as pd
from tqdm import tqdm

def count_moves(game):
    """Return the number of moves in a PGN game."""
    return sum(1 for _ in game.mainline_moves())

def parse_pgn_file(file_path):
    """Parse all games in a single .txt or .pgn file."""
    move_counts = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            move_counts.append(count_moves(game))
    return move_counts

def process_directory(directory_path):
    """Process all .txt/.pgn files in a directory and collect move counts."""
    all_moves = []
    files = [f for f in os.listdir(directory_path) if f.lower().endswith((".txt", ".pgn"))]
    
    for file in tqdm(files, desc=f"Processing {directory_path}"):
        path = os.path.join(directory_path, file)
        all_moves.extend(parse_pgn_file(path))
    
    return all_moves

def analyze_move_distribution(move_counts):
    """Show how many games exceed certain move thresholds."""
    df = pd.Series(move_counts, name="NumMoves")
    print(f"\nTotal games: {len(df)}")
    print(f"Average length: {df.mean():.2f} moves")
    print(f"Median length: {df.median():.0f} moves")

    # thresholds: 50, 60, 70, ... up to max moves rounded to nearest 10
    thresholds = list(range(50, int(df.max()) + 10, 10))
    print("\n=== Games exceeding move thresholds ===")
    for t in thresholds:
        count = (df > t).sum()
        print(f"> {t} moves: {count} games")

if __name__ == "__main__":
    data_dir = r'/Users/ashu/Downloads/chess-games-dataset-main/output_magnus'
    move_counts = process_directory(data_dir)
    analyze_move_distribution(move_counts)
