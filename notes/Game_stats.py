import os
import chess.pgn
import pandas as pd
from tqdm import tqdm

def extract_metadata(game):
    """Extract metadata and move count from a PGN game."""
    headers = game.headers
    board = game.board()
    
    num_moves = sum(1 for _ in game.mainline_moves())
    
    return {
        "Event": headers.get("Event", ""),
        "Site": headers.get("Site", ""),
        "Date": headers.get("Date", ""),
        "White": headers.get("White", ""),
        "Black": headers.get("Black", ""),
        "Result": headers.get("Result", ""),
        "WhiteElo": headers.get("WhiteElo", ""),
        "BlackElo": headers.get("BlackElo", ""),
        "WhiteRatingDiff": headers.get("WhiteRatingDiff", ""),
        "BlackRatingDiff": headers.get("BlackRatingDiff", ""),
        "WhiteTitle": headers.get("WhiteTitle", ""),
        "BlackTitle": headers.get("BlackTitle", ""),
        "Variant": headers.get("Variant", ""),
        "TimeControl": headers.get("TimeControl", ""),
        "ECO": headers.get("ECO", ""),
        "Opening": headers.get("Opening", ""),
        "Termination": headers.get("Termination", ""),
        "NumMoves": num_moves
    }

def parse_pgn_file(file_path):
    """Parse all games in a single .txt or .pgn file."""
    games = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(extract_metadata(game))
    return games

def process_directory(directory_path):
    """Process all .txt/.pgn files in a directory."""
    all_data = []
    files = [f for f in os.listdir(directory_path) if f.lower().endswith((".txt", ".pgn"))]
    
    for file in tqdm(files, desc=f"Processing {directory_path}"):
        path = os.path.join(directory_path, file)
        all_data.extend(parse_pgn_file(path))
    
    return pd.DataFrame(all_data)

def analyze_statistics(df):
    """Compute statistical summaries including move counts."""
    df["WhiteElo"] = pd.to_numeric(df["WhiteElo"], errors="coerce")
    df["BlackElo"] = pd.to_numeric(df["BlackElo"], errors="coerce")
    df["NumMoves"] = pd.to_numeric(df["NumMoves"], errors="coerce")

    print("\n=== General Summary ===")
    print(f"Total Games: {len(df)}")
    print(f"Unique Players: {len(set(df['White']).union(set(df['Black'])))}")

    print("\n=== Ratings ===")
    print(f"Average White Elo: {df['WhiteElo'].mean():.1f}")
    print(f"Average Black Elo: {df['BlackElo'].mean():.1f}")

    print("\n=== Results ===")
    print(df["Result"].value_counts())

    print("\n=== Move Count Statistics ===")
    print(f"Average number of moves: {df['NumMoves'].mean():.1f}")
    print(f"Median number of moves: {df['NumMoves'].median():.1f}")
    print(f"Shortest game: {df['NumMoves'].min()} moves")
    print(f"Longest game: {df['NumMoves'].max()} moves")

    print("\nTop 10 Longest Games:")
    print(df.nlargest(10, "NumMoves")[["White", "Black", "NumMoves", "Result", "Opening"]])

    print("\nTop 10 Shortest Games:")
    print(df.nsmallest(10, "NumMoves")[["White", "Black", "NumMoves", "Result", "Opening"]])

if __name__ == "__main__":
    data_dir = r'/Users/ashu/Downloads/chess-games-dataset-main/output_magnus'
    df = process_directory(data_dir)
    output_path = os.path.join(data_dir, "metadata_with_moves.csv")
    df.to_csv(output_path, index=False)
    analyze_statistics(df)