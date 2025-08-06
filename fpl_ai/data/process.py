"""
Turn raw API payloads into a per-Gameweek player table.
Write player_weeks.parquet into data/processed/.
"""
from pathlib import Path
import pandas as pd, json
from fpl_ai.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

RAW  = Path(RAW_DATA_DIR)
PROC = Path(PROCESSED_DATA_DIR)
PROC.mkdir(parents=True, exist_ok=True)

def build_weekly():
    static = json.loads((RAW / "bootstrap_static.json").read_text())
    elems  = pd.DataFrame(static["elements"])
    # simple features
    df = elems[["id", "first_name", "second_name",
                "now_cost", "total_points", "minutes",
                "influence", "creativity", "threat"]]
    df["ppg"] = df["total_points"] / 38
    df["target"] = df["ppg"].shift(-1)  # LABEL: naïve next-GW points
    return df

def main():
    build_weekly().to_parquet(PROC / "player_weeks.parquet")

if __name__ == "__main__":
    main()
