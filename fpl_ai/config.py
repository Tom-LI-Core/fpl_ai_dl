# === Global constants & paths ===
FPL_API_URL = "https://fantasy.premierleague.com/api/"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models/saved"

BUDGET = 100.0  # squad budget in £m
POSITIONS   = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_LIMIT = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
TEAM_LIMIT  = 3  # max players per EPL club
