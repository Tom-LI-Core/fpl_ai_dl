import pandas as pd, pickle
from pathlib import Path
from fpl_ai.config import MODEL_DIR

class RollingAverage:
    def __init__(self, window=3):
        self.window = window

    def fit(self, df: pd.DataFrame):
        pass  # nothing to fit

    def predict(self, df: pd.DataFrame):
        return df["ppg"].rolling(self.window, min_periods=1).mean().shift(1)

    def save(self):
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        pickle.dump(self, open(Path(MODEL_DIR)/"baseline.pkl", "wb"))
