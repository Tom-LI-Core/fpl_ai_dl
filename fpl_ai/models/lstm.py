"""
Train an LSTM on player_weeks.parquet.
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np
from pathlib import Path
from fpl_ai.config import PROCESSED_DATA_DIR, MODEL_DIR

class PlayerDS(Dataset):
    def __init__(self, parquet, seq_len=10):
        self.df = pd.read_parquet(parquet)
        self.seq_len = seq_len
        self.players = self.df["id"].unique()
        self.feats = [c for c in self.df.columns if c not in ("target","id")]

    def __len__(self):
        return len(self.players)

    def __getitem__(self, idx):
        pid = self.players[idx]
        hist = self.df[self.df.id == pid].tail(self.seq_len)
        x = hist[self.feats].to_numpy(dtype=np.float32)
        y = hist["target"].iloc[-1:].to_numpy(dtype=np.float32)
        # pad
        if x.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - x.shape[0], x.shape[1]), np.float32)
            x = np.vstack([pad, x])
        return torch.tensor(x), torch.tensor(y)

class LSTMReg(nn.Module):
    def __init__(self, n_feats, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1])

def train(seq_len=10, epochs=5, lr=1e-3):
    ds = PlayerDS(f"{PROCESSED_DATA_DIR}/player_weeks.parquet", seq_len)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    model = LSTMReg(len(ds.feats)).cuda()
    opt = optim.Adam(model.parameters(), lr)
    mse = nn.MSELoss()
    for e in range(epochs):
        tot = 0
        for x,y in dl:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            loss = mse(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()*len(y)
        print(f"E{e+1}: {(tot/len(ds)):.4f}")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(MODEL_DIR)/"lstm_xp.pt")

if __name__ == "__main__":
    train()
