"""
End-to-end weekly pipeline.
"""
import argparse, json, pandas as pd, torch
from fpl_ai.data import fetch, process
from fpl_ai.models.lstm import LSTMReg
from fpl_ai.optimiser import solver
from fpl_ai.config import PROCESSED_DATA_DIR, MODEL_DIR

def load_model(path):
    dummy = torch.randn(1,10,5)
    state  = torch.load(path, map_location="cpu")
    model  = LSTMReg(dummy.size(2)); model.load_state_dict(state); model.eval()
    return model

def main(team_json, model_path):
    fetch.download_static()
    process.main()
    df = pd.read_parquet(f"{PROCESSED_DATA_DIR}/player_weeks.parquet")
    feats = [c for c in df.columns if c not in ("target","id")]
    model = load_model(model_path)
    xp = model(torch.tensor(df[feats].to_numpy(dtype=float)
                            .reshape(len(df),1,len(feats))).float()).squeeze().detach()
    df["xp"] = xp
    squad_ids = solver.build_team(df.set_index("id")["xp"], df.set_index("id"))
    print("Recommended 15-man squad IDs:", squad_ids)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_json", required=True)
    ap.add_argument("--model_path", default=f"{MODEL_DIR}/lstm_xp.pt")
    args = ap.parse_args()
    main(args.team_json, args.model_path)
