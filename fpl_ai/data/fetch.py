"""
Download raw JSON from the FPL API (bootstrap + fixtures)
and cache to data/raw/.
"""
from pathlib import Path
import json, time, requests
from fpl_ai.config import FPL_API_URL, RAW_DATA_DIR

RAW = Path(RAW_DATA_DIR)
RAW.mkdir(parents=True, exist_ok=True)

def _get(endpoint: str, retries: int = 5):
    url = f"{FPL_API_URL}{endpoint}"
    for _ in range(retries):
        r = requests.get(url, timeout=10)
        if r.ok:
            return r.json()
        time.sleep(2)
    r.raise_for_status()

def download_static():
    out = RAW / "bootstrap_static.json"
    out.write_text(json.dumps(_get("bootstrap-static/")))

def download_fixtures():
    out = RAW / "fixtures.json"
    out.write_text(json.dumps(_get("fixtures/")))

if __name__ == "__main__":
    download_static()
    download_fixtures()
