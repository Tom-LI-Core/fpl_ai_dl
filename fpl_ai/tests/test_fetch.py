from pathlib import Path
from fpl_ai.data import fetch
import json, tempfile, pytest

def test_bootstrap_download(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("fpl_ai.data.fetch.RAW_DATA_DIR", tmp)
        fetch.download_static()
        f = Path(tmp)/"bootstrap_static.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert "elements" in data
