from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models")).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)