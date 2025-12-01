from pathlib import Path

# 1. Get the Project Root folder dynamically
# (This goes up two levels from src/config.py to the root folder)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# 2. Define Data Paths
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# 3. Define File Names
RAW_DATA_FILES = list(RAW_DATA_DIR.glob("*.pdf"))
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "godfather_corpus.txt"
MODEL_FILE = MODELS_DIR / "godfather_w2v.model"

# Create directories if they don't exist (safety check)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


