from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = BASE_DIR / "images"

TRAIN_FILE = DATA_RAW_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_RAW_DIR / "KDDTest+.txt"

MODEL_FILE = MODELS_DIR / "random_forest_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.csv"
ALERTS_FILE = OUTPUTS_DIR / "alerts.csv"

RANDOM_STATE = 42

for path in [DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR]:
    path.mkdir(parents=True, exist_ok=True)