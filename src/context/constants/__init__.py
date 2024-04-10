"""File containing useful constants."""
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
MLFLOW_URI = "https://dagshub.com/marcolongfils90/MLOps.mlflow"
MODEL_PATH = Path("artifacts/model/trained_model.pt")
CAT_CLASS_ID = 17
MIN_SCORE = 0.9
CLASSES = [
    'Abyssinian',
    'Bengal',
    'Birman',
    'Bombay',
    'British',
    'Egyptian',
    'Maine',
    'Persian',
    'Ragdoll',
    'Russian',
    'Siamese',
    'Sphynx'
]