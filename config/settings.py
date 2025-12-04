from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

@dataclass
class Paths:
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    logs_dir: Path = BASE_DIR / "logs"

    classifier_path: Path = BASE_DIR / "models" / "news_classifier.joblib"
    embeddings_path: Path = BASE_DIR / "models" / "embeddings_model"
    lda_model_path: Path = BASE_DIR / "models" / "lda_model.joblib"
    nmf_model_path: Path = BASE_DIR / "models" / "nmf_model.joblib"


@dataclass
class AppConfig:
    random_state: int = 42
    test_size: float = 0.2
    language_default: str = "en"
    max_features_tfidf: int = 20000
    num_topics: int = 10


paths = Paths()
config = AppConfig()
