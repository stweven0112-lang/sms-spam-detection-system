import os
import pandas as pd

from src.config import TrainConfig
from src.paths import ProjectPaths


def load_sms_dataset(paths: ProjectPaths, cfg: TrainConfig) -> pd.DataFrame:
    if not os.path.exists(paths.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {paths.dataset_path}\n"
            f"Keep your dataset at: data/spam.csv"
        )

    df = pd.read_csv(paths.dataset_path, encoding="latin-1")
    df = df.iloc[:, :2].copy()
    df.columns = ["label", "text"]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    df = df[df["label"].isin([cfg.label_ham, cfg.label_spam])].reset_index(drop=True)
    return df
