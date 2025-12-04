from typing import List
import pandas as pd


REQUIRED_COLUMNS = ["text", "label"]


def validate_dataset(df: pd.DataFrame, required_cols: List[str] = None) -> None:
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if df[required_cols].isnull().any().any():
        raise ValueError("Dataset contains nulls in required columns.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 unique labels for classification.")
