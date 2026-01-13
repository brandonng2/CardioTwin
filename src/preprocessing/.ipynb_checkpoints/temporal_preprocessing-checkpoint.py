import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


CONFIG_PATH = Path("configs/temporal_preprocessing.json")
OUT_PATH = Path("data/processed/temporal_features.parquet")

def clean_cols_types(df):
    """
    Normalize column types:
      - Convert columns containing 'date' or 'time' in their names to datetime.
      - Convert all other object columns to Pandas string dtype.
    """
    time_keywords = ("date", "time", "dod")

    for col in df.columns:
        col_lower = col.lower()

        if any(k in col_lower for k in time_keywords):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            continue

        # Convert object columns to Pandas string
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    return df

def flatten_columns(df, cols, output_col="flattened"):
    """
    Combine multiple report columns into one list column.
    """
    df[output_col] = df[cols].apply(
        lambda row: [s.strip() for s in row if pd.notna(s) and s.strip()],
        axis=1
    )
    return df.drop(columns=cols)

def preprocess_ecg_data(df):
    """
    Full pipeline to clean and flatten ECG report fields.
    Makes a copy of the input to avoid modifying the original.
    """
    report_cols = [col for col in df.columns if col.startswith("report_")]
    invalid_phrases = ["Uncertain rhythm: review", "All 12 leads are missing"]

    # Normalize column types
    df = clean_cols_types(df)

    # Flatten report columns into a single list column
    df = flatten_columns(df, report_cols, "full_report")

    # Remove rows containing invalid machine messages
    df = df[df["full_report"].apply(
        lambda lst: all(p not in lst for p in invalid_phrases)
    )]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df

