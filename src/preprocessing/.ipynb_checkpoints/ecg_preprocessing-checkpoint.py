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


def merge_hosp_to_ecg(hosp_master_df, record_list_df): # FIX
    df = hosp_master_df.reset_index(drop=True).copy()
    df["_row_idx"] = df.index

    ecg = record_list_df.copy()
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"])

    # make sure time columns are datetime
    time_cols = ["hosp_admittime","hosp_dischtime","ed_intime","ed_outtime"]
    df[time_cols] = df[time_cols].apply(pd.to_datetime)

    merged = df.merge(ecg, on="subject_id", how="left")
    merged = merged[merged["ecg_time"].notna()].copy()

    hosp_mask = (
        merged["hosp_admittime"].notna() &
        merged["hosp_dischtime"].notna() &
        merged["ecg_time"].between(merged["hosp_admittime"], merged["hosp_dischtime"])
    )

    ed_mask = (
        merged["ed_intime"].notna() &
        merged["ed_outtime"].notna() &
        merged["ecg_time"].between(merged["ed_intime"], merged["ed_outtime"])
    )

    merged = merged[hosp_mask | ed_mask].copy()

    hosp_win = merged["hosp_dischtime"] - merged["hosp_admittime"]
    ed_win = merged["ed_outtime"] - merged["ed_intime"]

    merged["window_size"] = hosp_win.combine(
        ed_win,
        lambda h, e: h if pd.notna(h) and (pd.isna(e) or h <= e) else e
    )

    merged = (
    merged.sort_values(["study_id", "window_size"])
          .drop_duplicates("study_id")
    )

    grouped = (
        merged.groupby("_row_idx", as_index=False)
            .agg(ecg_study_ids=("study_id", list))
    )

    df = df.merge(grouped, on="_row_idx", how="left")
    df["ecg_study_ids"] = df["ecg_study_ids"].apply(lambda x: x if isinstance(x, list) else [])

    return df.drop(columns="_row_idx")