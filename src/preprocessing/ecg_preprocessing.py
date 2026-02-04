import json
from pathlib import Path
import pandas as pd
import numpy as np
import re
import ast
from .machine_measurements_labels import report_label_map

def load_config(config_path):
    """Load preprocessing configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def load_ecg_data(raw_dir, processed_dir, config):
    """Load ECG data from CSV files."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    s = config["sources"]
    
    ecg_record = pd.read_csv(raw_dir / s["record"], dtype=str, low_memory=False)
    machine_measurements = pd.read_csv(raw_dir / s["machine_measurements"], dtype=str, low_memory=False)
    clinical_encounters = pd.read_csv(processed_dir / s["clinical_encounters"], dtype=str, low_memory=False)
    
    return ecg_record, machine_measurements, clinical_encounters

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
    Ignores any NaN, <NA>, or empty strings.
    """
    df[output_col] = [
        [str(s).strip() for s in row if pd.notna(s) and str(s).strip() and str(s).lower() != "<na>"]
        for row in df[cols].to_numpy()
    ]
    return df.drop(columns=cols)

def preprocess_ecg_reports(df):
    """
    Full pipeline to clean and flatten ECG report fields.
    Cleans each report string before combining into a list column.
    """
    # Identify report columns
    report_cols = [col for col in df.columns if col.startswith("report_")]

    # Standardize column types
    df = clean_cols_types(df)

    # Clean each report column in place
    for col in report_cols:
        df[col] = df[col].astype("string").apply(lambda x: clean_report_text(x))

    # Flatten report columns into a single list column
    df = flatten_columns(df, report_cols, output_col="full_report")

    # Remove rows containing invalid machine messages
    invalid_phrases = ["uncertain rhythm review", "all 12 leads are missing"]
    df = df[df["full_report"].apply(lambda lst: all(p not in lst for p in invalid_phrases))]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df

# -------------------------
# Label extraction
# -------------------------

def clean_report_text(s):
    """
    Clean a single report string:
      - lowercase
      - remove non-alphabetic characters
      - collapse whitespace
      - strip leading/trailing spaces
    """
    s = str(s).lower()
    s = re.sub(r'[^a-z\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def apply_phrase_labels(df):
    """
    Apply binary labels to dataframe based on pattern matching.
    """
    def contains_any(text, phrases):
        """Check if text contains any of the phrases."""
        return any(p in text for p in phrases)

    # Create all label columns at once using a dictionary with 'report_' prefix
    label_columns = {
        f'report_{label}': df['full_report'].apply(lambda x: contains_any(x, phrases)).astype(int)
        for label, phrases in report_label_map.items()
    }
    
    # Concatenate all new columns at once
    new_columns = pd.DataFrame(label_columns, index=df.index)
    df = pd.concat([df, new_columns], axis=1)
    
    return df

# -------------------------
# ECG Record List
# -------------------------
def match_ecg_to_encounters(ecg_record_list_df, encounter_df):
    """
    Match ECG records to hospital/ED encounters based on time windows.
    """
    # Create copies to avoid modifying original dataframes
    ecg_df = ecg_record_list_df.copy()
    enc_df = encounter_df.copy()
    
    # Ensure proper data types
    ecg_df["ecg_time"] = pd.to_datetime(ecg_df["ecg_time"])
    
    # Ensure time columns are datetime
    time_cols = ["hosp_admittime", "hosp_dischtime", "ed_intime", "ed_outtime"]
    enc_df[time_cols] = enc_df[time_cols].apply(pd.to_datetime)
    
    # Define columns to keep from encounter data
    encounter_time_cols = ['hadm_id', "hosp_admittime", "hosp_dischtime", 
                           "ed_intime", "ed_outtime",  'ed_stay_id', 
                           'icu_stay_id', 'icu_intime', 'icu_outtime']
    
    all_columns = list(ecg_df.columns) + encounter_time_cols
    
    # Merge ECG records with hospital/ED data
    merged = ecg_df.merge(enc_df, on="subject_id", how="left")[all_columns]
    
    # Check if ECG falls within hospital encounter window
    hosp_mask = (
        merged["hosp_admittime"].notna() &
        merged["hosp_dischtime"].notna() &
        merged["ecg_time"].between(merged["hosp_admittime"], merged["hosp_dischtime"])
    )
    
    # Check if ECG falls within ED encounter window
    ed_mask = (
        merged["ed_intime"].notna() &
        merged["ed_outtime"].notna() &
        merged["ecg_time"].between(merged["ed_intime"], merged["ed_outtime"])
    )
    
    # Keep only ECGs that fall within either hospital or ED encounter windows
    merged = merged[hosp_mask | ed_mask].copy()

    # Create binary indicators before filtering
    merged["in_hosp"] = hosp_mask[hosp_mask | ed_mask].astype(int)
    merged["in_ed"] = ed_mask[hosp_mask | ed_mask].astype(int)
    
    # Calculate window sizes for both encounter types
    hosp_win = merged["hosp_dischtime"] - merged["hosp_admittime"]
    ed_win = merged["ed_outtime"] - merged["ed_intime"]
    
    # Select the smaller window (prefer shorter, more specific encounters)
    merged["window_size"] = hosp_win.combine(
        ed_win,
        lambda h, e: h if pd.notna(h) and (pd.isna(e) or h <= e) else e
    )
    
    # For each study_id, keep only the row with the smallest window
    # (most specific encounter match)
    merged = (
        merged.sort_values(["study_id", "window_size"])
              .drop_duplicates("study_id", keep="first")
              .drop(["window_size", 'hosp_admittime', 'hosp_dischtime', 'ed_intime', 'ed_outtime'], axis=1)
    )
    
    return merged

def add_icu_indicator(merged_df):
    """
    Add in_icu binary column based on whether ECG was taken during ICU stay.
    Add icu_within_12hrs and icu_within_24hrs columns to indicate if patient
    transferred to ICU within those time windows after ECG.
    Drops icu_intime and icu_outtime columns after processing.
    """
    df = merged_df.copy()
    
    def parse_timestamp_list(ts_value):
        """Parse timestamp value into list of datetimes"""
        # Check type first before pd.isna
        if isinstance(ts_value, list):
            return [pd.to_datetime(t) for t in ts_value if pd.notna(t)]
        
        if not isinstance(ts_value, str) and pd.isna(ts_value):
            return []
        
        if isinstance(ts_value, str):
            try:
                ts_value = ast.literal_eval(ts_value)
                if isinstance(ts_value, list):
                    return [pd.to_datetime(t) for t in ts_value if pd.notna(t)]
            except (ValueError, SyntaxError):
                return []
        
        return []
    
    def check_all_icu_indicators(row):
        """
        Check all ICU indicators in a single pass for efficiency.
        Returns (in_icu, icu_within_12hrs, icu_within_24hrs)
        """
        in_icu = 0
        icu_within_12hrs = 0
        icu_within_24hrs = 0
        
        icu_intimes = parse_timestamp_list(row["icu_intime"])
        icu_outtimes = parse_timestamp_list(row["icu_outtime"])
        
        if not icu_intimes:
            return pd.Series([in_icu, icu_within_12hrs, icu_within_24hrs])
        
        ecg_time = row["ecg_time"]
        target_12hr = ecg_time + pd.Timedelta(hours=12)
        target_24hr = ecg_time + pd.Timedelta(hours=24)
        
        # Sort ICU admission times once
        icu_intimes_sorted = sorted([t for t in icu_intimes if pd.notna(t)])
        
        for i, icu_in in enumerate(icu_intimes_sorted):
            # Check in_icu (only if in_hosp=1 and we have matching outtime)
            if row["in_hosp"] == 1 and i < len(icu_outtimes):
                icu_out = icu_outtimes[i]
                if pd.notna(icu_out) and icu_in <= ecg_time <= icu_out:
                    in_icu = 1
            
            # Check transfer windows (regardless of in_hosp)
            if ecg_time < icu_in:
                if icu_in <= target_12hr:
                    icu_within_12hrs = 1
                if icu_in <= target_24hr:
                    icu_within_24hrs = 1
                else:
                    # Optimization: passed 24hr window, stop checking
                    break
        
        return pd.Series([in_icu, icu_within_12hrs, icu_within_24hrs])
    
    # Compute all indicators in single apply call
    df[["in_icu", "icu_within_12hrs", "icu_within_24hrs"]] = df.apply(
        check_all_icu_indicators, axis=1
    )
    
    # Drop the original timestamp columns
    df = df.drop(["icu_intime", "icu_outtime"], axis=1)
    
    return df

# -------------------------
# Pipeline runner
# -------------------------

def run_ecg_preprocessing(in_dir, config_path, out_path):
    print("Running ECG preprocessing...")
    
    print("\n[1/5] Loading configuration...")
    config = load_config(config_path)

    print("[2/5] Loading raw data...")
    processed_dir = Path(config["paths"]["processed_dir"])
    ecg_record, machine_measurements, clinical_encounters = load_ecg_data(in_dir, processed_dir, config)

    print("[3/5] Preprocessing machine measurements...")
    cleaned_machine_measurements = preprocess_ecg_reports(machine_measurements)
    final_machine_measurements = apply_phrase_labels(cleaned_machine_measurements)

    print("[4/5] Preprocessing ECG records...")
    merged = match_ecg_to_encounters(ecg_record, clinical_encounters)
    final_ecg_record_list = add_icu_indicator(merged)

    print("[5/5] Creating ECG dataset...")
    master_ecg = final_ecg_record_list.merge(final_machine_measurements, on=['subject_id', 'study_id', 'ecg_time'], how='inner')

    print("-" * 60)
    print(f"Final dataset shape: {master_ecg.shape}")
    print(f"Number of columns: {len(master_ecg.columns)}")
    print(f'Hours of ECG data: {master_ecg.shape[0] * 10 / 60 / 60}')
    print(f"Saving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_ecg.to_csv(out_path, index=False)
    print("ECG preprocessing complete!")
    print("-" * 60)

    return master_ecg
