import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_config(config_path):
    """Load configuration from JSON file."""

    with open(config_path, "r") as f:
        return json.load(f)


def load_static_data(in_dir, config):
    """Load all static CSV files specified in config."""

    in_dir = Path(in_dir)
    s = config["static_sources"]
    
    patients = pd.read_csv(in_dir / s["patients"])
    admissions = pd.read_csv(in_dir / s["admissions"])
    hosp_diagnosis = pd.read_csv(in_dir / s["hosp_diagnosis"])
    drgcodes = pd.read_csv(in_dir / s["drgcodes"])
    icustays = pd.read_csv(in_dir / s["icustays"])
    edstays = pd.read_csv(in_dir / s["edstays"])
    ed_diagnosis = pd.read_csv(in_dir / s["ed_diagnosis"])
    record_list = pd.read_csv(in_dir / s["record_list"])
    
    return patients, admissions, hosp_diagnosis, drgcodes, icustays, edstays, ed_diagnosis, record_list


def clean_cols_types(df):
    """Convert date/time columns to datetime and object columns to string dtype."""

    time_keywords = ("date", "time", "dod")

    for col in df.columns:
        col_lower = col.lower()
        
        if any(k in col_lower for k in time_keywords):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif df[col].dtype == "object":
            df[col] = df[col].astype("string")

    return df


def add_prefix_to_columns(df, prefix):
    """Add prefix to all columns except subject_id and hadm_id."""

    exclude_cols = ['subject_id', 'hadm_id']
    
    rename_dict = {
        col: f"{prefix}_{col}" 
        for col in df.columns 
        if col not in exclude_cols
    }
    
    return df.rename(columns=rename_dict)


def preprocess_patient(df):
    """Remove anchor columns and clean column types."""

    df = df.drop(columns=['anchor_year', 'anchor_year_group'])
    return clean_cols_types(df)


def preprocess_admissions(df):
    """Remove unnecessary columns, rename time columns, and clean types."""

    df = df.drop(columns=[
        'insurance', 'admission_location', 'marital_status',
        'hospital_expire_flag', 'language', 'admit_provider_id'
    ])
    df = clean_cols_types(df)
    df = df.rename(columns={
        'admittime': 'hosp_admittime',
        'dischtime': 'hosp_dischtime'
    })
    return df


def preprocess_drgcodes(df):
    """Flatten DRG codes to one row per admission with APR and HCFA data."""
    cols = ["subject_id", "hadm_id", "drg_type", "drg_code", 
            "description", "drg_severity", "drg_mortality"]
    df = df[cols].copy()

    df_flat = df.pivot(
        index=["subject_id", "hadm_id"],
        columns="drg_type",
        values=["drg_code", "description", "drg_severity", "drg_mortality"]
    )

    df_flat.columns = [f"{val}_{typ.lower()}" for val, typ in df_flat.columns]

    columns_to_keep = [
        "drg_code_apr", "description_apr", "drg_severity_apr", "drg_mortality_apr",
        "drg_code_hcfa", "description_hcfa"
    ]
    df_flat = df_flat[columns_to_keep].reset_index()
    return clean_cols_types(df_flat)


def clean_diagnosis_data(df, prefix):
    """Aggregate diagnosis codes and titles into lists per admission, optionally keeping stay_id for ED."""
    df = clean_cols_types(df)

    # Determine sorting and grouping columns
    if prefix == "ed":
        group_cols = ["subject_id", "stay_id"]
    else:
        group_cols = ["subject_id", "hadm_id"]

    df = df.sort_values(group_cols + ['seq_num'])

    # Determine diagnosis title column
    title_col = 'long_title' if 'long_title' in df.columns else 'icd_title'

    # Aggregate codes and titles into lists
    df_diag_agg = (
        df.groupby(group_cols, as_index=False, sort=False)
        .agg({
            "icd_code": list,
            title_col: list
        })
        .rename(columns={
            "icd_code": f"{prefix}_icd_codes_diagnosis",
            title_col: f"{prefix}_diagnosis",
            "stay_id": f"{prefix}_stay_id"
        })
    )
    return df_diag_agg


def preprocess_icustays(df):
    """Aggregate ICU stays per admission into lists and add stay count."""
    df = clean_cols_types(df)
    
    temporal_cols = ['stay_id', 'first_careunit', 'last_careunit', 'los']
    
    df = df.groupby(['subject_id', 'hadm_id'], sort=False).agg({
        col: list for col in temporal_cols
    }).reset_index()
    
    df['count'] = df['stay_id'].str.len()
    
    return add_prefix_to_columns(df, 'icu')


def merge_ecg_to_hosp(hosp_master_df, record_list_df):
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



def merge_hosp_to_ed(admissions_df, patients_df, diagnosis_df, drgcodes_df, icustays_df, edstays_df, ed_diagnosis_df):
    """
    Merge all hospital-related dataframes into a master hospital dataframe.
    
    Args:
        admissions_df: Preprocessed admissions data
        patients_df: Preprocessed patients data
        diagnosis_df: Cleaned diagnosis data
        drgcodes_df: Preprocessed DRG codes data
        icustays_df: Aggregated ICU stays data
        edstays_df: ED stays data
        ed_diagnosis_df: Cleaned ED diagnosis
    
    Returns:
        Merged master hospital dataframe
    """
    hosp_master_df = admissions_df.merge(patients_df, on="subject_id", how="left")
    
    if 'deathtime' in hosp_master_df.columns and 'dod' in hosp_master_df.columns:
        hosp_master_df['death_time'] = hosp_master_df['deathtime'].combine_first(hosp_master_df['dod'])
        hosp_master_df = hosp_master_df.drop(columns=['dod', 'deathtime'])
    
    hosp_master_df = hosp_master_df.merge(diagnosis_df, on=["subject_id", "hadm_id"], how="left")
    hosp_master_df = hosp_master_df.merge(drgcodes_df, on=["subject_id", "hadm_id"], how="left")
    hosp_master_df = hosp_master_df.merge(icustays_df, on=["subject_id", "hadm_id"], how="left")

    ed_master_df = edstays_df.merge(
        ed_diagnosis_df,
        on=["subject_id", "ed_stay_id"],
        how="left"
    )

    # Outer join to capture ED-only visits
    hosp_master_df = ed_master_df.merge(
        hosp_master_df,
        on=['subject_id', 'hadm_id'],
        how="outer"
    )

    return hosp_master_df


def run_static_preprocessing(in_dir, config_path, out_path):
    """
    Main preprocessing pipeline for static features.
    
    Args:
        in_dir: Path to directory containing raw CSV files
        config_path: Path to preprocessing configuration JSON
        out_path: Path to save processed parquet file
    
    Returns:
        DataFrame with all processed static features
    """
    print("Running static preprocessing...")
    
    print("\n[1/12] Loading configuration...")
    config = load_config(config_path)
    
    print("[2/12] Loading raw data...")
    (patients, admissions, hosp_diagnosis, drgcodes, 
     icustays, edstays, ed_diagnosis, record_list) = load_static_data(in_dir, config)
    

    print("[3/12] Preprocessing patients...")
    record_list_processed = clean_cols_types(record_list)

    print("[4/12] Preprocessing patients...")
    patients_processed = preprocess_patient(patients)
    
    print("[5/12] Preprocessing admissions...")
    admissions_processed = preprocess_admissions(admissions)
    
    print("[6/12] Preprocessing DRG codes...")
    drgcodes_processed = preprocess_drgcodes(drgcodes)
    
    print("[7/12] Cleaning hospital diagnosis data...")
    hosp_diagnosis_cleaned = clean_diagnosis_data(hosp_diagnosis, prefix="hosp")
    
    print("[8/12] Cleaning ED diagnosis data...")
    ed_diagnosis_cleaned = clean_diagnosis_data(ed_diagnosis, prefix="ed")
    
    print("[9/12] Preprocessing ICU stays...")
    icustays_agg = preprocess_icustays(icustays)
    
    print("[10/12] Cleaning ED stays...")
    edstays_cleaned = clean_cols_types(edstays)
    edstays_cleaned = add_prefix_to_columns(edstays_cleaned, 'ed')
    
    print("[11/12] Merging hospital data...")
    hosp_master = merge_hosp_to_ed(
        admissions_processed,
        patients_processed,
        hosp_diagnosis_cleaned,
        drgcodes_processed,
        icustays_agg,
        edstays_cleaned,
        ed_diagnosis_cleaned
    )
    
    print("[12/12] Merging ECG records...")
    static_master = merge_ecg_to_hosp(hosp_master, record_list_processed)
    
    print("\n" + "=" * 60)
    print(f"Final dataset shape: {static_master.shape}")
    print(f"Number of columns: {len(static_master.columns)}")
    print("=" * 60)
    
    print(f"\nSaving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    static_master.to_csv(out_path, index=False)
    
    print("Static preprocessing complete!")
    return static_master