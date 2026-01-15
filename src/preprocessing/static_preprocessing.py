import json
from pathlib import Path
import pandas as pd

def load_config(config_path):
    """Load configuration from JSON file."""

    with open(config_path, "r") as f:
        return json.load(f)


def load_static_data(in_dir, config):
    """Load all static CSV files specified in config."""

    in_dir = Path(in_dir)
    s = config["sources"]
    
    patients = pd.read_csv(in_dir / s["patients"])
    admissions = pd.read_csv(in_dir / s["admissions"])
    hosp_diagnosis = pd.read_csv(in_dir / s["hosp_diagnosis"])
    icustays = pd.read_csv(in_dir / s["icustays"])
    edstays = pd.read_csv(in_dir / s["edstays"])
    ed_diagnosis = pd.read_csv(in_dir / s["ed_diagnosis"])
    
    return patients, admissions, hosp_diagnosis, icustays, edstays, ed_diagnosis


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
        'hospital_expire_flag', 'language', 'admit_provider_id',
        'admission_type', 'discharge_location'
    ])
    df = clean_cols_types(df)
    df = df.rename(columns={
        'admittime': 'hosp_admittime',
        'dischtime': 'hosp_dischtime'
    })
    return df


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
    temporal_cols = ['stay_id', 'intime', 'outtime']
    
    df = df.groupby(['subject_id', 'hadm_id'], sort=False).agg({
        col: list for col in temporal_cols
    }).reset_index()
    
    df['count'] = df['stay_id'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return add_prefix_to_columns(df, 'icu')

def preprocess_edstays(df):
    """Aggregate ED stays per admission into lists and add stay count."""
    df = clean_cols_types(df)
    
    # df = df.drop(columns=['arrival_transport'])

    df = df.drop(columns=['arrival_transport', 'disposition'])
    
    return add_prefix_to_columns(df, 'ed')


def merge_hosp_to_ed(admissions_df, patients_df, diagnosis_df, icustays_df, edstays_df, ed_diagnosis_df):
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

    # Keep rows where timestamps match, or one side is missing
    mask_intime = (
        (hosp_master_df['ed_intime'] == hosp_master_df['edregtime']) |
        hosp_master_df['ed_intime'].isna() |
        hosp_master_df['edregtime'].isna()
    )
    mask_outtime = (
        (hosp_master_df['ed_outtime'] == hosp_master_df['edouttime']) |
        hosp_master_df['ed_outtime'].isna() |
        hosp_master_df['edouttime'].isna()
    )

    # Only keep rows where both intime and outtime pass the filter
    hosp_master_df = hosp_master_df[mask_intime & mask_outtime].copy()

    return hosp_master_df


def clean_master_df(df):
    df['gender'] = df['gender'].combine_first(df['ed_gender'])
    df['race'] = df['race'].combine_first(df['ed_race'])
    df['diagnosis'] = df['hosp_diagnosis'].combine_first(df['ed_diagnosis'])
    df['icd_codes'] = df['hosp_icd_codes_diagnosis'].combine_first(df['ed_icd_codes_diagnosis'])

    df = df.drop(columns=['edregtime', 'edouttime',
                          'ed_gender', 'ed_race', 
                          'hosp_diagnosis', 'ed_diagnosis',
                          'hosp_icd_codes_diagnosis', 'ed_icd_codes_diagnosis'])

    return df


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
    print("Running clinical encounters preprocessing...")
    
    print("\n[1/10] Loading configuration...")
    config = load_config(config_path)
    
    print("[2/10] Loading raw data...")
    (patients, admissions, hosp_diagnosis, 
     icustays, edstays, ed_diagnosis) = load_static_data(in_dir, config)

    print("[3/10] Preprocessing patients...")
    patients_processed = preprocess_patient(patients)
    
    print("[4/10] Preprocessing admissions...")
    admissions_processed = preprocess_admissions(admissions)
    
    print("[5/10] Cleaning hospital diagnosis data...")
    hosp_diagnosis_cleaned = clean_diagnosis_data(hosp_diagnosis, prefix="hosp")
    
    print("[6/10] Cleaning ED diagnosis data...")
    ed_diagnosis_cleaned = clean_diagnosis_data(ed_diagnosis, prefix="ed")
    
    print("[7/10] Preprocessing ICU stays...")
    icustays_agg = preprocess_icustays(icustays)

    print("[8/10] Cleaning ED stays...")
    edstays_cleaned = preprocess_edstays(edstays)
    
    print("[9/10] Merging emergency and hospital data...")
    hosp_master = merge_hosp_to_ed(
        admissions_processed,
        patients_processed,
        hosp_diagnosis_cleaned,
        icustays_agg,
        edstays_cleaned,
        ed_diagnosis_cleaned
    )

    print("[10/10] Cleaning merged dataset...")
    master_clinical_encounters = clean_master_df(hosp_master)
    
    print("-" * 60)
    print(f"Final dataset shape: {master_clinical_encounters.shape}")
    print(f"Number of columns: {len(master_clinical_encounters.columns)}")
    print(f"Saving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_clinical_encounters.to_csv(out_path, index=False)
    print("Clinical encounters preprocessing complete!")
    print("-" * 60)

    return master_clinical_encounters