import pandas as pd
from pathlib import Path
import json
import sys
from tqdm import tqdm


def load_config(config_path):
    """Load JSON configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def preprocess_omr(in_dir, config):
    """Load and pivot OMR vitals."""
    in_dir = Path(in_dir)
    omr_file = in_dir / config["sources"]["omr"]
    
    omr = pd.read_csv(omr_file)
    
    # Standardize result names
    omr['result_name'] = omr['result_name'].replace({
        'Weight': 'Hosp Weight (Lbs)',
        'Height': 'Hosp Height (Inches)',
        'BMI': 'Hosp BMI (kg/m2)'
    })
    
    # Pivot table
    omr_pivot = (
        omr
        .pivot_table(
            index=['subject_id', 'chartdate'],
            columns='result_name',
            values='result_value',
            aggfunc='first'
        )
        .reset_index()
    )
    
    # Convert date
    omr_pivot['chartdate'] = pd.to_datetime(omr_pivot['chartdate'])
    
    # Rename BP
    if 'Blood Pressure' in omr_pivot.columns:
        omr_pivot = omr_pivot.rename(columns={'Blood Pressure': 'Hosp Blood Pressure'})
    
    # Group to keep first per patient/date
    omr_cleaned = (
        omr_pivot
        .groupby(['subject_id', 'chartdate'], as_index=False)
        .agg({
            'Hosp BMI (kg/m2)': 'first',
            'Hosp Height (Inches)': 'first',
            'Hosp Weight (Lbs)': 'first',
            'Hosp Blood Pressure': 'first'
        })
    )
    
    return omr_cleaned


def preprocess_ed_vitals(in_dir, config):
    """Load ED vitals and normalize chartdate."""
    in_dir = Path(in_dir)
    s = config["sources"]
    ed_file = in_dir / s["ed_vitals"]

    ed_vitals = pd.read_csv(ed_file)
    
    # Convert to datetime
    ed_vitals['charttime'] = pd.to_datetime(ed_vitals['charttime'])
    ed_vitals['chartdate'] = ed_vitals['charttime'].dt.normalize()
    
    # Rename columns for clarity
    ed_vitals = ed_vitals.rename(columns={
        'temperature': 'ED Temperature (F)',
        'heartrate': 'ED Heart Rate',
        'resprate': 'ED Respitory Rate',
        'o2sat': 'ED Oxygen Saturation %',
        'sbp': 'ED sbp',
        'dbp': 'ED dbp',
        'charttime': 'charttime_ed'
    })
    
    return ed_vitals


def preprocess_lab_events(in_dir, config):
    """Load lab events and pivot."""
    in_dir = Path(in_dir)
    s = config["sources"]
    lab_file = in_dir / s["lab_events"]
    
    lab_events = pd.read_csv(lab_file)
    
    # Clean missing and extract numeric values
    lab_events['value'] = lab_events['value'].replace(['___', ''], pd.NA)
    lab_events['valuenum'] = lab_events['valuenum'].fillna(
        pd.to_numeric(
            lab_events['value'].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0],
            errors='coerce'
        )
    )
    
    # Pivot table
    lab_events_vitals = (
        lab_events
        .pivot_table(
            index=['subject_id', 'hadm_id', 'charttime'],
            columns='label',
            values='valuenum',
            aggfunc='first'
        )
        .reset_index()
    )
    
    # Normalize date and rename
    lab_events_vitals['charttime'] = pd.to_datetime(lab_events_vitals['charttime'])
    lab_events_vitals['chartdate'] = lab_events_vitals['charttime'].dt.normalize()
    
    lab_events_vitals = lab_events_vitals.rename(columns={
        'Oxygen Saturation': 'Lab Oxygen Saturation %',
        'Temperature': 'Lab Temperature (C)',
        'charttime': 'charttime_lab'
    })
    
    return lab_events_vitals


def merge_vitals(omr_cleaned, ed_vitals, lab_events_vitals):
    """Merge OMR, ED vitals, and lab events preserving all OMR patient/days."""
    
    # Merge ED + OMR (right join keeps all OMR)
    ed_omr = ed_vitals.merge(
        omr_cleaned,
        on=['subject_id', 'chartdate'],
        how='right'
    )
    
    # Merge lab events
    final = lab_events_vitals.merge(
        ed_omr,
        on=['subject_id', 'chartdate'],
        how='right'
    )
    
    # Sort for clarity
    final = final.sort_values(['subject_id', 'chartdate', 'charttime_lab', 'charttime_ed'])
    
    return final


def run_vitals_preprocessing(in_dir, config_path, out_path):
    """Main vitals preprocessing pipeline."""
    steps = [
        "Loading configuration",
        "Processing OMR vitals",
        "Processing ED vitals",
        "Processing Lab Events vitals",
        "Merging all vitals"
    ]
    
    print("Running Vitals Preprocessing...")
    print()
    
    pbar = tqdm(total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
    
    try:
        pbar.set_description(f"[1/5] {steps[0]}")
        config = load_config(config_path)
        pbar.update(1)
        
        pbar.set_description(f"[2/5] {steps[1]}")
        omr_cleaned = preprocess_omr(in_dir, config)
        pbar.update(1)
        
        pbar.set_description(f"[3/5] {steps[2]}")
        ed_vitals = preprocess_ed_vitals(in_dir, config)
        pbar.update(1)
        
        pbar.set_description(f"[4/5] {steps[3]}")
        lab_events_vitals = preprocess_lab_events(in_dir, config)
        pbar.update(1)
        
        pbar.set_description(f"[5/5] {steps[4]}")
        final_vitals = merge_vitals(omr_cleaned, ed_vitals, lab_events_vitals)
        pbar.update(1)
        
    finally:
        pbar.close()
    
    print()
    print(f"Saving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_vitals.to_csv(out_path, index=False)
    print(f"✓ Vitals preprocessing completed. Saved to {out_path}")
    
    return final_vitals