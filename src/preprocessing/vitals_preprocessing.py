import pandas as pd
from pathlib import Path
import json
import sys
from tqdm import tqdm


def load_config(config_path):
    """Load JSON configuration."""
    with open(config_path, "r") as f:
        return json.load(f)
    
def preprocess_vitals(ed_vitals):
    """
    Forward fill vital signs within each patient stay.
    
    Args:
        ed_vitals: Raw ED vitals DataFrame
        
    Returns:
        DataFrame with forward-filled vital signs
    """
    cols_to_fill = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    
    ed_vitals = ed_vitals.copy()
    ed_vitals[cols_to_fill] = ed_vitals.groupby(
        ['subject_id', 'stay_id']
    )[cols_to_fill].ffill()
    
    return ed_vitals

def load_ed_vitals(in_dir, config):
    """Load ED vitals and normalize chartdate."""
    in_dir = Path(in_dir)
    s = config["sources"]
    ed_file = in_dir / s["ed_vitals"]

    ed_vitals = pd.read_csv(ed_file)
    
    # Convert to datetime
    ed_vitals['charttime'] = pd.to_datetime(ed_vitals['charttime'])
    ed_vitals['chartdate'] = ed_vitals['charttime'].dt.normalize()
    
    return ed_vitals

def run_vitals_preprocessing(in_dir, config_path, out_path):
    """Main vitals preprocessing pipeline."""
    steps = [
        "Loading configuration",
        "Loading ED vitals",
        "Preprocessing vitals",
    ]
    
    print("Running Vitals Preprocessing...")
    print()
    
    pbar = tqdm(total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
    
    try:
        pbar.set_description(f"{steps[0]}")
        config = load_config(config_path)
        pbar.update(1)
        
        
        pbar.set_description(f"{steps[1]}")
        ed_vitals = load_ed_vitals(in_dir, config)
        pbar.update(1)

        pbar.set_description(f"{steps[2]}")
        ed_vitals = preprocess_vitals(ed_vitals)
        pbar.update(1)
        
    finally:
        pbar.close()
    
    print()
    print(f"Saving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ed_vitals.to_csv(out_path, index=False)
    print(f"✓ Vitals preprocessing completed. Saved to {out_path}")
    
    return ed_vitals