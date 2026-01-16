import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_config(config_path):
    """Load preprocessing configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def load_ecg_data(raw_dir, processed_dir, config):
    """Load ECG data from CSV files."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    s = config["sources"]
    
    # ecg_record = pd.read_csv(raw_dir / s["ecg_record"])
    # return ecg_record
    
    return