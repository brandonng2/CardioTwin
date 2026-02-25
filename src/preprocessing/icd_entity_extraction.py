import json
from pathlib import Path
import pandas as pd
import ast
import numpy as np
import sys
from tqdm import tqdm
from .icd_code_labels import icd_code_labels, noncardiovascular_labels, cardiovascular_labels


def load_config(config_path):
    """Load preprocessing configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_clinical_data(in_dir, config):
    """Load clinical data from CSV files."""
    in_dir = Path(in_dir)
    s = config["sources"]
    clinical_encounters = pd.read_csv(in_dir / s["clinical_encounters"], dtype=str, low_memory=False)
    
    return clinical_encounters


def normalize_icd(code):
    """Normalize ICD code format."""
    if code is None:
        return None
    code = str(code)
    if len(code) > 3 and "." not in code:
        return code[:3] + "." + code[3:]
    return code


def clean_icd_codes(codes):
    """Clean and normalize ICD codes from various formats."""
    if codes is None or (isinstance(codes, float) and np.isnan(codes)):
        return None
    
    # If it's already a list, use it
    if isinstance(codes, list):
        codes_list = codes
    # If it's a string representation of a list, parse it
    elif isinstance(codes, str) and codes.startswith("[") and codes.endswith("]"):
        codes_list = ast.literal_eval(codes)
    else:
        return None
        
    return [normalize_icd(code) for code in codes_list]


def map_codes_to_labels(icd_codes):
    """
    Map ICD-10 codes to condition labels.
    Returns 'unspecified_cardiac' only if code starts with 'I' and no labels are matched.
    """
    if not isinstance(icd_codes, list):
        return []
    
    labels = []
    has_cardiac_code = False
    
    # Convert codes to strings and strip whitespace
    icd_codes_str = [str(code).strip() for code in icd_codes]
    
    # Check if any code starts with 'I' (cardiovascular ICD-10 codes)
    has_cardiac_code = any(code.startswith('I') for code in icd_codes_str)
    
    for label, code_prefixes in icd_code_labels.items():
        for icd_code in icd_codes_str:
            # Check if any prefix matches the beginning of the ICD code
            if any(icd_code.startswith(prefix) for prefix in code_prefixes):
                labels.append(label)
                break  # Found a match for this label, move to next label
    
    # Return 'unspecified_cardiac' only if has cardiac code but no labels matched
    if not labels and has_cardiac_code:
        return ["unspecified_cardiac"]
    
    return labels

def is_cardiovascular_encounter(labels):
    return any(label in cardiovascular_labels for label in labels)


def run_entity_extraction(in_dir, config_path, out_path):
    """Run full clinical entity extraction and save results."""
    steps = [
        "Loading configuration",
        "Loading clinical data",
        "Cleaning ICD codes",
        "Extracting cardiovascular labels",
    ]
    
    print("Running ICD Code Extraction...")
    print()
    
    pbar = tqdm(total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
    
    try:
        pbar.set_description(f"{steps[0]}")
        config = load_config(config_path)
        pbar.update(1)

        pbar.set_description(f"{steps[1]}")
        clinical_encounter = load_clinical_data(in_dir, config)
        pbar.update(1)

        pbar.set_description(f"{steps[2]}")
        clinical_encounter['icd_codes'] = clinical_encounter['icd_codes'].apply(clean_icd_codes)
        pbar.update(1)

        pbar.set_description(f"{steps[3]}")
        clinical_encounter['diagnosis_labels'] = clinical_encounter['icd_codes'].apply(map_codes_to_labels)
        clinical_encounter['is_cardiovascular'] = clinical_encounter['diagnosis_labels'].apply(is_cardiovascular_encounter)

        pbar.update(1)
        
    finally:
        pbar.close()

    print()
    print("-" * 60)
    print(f"Final dataset shape: {clinical_encounter.shape}")
    print(f"Number of columns: {len(clinical_encounter.columns)}")
    print(f"Saving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clinical_encounter.to_csv(out_path, index=False)
    print("✓ Clinical entity extraction complete!")
    print("-" * 60)

    return clinical_encounter