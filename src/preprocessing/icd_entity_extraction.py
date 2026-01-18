import json
from pathlib import Path
import pandas as pd
import ast
import numpy as np
from .icd_code_labels import icd_code_labels
from .icd_code_labels import noncardiovascular_labels
from .icd_code_labels import cardiovascular_labels


def load_config(config_path):
    """Load preprocessing configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def load_clinical_data(config):
    """Load clinical data from CSV files."""
    in_dir = Path(config["in_dir"])
    s = config["sources"]
    clinical_encounters = pd.read_csv(in_dir / s["clinical_encounters"], dtype=str, low_memory=False)
    
    return clinical_encounters


def normalize_icd(code):
    if code is None:
        return None
    code = str(code)
    if len(code) > 3 and "." not in code:
        return code[:3] + "." + code[3:]
    return code

def clean_icd_codes(codes):
    if codes is None or (isinstance(codes, float) and np.isnan(codes)):
        return None
     # If it's already a list, use it
    if isinstance(codes, list):
        codes_list = codes
    # If it's a string representation of a list, parse it
    elif isinstance(codes, str) and codes.startswith("[") and codes.endswith("]"):
        codes_list = ast.literal_eval(codes)
        
    return [normalize_icd(code) for code in codes_list]

# -----------------------------
# 6. Map phrases to canonical labels
# -----------------------------
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


def onehot_labels(df, label_column='labels', prefix='label_'):
    """
    One-hot encode the labels column into separate binary columns.
    Also creates summary columns for cardiovascular conditions.
    """
    import pandas as pd
    
    # Get all unique labels across all rows
    all_labels = set()
    for labels_list in df[label_column]:
        if isinstance(labels_list, list):
            all_labels.update(labels_list)
    
    # Sort labels for consistent column ordering
    all_labels = sorted(all_labels)
    
    # Create one-hot encoded columns
    onehot_data = {}
    for label in all_labels:
        onehot_data[f'{prefix}{label}'] = df[label_column].apply(
            lambda x: 1 if isinstance(x, list) and label in x else 0
        )
    
    # Add one-hot columns to original dataframe
    onehot_df = pd.DataFrame(onehot_data, index=df.index)
    result = pd.concat([df, onehot_df], axis=1)
    
    # Create is_cardiovascular column - 1 if has any CV label (including unspecified)
    result['is_cardiovascular'] = result[label_column].apply(
        lambda x: 1 if isinstance(x, list) and len(x) > 0 and 
        any(label not in noncardiovascular_labels for label in x) else 0
    )
    
    # Create is_specified_cardiac column - 1 if has specific cardiac diagnosis (not just unspecified)
    result['is_specified_cardiac'] = result[label_column].apply(
        lambda x: 1 if isinstance(x, list) and len(x) > 0 and 
        any(label in cardiovascular_labels and label != 'unspecified_cardiac' for label in x) else 0
    )
    
    return result


# -----------------------------
# 7. Main function
# -----------------------------
def run_entity_extraction(config_path):
    """
    Run full clinical entity extraction and save results.
    """
    print("Running ICD Code Extraction...")

    config = load_config(config_path)

    clinical_encounter = load_clinical_data(config)

    clinical_encounter['icd_codes'] = clinical_encounter['icd_codes'].apply(clean_icd_codes)

    print("[1/2] Extracting Cardiovascular labels from ICD codes...")
    clinical_encounter['labels'] = clinical_encounter['icd_codes'].apply(map_codes_to_labels)

    print("[2/2] One-Hot Encoding labels...")
    clinical_encounter_extracted = onehot_labels(clinical_encounter)

    print("-" * 60)
    print(f"Final dataset shape: {clinical_encounter_extracted.shape}")
    print(f"Number of columns: {len(clinical_encounter_extracted.columns)}")
    out_path = Path(config["out_path"])

    print(f"Saving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clinical_encounter_extracted.to_csv(out_path, index=False)
    print("Clinical entity extraction complete!")
    print("-" * 60)

    return clinical_encounter_extracted
