import json
import sys
from pathlib import Path
import pandas as pd
from src.preprocessing.static_preprocessing import run_static_preprocessing
from src.preprocessing.ecg_preprocessing import run_ecg_machine_measurements_preprocessing


def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    # Check for skip flags
    skip_static = "--skip-static" in sys.argv

    print("=" * 60)
    print("MIMIC-IV PREPROCESSING PIPELINE")
    print("=" * 60)
    print()
    
    
    # --- Static Preprocessing ---
    if not skip_static:
        static_config = load_config("configs/static_preprocessing_params.json")
        
        in_dir = Path(static_config["paths"]["in_dir"])
        out_path = Path(static_config["paths"]["out_dir"])
        
        run_static_preprocessing(in_dir, "configs/static_preprocessing_params.json", out_path)
    else:
        print("Skipping static preprocessing")

    # --- ECG Machine Measurements ---
    skip_ecg_mm = "--skip-ecg-mm" in sys.argv

    if not skip_ecg_mm:
        ecg_config = load_config("configs/ecg_preprocessing_params.json")

        in_dir = Path(ecg_config["paths"]["in_dir"])
        out_path = Path(ecg_config["paths"]["out_dir"])

        run_ecg_machine_measurements_preprocessing(in_dir, "configs/ecg_machine_measurements_params.json", out_path)
    else:
        print("Skipping ECG machine measurements preprocessing")


    # --- Clinical Entity Extraction ---
    if not skip_static:
        # Output from static preprocessing
        static_master_path = out_path
        entity_out_path = out_path.parent / "static_master_with_entities.csv"

        # Load static master
        static_master = pd.read_csv(static_master_path)

        # Run the new modular clinical entity extraction
        from src.preprocessing.icd_entity_extraction import run_entity_extraction

        static_master = run_entity_extraction(static_master, entity_out_path)
        print(f"Clinical entity extraction completed. Saved to {entity_out_path}")
    else:
        print("Skipping clinical entity extraction since static preprocessing was skipped")

    
    print("\n" + "=" * 60)
    print("ALL PREPROCSSING COMPLETED!")
    print("=" * 60)
    

if __name__ == "__main__":
    main()
