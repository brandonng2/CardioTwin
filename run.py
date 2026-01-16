import json
import sys
import argparse
from pathlib import Path
import pandas as pd
from src.preprocessing.static_preprocessing import run_static_preprocessing
from src.preprocessing.ecg_preprocessing import run_ecg_preprocessing
from src.preprocessing.vitals_preprocessing import run_vitals_preprocessing



def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_static(args):
    """Run static preprocessing."""
    print("\n" + "=" * 60)
    print("STATIC PREPROCESSING")
    print("=" * 60)
    
    static_config = load_config("configs/static_preprocessing_params.json")
    in_dir = Path(static_config["paths"]["in_dir"])
    out_path = Path(static_config["paths"]["out_dir"])
    
    run_static_preprocessing(in_dir, "configs/static_preprocessing_params.json", out_path)
    print("✓ Static preprocessing completed")
    return out_path


def run_ecg(args):
    """Run ECG preprocessing."""
    print("\n" + "=" * 60)
    print("ECG PREPROCESSING")
    print("=" * 60)
    
    ecg_config = load_config("configs/ecg_preprocessing_params.json")
    in_dir = Path(ecg_config["paths"]["in_dir"])
    out_path = Path(ecg_config["paths"]["out_dir"])
    
    run_ecg_preprocessing(in_dir, "configs/ecg_preprocessing_params.json", out_path)
    print("✓ ECG preprocessing completed")
    return out_path

def run_vitals(args):
    """Run vitals preprocessing."""
    print("\n" + "=" * 60)
    print("VITALS PREPROCESSING")
    print("=" * 60)
    
    vitals_config_path = "configs/vitals_preprocessing_params.json"
    run_vitals_preprocessing(vitals_config_path)
    
    print("✓ Vitals preprocessing completed")

def run_entity_extraction(args, static_master_path=None):
    """Run clinical entity extraction."""
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION")
    print("=" * 60)
    
    if static_master_path is None:
        static_config = load_config("configs/static_preprocessing_params.json")
        static_master_path = Path(static_config["paths"]["out_dir"])
    
    entity_out_path = static_master_path.parent / "static_master_with_entities.csv"
    static_master = pd.read_csv(static_master_path)
    
    from src.preprocessing.icd_entity_extraction import run_entity_extraction
    static_master = run_entity_extraction(static_master, entity_out_path)
    print(f"✓ Entity extraction completed. Saved to {entity_out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Patient-Specific Cardiovascular Digital Twin for Personalized Cardiac Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  Examples:
  # Run everything
  python run.py --all
  
  # Run specific steps
  python run.py --static --ecg
  python run.py --ecg
  
  # Run everything except certain steps
  python run.py --all --skip-static
        """
    )
    
    # Run specific components
    parser.add_argument("--all", action="store_true", help="Run all preprocessing steps")
    parser.add_argument("--static", action="store_true", help="Run static preprocessing")
    parser.add_argument("--ecg", action="store_true", help="Run ECG preprocessing")
    parser.add_argument("--vitals", action="store_true", help="Run vitals preprocessing")
    parser.add_argument("--entities", action="store_true", help="Run entity extraction")  
    
    # Skip flags (for use with --all)
    parser.add_argument("--skip-static", action="store_true", help="Skip static preprocessing")
    parser.add_argument("--skip-ecg", action="store_true", help="Skip ECG preprocessing")
    parser.add_argument("--skip-vitals", action="store_true", help="Skip vitals preprocessing")  
    parser.add_argument("--skip-entities", action="store_true", help="Skip entity extraction")
        
    args = parser.parse_args()
    
    # Determine what to run
    run_all = args.all or not any([args.static, args.ecg, args.vitals, args.entities])
    
    print("=" * 60)
    print("MIMIC-IV PREPROCESSING PIPELINE")
    print("=" * 60)
    
    static_master_path = None
    
    # Static preprocessing
    if (run_all and not args.skip_static) or args.static:
        static_master_path = run_static(args)
    
    # ECG preprocessing
    if (run_all and not args.skip_ecg) or args.ecg:
        run_ecg(args)

    # Vitals preprocessing
    if (run_all and not args.skip_vitals) or args.vitals:
        run_vitals(args)
    
    # Entity extraction
    if (run_all and not args.skip_entities) or args.entities:
        run_entity_extraction(args, static_master_path)
    
    print("\n" + "=" * 60)
    print("ALL PREPROCESSING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()