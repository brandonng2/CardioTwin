import json
import argparse
from pathlib import Path
from src.models.xgboost_baseline import run_xgboost_baseline_pipeline
from src.preprocessing.static_preprocessing import run_static_preprocessing
from src.preprocessing.ecg_preprocessing import run_ecg_preprocessing
from src.preprocessing.vitals_preprocessing import run_vitals_preprocessing
from src.preprocessing.icd_entity_extraction import run_entity_extraction


def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_static(args):
    """Run static preprocessing."""
    print("\n" + "=" * 60)
    print("STATIC PREPROCESSING")
    print("=" * 60)
    
    config = load_config("configs/static_preprocessing_params.json")
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])
    
    run_static_preprocessing(in_dir, "configs/static_preprocessing_params.json", out_path)
    print("✓ Static preprocessing completed")


def run_ecg(args):
    """Run ECG preprocessing."""
    print("\n" + "=" * 60)
    print("ECG PREPROCESSING")
    print("=" * 60)
    
    config = load_config("configs/ecg_preprocessing_params.json")
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])
    
    run_ecg_preprocessing(in_dir, "configs/ecg_preprocessing_params.json", out_path)
    print("✓ ECG preprocessing completed")


def run_vitals(args):
    """Run vitals preprocessing."""
    print("\n" + "=" * 60)
    print("VITALS PREPROCESSING")
    print("=" * 60)

    config_path = "configs/vitals_preprocessing_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])
    
    run_vitals_preprocessing(in_dir, config_path, out_path)
    print("✓ Vitals preprocessing completed")


def run_icd_extraction(args):
    """Run clinical entity extraction."""
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION")
    print("=" * 60)
    
    config_path = "configs/icdcode_extractor_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])
    
    run_entity_extraction(in_dir, config_path, out_path)
    print("✓ ICD code extraction completed")


def run_xgboost_baseline(args):
    """Run XGBoost baseline model."""
    print("\n" + "=" * 60)
    print("XGBOOST BASELINE MODEL")
    print("=" * 60)
    
    config_path = "configs/xgboost_baseline_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])
    
    # Determine target_type from command-line argument
    target_type = None
    if hasattr(args, 'xgb_target') and args.xgb_target and args.xgb_target != 'default':
        # Map friendly names to internal values
        target_mapping = {
            'diagnosis': 'labels',
            'label': 'labels',
            'labels': 'labels',
            'report': 'reports',
            'reports': 'reports'
        }
        target_type = target_mapping.get(args.xgb_target.lower())
        if target_type is None:
            print(f"Warning: Invalid target type '{args.xgb_target}'. Using config default.")

    run_xgboost_baseline_pipeline(in_dir, config_path, out_path, target_type=target_type)


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
  
  # Run XGBoost with specific target
  python run.py --xgbaseline label      # Predict diagnosis labels (default)
  python run.py --xgbaseline report     # Predict ECG reports
  python run.py --xgbaseline            # Use config default
        """
    )
    
    # Run specific components
    parser.add_argument("--all", action="store_true", help="Run all preprocessing steps")
    parser.add_argument("--static", action="store_true", help="Run static preprocessing")
    parser.add_argument("--ecg", action="store_true", help="Run ECG preprocessing")
    parser.add_argument("--vitals", action="store_true", help="Run vitals preprocessing")
    parser.add_argument("--entities", action="store_true", help="Run entity extraction")
    parser.add_argument("--xgbaseline", nargs='?', const='default', dest='xgb_target',
                       help="Run XGBoost baseline model. Optional: 'label' (default) or 'report'")
    
    # Skip flags (for use with --all)
    parser.add_argument("--skip-static", action="store_true", help="Skip static preprocessing")
    parser.add_argument("--skip-ecg", action="store_true", help="Skip ECG preprocessing")
    parser.add_argument("--skip-vitals", action="store_true", help="Skip vitals preprocessing")
    parser.add_argument("--skip-entities", action="store_true", help="Skip entity extraction")
    parser.add_argument("--skip-xgbaseline", action="store_true", help="Skip XGBoost baseline model")

    args = parser.parse_args()
    
    # Determine what to run
    run_all = args.all or not any([args.static, args.ecg, args.vitals, args.entities, args.xgb_target])
    
    print("=" * 60)
    print("MIMIC-IV PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Static preprocessing
    if (run_all and not args.skip_static) or args.static:
        run_static(args)
    
    # ECG preprocessing
    if (run_all and not args.skip_ecg) or args.ecg:
        run_ecg(args)

    # Vitals preprocessing
    if (run_all and not args.skip_vitals) or args.vitals:
        run_vitals(args)
    
    # Entity extraction
    if (run_all and not args.skip_entities) or args.entities:
        run_icd_extraction(args)
    
    print("\n" + "=" * 60)
    print("✓ ALL PREPROCESSING COMPLETED!")
    print("=" * 60)

    # XGBoost Baseline Model
    if (run_all and not args.skip_xgbaseline) or args.xgb_target:
        run_xgboost_baseline(args)


if __name__ == "__main__":
    main()