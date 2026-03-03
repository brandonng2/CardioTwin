import json
import argparse
from pathlib import Path
from src.models.mlp import (
    run_mlp_base_pipeline, 
    run_mlp_smote_pipeline,
    run_mlp_weighted_pipeline
)
from src.models.xgboost import (
    run_xgboost_base_pipeline,
    run_xgboost_weighted_pipeline,
    run_xgboost_smote_pipeline,
)
from src.models.xgboost_embedding import run_xgboost_embedding_pipeline
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


def run_xgboost_base(args):
    """Run XGBoost base model."""
    print("\n" + "=" * 60)
    print("XGBOOST BASE MODEL")
    print("=" * 60)

    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_xgboost_base_pipeline(in_dir, config_path, out_path)


def run_xgboost_weighted(args):
    """Run XGBoost weighted model."""
    print("\n" + "=" * 60)
    print("XGBOOST WEIGHTED MODEL")
    print("=" * 60)

    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_xgboost_weighted_pipeline(in_dir, config_path, out_path)


def run_xgboost_smote(args):
    """Run XGBoost SMOTE model."""
    print("\n" + "=" * 60)
    print("XGBOOST SMOTE MODEL")
    print("=" * 60)

    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_xgboost_smote_pipeline(in_dir, config_path, out_path)


def run_xgboost_embedding(args):
    """Run XGBoost with Embeddings model."""
    print("\n" + "=" * 60)
    print("XGBOOST W/ EMBEDDINGS MODEL")
    print("=" * 60)

    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_xgboost_embedding_pipeline(in_dir, config_path, out_path)


def run_mlp_baseline(args):
    """Run MLP baseline model."""
    print("\n" + "=" * 60)
    print("MLP BASELINE MODEL")
    print("=" * 60)

    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_mlp_base_pipeline(in_dir, config_path, out_path)

def run_mlp_weighted(args):
    """Run MLP weighted model."""
    print("\n" + "=" * 60)
    print("MLP WEIGHTED MODEL")
    print("=" * 60)

    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_mlp_weighted_pipeline(in_dir, config_path, out_path)

def run_mlp_smote(args):
    """Run MLP SMOTE model."""
    print("\n" + "=" * 60)
    print("MLP SMOTE MODEL")
    print("=" * 60)

    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    in_dir = Path(config["paths"]["in_dir"])
    out_path = Path(config["paths"]["out_dir"])

    run_mlp_smote_pipeline(in_dir, config_path, out_path)


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

            # Run XGBoost variants
            python run.py --xgboost-base        # Baseline
            python run.py --xgboost-weighted    # Per-label scale_pos_weight
            python run.py --xgboost-smote       # SMOTE on rare labels
            python run.py --xgboost-embedding   # ECG-FM Embeddings
            
            # Run MLP variants
            python run.py --mlp-baseline
            python run.py --mlp-weighted
            python run.py --mlp-smote
        """,
    )

    # Run specific components
    parser.add_argument("--all", action="store_true", help="Run all preprocessing steps")
    parser.add_argument("--static", action="store_true", help="Run static preprocessing")
    parser.add_argument("--ecg", action="store_true", help="Run ECG preprocessing")
    parser.add_argument("--vitals", action="store_true", help="Run vitals preprocessing")
    parser.add_argument("--entities", action="store_true", help="Run entity extraction")
    parser.add_argument(
        "--xgboost-base", action="store_true", dest="xgb_base",
        help="Run XGBoost base model (default)",
    )
    parser.add_argument(
        "--xgboost-weighted", action="store_true", dest="xgb_weighted",
        help="Run XGBoost weighted model — per-label scale_pos_weight",
    )
    parser.add_argument(
        "--xgboost-smote", action="store_true", dest="xgb_smote",
        help="Run XGBoost SMOTE model — SMOTE on labels < 8% prevalence",
    )
    parser.add_argument(
        "--xgboost-embedding", action="store_true", dest="xgb_embedding",
        help="Run XGBoost embedding model — Creating ECG embeddings instead of derived features",
    )
    parser.add_argument(
        "--mlp-baseline", action="store_true", dest="mlp_baseline",
        help="Run MLP baseline model (predicts diagnosis labels)",
    )
    parser.add_argument(
        "--mlp-weighted", action="store_true", dest="mlp_weighted",
        help="Run MLP weighted model (per-label class weights)",
    )
    parser.add_argument(
        "--mlp-smote", action="store_true", dest="mlp_smote",
        help="Run MLP SMOTE model (SMOTE on rare labels)",
    )

    # Skip flags (for use with --all)
    parser.add_argument("--skip-static", action="store_true", help="Skip static preprocessing")
    parser.add_argument("--skip-ecg", action="store_true", help="Skip ECG preprocessing")
    parser.add_argument("--skip-vitals", action="store_true", help="Skip vitals preprocessing")
    parser.add_argument("--skip-entities", action="store_true", help="Skip entity extraction")
    parser.add_argument("--skip-xgboost-base", action="store_true", help="Skip XGBoost base model")
    parser.add_argument("--skip-xgboost-embedding", action="store_true", help="Skip XGBoost embedding model")
    parser.add_argument("--skip-mlp-baseline", action="store_true", help="Skip MLP baseline model")
    parser.add_argument("--skip-mlp-weighted", action="store_true", help="Skip MLP weighted model")
    parser.add_argument("--skip-mlp-smote", action="store_true", help="Skip MLP SMOTE model")

    args = parser.parse_args()

    # Determine what to run
    run_all = args.all or not any([
        args.static, args.ecg, args.vitals, args.entities,
        args.xgb_base, args.xgb_weighted, args.xgb_smote,
        args.xgb_embedding, args.mlp_baseline, args.mlp_weighted, args.mlp_smote,
    ])

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

    # XGBoost Models
    if (run_all and not args.skip_xgboost_base) or args.xgb_base:
        run_xgboost_base(args)

    if (run_all and not args.skip_xgboost_weighted) or args.xgb_weighted:
        run_xgboost_weighted(args)

    if (run_all and not args.skip_xgboost_smote) or args.xgb_smote:
        run_xgboost_smote(args)

    if (run_all and not args.skip_xgboost_embedding) or args.xgb_embedding:
        run_xgboost_embedding(args)

    # MLP Baseline Model
    if (run_all and not args.skip_mlp_baseline) or args.mlp_baseline:
        run_mlp_baseline(args)

    # MLP Weighted Model
    if (run_all and not args.skip_mlp_weighted) or args.mlp_weighted:
        run_mlp_weighted(args)

    # MLP SMOTE Model
    if (run_all and not args.skip_mlp_smote) or args.mlp_smote:
        run_mlp_smote(args)
        


if __name__ == "__main__":
    main()