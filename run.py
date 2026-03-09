import json
import argparse
from pathlib import Path

from src.preprocessing.static_preprocessing import run_static_preprocessing
from src.preprocessing.ecg_preprocessing import run_ecg_preprocessing
from src.preprocessing.vitals_preprocessing import run_vitals_preprocessing
from src.preprocessing.icd_entity_extraction import run_entity_extraction

from src.models.xgboost import (
    run_xgboost_base_pipeline,
    run_xgboost_weighted_pipeline,
    run_xgboost_smote_pipeline,
)
from src.models.xgboost_embedding import run_xgboost_embedding_pipeline

from src.models.mlp import (
    run_mlp_base_pipeline,
    run_mlp_smote_pipeline,
    run_mlp_weighted_pipeline,
    run_mlp_embedding_pipeline,
    run_mlp_embedding_weighted_pipeline,
)

from src.models.CardioTwin import run_cardiotwin_final

from src.models.cardio_digital_twin import (
    run_cardiotwin_ablation_pipeline,
)


# =============================================================================
# Config helper
# =============================================================================

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


# =============================================================================
# Preprocessing runners
# =============================================================================

def run_static(args):
    print("\n" + "=" * 60)
    print("STATIC PREPROCESSING")
    print("=" * 60)
    config = load_config("configs/static_preprocessing_params.json")
    run_static_preprocessing(
        Path(config["paths"]["in_dir"]),
        "configs/static_preprocessing_params.json",
        Path(config["paths"]["out_dir"]),
    )
    print("✓ Static preprocessing completed")


def run_ecg(args):
    print("\n" + "=" * 60)
    print("ECG PREPROCESSING")
    print("=" * 60)
    config = load_config("configs/ecg_preprocessing_params.json")
    run_ecg_preprocessing(
        Path(config["paths"]["in_dir"]),
        "configs/ecg_preprocessing_params.json",
        Path(config["paths"]["out_dir"]),
    )
    print("✓ ECG preprocessing completed")


def run_vitals(args):
    print("\n" + "=" * 60)
    print("VITALS PREPROCESSING")
    print("=" * 60)
    config_path = "configs/vitals_preprocessing_params.json"
    config = load_config(config_path)
    run_vitals_preprocessing(
        Path(config["paths"]["in_dir"]),
        config_path,
        Path(config["paths"]["out_dir"]),
    )
    print("✓ Vitals preprocessing completed")


def run_icd_extraction(args):
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION")
    print("=" * 60)
    config_path = "configs/icdcode_extractor_params.json"
    config = load_config(config_path)
    run_entity_extraction(
        Path(config["paths"]["in_dir"]),
        config_path,
        Path(config["paths"]["out_dir"]),
    )
    print("✓ ICD code extraction completed")


# =============================================================================
# XGBoost runners
# =============================================================================

def run_xgboost_base(args):
    print("\n" + "=" * 60)
    print("XGBOOST BASELINE MODEL")
    print("=" * 60)
    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    run_xgboost_base_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_xgboost_weighted(args):
    print("\n" + "=" * 60)
    print("XGBOOST WEIGHTED MODEL")
    print("=" * 60)
    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    run_xgboost_weighted_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_xgboost_smote(args):
    print("\n" + "=" * 60)
    print("XGBOOST SMOTE MODEL")
    print("=" * 60)
    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    run_xgboost_smote_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_xgboost_embedding(args):
    print("\n" + "=" * 60)
    print("XGBOOST EMBEDDING MODEL")
    print("=" * 60)
    config_path = "configs/xgboost_params.json"
    config = load_config(config_path)
    run_xgboost_embedding_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_xgboost_all_ablation(args):
    """Run all XGBoost variants: baseline, weighted, SMOTE, embedding."""
    print("\n" + "=" * 60)
    print("XGBOOST — FULL ABLATION (baseline / weighted / smote / embedding)")
    print("=" * 60)
    run_xgboost_base(args)
    run_xgboost_weighted(args)
    run_xgboost_smote(args)
    run_xgboost_embedding(args)


# =============================================================================
# MLP runners
# =============================================================================

def run_mlp_baseline(args):
    print("\n" + "=" * 60)
    print("MLP BASELINE MODEL")
    print("=" * 60)
    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    run_mlp_base_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_mlp_weighted(args):
    print("\n" + "=" * 60)
    print("MLP WEIGHTED MODEL")
    print("=" * 60)
    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    run_mlp_weighted_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_mlp_smote(args):
    print("\n" + "=" * 60)
    print("MLP SMOTE MODEL")
    print("=" * 60)
    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    run_mlp_smote_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_mlp_embedding(args):
    print("\n" + "=" * 60)
    print("MLP EMBEDDING MODEL")
    print("=" * 60)
    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    run_mlp_embedding_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_mlp_embedding_weighted(args):
    print("\n" + "=" * 60)
    print("MLP EMBEDDING WEIGHTED MODEL")
    print("=" * 60)
    config_path = "configs/mlp_params.json"
    config = load_config(config_path)
    run_mlp_embedding_weighted_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_mlp_all_ablation(args):
    """Run all MLP variants: baseline, weighted, SMOTE, embedding, embedding_weighted."""
    print("\n" + "=" * 60)
    print("MLP — FULL ABLATION (baseline / weighted / smote / embedding / embedding_weighted)")
    print("=" * 60)
    run_mlp_baseline(args)
    run_mlp_weighted(args)
    run_mlp_smote(args)
    run_mlp_embedding(args)
    run_mlp_embedding_weighted(args)


# =============================================================================
# CardioTwin runners
# =============================================================================

def run_cardiotwin(args):
    """Main pipeline — baseline only (128-dim, gated, BCE, no sampler)."""
    print("\n" + "=" * 60)
    print("CARDIOTWIN — FINAL MODEL")
    print("=" * 60)
    config_path = "configs/CardioTwin_model_params.json"
    config = load_config(config_path)
    run_cardiotwin_final(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


def run_cardiotwin_ablation(args):
    """Full ablation — all variants x all loss types x all sampler modes."""
    print("\n" + "=" * 60)
    print("CARDIOTWIN — FULL ABLATION")
    print("=" * 60)
    config_path = "configs/cardiotwin_params.json"
    config = load_config(config_path)
    run_cardiotwin_ablation_pipeline(
        Path(config["paths"]["in_dir"]), config_path, Path(config["paths"]["out_dir"])
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Patient-Specific Cardiovascular Digital Twin — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Main pipeline: preprocessing + CardioTwin final model
  python run.py --all

  # Preprocessing only
  python run.py --preprocess

  # Individual preprocessing steps
  python run.py --static
  python run.py --ecg
  python run.py --vitals
  python run.py --entities

  # XGBoost variants
  python run.py --xgboost-baseline
  python run.py --xgboost-weighted
  python run.py --xgboost-smote
  python run.py --xgboost-embedding
  python run.py --xgboost-ablation        # all four above in sequence

  # MLP variants
  python run.py --mlp-baseline
  python run.py --mlp-weighted
  python run.py --mlp-smote
  python run.py --mlp-embedding
  python run.py --mlp-embedding-weighted
  python run.py --mlp-ablation            # all five above in sequence

  # CardioTwin
  python run.py --cardiotwin              # final model only (main pipeline)
  python run.py --cardiotwin-ablation     # all variants x losses x samplers
        """,
    )

    # -----------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------
    parser.add_argument(
        "--all", action="store_true",
        help="Run full main pipeline: all preprocessing + CardioTwin final model",
    )

    # -----------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Run all preprocessing steps (static, ECG, vitals, entities)",
    )
    parser.add_argument("--static", action="store_true", help="Run static preprocessing")
    parser.add_argument("--ecg", action="store_true", help="Run ECG preprocessing")
    parser.add_argument("--vitals", action="store_true", help="Run vitals preprocessing")
    parser.add_argument("--entities", action="store_true", help="Run ICD entity extraction")

    # -----------------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------------
    parser.add_argument("--xgboost-baseline", action="store_true", dest="xgb_base",
                        help="XGBoost baseline (normalized, no weighting)")
    parser.add_argument("--xgboost-weighted", action="store_true", dest="xgb_weighted",
                        help="XGBoost with per-label scale_pos_weight")
    parser.add_argument("--xgboost-smote", action="store_true", dest="xgb_smote",
                        help="XGBoost with capped SMOTE on rare labels")
    parser.add_argument("--xgboost-embedding", action="store_true", dest="xgb_embedding",
                        help="XGBoost with ECG-FM 1536-dim embeddings")
    parser.add_argument("--xgboost-ablation", action="store_true", dest="xgb_ablation",
                        help="Run all XGBoost variants in sequence")

    # -----------------------------------------------------------------
    # MLP
    # -----------------------------------------------------------------
    parser.add_argument("--mlp-baseline", action="store_true", dest="mlp_baseline",
                        help="MLP baseline (normalized, uniform BCE)")
    parser.add_argument("--mlp-weighted", action="store_true", dest="mlp_weighted",
                        help="MLP with per-label pos_weight")
    parser.add_argument("--mlp-smote", action="store_true", dest="mlp_smote",
                        help="MLP with capped SMOTE on rare labels")
    parser.add_argument("--mlp-embedding", action="store_true", dest="mlp_embedding",
                        help="MLP with ECG-FM 1536-dim embeddings")
    parser.add_argument("--mlp-embedding-weighted", action="store_true", dest="mlp_embedding_weighted",
                        help="MLP with ECG-FM embeddings + pos_weight")
    parser.add_argument("--mlp-ablation", action="store_true", dest="mlp_ablation",
                        help="Run all MLP variants in sequence")

    # -----------------------------------------------------------------
    # CardioTwin
    # -----------------------------------------------------------------
    parser.add_argument(
        "--cardiotwin", action="store_true", dest="cardiotwin",
        help="Run CardioTwin final model (baseline: 128-dim, gated, BCE, no sampler)",
    )
    parser.add_argument(
        "--cardiotwin-ablation", action="store_true", dest="cardiotwin_ablation",
        help="Run full CardioTwin ablation (all variants x loss types x samplers)",
    )

    args = parser.parse_args()

    # --all = preprocess + CardioTwin final
    run_all = args.all

    # --preprocess = all four preprocessing steps
    run_preprocess = run_all or args.preprocess

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------
    if run_preprocess or args.static:
        run_static(args)
    if run_preprocess or args.ecg:
        run_ecg(args)
    if run_preprocess or args.vitals:
        run_vitals(args)
    if run_preprocess or args.entities:
        run_icd_extraction(args)

    if run_preprocess:
        print("\n" + "=" * 60)
        print("✓ ALL PREPROCESSING COMPLETED")
        print("=" * 60)

    # ---------------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------------
    if args.xgb_ablation:
        run_xgboost_all_ablation(args)
    else:
        if args.xgb_base:
            run_xgboost_base(args)
        if args.xgb_weighted:
            run_xgboost_weighted(args)
        if args.xgb_smote:
            run_xgboost_smote(args)
        if args.xgb_embedding:
            run_xgboost_embedding(args)

    # ---------------------------------------------------------------
    # MLP
    # ---------------------------------------------------------------
    if args.mlp_ablation:
        run_mlp_all_ablation(args)
    else:
        if args.mlp_baseline:
            run_mlp_baseline(args)
        if args.mlp_weighted:
            run_mlp_weighted(args)
        if args.mlp_smote:
            run_mlp_smote(args)
        if args.mlp_embedding:
            run_mlp_embedding(args)
        if args.mlp_embedding_weighted:
            run_mlp_embedding_weighted(args)

    # ---------------------------------------------------------------
    # CardioTwin
    # ---------------------------------------------------------------
    if args.cardiotwin_ablation:
        run_cardiotwin_ablation(args)
    elif run_all or args.cardiotwin:
        run_cardiotwin(args)


if __name__ == "__main__":
    main()