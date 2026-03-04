import sys
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

from src.models.ecg_fm import run_pooled_ecg_extraction
from src.models.tabular_utils import (
    load_config,
    load_data_files,
    filter_ed_encounters,
    filter_ed_ecg_records,
    extract_earliest_ecg_per_stay,
    aggregate_vitals_to_ecg_time,
    create_model_df,
    prepare_model_features,
    create_train_test_set,
    scale_features,
    evaluate_and_visualize_multilabel_model,
)


logger = logging.getLogger(__name__)

# ECG-FM config is always loaded from ecg_fm_params.json (sibling of this file)
ECG_FM_CONFIG = Path(__file__).parent.parent.parent / "configs" / "ecg_fm_params.json"


# =============================================================================
# ECG-FM embedding extraction
# =============================================================================

def extract_ecg_embeddings(df, xgboost_config):
    """
    Extract 1536-dim ECG-FM embeddings (2 x 768-dim segments) for each row
    in df using run_pooled_ecg_extraction from ecg_fm_playground.

    ECG-FM model settings are always read from ecg_fm_params.json — the
    xgboost_config is only used to resolve base_records_dir for record paths.

    Returns df with emb_0 … emb_1535 columns appended.
    """
    base_path = xgboost_config["paths"]["base_records_dir"]
    paths = []
    for p in df["path"]:
        p = os.path.splitext(p)[0]
        if p.startswith("files/"):
            p = p[len("files/"):]
        paths.append(os.path.join(base_path, p))

    subject_df = df.copy().reset_index(drop=True)
    subject_df["ecg_path"] = paths

    return run_pooled_ecg_extraction(str(ECG_FM_CONFIG), subject_df)

# =============================================================================
# Feature preparation
# =============================================================================

def prepare_embedding_features(model_df):
    """
    Extend the base feature set from tabular_utils.prepare_model_features
    with ECG-FM embedding columns (emb_0 … emb_1535).

    Embedding columns are treated as continuous and included in cols_to_scale
    so they are normalized alongside ECG intervals and vitals.
    """
    X, y, y_features, cols_to_scale = prepare_model_features(model_df)

    embedding_cols = [col for col in model_df.columns if col.startswith("emb_")]

    emb_df = model_df[embedding_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = pd.concat([X.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    cols_to_scale = cols_to_scale + [c for c in embedding_cols if c in X.columns]

    return X, y, y_features, cols_to_scale


# =============================================================================
# K-Fold cross-validation loss curves
# =============================================================================

def plot_kfold_loss_curves(
    X,
    y,
    out_path,
    model_name="xgboost_embedding",
    n_splits=5,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
):
    """
    Train one XGBClassifier per label using k-fold cross-validation and plot
    the mean train/validation log-loss across folds at each boosting round.

    Produces one PNG per label (saved in out_path/model_name/) and one overall
    PNG averaging loss across all labels (saved in out_path/).
    """
    label_out = Path(out_path) / model_name
    label_out.mkdir(parents=True, exist_ok=True)

    X_arr = X.values
    cv_results = {}
    rounds = np.arange(1, n_estimators + 1)

    all_mean_train = []
    all_mean_val = []

    for label in y.columns:
        y_arr = y[label].values

        if len(np.unique(y_arr)) < 2:
            print(f"  Skipping {label}: only one class present")
            continue

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        train_loss_folds = np.zeros((n_splits, n_estimators))
        val_loss_folds = np.zeros((n_splits, n_estimators))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="logloss",
                random_state=random_state,
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                verbose=False,
            )

            evals = clf.evals_result()
            train_loss_folds[fold_idx] = evals["validation_0"]["logloss"]
            val_loss_folds[fold_idx] = evals["validation_1"]["logloss"]

        mean_train = train_loss_folds.mean(axis=0)
        mean_val = val_loss_folds.mean(axis=0)
        best_round = int(np.argmin(mean_val))

        cv_results[label] = {
            "train_loss_folds": train_loss_folds,
            "val_loss_folds": val_loss_folds,
            "mean_train_loss": mean_train,
            "mean_val_loss": mean_val,
            "best_round": best_round,
        }

        all_mean_train.append(mean_train)
        all_mean_val.append(mean_val)

        # --- Per-label plot ---
        short = label.replace("label_", "").replace("report_", "")
        fig, ax = plt.subplots(figsize=(10, 5))

        for fold_idx in range(n_splits):
            ax.plot(rounds, train_loss_folds[fold_idx], color="#2E5090", alpha=0.15, linewidth=1)
            ax.plot(rounds, val_loss_folds[fold_idx], color="#D32F2F", alpha=0.15, linewidth=1)

        ax.plot(rounds, mean_train, color="#2E5090", linewidth=2.5, label="Train loss (mean)")
        ax.plot(rounds, mean_val, color="#D32F2F", linewidth=2.5, label="Val loss (mean)")
        ax.axvline(best_round + 1, color="gray", linestyle="--", linewidth=1.5, label=f"Best round: {best_round + 1} (val={mean_val[best_round]:.4f})")

        ax.set_xlabel("Boosting Round", fontsize=12)
        ax.set_ylabel("Log Loss", fontsize=12)
        ax.set_title(f"{model_name} — {short}\n{n_splits}-Fold CV Loss Curves", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = label_out / f"{model_name}_kfold_loss_{short}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}")

    # --- Overall plot ---
    if all_mean_train:
        overall_train = np.mean(all_mean_train, axis=0)
        overall_val = np.mean(all_mean_val, axis=0)
        best_round_overall = int(np.argmin(overall_val))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, overall_train, color="#2E5090", linewidth=2.5, label="Train loss (mean across labels)")
        ax.plot(rounds, overall_val, color="#D32F2F", linewidth=2.5, label="Val loss (mean across labels)")
        ax.axvline(best_round_overall + 1, color="gray", linestyle="--", linewidth=1.5, label=f"Best round: {best_round_overall + 1} (val={overall_val[best_round_overall]:.4f})")

        ax.set_xlabel("Boosting Round", fontsize=12)
        ax.set_ylabel("Log Loss", fontsize=12)
        ax.set_title(f"{model_name} — Overall\n{n_splits}-Fold CV Loss Curves (averaged across {len(all_mean_train)} labels)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        overall_path = Path(out_path) / f"{model_name}_kfold_loss_overall.png"
        plt.savefig(overall_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {overall_path}")

    print(f"\nK-fold loss curves saved to '{out_path}'")
    return cv_results


# =============================================================================
# Training
# =============================================================================

def _train_xgboost(X_train, y_train):
    """Train one XGBClassifier per label (multi-output via MultiOutputClassifier)."""
    estimators = []
    for col in y_train.columns:
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X_train, y_train[col])
        estimators.append(clf)

    multi_xgb = MultiOutputClassifier(XGBClassifier(), n_jobs=-1)
    multi_xgb.estimators_ = estimators
    multi_xgb.n_outputs_ = len(estimators)
    return multi_xgb


# =============================================================================
# Pipeline
# =============================================================================

def run_xgboost_embedding_pipeline(in_dir, config_path, out_path):
    """
    XGBoost pipeline with ECG-FM embeddings (1536-dim) + StandardScaler normalization.
    ECG-FM config is always read from ecg_fm_params.json via run_pooled_ecg_extraction.
    """
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Extracting ECG-FM embeddings",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features (tabular + embeddings)",
        "Train/test split & scaling",
        "Training XGBoost model",
        "K-fold loss curves",
        "Evaluating model",
    ]

    print("Running XGBoost Embedding model...")
    print()

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        pbar.set_description(steps[0])
        config = load_config(config_path)
        ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)
        pbar.update(1)

        pbar.set_description(steps[1])
        ed_encounters  = filter_ed_encounters(clinical_encounters)
        ed_ecg_records = filter_ed_ecg_records(ecg_records)
        pbar.update(1)

        pbar.set_description(steps[2])
        earliest_ecgs = extract_earliest_ecg_per_stay(ed_ecg_records)
        earliest_ecgs = earliest_ecgs.reset_index(drop=True)
        pbar.update(1)

        pbar.set_description(steps[3])
        earliest_ecgs = extract_ecg_embeddings(earliest_ecgs, config)
        pbar.update(1)

        pbar.set_description(steps[4])
        ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(
            ed_vitals, earliest_ecgs, agg_window_hours=4.0
        )
        pbar.update(1)

        pbar.set_description(steps[5])
        model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)
        pbar.update(1)

        pbar.set_description(steps[6])
        X, y, y_features, cols_to_scale = prepare_embedding_features(model_df)
        pbar.update(1)

        pbar.set_description(steps[7])
        X_train, X_test, y_train, y_test = create_train_test_set(model_df, X, y)
        X_train, X_test, _ = scale_features(X_train, X_test, cols_to_scale)
        pbar.update(1)

        pbar.set_description(steps[8])
        multi_xgb = _train_xgboost(X_train, y_train)
        pbar.update(1)

        pbar.set_description(steps[9])
        plot_kfold_loss_curves(X_train, y_train, out_path=out_path)
        pbar.update(1)

        pbar.set_description(steps[10])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, "xgboost_embedding",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise

    print()
    print("✓ XGBoost embedding model complete (predicted diagnosis labels)!")
    return results_df