"""
XGBoost multi-label classifier augmented with ECG-FM waveform embeddings.

Extracts 512-dim embeddings from raw 10-second ECG signals using the ECG-FM
foundation model (fairseq_signals), then trains a normalized XGBoost classifier
on the combined feature set (embeddings + ECG measurements + vitals + demographics).

Pipeline:
    Load data → Extract ECG embeddings → Aggregate vitals → Build model df
    → Prepare features → Normalize → Train XGBoost → Evaluate
"""

import sys
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch
import wfdb
from tqdm import tqdm
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from fairseq_signals.models import build_model_from_checkpoint


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


# =============================================================================
# ECG-FM embedding extraction
# =============================================================================

def _build_record_paths(df, base_path):
    """
    Resolve full WFDB record paths from the 'path' column in the ECG DataFrame.

    Args:
        df        : DataFrame with a 'path' column of relative record paths
        base_path : Root directory containing WFDB records

    Returns:
        List of absolute record paths (no file extension) in the same row order as df
    """
    paths = []
    for p in df["path"]:
        p = os.path.splitext(p)[0]
        if p.startswith("files/"):
            p = p[len("files/"):]
        paths.append(os.path.join(base_path, p))
    return paths


def _read_ecg_signal(path, n_channels=12, target_samples=10000):
    """
    Read a WFDB record and return a (channels, target_samples) array.
    Pads with zeros if signal is shorter than target_samples.

    Args:
        path           : WFDB record path (no extension)
        n_channels     : Expected number of ECG leads (default: 12)
        target_samples : Total samples expected for the full recording (default: 10000 = 10s @ 500Hz)

    Returns:
        sig : np.ndarray of shape (n_channels, target_samples)

    Raises:
        RuntimeError if the record cannot be read
    """
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal.T  # (channels, time)

    if sig.shape[1] < target_samples:
        pad_width = target_samples - sig.shape[1]
        sig = np.pad(sig, ((0, 0), (0, pad_width)))

    return sig[:, :target_samples]


def extract_ecg_embeddings(df, config, batch_size=16):
    """
    Extract 512-dim ECG-FM embeddings for each row in df by splitting each
    10-second recording into two 5-second segments, embedding each independently,
    and concatenating the results.

    Embeddings are appended as columns emb_0 … emb_{emb_dim-1} to a copy of df,
    aligned by positional index (not df.index) to avoid silent misalignment if
    upstream steps dropped or reordered rows.

    Failed records are filled with zeros and counted; a warning is printed at
    the end if any failures occurred.

    Args:
        df         : DataFrame with a 'path' column; must have a contiguous
                     integer positional index (call .reset_index(drop=True) first)
        config     : Config dict with:
                       config["model"]["checkpoint_path"]
                       config["paths"]["base_records_dir"]
                       config["model"].get("embedding_dim", 512)
        batch_size : Number of records to embed per GPU batch (default: 16)

    Returns:
        DataFrame with embedding columns appended (same row order as input)
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = config["model"]["checkpoint_path"]
    base_path  = config["paths"]["base_records_dir"]
    emb_dim    = config["model"].get("embedding_dim", 512)

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    record_paths = _build_record_paths(df, base_path)
    n_records    = len(record_paths)

    # Pre-allocate output array — positional alignment guaranteed
    all_embeddings = np.zeros((n_records, emb_dim), dtype=np.float32)
    n_failed = 0

    for batch_start in tqdm(range(0, n_records, batch_size), desc="Extracting ECG embeddings"):
        batch_end   = min(batch_start + batch_size, n_records)
        batch_paths = record_paths[batch_start:batch_end]

        for offset, path in enumerate(batch_paths):
            row_idx = batch_start + offset
            try:
                sig      = _read_ecg_signal(path)
                first_5s = sig[:, :5000]
                last_5s  = sig[:, 5000:10000]

                x1 = torch.tensor(first_5s).float().unsqueeze(0).to(device)
                x2 = torch.tensor(last_5s).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    f1 = model(source=x1, features_only=True)["features"].mean(dim=1)
                    f2 = model(source=x2, features_only=True)["features"].mean(dim=1)
                    emb = torch.cat([f1, f2], dim=1).cpu().numpy().flatten()

                all_embeddings[row_idx] = emb

            except Exception as e:
                logger.warning(f"Failed to embed record {path}: {e}")
                n_failed += 1
                # Row stays as zeros (pre-allocated above)

        # Free GPU memory after each batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_records} ECG records failed to embed and were filled with zeros. "
            "Check logs for details.",
            UserWarning,
        )

    # Clean up model from GPU
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df   = pd.DataFrame(all_embeddings, columns=emb_cols)  # positional index

    # Reset df index to positional before concat to guarantee alignment
    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


# =============================================================================
# Feature preparation (extends tabular_utils.prepare_model_features)
# =============================================================================

def prepare_embedding_features(model_df):
    """
    Extend the base feature set from tabular_utils.prepare_model_features
    with ECG-FM embedding columns (emb_0 … emb_N).

    Embedding columns are treated as continuous and included in cols_to_scale
    so they are normalized alongside ECG intervals and vitals.

    Args:
        model_df : Combined model DataFrame (output of create_model_df),
                   must already contain emb_* columns from extract_ecg_embeddings

    Returns:
        X             : Feature DataFrame
        y             : Label DataFrame
        y_features    : List of label column names
        cols_to_scale : List of continuous columns to normalize (ECG + vitals + embeddings)
    """
    X, y, y_features, cols_to_scale = prepare_model_features(model_df)

    embedding_cols = [col for col in model_df.columns if col.startswith("emb_")]

    if not embedding_cols:
        warnings.warn(
            "No embedding columns (emb_*) found in model_df. "
            "Did you call extract_ecg_embeddings before prepare_embedding_features?",
            UserWarning,
        )
        return X, y, y_features, cols_to_scale

    # Add embeddings to X and to the scale list
    emb_df = model_df[embedding_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X      = pd.concat([X.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
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

    Because XGBoost trains one binary classifier per label, this function
    produces one plot per label with n_splits fold curves plus a bold mean line
    for both train and validation loss.

    Args:
        X            : Feature DataFrame (post-scaling)
        y            : Label DataFrame (binary, one column per label)
        out_path     : Directory to save plots
        model_name   : Prefix for output filenames
        n_splits     : Number of CV folds (default: 5)
        n_estimators : Boosting rounds per classifier (default: 100)
        max_depth    : XGBoost max tree depth (default: 5)
        learning_rate: XGBoost learning rate (default: 0.1)
        random_state : Random seed for fold splitting (default: 42)

    Returns:
        cv_results : dict mapping label name → dict with keys:
                       "train_loss_folds"  : (n_splits, n_estimators) array
                       "val_loss_folds"    : (n_splits, n_estimators) array
                       "mean_train_loss"   : (n_estimators,) array
                       "mean_val_loss"     : (n_estimators,) array
                       "best_round"        : int, round with lowest mean val loss
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    X_arr = X.values
    cv_results = {}

    for label in y.columns:
        y_arr = y[label].values

        # Skip labels with only one class — CV would fail
        if len(np.unique(y_arr)) < 2:
            print(f"  Skipping {label}: only one class present")
            continue

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        train_loss_folds = np.zeros((n_splits, n_estimators))
        val_loss_folds   = np.zeros((n_splits, n_estimators))

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
            val_loss_folds[fold_idx]   = evals["validation_1"]["logloss"]

        mean_train = train_loss_folds.mean(axis=0)
        mean_val   = val_loss_folds.mean(axis=0)
        best_round = int(np.argmin(mean_val))

        cv_results[label] = {
            "train_loss_folds": train_loss_folds,
            "val_loss_folds":   val_loss_folds,
            "mean_train_loss":  mean_train,
            "mean_val_loss":    mean_val,
            "best_round":       best_round,
        }

        # --- Plot ---
        rounds = np.arange(1, n_estimators + 1)
        short  = label.replace("label_", "").replace("report_", "")

        fig, ax = plt.subplots(figsize=(10, 5))

        # Individual fold curves (faint)
        for fold_idx in range(n_splits):
            ax.plot(rounds, train_loss_folds[fold_idx],
                    color="#2E5090", alpha=0.15, linewidth=1)
            ax.plot(rounds, val_loss_folds[fold_idx],
                    color="#D32F2F", alpha=0.15, linewidth=1)

        # Mean curves (bold)
        ax.plot(rounds, mean_train,
                color="#2E5090", linewidth=2.5, label=f"Train loss (mean)")
        ax.plot(rounds, mean_val,
                color="#D32F2F", linewidth=2.5, label=f"Val loss (mean)")

        # Mark best round
        ax.axvline(best_round + 1, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Best round: {best_round + 1} (val={mean_val[best_round]:.4f})")

        ax.set_xlabel("Boosting Round", fontsize=12)
        ax.set_ylabel("Log Loss", fontsize=12)
        ax.set_title(
            f"{model_name} — {short}\n"
            f"{n_splits}-Fold CV Loss Curves",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = Path(out_path) / f"{model_name}_kfold_loss_{short}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}")

    print(f"\nK-fold loss curves saved to '{out_path}'")
    return cv_results



# =============================================================================
# Training
# =============================================================================

def _train_xgboost(X_train, y_train):
    """Train unweighted XGBoost multi-output classifier."""
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
    multi_xgb.n_outputs_  = len(estimators)
    return multi_xgb


# =============================================================================
# Pipeline
# =============================================================================

def run_xgboost_embedding_pipeline(in_dir, config_path, out_path):
    """
    XGBoost pipeline with ECG-FM embeddings + StandardScaler normalization.

    Steps:
        1. Load config & data
        2. Filter ED encounters & ECG records
        3. Extract earliest ECG per stay
        4. Extract ECG-FM embeddings (GPU if available)
        5. Aggregate vitals to ECG time
        6. Build model DataFrame
        7. Prepare features (tabular + embeddings)
        8. Train/test split & normalize
        9. Train XGBoost
        10. Evaluate & visualize
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
        # Reset index before embedding to guarantee positional alignment
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
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, "xgboost_embedding",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ XGBoost embedding model complete (predicted diagnosis labels)!")
    return results_df