import sys
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import torch
import wfdb
from tqdm import tqdm
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
# Helpers
# =============================================================================

def _build_record_paths(df, base_path):
    """
    Resolve full WFDB record paths from the 'path' column in the ECG DataFrame.
    """
    paths = []
    for p in df["path"]:
        p = os.path.splitext(p)[0]
        if p.startswith("files/"):
            p = p[len("files/"):]
        paths.append(os.path.join(base_path, p))
    return paths


def _read_ecg_signal(path, target_samples=10000):
    """
    Read a WFDB record and return a (channels, target_samples) float32 array.
    Pads with zeros if the signal is shorter than target_samples.
    """
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal.T.astype(np.float32)
    if sig.shape[1] < target_samples:
        sig = np.pad(sig, ((0, 0), (0, target_samples - sig.shape[1])))
    return sig[:, :target_samples]


# =============================================================================
# ECG-FM embedding extraction
# =============================================================================

def extract_ecg_embeddings(df, config, batch_size=64, io_workers=8):
    """
    Extract 512-dim ECG-FM embeddings for each row in df by splitting each
    10-second recording into two 5-second segments, embedding each in a single
    batched GPU pass, and concatenating the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = config["model"]["checkpoint_path"]
    base_path = config["paths"]["base_records_dir"]
    emb_dim = config["model"].get("embedding_dim", 512)

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    record_paths = _build_record_paths(df, base_path)
    n_records = len(record_paths)
    all_embeddings = np.zeros((n_records, emb_dim), dtype=np.float32)

    def _load_record(args):
        row_idx, path = args
        try:
            sig = _read_ecg_signal(path)
            return row_idx, sig[:, :5000], sig[:, 5000:10000]
        except Exception as e:
            logger.warning(f"Skipping record {row_idx} ({path}): {e}")
            return None

    for batch_start in tqdm(
        range(0, n_records, batch_size), desc="Extracting ECG embeddings"
    ):
        batch_end = min(batch_start + batch_size, n_records)
        batch_args = [
            (row_idx, record_paths[row_idx])
            for row_idx in range(batch_start, batch_end)
        ]

        # --- Parallel IO ---
        loaded = {}   # row_idx → (seg1, seg2)
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            futures = {pool.submit(_load_record, args): args[0] for args in batch_args}
            for future in as_completed(futures):
                row_idx = futures[future]
                result = future.result()
                if result is not None:
                    idx, seg1, seg2 = result
                    loaded[idx] = (seg1, seg2)

        if not loaded:
            continue

        # Preserve order — sort by row_idx so embedding rows match indices
        sorted_items = sorted(loaded.items())
        row_indices = [idx for idx, _ in sorted_items]
        segs1 = np.stack([s1 for _, (s1, _) in sorted_items])  # (B, C, 5000)
        segs2 = np.stack([s2 for _, (_, s2) in sorted_items])  # (B, C, 5000)

        # --- Batched GPU inference (single forward pass per segment) ---
        x1 = torch.from_numpy(segs1).to(device)
        x2 = torch.from_numpy(segs2).to(device)

        with torch.no_grad():
            f1 = model(source=x1, features_only=True)["features"].mean(dim=1)  # (B, 256)
            f2 = model(source=x2, features_only=True)["features"].mean(dim=1)  # (B, 256)
            batch_embs = torch.cat([f1, f2], dim=1).cpu().numpy()               # (B, 512)

        for i, row_idx in enumerate(row_indices):
            all_embeddings[row_idx] = batch_embs[i]

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Cleanup GPU memory before returning
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(all_embeddings, columns=emb_cols)   # positional index

    # reset_index on df guarantees positional alignment even if upstream dropped rows
    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


# =============================================================================
# Feature preparation
# =============================================================================

def prepare_embedding_features(model_df):
    """
    Extend the base feature set from tabular_utils.prepare_model_features
    with ECG-FM embedding columns (emb_0 … emb_N).

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
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)

    X_arr = X.values
    cv_results = {}

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

        rounds = np.arange(1, n_estimators + 1)
        short = label.replace("label_", "").replace("report_", "")

        fig, ax = plt.subplots(figsize=(10, 5))

        for fold_idx in range(n_splits):
            ax.plot(rounds, train_loss_folds[fold_idx], color="#2E5090", alpha=0.15, linewidth=1)
            ax.plot(rounds, val_loss_folds[fold_idx], color="#D32F2F", alpha=0.15, linewidth=1)

        ax.plot(rounds, mean_train, color="#2E5090", linewidth=2.5, label="Train loss (mean)")
        ax.plot(rounds, mean_val, color="#D32F2F", linewidth=2.5, label="Val loss (mean)")
        ax.axvline(best_round + 1, color="gray", linestyle="--", linewidth=1.5, label=f"Best round: {best_round + 1} (val {mean_val[best_round]:.4f})")

        ax.set_xlabel("Boosting Round", fontsize=12)
        ax.set_ylabel("Log Loss", fontsize=12)
        ax.set_title(f"{model_name} — {short}\n{n_splits}-Fold CV Loss Curves", fontsize=14, fontweight="bold")

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = Path(out_path) / model_name / f"{model_name}_kfold_loss_{short}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}")

    print(f"\nK-fold loss curves saved to '{plot_path}'")
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
    XGBoost pipeline with ECG-FM embeddings + StandardScaler normalization.
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