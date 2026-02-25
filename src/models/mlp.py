import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.preprocessing.icd_code_labels import cardiovascular_labels


# =============================================================================
# Model
# =============================================================================

class MultilabelMLP(nn.Module):
    """
    Feedforward MLP for multi-label binary classification.
    Outputs one logit per label — use with BCEWithLogitsLoss.
    Apply torch.sigmoid to logits to get probabilities.
    """

    def __init__(self, in_features: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_labels = num_labels

        self.backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =============================================================================
# Low-level training utilities
# =============================================================================

def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    if hasattr(x, "values"):
        return torch.from_numpy(x.values).to(device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
    return total_loss / n if n else 0.0


def predict_proba(model, X, device, batch_size=256):
    """Return probability matrix (n_samples, num_labels)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = _to_tensor(X[i : i + batch_size], device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.vstack(probs)


def predict(model, X, device, threshold=0.5, batch_size=256):
    """Return binary predictions (n_samples, num_labels)."""
    return (predict_proba(model, X, device, batch_size) >= threshold).astype(np.int64)


def fit_multilabel_mlp(
    X_train,
    y_train,
    dropout=0.3,
    pos_weight=None,
    lr=1e-3,
    epochs=100,
    batch_size=64,
    device=None,
    validation_data=None,
    early_stopping_patience=10,
):
    """
    Train a MultilabelMLP with BCEWithLogitsLoss and optional early stopping.

    Args:
        X_train: Training features (ndarray or DataFrame)
        y_train: Training labels (ndarray or DataFrame)
        dropout: Dropout rate
        pos_weight: Per-label positive weights for BCEWithLogitsLoss (ndarray, length = num_labels)
        lr: Learning rate
        epochs: Maximum training epochs
        batch_size: Mini-batch size
        device: torch.device — auto-detected if None
        validation_data: Optional (X_val, y_val) tuple for early stopping
        early_stopping_patience: Epochs to wait before stopping if val loss doesn't improve

    Returns:
        Trained MultilabelMLP (loaded with best weights)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_features = X.shape[1]
    num_labels = y.shape[1]

    # Build pos_weight tensor
    pw_tensor = None
    if pos_weight is not None:
        pw = np.asarray(pos_weight, dtype=np.float32)
        if pw.ndim == 1 and len(pw) == num_labels:
            pw_tensor = torch.from_numpy(pw).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    model = MultilabelMLP(
        in_features=n_features,
        num_labels=num_labels,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, loader, criterion, optimizer, device)

        if validation_data is None:
            if best_state is None or train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            continue

        X_val = np.asarray(validation_data[0], dtype=np.float32)
        y_val = np.asarray(validation_data[1], dtype=np.float32)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(y_val)
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
                n_val += X_b.size(0)
        val_loss = val_loss / n_val if n_val else float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# Data loading & feature engineering  (mirrors xgboost_baseline.py exactly)
# =============================================================================

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_data_files(in_dir, config):
    ed_vitals = pd.read_csv(os.path.join(in_dir, config["sources"]["vitals"]))
    clinical_encounters = pd.read_csv(
        os.path.join(in_dir, config["sources"]["clinical_encounters"]),
        dtype=str,
        low_memory=False,
    )
    ecg_records = pd.read_csv(os.path.join(in_dir, config["sources"]["ecg_records"]))
    return ed_vitals, clinical_encounters, ecg_records


def filter_label_columns(df, prefix: str = "label_"):
    label_cols = [col for col in df.columns if prefix in col]
    numeric_labels = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    col_sums = numeric_labels.sum()
    cols_to_drop = col_sums[col_sums == 0].index.tolist()
    return df.drop(columns=cols_to_drop)


def filter_ed_encounters(clinical_encounters):
    ed_encounters = clinical_encounters[clinical_encounters["ed_stay_id"].notna()].copy()
    return filter_label_columns(ed_encounters, prefix="label_")


def filter_ed_ecg_records(ecg_records):
    ed_ecg_records = ecg_records[ecg_records["in_ed"] == 1]
    machine_report_cols = [col for col in ed_ecg_records.columns if col.startswith("report_")]
    numeric_labels = (
        ed_ecg_records[machine_report_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    col_sums = numeric_labels.sum()
    cols_to_drop = col_sums[col_sums == 0].index.tolist()
    return ed_ecg_records.drop(columns=cols_to_drop)


def extract_earliest_ecg_per_stay(ecg_records_df):
    ecg_records_df["ecg_time"] = pd.to_datetime(ecg_records_df["ecg_time"])
    ecg_sorted = ecg_records_df.sort_values(["subject_id", "ed_stay_id", "ecg_time"])
    earliest = ecg_sorted.groupby(["subject_id", "ed_stay_id"], as_index=False).first()
    earliest["qrs_duration"] = earliest["qrs_end"] - earliest["qrs_onset"]
    earliest["pr_interval"]  = earliest["qrs_onset"] - earliest["p_onset"]
    earliest["qt_proxy"]     = earliest["t_end"] - earliest["qrs_onset"]
    return earliest


def aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs, agg_window_hours: float = 4.0):
    ed_vitals = ed_vitals.copy()
    ed_vitals["charttime"] = pd.to_datetime(ed_vitals["charttime"])
    earliest_ecgs = earliest_ecgs.copy()
    earliest_ecgs["ecg_time"] = pd.to_datetime(earliest_ecgs["ecg_time"])

    vitals_with_ecg = ed_vitals.merge(
        earliest_ecgs[["subject_id", "ed_stay_id", "ecg_time"]],
        left_on=["subject_id", "stay_id"],
        right_on=["subject_id", "ed_stay_id"],
        how="inner",
    )

    time_delta = pd.Timedelta(hours=agg_window_hours)
    vitals_before_ecg = vitals_with_ecg[
        (vitals_with_ecg["charttime"] <= vitals_with_ecg["ecg_time"])
        & (vitals_with_ecg["charttime"] >= vitals_with_ecg["ecg_time"] - time_delta)
    ].copy()

    vitals_before_ecg["time_diff"] = (
        vitals_before_ecg["ecg_time"] - vitals_before_ecg["charttime"]
    )

    closest_vitals = vitals_before_ecg.loc[
        vitals_before_ecg.groupby(["subject_id", "ed_stay_id"])["time_diff"].idxmin()
    ]

    vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

    closest_vitals_df = closest_vitals[
        ["subject_id", "ed_stay_id"] + vital_cols + ["time_diff"]
    ].copy()
    closest_vitals_df = closest_vitals_df.rename(
        columns={col: f"{col}_closest" for col in vital_cols}
    )
    closest_vitals_df = closest_vitals_df.rename(
        columns={"time_diff": "vitals_time_before_ecg"}
    )

    agg_dict = {}
    for col in vital_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"]  = (col, "std")
        agg_dict[f"{col}_min"]  = (col, "min")
        agg_dict[f"{col}_max"]  = (col, "max")

    vitals_agg = (
        vitals_before_ecg.groupby(["subject_id", "ed_stay_id"]).agg(**agg_dict).reset_index()
    )
    vitals_combined = vitals_agg.merge(
        closest_vitals_df, on=["subject_id", "ed_stay_id"], how="left"
    )
    return earliest_ecgs.merge(vitals_combined, on=["subject_id", "ed_stay_id"], how="left")


def create_model_df(ed_encounters, ecg_aggregate_vitals):
    ecg_aggregate_vitals = ecg_aggregate_vitals.copy()
    ed_encounters = ed_encounters.copy()

    ecg_aggregate_vitals["subject_id"] = pd.to_numeric(
        ecg_aggregate_vitals["subject_id"], errors="coerce"
    ).astype("Int64")
    ed_encounters["subject_id"] = pd.to_numeric(
        ed_encounters["subject_id"], errors="coerce"
    ).astype("Int64")
    ecg_aggregate_vitals["ed_stay_id"] = pd.to_numeric(
        ecg_aggregate_vitals["ed_stay_id"], errors="coerce"
    ).astype("Int64")
    ed_encounters["ed_stay_id"] = pd.to_numeric(
        ed_encounters["ed_stay_id"], errors="coerce"
    ).astype("Int64")

    return ecg_aggregate_vitals.merge(
        ed_encounters, on=["subject_id", "ed_stay_id"], how="inner"
    )


def prepare_model_features(model_df, ed_ecg_records):
    """Prepare feature matrix X and target matrix y for diagnosis label prediction."""
    model_df_encoded = pd.get_dummies(model_df, columns=["race", "gender"], drop_first=True)

    ecg_features = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end",
        "t_end", "p_axis", "qrs_axis", "t_axis", "qrs_duration", "pr_interval", "qt_proxy",
    ]
    machine_cols = [col for col in ed_ecg_records.columns if col.startswith("report_")]
    vital_features = [
        col for col in model_df_encoded.columns
        if any(kw in col for kw in ["_mean", "_std", "_min", "_max", "_closest", "vitals_time_before_ecg"])
    ]
    label_cols = [
        col for col in model_df_encoded.columns
        if col.startswith("label_")
        and any(cv in col for cv in cardiovascular_labels.keys())
    ]
    demo_features = ["anchor_age"] + [
        c for c in model_df_encoded.columns if c.startswith(("race_", "gender_"))
    ]

    X_features = machine_cols + vital_features + demo_features + ecg_features
    y_features = label_cols

    X = model_df_encoded[X_features].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    y = model_df_encoded[y_features].copy().apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    return X, y, y_features


def create_train_test_set(model_df, X, y, test_size=0.2, random_state=42):
    groups = model_df["subject_id"].astype(int).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def compute_pos_weights(y_train):
    """Compute per-label pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    weights = []
    for col in y_train.columns:
        n_pos = y_train[col].sum()
        n_neg = len(y_train) - n_pos
        weights.append(float(n_neg / n_pos) if n_pos > 0 else 1.0)
    return np.array(weights, dtype=np.float32)


def train_mlp_model(X_train, y_train, X_val=None, y_val=None):
    """Wrap fit_multilabel_mlp with pos_weight and optional validation split."""
    pos_weight = compute_pos_weights(y_train)
    validation_data = (
        (X_val.values if hasattr(X_val, "values") else X_val,
         y_val.values if hasattr(y_val, "values") else y_val)
        if X_val is not None else None
    )
    return fit_multilabel_mlp(
        X_train=X_train.values if hasattr(X_train, "values") else X_train,
        y_train=y_train.values if hasattr(y_train, "values") else y_train,
        dropout=0.3,
        pos_weight=pos_weight,
        lr=1e-3,
        epochs=100,
        batch_size=64,
        device=None,
        validation_data=validation_data,
        early_stopping_patience=10,
    )


# =============================================================================
# Evaluation & visualisation  (same charts as xgboost_baseline.py)
# =============================================================================

def evaluate_and_visualize_mlp(
    model,
    X_test,
    y_test,
    y_features,
    out_path="../data/model_results/",
    label_group_name="Diagnosis Labels",
):
    """
    Evaluate MLP and produce the same three charts as the XGBoost pipeline:
      1. ROC curves
      2. Precision-Recall curves
      3. Aggregated confusion matrix
    Plus the label co-occurrence heatmap.
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_np = np.asarray(X_test.values if hasattr(X_test, "values") else X_test, dtype=np.float32)

    proba_matrix = predict_proba(model, X_np, device)     # (n_samples, n_labels)
    pred_matrix  = (proba_matrix >= 0.5).astype(int)

    # --- Per-label metrics ---
    results = []
    for i, target in enumerate(y_features):
        n_pos = y_test[target].sum()
        if y_test[target].nunique() > 1:
            auc = roc_auc_score(y_test[target], proba_matrix[:, i])
            ap  = average_precision_score(y_test[target], proba_matrix[:, i])
        else:
            auc = ap = np.nan
        results.append({
            "target":     target,
            "n_test_pos": int(n_pos),
            "pos_rate":   y_test[target].mean(),
            "roc_auc":    auc,
            "pr_auc":     ap,
        })

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    valid_labels = [l for l in y_features if y_test[l].nunique() > 1]
    print(f"\nPlotting {len(valid_labels)} labels with valid metrics")

    if label_group_name is None:
        label_group_name = f"All {len(valid_labels)} Labels"

    # -----------------------------------------------------------------------
    # Figure 1: ROC | PR | Confusion matrix
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for label in valid_labels:
        idx = list(y_features).index(label)
        fpr, tpr, _ = roc_curve(y_test[label], proba_matrix[:, idx])
        roc_auc = results_df[results_df["target"] == label].iloc[0]["roc_auc"]
        color = "#2E5090" if roc_auc >= 0.95 else ("#6B46C1" if roc_auc >= 0.85 else "#D32F2F")
        alpha = 0.3       if roc_auc >= 0.95 else (0.4       if roc_auc >= 0.85 else 0.6)
        axes[0].plot(fpr, tpr, linewidth=1.5, alpha=alpha, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=2, label="Random (AUC=0.5)")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title(
        f"ROC Curves for {len(valid_labels)} {label_group_name}\n"
        f"Mean AUC: {results_df['roc_auc'].mean():.3f}", fontsize=14
    )
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for label in valid_labels:
        idx = list(y_features).index(label)
        prec, rec, _ = precision_recall_curve(y_test[label], proba_matrix[:, idx])
        pr_auc = results_df[results_df["target"] == label].iloc[0]["pr_auc"]
        color = "#2E5090" if pr_auc >= 0.7 else ("#6B46C1" if pr_auc >= 0.3 else "#D32F2F")
        alpha = 0.3       if pr_auc >= 0.7 else (0.4       if pr_auc >= 0.3 else 0.6)
        axes[1].plot(rec, prec, linewidth=1.5, alpha=alpha, color=color)

    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title(
        f"Precision-Recall Curves for {len(valid_labels)} {label_group_name}\n"
        f"Mean PR-AUC: {results_df['pr_auc'].mean():.3f}", fontsize=14
    )
    axes[1].grid(True, alpha=0.3)

    total_cm = np.zeros((2, 2))
    y_true_all, y_pred_all = [], []
    for label in valid_labels:
        idx = list(y_features).index(label)
        cm = confusion_matrix(y_test[label], pred_matrix[:, idx])
        total_cm += cm
        y_true_all.extend(y_test[label].values)
        y_pred_all.extend(pred_matrix[:, idx])

    sns.heatmap(
        total_cm, annot=True, fmt=".0f", cmap="Blues", ax=axes[2],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar_kws={"label": "Count"},
    )
    axes[2].set_title(
        f"Aggregated Confusion Matrix\n(Sum Across {len(valid_labels)} {label_group_name})",
        fontsize=14,
    )
    axes[2].set_ylabel("True Label", fontsize=12)
    axes[2].set_xlabel("Predicted Label", fontsize=12)

    accuracy  = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall    = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1        = f1_score(y_true_all, y_pred_all, zero_division=0)

    axes[2].text(
        1.5, -0.5,
        f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plot_path = Path(out_path) / "mlp_baseline_diagnosis_evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to '{plot_path}'")

    # -----------------------------------------------------------------------
    # Figure 2: Label co-occurrence matrix
    # -----------------------------------------------------------------------
    n_labels = len(valid_labels)
    label_cm = np.zeros((n_labels, n_labels))
    y_pred_all_labels = np.zeros((len(X_np), n_labels))
    for idx, label in enumerate(valid_labels):
        label_idx = list(y_features).index(label)
        y_pred_all_labels[:, idx] = pred_matrix[:, label_idx]

    for i, true_label in enumerate(valid_labels):
        true_label_idx = list(y_features).index(true_label)
        positive_mask = y_test[true_label].values == 1
        if positive_mask.sum() > 0:
            for j in range(n_labels):
                label_cm[i, j] = y_pred_all_labels[positive_mask, j].sum()

    fig, ax = plt.subplots(figsize=(max(12, n_labels * 0.6), max(10, n_labels * 0.5)))
    im = ax.imshow(label_cm, cmap="Greens", aspect="auto")

    shortened = [
        l.replace("label_", "").replace("report_", "") for l in valid_labels
    ]
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(shortened, rotation=90, ha="right", fontsize=9)
    ax.set_yticklabels(shortened, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prediction Count", rotation=270, labelpad=20, fontsize=11)

    if n_labels <= 30:
        for i in range(n_labels):
            for j in range(n_labels):
                count = int(label_cm[i, j])
                if count > 0:
                    text_color = "white" if label_cm[i, j] > label_cm.max() / 2 else "black"
                    ax.text(j, i, f"{count}", ha="center", va="center",
                            color=text_color, fontsize=7)

    ax.set_title(
        f"Label Co-occurrence Matrix\n{label_group_name}\n"
        "(When true label on Y-axis is positive, how often is predicted label on X-axis positive?)",
        fontsize=13, pad=20,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    label_cm_path = Path(out_path) / "mlp_baseline_diagnosis_label_confusion_matrix.png"
    plt.savefig(label_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Label co-occurrence matrix saved to '{label_cm_path}'")

    # --- Console summary ---
    tn, fp, fn, tp = total_cm.ravel()
    print(f"\nAggregated Metrics (across all {len(valid_labels)} labels):")
    print(f"  Total Predictions: {int(total_cm.sum())}")
    print(f"  True Negatives:  {int(tn)} ({tn/total_cm.sum()*100:.1f}%)")
    print(f"  False Positives: {int(fp)} ({fp/total_cm.sum()*100:.1f}%)")
    print(f"  False Negatives: {int(fn)} ({fn/total_cm.sum()*100:.1f}%)")
    print(f"  True Positives:  {int(tp)} ({tp/total_cm.sum()*100:.1f}%)")
    print(f"  Overall Accuracy:  {accuracy:.3f}")
    print(f"  Overall Precision: {precision:.3f}")
    print(f"  Overall Recall:    {recall:.3f}")
    print(f"  Overall F1-Score:  {f1:.3f}")

    csv_path = Path(out_path) / "mlp_baseline_diagnosis_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")

    return results_df


# =============================================================================
# Main pipeline
# =============================================================================

def run_mlp_baseline_pipeline(in_dir, config_path, out_path):
    """
    End-to-end MLP pipeline predicting cardiovascular diagnosis labels.
    Identical data path as run_xgboost_baseline_pipeline.
    Differences: StandardScaler on features, MLP model, BCEWithLogitsLoss.
    """
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features & scaling",
        "Creating train/test split",
        "Training MLP model",
        "Evaluating model",
    ]

    print("Running MLP Baseline model...")
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
        pbar.update(1)

        pbar.set_description(steps[3])
        ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(
            ed_vitals, earliest_ecgs, agg_window_hours=4.0
        )
        pbar.update(1)

        pbar.set_description(steps[4])
        model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)
        pbar.update(1)

        pbar.set_description(steps[5])
        X, y, y_features = prepare_model_features(model_df, ed_ecg_records)
        # StandardScaler — MLPs are sensitive to feature scale; XGBoost is not
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )
        pbar.update(1)

        pbar.set_description(steps[6])
        X_train, X_test, y_train, y_test = create_train_test_set(model_df, X_scaled, y)
        pbar.update(1)

        pbar.set_description(steps[7])
        # Carve 10 % of training data as validation set for early stopping
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_groups = model_df.loc[y_train.index, "subject_id"].astype(int).values
        tr_idx, val_idx = next(val_splitter.split(X_train, y_train, groups=train_groups))
        model = train_mlp_model(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            X_val=X_train.iloc[val_idx], y_val=y_train.iloc[val_idx],
        )
        pbar.update(1)

        pbar.set_description(steps[8])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_mlp(
            model, X_test, y_test, y_features,
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ MLP baseline model complete (predicted diagnosis labels)!")

    return results_df