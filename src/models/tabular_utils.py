import json
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
# Config & data loading
# =============================================================================

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_data_files(in_dir, config):
    """Load all required data files based on configuration."""
    ed_vitals = pd.read_csv(os.path.join(in_dir, config["sources"]["vitals"]))
    clinical_encounters = pd.read_csv(
        os.path.join(in_dir, config["sources"]["clinical_encounters"]),
        dtype=str,
        low_memory=False,
    )
    ecg_records = pd.read_csv(os.path.join(in_dir, config["sources"]["ecg_records"]))
    return ed_vitals, clinical_encounters, ecg_records


# =============================================================================
# Filtering
# =============================================================================

def filter_label_columns(df, prefix: str = "label_"):
    """Remove label columns that have zero positive cases."""
    label_cols = [col for col in df.columns if prefix in col]
    numeric_labels = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    col_sums = numeric_labels.sum()
    cols_to_drop = col_sums[col_sums == 0].index.tolist()
    return df.drop(columns=cols_to_drop)


def filter_ed_encounters(clinical_encounters):
    """Filter clinical encounters to only ED stays and clean labels."""
    ed_encounters = clinical_encounters[clinical_encounters["ed_stay_id"].notna()].copy()
    return filter_label_columns(ed_encounters, prefix="label_")


def filter_ed_ecg_records(ecg_records):
    """Filter ECG records to ED stays only; drop report columns with no observations."""
    ed_ecg_records = ecg_records[ecg_records["in_ed"] == 1]
    machine_report_cols = [col for col in ed_ecg_records.columns if col.startswith("report_")]
    numeric_labels = (
        ed_ecg_records[machine_report_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    col_sums = numeric_labels.sum()
    cols_to_drop = col_sums[col_sums == 0].index.tolist()
    return ed_ecg_records.drop(columns=cols_to_drop)


# =============================================================================
# ECG & vitals feature engineering
# =============================================================================

def extract_earliest_ecg_per_stay(ecg_records_df):
    """
    Get the earliest ECG recording per ED stay with derived intervals.
    """
    ecg_records_df["ecg_time"] = pd.to_datetime(ecg_records_df["ecg_time"])
    ecg_sorted = ecg_records_df.sort_values(["subject_id", "ed_stay_id", "ecg_time"])
    earliest = ecg_sorted.groupby(["subject_id", "ed_stay_id"], as_index=False).first()

    earliest["qrs_duration"] = earliest["qrs_end"] - earliest["qrs_onset"]
    earliest["pr_interval"]  = earliest["qrs_onset"] - earliest["p_onset"]
    earliest["qt_proxy"]     = earliest["t_end"] - earliest["qrs_onset"]
    return earliest


def aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs, agg_window_hours: float = 4.0):
    """
    Aggregate vital signs in a window before each stay's first ECG and merge.
    """
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


# =============================================================================
# Model dataframe construction
# =============================================================================

def create_model_df(ed_encounters, ecg_aggregate_vitals):
    """Merge ED encounters with ECG and vital signs data."""
    ecg_aggregate_vitals = ecg_aggregate_vitals.copy()
    ed_encounters = ed_encounters.copy()

    merge_keys = ["subject_id", "ed_stay_id", "hadm_id", "icu_stay_id"]
    for key in merge_keys:
        ecg_aggregate_vitals[key] = pd.to_numeric(ecg_aggregate_vitals[key], errors="coerce").astype("Int64")
        ed_encounters[key]         = pd.to_numeric(ed_encounters[key], errors="coerce").astype("Int64")

    return ecg_aggregate_vitals.merge(ed_encounters, on=merge_keys, how="inner")


def onehot_labels(df, label_column="labels", prefix="label_"):
    """One-hot encode a column containing lists (or string representations of lists)."""
    df[label_column] = df[label_column].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    all_labels = sorted({
        label
        for labels_list in df[label_column]
        if isinstance(labels_list, list)
        for label in labels_list
    })
    onehot_df = pd.DataFrame(
        {
            f"{prefix}{label}": df[label_column].apply(
                lambda x: int(isinstance(x, list) and label in x)
            )
            for label in all_labels
        },
        index=df.index,
    )
    return pd.concat([df, onehot_df], axis=1)


def prepare_model_features(model_df):
    """
    Prepare feature matrix X, target matrix y, and metadata for modeling.
    """
    model_df = onehot_labels(model_df, label_column="full_report", prefix="report_")
    model_df = onehot_labels(model_df, label_column="diagnosis_labels", prefix="label_")
    model_df_encoded = pd.get_dummies(model_df, columns=["race", "gender"], drop_first=True)

    ecg_features = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end",
        "t_end", "p_axis", "qrs_axis", "t_axis", "qrs_duration", "pr_interval", "qt_proxy",
    ]
    machine_cols = [col for col in model_df_encoded.columns if col.startswith("report_")]
    vital_features = [
        col for col in model_df_encoded.columns
        if any(kw in col for kw in ["_mean", "_std", "_min", "_max", "_closest", "vitals_time_before_ecg"])
    ]
    label_cols = [
        col for col in model_df_encoded.columns
        if col.startswith("label_") and any(cv in col for cv in cardiovascular_labels.keys())
    ]
    demo_features = ["anchor_age"] + [
        c for c in model_df_encoded.columns if c.startswith(("race_", "gender_"))
    ]

    X_features = machine_cols + vital_features + demo_features + ecg_features
    X = model_df_encoded[X_features].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    y = model_df_encoded[label_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Continuous columns that benefit from normalization
    cols_to_scale = [c for c in (ecg_features + vital_features) if c in X.columns]

    return X, y, label_cols, cols_to_scale


# =============================================================================
# Train / test split
# =============================================================================

def create_train_test_set(model_df, X, y, test_size=0.2, random_state=42):
    """
    Patient-aware train/test split — no patient appears in both sets.
    """
    groups = model_df["subject_id"].astype(int).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# =============================================================================
# Feature scaling (used by normalized variant)
# =============================================================================

def scale_features(X_train, X_test, cols_to_scale):
    """
    Fit StandardScaler on training data only and apply to both splits.
    Only scales continuous ECG and vital sign columns; binary/categorical
    columns (report_, race_, gender_, anchor_age) are left untouched.
    """
    cols = [c for c in cols_to_scale if c in X_train.columns]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[cols] = scaler.fit_transform(X_train[cols])
    X_test_scaled[cols]  = scaler.transform(X_test[cols])

    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# SMOTE resampling (used by SMOTE variant)
# =============================================================================

def smote_resample_low_prevalence(X_train, y_train, prevalence_threshold=0.08, random_state=42):
    """
    Apply SMOTE independently per label to training data for labels whose
    positive prevalence falls below `prevalence_threshold`.

    Because SMOTE requires a single binary target, we iterate over each
    low-prevalence label, resample X/y for that label, then stitch the
    synthetic rows back into the full training set.
    """
    low_prev_labels = [
        col for col in y_train.columns
        if y_train[col].mean() < prevalence_threshold and y_train[col].sum() > 1
    ]

    if not low_prev_labels:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    X_synthetic_all = []
    y_synthetic_all = []

    for col in low_prev_labels:
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train[col])

        # Only keep the newly generated synthetic rows (appended at the end by SMOTE)
        n_synthetic = len(X_res) - len(X_train)
        if n_synthetic <= 0:
            continue

        X_new = pd.DataFrame(X_res[-n_synthetic:], columns=X_train.columns)

        # For synthetic rows: set this label to 1, all other labels to 0
        y_new = pd.DataFrame(0, index=range(n_synthetic), columns=y_train.columns)
        y_new[col] = 1

        X_synthetic_all.append(X_new)
        y_synthetic_all.append(y_new)

    if X_synthetic_all:
        X_resampled = pd.concat([X_train] + X_synthetic_all, ignore_index=True)
        y_resampled = pd.concat([y_train] + y_synthetic_all, ignore_index=True)
        n_added = len(X_resampled) - len(X_train)
    else:
        X_resampled = X_train.reset_index(drop=True)
        y_resampled = y_train.reset_index(drop=True)

    return X_resampled, y_resampled, n_added, low_prev_labels


# =============================================================================
# Weighted loss helper (used by weighted & normalized variants)
# =============================================================================

def compute_scale_pos_weights(y_train):
    """
    Compute per-label scale_pos_weight = n_negative / n_positive for XGBoost.
    Falls back to 1.0 for labels with no positive examples.
    """
    weights = {}
    for col in y_train.columns:
        n_pos = y_train[col].sum()
        n_neg = len(y_train) - n_pos
        weights[col] = n_neg / n_pos if n_pos > 0 else 1.0
    return weights


# =============================================================================
# Evaluation & visualisation
# =============================================================================

def evaluate_and_visualize_multilabel_model(
    multi_xgb,
    X_test,
    y_test,
    y_features,
    model_name,
    out_path="../data/model_results/",
    label_group_name=None,
):
    """
    Evaluate a trained MultiOutputClassifier and produce:
      - ROC + PR + aggregated confusion matrix plot
      - Label co-occurrence heatmap
      - Results CSV
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)

    results = []
    for i, target in enumerate(y_features):
        y_pred_proba = multi_xgb.estimators_[i].predict_proba(X_test)[:, 1]
        n_pos_test   = y_test[target].sum()

        if y_test[target].nunique() > 1:
            auc = roc_auc_score(y_test[target], y_pred_proba)
            ap  = average_precision_score(y_test[target], y_pred_proba)
        else:
            auc = np.nan
            ap  = np.nan

        results.append({
            "target":    target,
            "n_test_pos": int(n_pos_test),
            "pos_rate":  y_test[target].mean(),
            "roc_auc":   auc,
            "pr_auc":    ap,
        })

    results_df   = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    valid_labels = [l for l in y_features if y_test[l].nunique() > 1]
    print(f"\nPlotting {len(valid_labels)} labels with valid metrics")

    if label_group_name is None:
        label_group_name = f"All {len(valid_labels)} Labels"

    # --- Figure 1: ROC / PR / Confusion matrix ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for label in valid_labels:
        label_idx    = list(y_features).index(label)
        y_pred_proba = multi_xgb.estimators_[label_idx].predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test[label], y_pred_proba)
        roc_auc      = results_df[results_df["target"] == label].iloc[0]["roc_auc"]
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
        label_idx    = list(y_features).index(label)
        y_pred_proba = multi_xgb.estimators_[label_idx].predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test[label], y_pred_proba)
        pr_auc       = results_df[results_df["target"] == label].iloc[0]["pr_auc"]
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

    total_cm   = np.zeros((2, 2))
    y_true_all = []
    y_pred_all = []
    for label in valid_labels:
        label_idx = list(y_features).index(label)
        y_pred    = multi_xgb.estimators_[label_idx].predict(X_test)
        total_cm += confusion_matrix(y_test[label], y_pred)
        y_true_all.extend(y_test[label].values)
        y_pred_all.extend(y_pred)

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

    fig.suptitle(f"{model_name} Results", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = Path(out_path) / f"{model_name}_evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to '{plot_path}'")

    # --- Figure 2: Label co-occurrence matrix ---
    n_labels          = len(valid_labels)
    label_co_matrix   = np.zeros((n_labels, n_labels))
    y_pred_all_labels = np.zeros((len(X_test), n_labels))

    for idx, label in enumerate(valid_labels):
        label_idx = list(y_features).index(label)
        y_pred_all_labels[:, idx] = multi_xgb.estimators_[label_idx].predict(X_test)

    for i, true_label in enumerate(valid_labels):
        positive_mask = y_test[true_label].values == 1
        if positive_mask.sum() > 0:
            for j in range(n_labels):
                label_co_matrix[i, j] = y_pred_all_labels[positive_mask, j].sum()

    fig, ax = plt.subplots(figsize=(max(12, n_labels * 0.6), max(10, n_labels * 0.5)))
    im = ax.imshow(label_co_matrix, cmap="Greens", aspect="auto")

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
                count = int(label_co_matrix[i, j])
                if count > 0:
                    text_color = "white" if label_co_matrix[i, j] > label_co_matrix.max() / 2 else "black"
                    ax.text(j, i, f"{count}", ha="center", va="center", color=text_color, fontsize=7)

    ax.set_title(
        f"Label Co-occurrence Matrix\n{label_group_name}\n"
        "(When true label on Y-axis is positive, how often is predicted label on X-axis positive?)",
        fontsize=13, pad=20,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.suptitle(f"{model_name} Results", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    label_cm_path = Path(out_path) / f"{model_name}_label_confusion_matrix.png"
    plt.savefig(label_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Label co-occurrence matrix saved to '{label_cm_path}'")

    tn, fp, fn, tp = total_cm.ravel()
    print(f"\nAggregated Metrics (across all {len(valid_labels)} labels):")
    print(f"  Total Predictions: {int(total_cm.sum())}")
    print(f"  True Negatives:  {int(tn)} ({tn / total_cm.sum() * 100:.1f}%)")
    print(f"  False Positives: {int(fp)} ({fp / total_cm.sum() * 100:.1f}%)")
    print(f"  False Negatives: {int(fn)} ({fn / total_cm.sum() * 100:.1f}%)")
    print(f"  True Positives:  {int(tp)} ({tp / total_cm.sum() * 100:.1f}%)")
    print(f"  Overall Accuracy:  {accuracy:.3f}")
    print(f"  Overall Precision: {precision:.3f}")
    print(f"  Overall Recall:    {recall:.3f}")
    print(f"  Overall F1-Score:  {f1:.3f}")

    csv_path = Path(out_path) / f"{model_name}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")

    return results_df