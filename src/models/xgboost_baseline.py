import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from xgboost import XGBClassifier

# ============================================================================
# Configuration and Data Loading
# ============================================================================

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_data_files(in_dir, config):
    """
    Load all required data files based on configuration.
    
    Args:
        in_dir: Input directory path
        config: Configuration dictionary
        
    Returns:
        Tuple of DataFrames: (ed_vitals, clinical_encounters, ecg_records)
    """
    ed_vitals = pd.read_csv(os.path.join(in_dir, config["sources"]["vitals"]))
    
    clinical_encounters = pd.read_csv(
        os.path.join(in_dir, config["sources"]["clinical_encounters"]), dtype=str, low_memory=False
    )
    
    ecg_records = pd.read_csv(
        os.path.join(in_dir, config["sources"]["ecg_records"])
    )
    
    return ed_vitals, clinical_encounters, ecg_records

def filter_label_columns(df, prefix: str = "label_"):
    """
    Remove label columns that have zero positive cases.
    
    Args:
        df: Input DataFrame
        prefix: Prefix to identify label columns
        
    Returns:
        DataFrame with zero-sum label columns removed
    """
    label_cols = [col for col in df.columns if prefix in col]
    
    # Convert to numeric safely
    numeric_labels = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # Identify and drop columns with sum == 0
    col_sums = numeric_labels.sum()
    cols_to_drop = col_sums[col_sums == 0].index.tolist()
    
    return df.drop(columns=cols_to_drop)

def filter_ed_encounters(clinical_encounters):
    """
    Filter clinical encounters to only ED stays and clean labels.
    
    Args:
        clinical_encounters: Full clinical encounters DataFrame
        
    Returns:
        Filtered DataFrame with only ED encounters
    """
    ed_encounters = clinical_encounters[clinical_encounters['ed_stay_id'].notna()].copy()
    
    # Filter out zero-sum labels
    ed_encounters = filter_label_columns(ed_encounters, prefix="label_")
    
    return ed_encounters

def filter_ed_ecg_records(ecg_records):
    """Filter ECG records to only those taken during ED stays and drop machine report columns with no observations."""
    ed_ecg_records = ecg_records[ecg_records['in_ed'] == 1]
    machine_report_cols = ed_ecg_records.columns[26:] # CHANGE LATER FOR 'REPORT_' PREFIX

    # convert to numeric safely (non-numeric -> NaN -> 0)
    numeric_labels = ed_ecg_records[machine_report_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # compute sums
    col_sums = numeric_labels.sum()

    # drop columns whose sum == 0
    cols_to_drop = col_sums[col_sums == 0].index.tolist()

    cleaned_ecg_records = ed_ecg_records.drop(columns=cols_to_drop)

    return cleaned_ecg_records

def get_earliest_ecg_per_stay(ecg_records):
    """
    Get the earliest ECG recording per ED stay.
    
    Args:
        ecg_records: ECG records DataFrame
        
    Returns:
        DataFrame with one row per ED stay (earliest ECG)
    """    
    # Convert time columns
    ecg_records['ecg_time'] = pd.to_datetime(ecg_records['ecg_time'])
    
    # Sort by time and get first per stay
    ecg_records_sorted = ecg_records.sort_values(
        ['subject_id', 'ed_stay_id', 'ecg_time']
    )
    
    earliest_ecgs = ecg_records_sorted.groupby(
        ['subject_id', 'ed_stay_id'], 
        as_index=False
    ).first()
        
    return earliest_ecgs

def preprocess_vitals(ed_vitals):
    """
    Forward fill vital signs within each patient stay.
    
    Args:
        ed_vitals: Raw ED vitals DataFrame
        
    Returns:
        DataFrame with forward-filled vital signs
    """
    cols_to_fill = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    
    ed_vitals = ed_vitals.copy()
    ed_vitals[cols_to_fill] = ed_vitals.groupby(
        ['subject_id', 'stay_id']
    )[cols_to_fill].ffill()
    
    
    return ed_vitals

def aggregate_vitals_to_ecg_time(
    ed_vitals,
    earliest_ecgs,
    agg_window_hours: float = 4.0
):
    """
    Aggregate vital signs up to ECG time for each stay and merge with ECG data.
    
    Args:
        ed_vitals: Preprocessed vitals DataFrame
        earliest_ecgs: Earliest ECG per stay DataFrame
        agg_window_hours: Time window before ECG to aggregate vitals
        
    Returns:
        DataFrame with all ECG data merged with aggregated vital statistics 
        and closest vitals
    """
    
    # Convert time columns
    ed_vitals = ed_vitals.copy()
    ed_vitals['charttime'] = pd.to_datetime(ed_vitals['charttime'])
    earliest_ecgs = earliest_ecgs.copy()
    earliest_ecgs['ecg_time'] = pd.to_datetime(earliest_ecgs['ecg_time'])
    
    # Merge to get ECG time for each vital
    vitals_with_ecg = ed_vitals.merge(
        earliest_ecgs[['subject_id', 'ed_stay_id', 'ecg_time']],
        left_on=['subject_id', 'stay_id'],
        right_on=['subject_id', 'ed_stay_id'],
        how='inner'
    )
    
    # Filter vitals within time window before ECG
    time_delta = pd.Timedelta(hours=agg_window_hours)
    vitals_before_ecg = vitals_with_ecg[
        (vitals_with_ecg['charttime'] <= vitals_with_ecg['ecg_time']) &
        (vitals_with_ecg['charttime'] >= vitals_with_ecg['ecg_time'] - time_delta)
    ].copy()
    
    # Get closest vitals to ECG time
    vitals_before_ecg['time_diff'] = (
        vitals_before_ecg['ecg_time'] - vitals_before_ecg['charttime']
    )
    
    closest_vitals = vitals_before_ecg.loc[
        vitals_before_ecg.groupby(['subject_id', 'ed_stay_id'])['time_diff'].idxmin()
    ]
    
    vital_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    
    # Prepare closest vitals with time_diff
    closest_vitals_df = closest_vitals[['subject_id', 'ed_stay_id'] + vital_cols + ['time_diff']].copy()
    closest_vitals_df = closest_vitals_df.rename(
        columns={col: f'{col}_closest' for col in vital_cols}
    )
    closest_vitals_df = closest_vitals_df.rename(
        columns={'time_diff': 'vitals_time_before_ecg'}
    )
    
    # Aggregate vital signs
    agg_dict = {}
    for col in vital_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')
        agg_dict[f'{col}_min'] = (col, 'min')
        agg_dict[f'{col}_max'] = (col, 'max')
    
    vitals_agg = vitals_before_ecg.groupby(
        ['subject_id', 'ed_stay_id']
    ).agg(**agg_dict).reset_index()
    
    # Fill NaN std with 0 (happens when only one measurement)
    std_cols = [col for col in vitals_agg.columns if '_std' in col]
    vitals_agg[std_cols] = vitals_agg[std_cols].fillna(0)
    
    # Merge aggregated vitals with closest vitals
    vitals_combined = vitals_agg.merge(
        closest_vitals_df,
        on=['subject_id', 'ed_stay_id'],
        how='left'
    )
    
    # Merge everything back to earliest_ecgs (preserves all ECGs)
    result = earliest_ecgs.merge(
        vitals_combined,
        on=['subject_id', 'ed_stay_id'],
        how='inner'
    )
    
    return result

def create_model_df(ecg_aggregate_vitals, ed_encounters):
    id_columns = ['subject_id', 'ed_stay_id', 'hadm_id']
    
    ecg_aggregate_vitals = ecg_aggregate_vitals.copy()
    ed_encounters = ed_encounters.copy()
        
    for col in id_columns:
        ecg_aggregate_vitals[col] = pd.to_numeric(
        ecg_aggregate_vitals[col], 
        errors='coerce'
        )
        ed_encounters[col] = pd.to_numeric(
        ed_encounters[col], 
        errors='coerce'
        )

    return ed_encounters.merge(ecg_aggregate_vitals, on=['subject_id', 'ed_stay_id', 'hadm_id'], how='inner')


# ============================================================================
# Modeling
# ============================================================================

def prepare_model_features(df, ed_ecg_records):
    """
    Select features and labels from merged ECG-vitals-encounters dataset.
    
    Args:
        df: Merged DataFrame with ECG, vitals, and encounter data
        ed_ecg_records: ECG records DataFrame containing machine report features
    
    Returns:
        X: Feature DataFrame (cleaned and encoded)
        y: Label DataFrame
        y_features: List of label column names
    """
    
    # Define feature groups
    ecg_feature_cols = [
        'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end',
        'p_axis', 'qrs_axis', 't_axis'
    ]
    
    vitals_feature_cols = [
        'temperature_mean', 'temperature_std', 'temperature_min', 'temperature_max',
        'heartrate_mean', 'heartrate_std', 'heartrate_min', 'heartrate_max',
        'resprate_mean', 'resprate_std', 'resprate_min', 'resprate_max',
        'o2sat_mean', 'o2sat_std', 'o2sat_min', 'o2sat_max',
        'sbp_mean', 'sbp_std', 'sbp_min', 'sbp_max',
        'dbp_mean', 'dbp_std', 'dbp_min', 'dbp_max',
        'temperature_closest', 'heartrate_closest', 'resprate_closest',
        'o2sat_closest', 'sbp_closest', 'dbp_closest'
    ]
    
    # Machine report features (ECG diagnoses from automated interpretation)
    machine_report_features = ed_ecg_records.columns[26:].tolist()
    
    demo_numeric = ['anchor_age']
    demo_categorical = ['gender', 'race']
    unique_identifiers = ['subject_id', 'ed_stay_id']
    
    # Get manual label features
    y_label_features = [col for col in df.columns if 'label_' in col]

    # Select features and labels
    feature_cols = (
        ecg_feature_cols + 
        vitals_feature_cols + 
        demo_numeric + 
        demo_categorical + 
        unique_identifiers +
        machine_report_features
    )
    y_features = y_label_features
    
    X = df[feature_cols].copy()
    y = df[y_features].fillna(0).astype(int)
    
    # Clean and encode features
    all_numeric = ecg_feature_cols + vitals_feature_cols + machine_report_features + demo_numeric
    for col in all_numeric:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=demo_categorical, drop_first=False)
    
    return X, y, y_features


def create_train_test_set(df, X, y):

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(X, y, groups=df['subject_id'])
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train):
    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    
    multi_xgb = MultiOutputClassifier(xgb_clf, n_jobs=-1)
    multi_xgb.fit(X_train, y_train)
    
    return multi_xgb


def evaluate_and_visualize_multilabel_model(multi_xgb, X_test, y_test, y_features, out_path='../data/model_results/'):
    """
    Evaluate multilabel classification model and create comprehensive visualizations.
    
    Args:
        multi_xgb: Trained MultiOutputClassifier with XGBoost estimators
        X_test: Test features
        y_test: Test labels DataFrame
        y_features: List/array of label column names
        out_path: Directory path to save results CSV and plots (default: 'data/model_results/')
    
    Returns:
        results_df: DataFrame with performance metrics for each label
    """
    
    # Create output directory if it doesn't exist
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, target in enumerate(y_features):
        # Get predictions for this specific label
        y_pred_proba = multi_xgb.estimators_[i].predict_proba(X_test)[:, 1]
        
        # Calculate metrics (only if both classes present in test set)
        n_pos_test = y_test[target].sum()
        
        if y_test[target].nunique() > 1:  # Both classes present
            auc = roc_auc_score(y_test[target], y_pred_proba)
            ap = average_precision_score(y_test[target], y_pred_proba)
        else:
            auc = np.nan
            ap = np.nan
        
        results.append({
            'target': target,
            'n_test_pos': int(n_pos_test),
            'pos_rate': y_test[target].mean(),
            'roc_auc': auc,
            'pr_auc': ap
        })
    
    # Convert to DataFrame and sort by PR-AUC
    results_df = pd.DataFrame(results).sort_values('pr_auc', ascending=False)
    
    # Get all valid labels (those with both classes in test set)
    valid_labels = [label for label in y_features if y_test[label].nunique() > 1]
    print(f"\nPlotting {len(valid_labels)} labels with valid metrics")
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Plot 1: All ROC Curves ---
    for label in valid_labels:
        label_idx = list(y_features).index(label)
        y_true = y_test[label]
        y_pred_proba = multi_xgb.estimators_[label_idx].predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        label_metrics = results_df[results_df['target'] == label].iloc[0]
        roc_auc = label_metrics['roc_auc']
        
        # Use different colors for different performance levels
        if roc_auc >= 0.95:
            color = '#2E5090'  # Dark blue for excellent
            alpha = 0.3
        elif roc_auc >= 0.85:
            color = '#6B46C1'  # Purple for good
            alpha = 0.4
        else:
            color = '#D32F2F'  # Red for poor
            alpha = 0.6
        
        axes[0].plot(fpr, tpr, linewidth=1.5, alpha=alpha, color=color)
    
    # Add random baseline and formatting
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title(f'ROC Curves for All {len(valid_labels)} Labels\nMean AUC: {results_df["roc_auc"].mean():.3f}', fontsize=14)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # --- Plot 2: All Precision-Recall Curves ---
    for label in valid_labels:
        label_idx = list(y_features).index(label)
        y_true = y_test[label]
        y_pred_proba = multi_xgb.estimators_[label_idx].predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        label_metrics = results_df[results_df['target'] == label].iloc[0]
        pr_auc = label_metrics['pr_auc']
        
        # Use different colors for different performance levels
        if pr_auc >= 0.7:
            color = '#2E5090'  # Dark blue for excellent
            alpha = 0.3
        elif pr_auc >= 0.3:
            color = '#6B46C1'  # Purple for moderate
            alpha = 0.4
        else:
            color = '#D32F2F'  # Red for poor
            alpha = 0.6
        
        axes[1].plot(recall, precision, linewidth=1.5, alpha=alpha, color=color)
    
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title(f'Precision-Recall Curves for All {len(valid_labels)} Labels\nMean PR-AUC: {results_df["pr_auc"].mean():.3f}', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # --- Plot 3: Aggregated Confusion Matrix (sum of all) ---
    total_cm = np.zeros((2, 2))
    y_true_all = []
    y_pred_all = []
    
    for label in valid_labels:
        label_idx = list(y_features).index(label)
        y_true = y_test[label]
        y_pred = multi_xgb.estimators_[label_idx].predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
        total_cm += cm
        
        # Collect all predictions for sklearn metrics
        y_true_all.extend(y_true.values)
        y_pred_all.extend(y_pred)
    
    # Plot aggregated confusion matrix
    sns.heatmap(total_cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[2],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    axes[2].set_title(f'Aggregated Confusion Matrix\n(Sum Across All {len(valid_labels)} Labels)', fontsize=14)
    axes[2].set_ylabel('True Label', fontsize=12)
    axes[2].set_xlabel('Predicted Label', fontsize=12)
    
    # Calculate overall metrics using sklearn
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    
    # Add text with overall metrics
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    axes[2].text(1.5, -0.5, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(out_path) / 'xgboost_baseline_evaluation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to '{plot_path}'")
    
    # Also calculate from confusion matrix for verification
    tn, fp, fn, tp = total_cm.ravel()
    
    print(f"\nAggregated Metrics (across all {len(valid_labels)} labels):")
    print(f"  Total Predictions: {int(total_cm.sum())}")
    print(f"  True Negatives:  {int(tn)} ({tn/total_cm.sum()*100:.1f}%)")
    print(f"  False Positives: {int(fp)} ({fp/total_cm.sum()*100:.1f}%)")
    print(f"  False Negatives: {int(fn)} ({fn/total_cm.sum()*100:.1f}%)")
    print(f"  True Positives:  {int(tp)} ({tp/total_cm.sum()*100:.1f}%)")
    print(f"  Overall Accuracy: {accuracy:.3f}")
    print(f"  Overall Precision: {precision:.3f}")
    print(f"  Overall Recall: {recall:.3f}")
    print(f"  Overall F1-Score: {f1:.3f}")

    # Save results to CSV
    csv_path = Path(out_path) / 'xgboost_baseline_diagnosis_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")
    
    return results_df

def run_xgboost_baseline_pipeline(in_dir, config_path, out_path):
    print("Running XGBoost Baseline model...")
    
    config = load_config(config_path)

    ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)

    ed_encounters = filter_ed_encounters(clinical_encounters)

    ed_ecg_records = filter_ed_ecg_records(ecg_records)

    earliest_ecgs = get_earliest_ecg_per_stay(ed_ecg_records)
    preprocessed_vitals = preprocess_vitals(ed_vitals)

    ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(preprocessed_vitals, earliest_ecgs, agg_window_hours=4.0)

    model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)

    X, y, y_features = prepare_model_features(model_df, ed_ecg_records)

    X_train, X_test, y_train, y_test = create_train_test_set(model_df, X, y)

    multi_xgb = train_xgboost_model(X_train, y_train)

    results_df = evaluate_and_visualize_multilabel_model(
        multi_xgb, X_test, y_test, y_features, out_path=out_path
    )
