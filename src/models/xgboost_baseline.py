import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm
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


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set default target_type if not specified
    if 'target_type' not in config:
        config['target_type'] = 'labels'
    
    return config


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
    machine_report_cols = [col for col in ed_ecg_records.columns if col.startswith('report_')]
    
    # Convert to numeric safely
    numeric_labels = ed_ecg_records[machine_report_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # Compute sums
    col_sums = numeric_labels.sum()
    
    # Drop columns whose sum == 0
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
    
    # Merge aggregated vitals with closest vitals
    vitals_combined = vitals_agg.merge(
        closest_vitals_df,
        on=['subject_id', 'ed_stay_id'],
        how='left'
    )
    
    # Merge with ECG data
    result = earliest_ecgs.merge(
        vitals_combined,
        on=['subject_id', 'ed_stay_id'],
        how='left'
    )
    
    return result


def create_model_df(ed_encounters, ecg_aggregate_vitals):
    """
    Merge ED encounters with ECG and vital signs data.
    
    Args:
        ed_encounters: Filtered ED encounters DataFrame
        ecg_aggregate_vitals: ECG data with aggregated vitals
        
    Returns:
        Combined DataFrame ready for modeling
    """
    # Ensure consistent data types for merge keys
    ecg_aggregate_vitals = ecg_aggregate_vitals.copy()
    ed_encounters = ed_encounters.copy()
    
    # Convert subject_id to int in both dataframes
    ecg_aggregate_vitals['subject_id'] = pd.to_numeric(ecg_aggregate_vitals['subject_id'], errors='coerce').astype('Int64')
    ed_encounters['subject_id'] = pd.to_numeric(ed_encounters['subject_id'], errors='coerce').astype('Int64')
    
    # Convert ed_stay_id to int in both dataframes
    ecg_aggregate_vitals['ed_stay_id'] = pd.to_numeric(ecg_aggregate_vitals['ed_stay_id'], errors='coerce').astype('Int64')
    ed_encounters['ed_stay_id'] = pd.to_numeric(ed_encounters['ed_stay_id'], errors='coerce').astype('Int64')
    
    model_df = ecg_aggregate_vitals.merge(
        ed_encounters,
        on=['subject_id', 'ed_stay_id'],
        how='inner'
    )
    
    return model_df


def prepare_model_features(model_df, ed_ecg_records, target_type='labels'):
    """
    Prepare feature matrix X and target matrix y for modeling.
    
    Args:
        model_df: Combined model DataFrame
        ed_ecg_records: ED ECG records with machine measurements
        target_type: Type of target to predict - 'labels' or 'reports'
                    - 'labels': Predict ICD diagnosis labels (label_*)
                    - 'reports': Predict ECG machine measurement reports (report_*)
        
    Returns:
        Tuple of (X, y, y_features, output_prefix) where:
            X: Feature matrix
            y: Target matrix
            y_features: List of target column names
            output_prefix: Prefix for output files
    """
    # Get machine measurement columns
    machine_cols = [col for col in ed_ecg_records.columns if col.startswith('report_')]
    
    # Get vital sign features
    vital_features = [col for col in model_df.columns if any(
        keyword in col for keyword in ['_mean', '_std', '_min', '_max', '_closest', 'vitals_time_before_ecg']
    )]
    
    # Get label columns
    label_cols = [col for col in model_df.columns if col.startswith('label_')]
    
    # Determine features and targets based on target_type
    if target_type == 'labels':
        # Predict labels using machine reports + vitals as features
        X_features = machine_cols + vital_features
        y_features = label_cols
        output_prefix = 'diagnosis'
    elif target_type == 'reports':
        # Predict machine reports using vitals + labels as features
        X_features = vital_features + label_cols
        y_features = machine_cols
        output_prefix = 'ecg_report'
    else:
        raise ValueError(f"target_type must be 'labels' or 'reports', got '{target_type}'")
    
    # Create feature and target matrices
    X = model_df[X_features].copy()
    y = model_df[y_features].copy()
    
    # Convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    return X, y, y_features, output_prefix


def create_train_test_set(model_df, X, y, test_size=0.2, random_state=42):
    """
    Create train/test split ensuring no patient appears in both sets.
    
    Args:
        model_df: Model DataFrame with subject_id
        X: Feature matrix
        y: Target matrix
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Get patient groups
    groups = model_df['subject_id'].astype(int).values
    
    # Split by patient
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    
    # Create train/test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train):
    """Train XGBoost multi-output classifier."""
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


def evaluate_and_visualize_multilabel_model(multi_xgb, X_test, y_test, y_features, output_prefix, out_path='../data/model_results/', label_group_name=None):
    """
    Evaluate multilabel classification model and create comprehensive visualizations.
    
    Args:
        multi_xgb: Trained MultiOutputClassifier with XGBoost estimators
        X_test: Test features
        y_test: Test labels DataFrame
        y_features: List/array of label column names
        output_prefix: Prefix for output files ('diagnosis' or 'ecg_report')
        out_path: Directory path to save results CSV and plots
        label_group_name: Human-readable name for label group (e.g., 'ECG Report Labels', 'Diagnosis Labels').
                         If None, defaults to 'All {N} Labels'
    
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
    
    # Set default label group name if not provided
    if label_group_name is None:
        label_group_name = f'All {len(valid_labels)} Labels'
    
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
    axes[0].set_title(f'ROC Curves for {len(valid_labels)} {label_group_name}\nMean AUC: {results_df["roc_auc"].mean():.3f}', fontsize=14)
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
    axes[1].set_title(f'Precision-Recall Curves for {len(valid_labels)} {label_group_name}\nMean PR-AUC: {results_df["pr_auc"].mean():.3f}', fontsize=14)
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
    axes[2].set_title(f'Aggregated Confusion Matrix\n(Sum Across {len(valid_labels)} {label_group_name})', fontsize=14)
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
    
    # Save the plot with appropriate filename
    plot_path = Path(out_path) / f'xgboost_baseline_{output_prefix}_evaluation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display
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
    
    # Save results to CSV with appropriate filename
    csv_path = Path(out_path) / f'xgboost_baseline_{output_prefix}_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")
    
    return results_df


def run_xgboost_baseline_pipeline(in_dir, config_path, out_path, target_type=None):
    """
    Main XGBoost baseline pipeline with progress tracking.
    
    Args:
        in_dir: Input directory path
        config_path: Path to configuration JSON file
        out_path: Output directory path
        target_type: Override for prediction target ('labels' or 'reports').
                    If None, uses value from config file (defaults to 'labels')
    
    Returns:
        results_df: DataFrame with performance metrics
    """
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Preprocessing & aggregating vitals",
        "Creating model dataframe",
        "Preparing features",
        "Creating train/test split",
        "Training XGBoost model",
        "Evaluating model"
    ]
    
    print("Running XGBoost Baseline model...")
    print()
    
    pbar = tqdm(total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
    
    try:
        pbar.set_description(f"[1/9] {steps[0]}")
        config = load_config(config_path)
        # Override config with command-line argument if provided
        if target_type is not None:
            config['target_type'] = target_type
        else:
            # Use config value if no argument provided
            target_type = config.get('target_type', 'labels')
        
        ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)
        pbar.update(1)
        
        pbar.set_description(f"[2/9] {steps[1]}")
        ed_encounters = filter_ed_encounters(clinical_encounters)
        ed_ecg_records = filter_ed_ecg_records(ecg_records)
        pbar.update(1)
        
        pbar.set_description(f"[3/9] {steps[2]}")
        earliest_ecgs = get_earliest_ecg_per_stay(ed_ecg_records)
        pbar.update(1)
        
        pbar.set_description(f"[4/9] {steps[3]}")
        preprocessed_vitals = preprocess_vitals(ed_vitals)
        ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(preprocessed_vitals, earliest_ecgs, agg_window_hours=4.0)
        pbar.update(1)
        
        pbar.set_description(f"[5/9] {steps[4]}")
        model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)
        pbar.update(1)
        
        pbar.set_description(f"[6/9] {steps[5]}")
        # Use target_type that was already set from args or config
        X, y, y_features, output_prefix = prepare_model_features(model_df, ed_ecg_records, target_type=target_type)
        pbar.update(1)
        
        pbar.set_description(f"[7/9] {steps[6]}")
        X_train, X_test, y_train, y_test = create_train_test_set(model_df, X, y)
        pbar.update(1)
        
        pbar.set_description(f"[8/9] {steps[7]}")
        multi_xgb = train_xgboost_model(X_train, y_train)
        pbar.update(1)
        
        pbar.set_description(f"[9/9] {steps[8]}")
        # Determine label group name based on target type
        if target_type == 'labels':
            label_group_name = 'Diagnosis Labels'
        elif target_type == 'reports':
            label_group_name = 'ECG Report Labels'
        else:
            label_group_name = 'Diagnosis Labels'
        
        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, output_prefix, out_path=out_path, label_group_name=label_group_name
        )
        pbar.update(1)
        
    finally:
        pbar.close()
    
    print()
    target_name = "diagnosis labels" if target_type == 'labels' else "ECG machine reports"
    print(f"✓ XGBoost baseline model complete (predicted {target_name})!")
    
    return results_df