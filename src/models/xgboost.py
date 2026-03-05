import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import warnings
import sys
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

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
    compute_scale_pos_weights,
    evaluate_and_visualize_multilabel_model,
)


# =============================================================================
# Shared pipeline steps (data loading through train/test split)
# =============================================================================

def _load_and_prepare(in_dir, config_path, pbar, steps):
    """Shared data loading, filtering, feature prep, and train/test split."""
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
    ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs, agg_window_hours=4.0)
    pbar.update(1)

    pbar.set_description(steps[4])
    model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)
    pbar.update(1)

    pbar.set_description(steps[5])
    X, y, y_features, cols_to_scale = prepare_model_features(model_df)
    pbar.update(1)

    pbar.set_description(steps[6])
    X_train, X_test, y_train, y_test = create_train_test_set(model_df, X, y)
    pbar.update(1)

    return X_train, X_test, y_train, y_test, y_features, cols_to_scale


# =============================================================================
# Base (normalized, no weighting)  —  default recommended variant
# =============================================================================

def run_xgboost_base_pipeline(in_dir, config_path, out_path):
    """XGBoost pipeline with StandardScaler normalization, no class weighting."""
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Creating train/test split & scaling",
        "Training XGBoost model",
        "Evaluating model",
    ]

    print("Running XGBoost Base (normalized) model...")
    print()

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        X_train, X_test, y_train, y_test, y_features, cols_to_scale = _load_and_prepare(
            in_dir, config_path, pbar, steps
        )
        # Normalize ECG and vital features — fit on train only, apply to both
        X_train, X_test, _ = scale_features(X_train, X_test, cols_to_scale)

        pbar.set_description(steps[7])
        estimators = []
        for col in y_train.columns:
            clf = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
            )
            clf.fit(X_train, y_train[col])
            estimators.append(clf)

        multi_xgb = MultiOutputClassifier(XGBClassifier(), n_jobs=-1)
        multi_xgb.estimators_ = estimators
        multi_xgb.n_outputs_  = len(estimators)
        pbar.update(1)

        pbar.set_description(steps[8])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, "xgboost_baseline",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ XGBoost base model complete (predicted diagnosis labels)!")
    return results_df


# =============================================================================
# Weighted (per-label scale_pos_weight, no scaling)
# =============================================================================

def run_xgboost_weighted_pipeline(in_dir, config_path, out_path):
    """XGBoost pipeline with StandardScaler normalization and per-label class-weighted loss."""
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Creating train/test split & scaling",
        "Training XGBoost model (weighted)",
        "Evaluating model",
    ]

    print("Running XGBoost Weighted model...")
    print()

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        X_train, X_test, y_train, y_test, y_features, cols_to_scale = _load_and_prepare(
            in_dir, config_path, pbar, steps
        )
        # Normalize ECG and vital features — fit on train only, apply to both
        X_train, X_test, _ = scale_features(X_train, X_test, cols_to_scale)

        pbar.set_description(steps[7])
        scale_pos_weights = compute_scale_pos_weights(y_train)
        estimators = []
        for col in y_train.columns:
            clf = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", scale_pos_weight=scale_pos_weights[col],
                random_state=42,
            )
            clf.fit(X_train, y_train[col])
            estimators.append(clf)

        multi_xgb = MultiOutputClassifier(XGBClassifier(), n_jobs=-1)
        multi_xgb.estimators_ = estimators
        multi_xgb.n_outputs_  = len(estimators)
        pbar.update(1)

        pbar.set_description(steps[8])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, "xgboost_weighted",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ XGBoost weighted model complete (predicted diagnosis labels)!")
    return results_df


# =============================================================================
# Capped SMOTE helper (mirrors MLP _cap_smote)
# =============================================================================

def _cap_smote(X_train, y_train, prevalence_threshold=0.03, max_prevalence=0.15, random_state=42):
    """
    Apply SMOTE per label with a hard cap on synthetic row generation.

    For each label below `prevalence_threshold`, generates synthetic rows until
    that label reaches `max_prevalence`. This prevents over-generation on
    extremely rare labels.
    """
    import pandas as pd
    from imblearn.over_sampling import SMOTE

    n_orig = len(X_train)
    low_prev_labels = [
        col for col in y_train.columns
        if y_train[col].mean() < prevalence_threshold and y_train[col].sum() > 1
    ]

    if not low_prev_labels:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True), 0, []

    X_synthetic_all = []
    y_synthetic_all = []

    for col in low_prev_labels:
        n_pos_orig = int(y_train[col].sum())

        # Solve for how many synthetic positives reach max_prevalence:
        # max_prevalence = (n_pos_orig + n_syn) / (n_orig + n_syn)
        # => n_syn = (max_prevalence * n_orig - n_pos_orig) / (1 - max_prevalence)
        n_syn_cap = int((max_prevalence * n_orig - n_pos_orig) / (1.0 - max_prevalence))
        if n_syn_cap <= 0:
            continue  # already at or above the cap

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train[col])

        n_synthetic = len(X_res) - n_orig
        if n_synthetic <= 0:
            continue

        # Only keep up to n_syn_cap synthetic rows for this label
        n_keep = min(n_synthetic, n_syn_cap)
        X_new = pd.DataFrame(X_res[n_orig: n_orig + n_keep], columns=X_train.columns)

        # Synthetic rows: positive for this label, zero for all others
        y_new = pd.DataFrame(0, index=range(n_keep), columns=y_train.columns)
        y_new[col] = 1

        X_synthetic_all.append(X_new)
        y_synthetic_all.append(y_new)

    if X_synthetic_all:
        X_resampled = pd.concat([X_train] + X_synthetic_all, ignore_index=True)
        y_resampled = pd.concat([y_train] + y_synthetic_all, ignore_index=True)
        n_added = len(X_resampled) - n_orig
    else:
        X_resampled = X_train.reset_index(drop=True)
        y_resampled = y_train.reset_index(drop=True)
        n_added = 0

    return X_resampled, y_resampled, n_added, low_prev_labels


# =============================================================================
# SMOTE (normalized + capped oversampling on labels < 3% prevalence)
# =============================================================================

def run_xgboost_smote_pipeline(in_dir, config_path, out_path):
    """
    XGBoost + SMOTE: normalize then oversample labels with prevalence < 3%.

    Synthetic rows are capped so each smoted label reaches at most 15%
    prevalence in the combined training set, preventing over-generation on
    extremely rare labels.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Creating train/test split & scaling",
        "Applying capped SMOTE",
        "Training XGBoost model",
        "Evaluating model",
    ]

    print("Running XGBoost SMOTE model...")
    print()

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        X_train, X_test, y_train, y_test, y_features, cols_to_scale = _load_and_prepare(
            in_dir, config_path, pbar, steps
        )
        # Normalize before SMOTE so synthetic samples are interpolated in scaled space
        X_train, X_test, _ = scale_features(X_train, X_test, cols_to_scale)

        pbar.set_description(steps[7])
        X_train, y_train, n_added, smoted_labels = _cap_smote(
            X_train, y_train,
            prevalence_threshold=0.03,
            max_prevalence=0.15,
        )
        print(
            f"\n  SMOTE: {n_added} synthetic rows added across "
            f"{len(smoted_labels)} labels (capped at 15% prevalence each)"
        )
        pbar.update(1)

        pbar.set_description(steps[8])
        estimators = []
        for col in y_train.columns:
            clf = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
            )
            clf.fit(X_train, y_train[col])
            estimators.append(clf)

        multi_xgb = MultiOutputClassifier(XGBClassifier(), n_jobs=-1)
        multi_xgb.estimators_ = estimators
        multi_xgb.n_outputs_  = len(estimators)
        pbar.update(1)

        pbar.set_description(steps[9])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, "xgboost_smote",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ XGBoost SMOTE model complete (predicted diagnosis labels)!")
    return results_df