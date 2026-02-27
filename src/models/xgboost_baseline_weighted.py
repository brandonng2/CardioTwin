"""
xgboost_weighted.py
-------------------
XGBoost multi-label classifier with per-label class-weighted loss
(scale_pos_weight = n_negative / n_positive). No feature scaling.
"""

import sys
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from models.tabular_utils import (
    load_config,
    load_data_files,
    filter_ed_encounters,
    filter_ed_ecg_records,
    extract_earliest_ecg_per_stay,
    aggregate_vitals_to_ecg_time,
    create_model_df,
    prepare_model_features,
    create_train_test_set,
    compute_scale_pos_weights,
    evaluate_and_visualize_multilabel_model,
)


def train_xgboost_model(X_train, y_train):
    """Train XGBoost multi-output classifier with per-label class-weighted loss."""
    scale_pos_weights = compute_scale_pos_weights(y_train)

    estimators = []
    for col in y_train.columns:
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weights[col],
            random_state=42,
        )
        clf.fit(X_train, y_train[col])
        estimators.append(clf)

    multi_xgb = MultiOutputClassifier(XGBClassifier(), n_jobs=-1)
    multi_xgb.estimators_ = estimators
    multi_xgb.n_outputs_  = len(estimators)
    return multi_xgb


def run_xgboost_weighted_pipeline(in_dir, config_path, out_path):
    """Main XGBoost pipeline with per-label weighted loss (unscaled features)."""
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Creating train/test split",
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
        X, y, y_features, output_prefix, _ = prepare_model_features(model_df)
        pbar.update(1)

        pbar.set_description(steps[6])
        X_train, X_test, y_train, y_test = create_train_test_set(model_df, X, y)
        pbar.update(1)

        pbar.set_description(steps[7])
        multi_xgb = train_xgboost_model(X_train, y_train)
        pbar.update(1)

        pbar.set_description(steps[8])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_multilabel_model(
            multi_xgb, X_test, y_test, y_features, output_prefix,
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ XGBoost weighted model complete (predicted diagnosis labels)!")
    return results_df