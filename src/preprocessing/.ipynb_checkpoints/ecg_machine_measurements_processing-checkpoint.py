# src/preprocessing/ecg_machine_measurements_processing.py

import json
from pathlib import Path
import pandas as pd
import re


# -------------------------
# Config + I/O
# -------------------------

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def load_ecg_data(in_path):
    return pd.read_csv(in_path)


# -------------------------
# Text construction
# -------------------------

def build_full_report(df, report_prefix="report_"):
    report_cols = [c for c in df.columns if c.startswith(report_prefix)]

    df["full_report"] = (
        df[report_cols]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )

    return df.drop(columns=report_cols)


def clean_report_text(df):
    df["report_clean"] = (
        df["full_report"]
        .str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


# -------------------------
# Label definitions
# -------------------------

def build_canonical_map():
    RHYTHM_LABELS = {
        "sinus_rhythm": ["sinus rhythm", "sinus rhythm normal"],
        "sinus_bradycardia": ["sinus bradycardia", "bradycardia"],
        "sinus_tachycardia": ["sinus tachycardia", "tachycardia"],
        "atrial_fibrillation": ["atrial fibrillation", "fibrillation"],
    }

    CONDUCTION_LABELS = {
        "bundle_branch_block": ["bundle branch block", "branch block"],
        "av_block": ["av block", "degree av block", "st degree av block"],
        "left_anterior_fascicular_block": [
            "left anterior fascicular block",
            "anterior fascicular block",
        ],
    }

    STRUCTURAL_LABELS = {
        "left_ventricular_hypertrophy": [
            "left ventricular hypertrophy",
            "ventricular hypertrophy",
        ],
        "axis_deviation": ["axis deviation", "left axis deviation"],
        "low_qrs_voltage": [
            "low qrs voltages",
            "low qrs",
            "low qrs voltages precordial",
        ],
    }

    REPOLARIZATION_LABELS = {
        "st_t_abnormality": [
            "stt changes",
            "stt changes nonspecific",
            "wave changes",
            "wave changes nonspecific",
            "t wave changes",
        ],
        "poor_r_wave_progression": ["poor wave progression"],
        "st_elevation_depression": ["st changes", "st degree"],
    }

    ISCHEMIA_LABELS = {
        "myocardial_ischemia": ["ischemia", "myocardial ischemia"],
        "infarct_pattern": [
            "inferior infarct",
            "anterior infarct",
            "infarct age undetermined",
        ],
    }

    ECTOPY_LABELS = {"pvcs": ["pvcs"]}

    IMPRESSION_LABELS = {
        "normal_ecg": [
            "normal ecg",
            "probable normal",
            "normal variant",
            "probable normal variant",
        ],
        "borderline_ecg": ["borderline ecg", "nonspecific borderline ecg"],
        "abnormal_ecg": [
            "abnormal ecg",
            "nonspecific abnormal ecg",
            "undetermined abnormal ecg",
        ],
    }

    PACED_RHYTHM_LABELS = {
        "atrial_paced_rhythm": ["atrial paced rhythm", "atrial paced"],
        "ventricular_paced_rhythm": ["ventricular paced rhythm", "ventricular paced"],
        "dual_paced_rhythm": [
            "dual paced rhythm",
            "a v dual paced",
            "atrial ventricular dual paced",
        ],
    }

    ADDITIONAL_RHYTHM_LABELS = {
        "sinus_arrhythmia": ["sinus arrhythmia", "slow sinus arrhythmia"],
        "ectopic_atrial_rhythm": [
            "ectopic atrial rhythm",
            "wandering atrial pacemaker",
        ],
        "junctional_rhythm": ["junctional rhythm"],
    }

    ADDITIONAL_FINDINGS = {
        "prolonged_qt": ["prolonged qt", "prolonged qt interval"],
        "intraventricular_conduction_delay": [
            "intraventricular conduction delay"
        ],
        "right_bundle_branch_block": [
            "rbbb",
            "right bundle branch block",
        ],
    }

    return {
        **RHYTHM_LABELS,
        **CONDUCTION_LABELS,
        **STRUCTURAL_LABELS,
        **REPOLARIZATION_LABELS,
        **ISCHEMIA_LABELS,
        **ECTOPY_LABELS,
        **IMPRESSION_LABELS,
        **PACED_RHYTHM_LABELS,
        **ADDITIONAL_RHYTHM_LABELS,
        **ADDITIONAL_FINDINGS,
    }


# -------------------------
# Label extraction
# -------------------------

def apply_phrase_labels(df, canonical_map):
    def contains_any(text, phrases):
        return any(p in text for p in phrases)

    for label, phrases in canonical_map.items():
        df[label] = (
            df["report_clean"]
            .apply(lambda x: contains_any(x, phrases))
            .astype(int)
        )
    return df


def apply_regex_fixes(df):
    df["atrial_paced_rhythm"] = df["full_report"].str.contains(
        r"\batrial[-\s]?paced\b", case=False, na=False
    ).astype(int)

    df["ventricular_paced_rhythm"] = df["full_report"].str.contains(
        r"\bventricular[-\s]?paced\b", case=False, na=False
    ).astype(int)

    df["dual_paced_rhythm"] = df["full_report"].str.contains(
        r"\b(dual|a[-\s]?v)[-\s]?paced\b", case=False, na=False
    ).astype(int)

    df["abnormal_r_wave_progression"] = df["full_report"].str.contains(
        r"\b(poor|abnormal)\s+r[-\s]?wave\s+progression\b",
        case=False,
        na=False,
    ).astype(int)

    return df


# -------------------------
# Pipeline runner
# -------------------------

def run_ecg_machine_measurements_preprocessing(
    in_path, config_path, out_path
):
    print("Running ECG machine measurement preprocessing...")

    config = load_config(config_path)
    df = load_ecg_data(in_path)

    df = build_full_report(df)
    df = clean_report_text(df)

    canonical_map = build_canonical_map()
    df = apply_phrase_labels(df, canonical_map)
    df = apply_regex_fixes(df)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"ECG machine measurements saved to {out_path}")
    return df
