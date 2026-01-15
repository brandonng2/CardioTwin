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
# -------------------------
# COMBINED LABEL MAP
# -------------------------
report_label_map = {
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
# HELPER FUNCTIONS
# -------------------------

def get_all_labels():
    """Return list of all label names."""
    return list(report_label_map.keys())

def get_labels_by_category():
    """Return labels organized by category."""
    return {
        'rhythm': list(RHYTHM_LABELS.keys()),
        'paced': list(PACED_RHYTHM_LABELS.keys()),
        'conduction': list(CONDUCTION_LABELS.keys()),
        'structural': list(STRUCTURAL_LABELS.keys()),
        'repolarization': list(REPOLARIZATION_LABELS.keys()),
        'ischemia': list(ISCHEMIA_LABELS.keys()),
        'ectopy': list(ECTOPY_LABELS.keys()),
        'impression': list(IMPRESSION_LABELS.keys()),
        'technical': list(ADDITIONAL_FINDINGS.keys()),
    }


def get_patterns_for_label(label_name):
    """Get all search patterns for a specific label."""
    return report_label_map.get(label_name, [])