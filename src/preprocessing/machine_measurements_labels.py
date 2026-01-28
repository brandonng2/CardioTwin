RHYTHM_LABELS = {
    "sinus_rhythm": ["sinus rhythm", "sinus rhythm normal"],
    "sinus_bradycardia": ["sinus bradycardia", "bradycardia"],
    "sinus_tachycardia": ["sinus tachycardia", "tachycardia"],
    "atrial_fibrillation": ["atrial fibrillation", "fibrillation", "aflutterfibrillation"],
    "atrial_flutter": ["atrial flutter", "aflutter"],
    "tachyarrhythmia": [  # CONSOLIDATED: SVT + VT
        "supraventricular tachycardia", "svt",
        "ventricular tachycardia", "wide complex tachycardia", "wideqrs tachycardia", "wide qrs tachycardia",
        "idioventricular rhythm", "accelerated idioventricular"
    ],
    "other_rhythm": [  # CONSOLIDATED: rare rhythms
        "ectopic atrial rhythm", "wandering atrial pacemaker",
        "junctional rhythm", "accelerated junctional rhythm"
    ],
}

CONDUCTION_LABELS = {
    "right_bundle_branch_block": ["rbbb", "right bundle branch block"],
    "left_bundle_branch_block": ["lbbb", "left bundle branch block", "ilbbb", "incomplete left bundle branch block"],
    "conduction_abnormality": [  # CONSOLIDATED: rare conduction defects
        "av block", "degree av block", "st degree av block", "nd degree av block", "rd degree av block", 
        "complete av block", "wenckebach", "mobitz",
        "bundle branch block", "branch block",
        "left posterior fascicular block", "posterior fascicular block", "lpfb",
        "intraventricular conduction delay", "ivcd", "intermittent conduction defect",
        "aberrant conduction", "aberrantly conducted", "aberrant complex", "aberrant ventricular conduction"
    ],
    "left_anterior_fascicular_block": ["left anterior fascicular block", "anterior fascicular block", "lafb"],
}

STRUCTURAL_LABELS = {
    "left_ventricular_hypertrophy": [
        "left ventricular hypertrophy",
        "ventricular hypertrophy",
        "lvh",
        "voltage criteria for left ventricular hypertrophy",
    ],
    "right_ventricular_hypertrophy": ["right ventricular hypertrophy", "rvh"],
    "atrial_abnormality": [  # CONSOLIDATED: all atrial enlargements
        "biatrial enlargement", "biatrial abnormality",
        "left atrial enlargement", "left atrial abnormality", "lae", "abnormal p terminal force",
        "right atrial enlargement", "right atrial abnormality", "rae",
        "tall inferior p waves", "abnormally tall p", "abnormally high p amplitudes",
        "abnormal p waves", "p wave abnormality"
    ],
    "axis_deviation": [
        "axis deviation",
        "left axis deviation",
        "right axis deviation",
        "extreme axis deviation",
        "abnormal extreme qrs axis deviation",
    ],
    "qrs_voltage_abnormality": [  # CONSOLIDATED: high + low voltage
        "low qrs voltages", "low qrs", "low qrs voltages precordial", "low voltage",
        "abnormally high qrs voltages", "high qrs voltage", "abnormally high sv rv"
    ],
}

REPOLARIZATION_LABELS = {
    "st_segment_abnormality": [  # CONSOLIDATED: ST elevation + depression + nonspecific ST-T
        "st elevation", "st elev", "early repolarization", "early repol",
        "st depression", "st depr", "junctional depression",
        "stt changes", "stt changes nonspecific", "wave changes", "wave changes nonspecific",
        "t wave changes", "st changes", "st t abnormality"
    ],
    "t_wave_abnormality": [  # CONSOLIDATED: T wave inversion + poor R wave progression
        "t wave inversion", "inverted t",
        "poor wave progression", "poor r wave progression", "anterior r decrease", "abnormal rwave progression"
    ],
    "qt_abnormality": [  # CONSOLIDATED: QT prolongation variants
        "prolonged qt", "prolonged qt interval", "qt prolongation"
    ],
}

ISCHEMIA_LABELS = {
    "acute_ischemia": [  # CONSOLIDATED: ischemia + injury (acute pathology)
        "ischemia", "myocardial ischemia", "consider ischemia", "probable ischemia", "suggest ischemia",
        "abnormal t consider ischemia", "abnormal t probable ischemia", "metabolicischemic abnrm",
        "myocardial injury", "injury", "st elevation consider", "cannot rule out myocardial injury"
    ],
    "infarct_pattern": [
        "infarct", "infarction",
        "inferior infarct", "anterior infarct", "lateral infarct", "septal infarct", "posterior infarct",
        "anterolateral infarct", "anteroseptal infarct", "extensive infarct",
        "age undetermined", "age indeterminate",
        "possibly acute", "probably acute", "acute infarct", "old infarct", "recent infarct",
        "abnormal q suggests", "abnormal inferior q waves",
    ],
}

ECTOPY_LABELS = {
    "ventricular_ectopy": [  # CONSOLIDATED: all ventricular ectopy patterns
        "pvcs", "ventricular premature", "ventricular extrasystoles", "multifocal pvcs", "frequent pvcs", "interpolated pvcs",
        "ventricular bigeminy", "bigeminy",
        "ventricular trigeminy", "trigeminy",
        "ventricular couplets", "couplets",
        "fusion complexes", "ventricular fusion"
    ],
    "supraventricular_ectopy": [  # CONSOLIDATED: PACs + SVE bigeminy
        "pacs", "atrial premature", "supraventricular extrasystoles", "atrial extrasystoles",
        "supraventricular bigeminy"
    ],
}

IMPRESSION_LABELS = {
    "normal_ecg": [
        "normal ecg",
        "probable normal",
        "normal variant",
        "probable normal variant",
        "within normal limits",
        "summary normal",
        "available leads normal",
    ],
    "borderline_ecg": [
        "borderline ecg",
        "nonspecific borderline ecg",
        "borderline abnormal",
        "summary borderline",
    ],
    "abnormal_ecg": [
        "abnormal ecg",
        "nonspecific abnormal ecg",
        "undetermined abnormal ecg",
        "summary abnormal",
    ],
}

PACED_RHYTHM_LABELS = {
    "paced_rhythm": [  # CONSOLIDATED: all pacing modes
        "ventricular paced rhythm", "ventricular paced", "ventricularpaced", "ventricular pacing",
        "atrial paced rhythm", "atrial paced",
        "dual paced rhythm", "a v dual paced", "atrial ventricular dual paced"
    ],
}

ADDITIONAL_RHYTHM_LABELS = {
    "sinus_arrhythmia": ["sinus arrhythmia", "slow sinus arrhythmia"],
    "undetermined_rhythm": ["undetermined rhythm", "unclassified ecg", "unknown rhythm"],
}

PREEXCITATION_LABELS = {
    "wpw_pattern": [
        "wpw pattern",
        "wolffparkinsonwhite",
        "ventricular preexcitation",
        "vent preexcit",
    ],
}

TECHNICAL_LABELS = {
    "technical_issue": [  # CONSOLIDATED: technical errors + lead reversal
        "technical error", "technically unsatisfactory", "analysis error", "all leads are missing",
        "data quality may affect",
        "lead reversal", "arm lead reversal", "limb lead reversal", "suggests dextrocardia"
    ],
}

ACUTE_PATHOLOGY_LABELS = {
    "pericarditis": [
        "pericarditis",
        "suggests acute pericarditis",
        "strongly suggests pericarditis",
        "post operative pericarditis",
    ],
}

METABOLIC_LABELS = {
    "electrolyte_abnormality": [  # CONSOLIDATED: hyperkalemia + digitalis
        "hyperkalemia", "tall t waves consider hyperkalemia", "tall t waves suggest hyperkalemia",
        "digitalis effect", "possible digitalis effect"
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
    **PREEXCITATION_LABELS,
    **TECHNICAL_LABELS,
    **ACUTE_PATHOLOGY_LABELS,
    **METABOLIC_LABELS,
}