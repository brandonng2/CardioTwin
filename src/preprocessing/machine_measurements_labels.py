RHYTHM_LABELS = {
    "sinus_rhythm": ["sinus rhythm", "sinus rhythm normal"],
    "sinus_bradycardia": ["sinus bradycardia", "bradycardia"],
    "sinus_tachycardia": ["sinus tachycardia", "tachycardia"],
    "atrial_fibrillation": ["atrial fibrillation", "fibrillation", "aflutterfibrillation"],
    "atrial_flutter": ["atrial flutter", "aflutter"],
    "supraventricular_tachycardia": ["supraventricular tachycardia", "svt"],
    "ventricular_tachycardia": ["ventricular tachycardia", "wide complex tachycardia", "wideqrs tachycardia", "wide qrs tachycardia"],
    "idioventricular_rhythm": ["idioventricular rhythm", "accelerated idioventricular"],
}

CONDUCTION_LABELS = {
    "bundle_branch_block": ["bundle branch block", "branch block"],
    "right_bundle_branch_block": ["rbbb", "right bundle branch block"],
    "left_bundle_branch_block": ["lbbb", "left bundle branch block", "ilbbb", "incomplete left bundle branch block"],
    "av_block": ["av block", "degree av block", "st degree av block", "nd degree av block", "rd degree av block", "complete av block", "wenckebach", "mobitz"],
    "left_anterior_fascicular_block": ["left anterior fascicular block", "anterior fascicular block", "lafb"],
    "left_posterior_fascicular_block": ["left posterior fascicular block", "posterior fascicular block", "lpfb"],
    "intraventricular_conduction_delay": ["intraventricular conduction delay", "ivcd", "intermittent conduction defect"],
    "aberrant_conduction": ["aberrant conduction", "aberrantly conducted", "aberrant complex", "aberrant ventricular conduction"],
}

STRUCTURAL_LABELS = {
    "left_ventricular_hypertrophy": [
        "left ventricular hypertrophy",
        "ventricular hypertrophy",
        "lvh",
        "voltage criteria for left ventricular hypertrophy",
    ],
    "right_ventricular_hypertrophy": ["right ventricular hypertrophy", "rvh"],
    "biatrial_enlargement": ["biatrial enlargement", "biatrial abnormality"],
    "left_atrial_enlargement": ["left atrial enlargement", "left atrial abnormality", "lae", "abnormal p terminal force"],
    "right_atrial_enlargement": [
        "right atrial enlargement",
        "right atrial abnormality",
        "rae",
        "tall inferior p waves",
        "abnormally tall p",
        "abnormally high p amplitudes",
    ],
    "axis_deviation": [
        "axis deviation",
        "left axis deviation",
        "right axis deviation",
        "extreme axis deviation",
        "abnormal extreme qrs axis deviation",
    ],
    "low_qrs_voltage": [
        "low qrs voltages",
        "low qrs",
        "low qrs voltages precordial",
        "low voltage",
    ],
    "high_qrs_voltage": ["abnormally high qrs voltages", "high qrs voltage", "abnormally high sv rv"],
}

REPOLARIZATION_LABELS = {
    "st_t_abnormality": [
        "stt changes",
        "stt changes nonspecific",
        "wave changes",
        "wave changes nonspecific",
        "t wave changes",
        "st changes",
        "st t abnormality",
    ],
    "poor_r_wave_progression": [
        "poor wave progression",
        "poor r wave progression",
        "anterior r decrease",
        "abnormal rwave progression",
    ],
    "st_elevation": [
        "st elevation",
        "st elev",
        "early repolarization",
        "early repol",
    ],
    "st_depression": [
        "st depression",
        "st depr",
        "junctional depression",
    ],
    "t_wave_inversion": ["t wave inversion", "inverted t"],
    "prolonged_qt": ["prolonged qt", "prolonged qt interval"],
    "qt_prolongation": ["qt prolongation"],
}

ISCHEMIA_LABELS = {
    "myocardial_ischemia": [
        "ischemia",
        "myocardial ischemia",
        "consider ischemia",
        "probable ischemia",
        "suggest ischemia",
        "abnormal t consider ischemia",
        "abnormal t probable ischemia",
        "metabolicischemic abnrm",
    ],
    "myocardial_injury": [
        "myocardial injury",
        "injury",
        "st elevation consider",
        "cannot rule out myocardial injury",
    ],
    "infarct_pattern": [
        "infarct",
        "infarction",
        "inferior infarct",
        "anterior infarct",
        "lateral infarct",
        "septal infarct",
        "posterior infarct",
        "anterolateral infarct",
        "anteroseptal infarct",
        "extensive infarct",
        "age undetermined",
        "age indeterminate",
        "possibly acute",
        "probably acute",
        "acute infarct",
        "old infarct",
        "recent infarct",
        "abnormal q suggests",
        "abnormal inferior q waves",
    ],
}

ECTOPY_LABELS = {
    "pvcs": ["pvcs", "ventricular premature", "ventricular extrasystoles", "multifocal pvcs", "frequent pvcs", "interpolated pvcs"],
    "pacs": ["pacs", "atrial premature", "supraventricular extrasystoles", "atrial extrasystoles"],
    "ventricular_bigeminy": ["ventricular bigeminy", "bigeminy"],
    "ventricular_trigeminy": ["ventricular trigeminy", "trigeminy"],
    "supraventricular_bigeminy": ["supraventricular bigeminy"],
    "ventricular_couplets": ["ventricular couplets", "couplets"],
    "fusion_complexes": ["fusion complexes", "ventricular fusion"],
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
    "atrial_paced_rhythm": ["atrial paced rhythm", "atrial paced"],
    "ventricular_paced_rhythm": [
        "ventricular paced rhythm",
        "ventricular paced",
        "ventricularpaced",
        "ventricular pacing",
    ],
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
    "junctional_rhythm": ["junctional rhythm", "accelerated junctional rhythm"],
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
    "technical_error": [
        "technical error",
        "technically unsatisfactory",
        "analysis error",
        "all leads are missing",
        "data quality may affect",
    ],
    "lead_reversal": [
        "lead reversal",
        "arm lead reversal",
        "limb lead reversal",
        "suggests dextrocardia",
    ],
}

PERICARDITIS_LABELS = {
    "pericarditis": [
        "pericarditis",
        "suggests acute pericarditis",
        "strongly suggests pericarditis",
        "post operative pericarditis",
    ],
}

ELECTROLYTE_LABELS = {
    "hyperkalemia": [
        "hyperkalemia",
        "tall t waves consider hyperkalemia",
        "tall t waves suggest hyperkalemia",
    ],
    "digitalis_effect": ["digitalis effect", "possible digitalis effect"],
}

P_WAVE_LABELS = {
    "abnormal_p_waves": [
        "abnormal p waves",
        "p wave abnormality",
        "abnormal p terminal force",
    ],
}

RATE_LABELS = {
    "ventricular_response": [
        "rapid ventricular response",
        "slow ventricular response",
        "uncontrolled ventricular response",
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
    **PERICARDITIS_LABELS,
    **ELECTROLYTE_LABELS,
    **P_WAVE_LABELS,
    **RATE_LABELS,
}