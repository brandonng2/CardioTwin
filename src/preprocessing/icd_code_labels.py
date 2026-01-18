icd_code_labels = {
    # ═══════════════════════════════════════
    # A. CARDIOVASCULAR (33)
    # ═══════════════════════════════════════
    
    # Acute Coronary Syndromes
    'stemi': ['I21.0', 'I21.1', 'I21.2', 'I21.3', 'I21.9'],
    'nstemi': ['I21.4'],
    'unstable_angina': ['I20.0'],
    'subsequent_mi': ['I22'],
    
    # Chronic Ischemic & Angina
    'stable_angina': ['I20.1', 'I20.8', 'I20.9'],
    'coronary_artery_disease': ['I25.1', 'I25.10', 'I25.11', 'I25.700', 'I25.701', 'I25.708', 'I25.709', 'I25.710', 'I25.718', 'I25.719', 'I25.72', 'I25.730', 'I25.731', 'I25.738', 'I25.739', 'I25.750', 'I25.751', 'I25.758', 'I25.759', 'I25.760', 'I25.761', 'I25.768', 'I25.769', 'I25.790', 'I25.791', 'I25.798', 'I25.799', 'I25.810', 'I25.811', 'I25.812'],
    'chronic_ischemic_heart_disease': ['I25.5', 'I25.6', 'I25.9'],
    'coronary_aneurysm_dissection': ['I25.41', 'I25.42'],
    'chronic_total_occlusion': ['I25.82'],
    
    # Heart Failure - NON-OVERLAPPING
    'acute_heart_failure': ['I50.21', 'I50.31', 'I50.41', 'I50.9'],  # ✅ Removed acute-on-chronic codes
    'chronic_heart_failure': ['I50.22', 'I50.32', 'I50.42'],
    'acute_on_chronic_heart_failure': ['I50.23', 'I50.33', 'I50.43'],  # ✅ Separated - no overlap
    'right_heart_failure': ['I50.810', 'I50.811', 'I50.812', 'I50.813', 'I50.814'],
    'high_output_heart_failure': ['I50.83'],
    'end_stage_heart_failure': ['I50.84'],
    'cardiogenic_shock': ['R57.0'],  # Note: Chapter 18 (Symptoms) - clinical finding
    
    # Cardiac Arrest
    'cardiac_arrest': ['I46', 'I46.2', 'I46.8', 'I46.9'],
    
    # Arrhythmias - NON-OVERLAPPING
    'atrial_fibrillation_flutter': ['I48', 'I48.0', 'I48.1', 'I48.2', 'I48.91'],
    'supraventricular_tachycardia': ['I47.1'],
    'ventricular_arrhythmias': ['I47.0', 'I47.2', 'I49.01', 'I49.02'],
    'premature_depolarization': ['I49.1', 'I49.2', 'I49.3', 'I49.40', 'I49.49'],
    'bradyarrhythmias_heart_block': ['I44', 'I45'],  # ✅ Removed I49.5 (sick sinus)
    'sick_sinus_syndrome': ['I49.5'],  # ✅ Separated - no overlap
    'long_qt_syndrome': ['I45.81'],
    'other_arrhythmias': ['I49.8', 'I49.9'],
    
    # Conduction Disorders
    'bundle_branch_block': ['I44.0', 'I44.1', 'I44.2', 'I44.4', 'I44.5', 'I44.6', 'I44.7', 'I45.0', 'I45.2'],
    'atrioventricular_block': ['I44.0', 'I44.1', 'I44.2', 'I44.30'],
    'other_conduction_disorders': ['I45.6', 'I45.89', 'I45.9'],
    
    # Structural - Common
    'valvular_heart_disease': ['I05', 'I06', 'I07', 'I08', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39'],
    'cardiomyopathy': ['I42', 'I42.0', 'I42.1', 'I42.2', 'I42.3', 'I42.4', 'I42.5', 'I42.6', 'I42.7', 'I42.8', 'I42.9', 'I43'],
    'myocarditis': ['I40', 'I40.0', 'I40.1', 'I40.8', 'I40.9', 'I41'],
    'pericarditis_pericardial_disease': ['I30', 'I30.0', 'I30.1', 'I30.8', 'I30.9', 'I31', 'I32'],
    'endocarditis': ['I33', 'I38', 'I39'],
    'cardiomegaly': ['I51.7'],
    
    # Structural - Rare (consider grouping for ML)
    'rare_cardiac_structural': [  # ✅ NEW: Grouped rare conditions
        'I31.4',   # Cardiac tamponade
        'I51.81',  # Takotsubo syndrome
        'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26',  # Congenital heart disease
        'I23',     # Post-MI complications
    ],
    
    # Other Heart Disease
    'other_heart_disease': ['I51.0', 'I51.1', 'I51.2', 'I51.3', 'I51.4', 'I51.5', 'I51.89', 'I51.9'],
    
    # Vascular
    'pulmonary_embolism': ['I26', 'I26.0', 'I26.01', 'I26.02', 'I26.09', 'I26.9', 'I26.90', 'I26.92', 'I26.93', 'I26.94', 'I26.99'],
    'deep_vein_thrombosis': ['I80', 'I82'],
    'aortic_disease': ['I71'],
    'peripheral_vascular_disease': ['I70.2', 'I73.9'],
    
    # Hypertension
    'hypertension': ['I10', 'I11', 'I12', 'I13'],
    'hypertensive_emergency': ['I16.0', 'I16.1'],
    
    # Stroke
    'ischemic_stroke': ['I63'],
    'hemorrhagic_stroke': ['I60', 'I61', 'I62'],
    'tia': ['G45'],  # Note: Chapter 6 (Nervous)
    
    # ═══════════════════════════════════════
    # B. RESPIRATORY (10)
    # ═══════════════════════════════════════
    
    'acute_hypoxic_respiratory_failure': ['J96.01'],
    'acute_hypercapnic_respiratory_failure': ['J96.02'],
    'pneumonia': ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18'],
    'copd_exacerbation': ['J44.1'],
    'asthma_exacerbation': ['J45.21', 'J45.31', 'J45.41', 'J45.51', 'J45.901'],
    'ards': ['J80'],
    'noncardiogenic_pulmonary_edema': ['J81'],
    'pleural_effusion': ['J90'],
    'pneumothorax': ['J93'],
    'pulmonary_hypertension': ['I27', 'I27.0', 'I27.1', 'I27.2', 'I27.20', 'I27.21', 'I27.22', 'I27.23', 'I27.24', 'I27.29', 'I27.8', 'I27.81', 'I27.82', 'I27.83', 'I27.89', 'I27.9'],
    
    # ═══════════════════════════════════════
    # C. RENAL (3)
    # ═══════════════════════════════════════
    
    'acute_kidney_injury': ['N17'],
    'chronic_kidney_disease': ['N18'],
    'dialysis_dependence': ['Z99.2'],  # Note: Chapter 21 (Factors)
    
    # ═══════════════════════════════════════
    # D. INFECTIOUS / INFLAMMATORY (3)
    # ═══════════════════════════════════════
    
    'sepsis': ['A40', 'A41'],
    'septic_shock': ['R65.21'],  # Note: Chapter 18 (Symptoms)
    'active_infection_nonsepsis': ['A49', 'B95', 'B96', 'B97'],
    
    # ═══════════════════════════════════════
    # E. ENDOCRINE & METABOLIC (2)
    # Note: These are conditions/lab findings that affect cardiac function
    # ═══════════════════════════════════════
    
    'diabetes_mellitus': ['E10', 'E11', 'E12', 'E13', 'E14'],
    'electrolyte_disturbance': ['E87'],  # Note: Lab finding, not primary diagnosis
    
    # ═══════════════════════════════════════
    # F. HEMATOLOGIC (1)
    # ═══════════════════════════════════════
    
    'anemia': ['D50', 'D51', 'D52', 'D53', 'D55', 'D59', 'D61', 'D62', 'D63', 'D64'],
    
    # ═══════════════════════════════════════
    # G. GASTROINTESTINAL / HEPATIC (1)
    # ═══════════════════════════════════════
    
    'gi_bleed_or_hepatic_failure': ['K92', 'K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77'],
}

# ═══════════════════════════════════════
# CARDIOVASCULAR-ONLY SUBSET
# Use this for pure cardiovascular digital twin modeling
# ═══════════════════════════════════════

noncardiovascular_labels = {
    'acute_hypoxic_respiratory_failure', 'acute_hypercapnic_respiratory_failure',
    'pneumonia', 'copd_exacerbation', 'asthma_exacerbation', 'ards',
    'noncardiogenic_pulmonary_edema', 'pleural_effusion', 'pneumothorax',
    'acute_kidney_injury', 'chronic_kidney_disease', 'dialysis_dependence',
    'sepsis', 'septic_shock', 'active_infection_nonsepsis',
    'diabetes_mellitus', 'electrolyte_disturbance',
    'anemia', 'gi_bleed_or_hepatic_failure'
}

cardiovascular_labels = {
    k: v for k, v in icd_code_labels.items() 
    if k not in noncardiovascular_labels
}