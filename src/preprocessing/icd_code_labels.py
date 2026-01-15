final_schema = {
    # ═══════════════════════════════════════
    # A. CARDIOVASCULAR (20)
    # ═══════════════════════════════════════
    
    # Acute Coronary Syndromes
    'stemi': ['I21.0', 'I21.1', 'I21.2', 'I21.3'],
    'nstemi': ['I21.4'],
    'unstable_angina': ['I20.0'],
    
    # Chronic Ischemic
    'stable_cad': ['I20.1', 'I20.8', 'I20.9', 'I25'],
    
    # Heart Failure
    'acute_heart_failure': ['I50.21', 'I50.23', 'I50.31', 'I50.33', 'I50.41', 'I50.43', 'I50.9'],
    'chronic_heart_failure': ['I50.22', 'I50.32', 'I50.42'],
    'cardiogenic_shock': ['R57.0'],  # Note: Chapter 18 (Symptoms)
    
    # Cardiac Arrest
    'cardiac_arrest': ['I46'],
    
    # Arrhythmias
    'atrial_fibrillation_flutter': ['I48'],
    'ventricular_arrhythmias': ['I47.2', 'I49.0'],
    'bradyarrhythmias_heart_block': ['I44', 'I45', 'I49.5'],
    
    # Structural
    'valvular_heart_disease': ['I05', 'I06', 'I07', 'I08', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39'],
    'myocarditis_pericarditis': ['I30', 'I40'],
    
    # Vascular
    'pulmonary_embolism': ['I26'],
    'deep_vein_thrombosis': ['I80', 'I82'],  # ✅ Added
    'aortic_disease': ['I71'],
    'peripheral_vascular_disease': ['I70.2', 'I73.9'],
    
    # Hypertension
    'hypertension': ['I10', 'I11', 'I12', 'I13'],  # ✅ Added chronic HTN
    'hypertensive_emergency': ['I16.0', 'I16.1'],
    
    # Stroke
    'ischemic_stroke': ['I63'],  # ✅ All ischemic, not just cardioembolic
    'hemorrhagic_stroke': ['I60', 'I61', 'I62'],  # ✅ Added
    'tia': ['G45'],  # Note: Chapter 6 (Nervous)
    
    # Device (optional - check prevalence)
    # 'cardiac_device_complications': ['T82.1', 'T82.5', 'T82.7'],  # Note: Chapter 19
    
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
    'pulmonary_hypertension': ['I27'],  # Note: Chapter 9, but in Respiratory section
    
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
    # ═══════════════════════════════════════
    
    'diabetes_mellitus': ['E10', 'E11', 'E12', 'E13', 'E14'],
    'electrolyte_disturbance': ['E87'],
    
    # ═══════════════════════════════════════
    # F. HEMATOLOGIC (1)
    # ═══════════════════════════════════════
    
    'anemia': ['D50', 'D51', 'D52', 'D53', 'D55', 'D59', 'D61', 'D62', 'D63', 'D64'],
    
    # ═══════════════════════════════════════
    # G. GASTROINTESTINAL / HEPATIC (1)
    # ═══════════════════════════════════════
    
    'gi_bleed_or_hepatic_failure': ['K92', 'K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77'],
}