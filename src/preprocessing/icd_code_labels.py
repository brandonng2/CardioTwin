icd_code_labels = {
    # ═══════════════════════════════════════
    # A. CARDIOVASCULAR (17)
    # ═══════════════════════════════════════
    
    # 1. Acute Myocardial Infarction (AMI)
    'ami_stemi': ['I21.0', 'I21.1', 'I21.2', 'I21.3', 'I21.9', 'I22'],
    'ami_nstemi': ['I21.4'],
    
    # 2. Ischemic Heart Disease (Unstable & Chronic)
    'unstable_angina_ac_ischemia': ['I20.0', 'I24', 'I23'], # Grouped acute complications
    'chronic_ischemic_disease': ['I20.1', 'I20.8', 'I20.9', 'I25'], # Grouped CAD/CTO/Aneurysm
    
    # 3. Heart Failure (Grouped by Severity/Type)
    'heart_failure_acute': ['I50.21', 'I50.31', 'I50.41', 'I50.9', 'I50.23', 'I50.33', 'I50.43', 'R57.0'], # Acute + Cardiogenic Shock
    'heart_failure_chronic': ['I50.22', 'I50.32', 'I50.42', 'I50.8', 'I51.7'], # Chronic + Cardiomegaly
    
    # 4. Arrhythmias (Electrophysiological Focus)
    'afib_aflutter': ['I48'], 
    'ventricular_arrhythmias_arrest': ['I47.0', 'I47.2', 'I49.0', 'I46'], # Lethal rhythms grouped
    'supraventricular_tachyarrhythmias': ['I47.1', 'I47.9'],
    'brady_heart_block_conduction': ['I44', 'I45', 'I49.5'], # Consolidated conduction/sick sinus
    
    # 5. Structural & Inflammatory
    'valvular_endocardial_disease': ['I05', 'I06', 'I07', 'I08', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39'],
    'cardiomyopathy_myocarditis': ['I42', 'I43', 'I40', 'I41'],
    'pericardial_disease_tamponade': ['I30', 'I31', 'I32'],

    # 6. Vascular & Hypertension
    'pe_dvt_venous_thromboembolism': ['I26', 'I80', 'I82'],
    'aortic_peripheral_vascular': ['I70', 'I71', 'I73'],
    'hypertension_crisis': ['I10', 'I11', 'I12', 'I13', 'I16'],

    # 7. Cerebrovascular
    'stroke_tia': ['I60', 'I61', 'I62', 'I63', 'G45'],

    # ═══════════════════════════════════════
    # B. NON-CARDIOVASCULAR (ML Risk Modifiers) (7)
    # ═══════════════════════════════════════
    
    'respiratory_failure_distress': ['J96', 'J80', 'J81'],
    'obstructive_lung_disease': ['J44', 'J45'],
    'infectious_pneumonia_sepsis': ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'A40', 'A41', 'R65.21'],
    'renal_failure_dialysis': ['N17', 'N18', 'Z99.2'],
    'metabolic_diabetes': ['E10', 'E11', 'E13', 'E14', 'E87'],
    'anemia_hematologic': ['D50', 'D64'],
    'gi_hepatic_complications': ['K92', 'K70', 'K74']
}

# ═══════════════════════════════════════
# CARDIOVASCULAR-ONLY SUBSET
# Use this for pure cardiovascular digital twin modeling
# ═══════════════════════════════════════

noncardiovascular_labels = {
    'respiratory_failure_distress',
    'obstructive_lung_disease',
    'infectious_pneumonia_sepsis',
    'renal_failure_dialysis',
    'metabolic_diabetes',
    'anemia_hematologic',
    'gi_hepatic_complications'
}

cardiovascular_labels = {
    k: v for k, v in icd_code_labels.items() 
    if k not in noncardiovascular_labels
}