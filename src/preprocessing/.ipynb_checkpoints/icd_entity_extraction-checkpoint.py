# file: src/preprocessing/clinical_entity_extraction.py
import pandas as pd
import ast
import icd10

# -----------------------------
# 1. Parsing ICD lists
# -----------------------------
def parse_icd_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return None
    return None

# -----------------------------
# 2. Normalize ICD codes
# -----------------------------
def normalize_icd(code):
    if code is None:
        return None
    if len(code) > 3 and "." not in code:
        return code[:3] + "." + code[3:]
    return code

# -----------------------------
# 3. ICD codes -> descriptions
# -----------------------------
def icd_codes_to_descriptions(codes):
    if not isinstance(codes, list):
        return []
    descs = []
    for c in codes:
        c_norm = normalize_icd(c)
        try:
            node = icd10.find(c_norm)
            if node:
                descs.append(node.description.lower())
        except:
            continue
    return descs

# -----------------------------
# 4. Heart-only ICD filters
# -----------------------------
heart_only_icd_prefixes = [f"I{str(i).zfill(2)}" for i in range(20, 53)]

def is_heart_icd(code):
    if not isinstance(code, str):
        return False
    return any(code.startswith(prefix) for prefix in heart_only_icd_prefixes)

def filter_heart_codes(icd_list):
    if not isinstance(icd_list, list):
        return []
    return [c for c in icd_list if is_heart_icd(c)]

# -----------------------------
# 5. Canonical mapping
# -----------------------------
canonical_map = {
    "heart failure": [
        "chronic diastolic (congestive) heart failure",
        "acute on chronic diastolic (congestive) heart failure",
        "chronic systolic (congestive) heart failure",
        "acute on chronic systolic (congestive) heart failure",
        "heart failure, unspecified",
        "end stage heart failure",
        "biventricular heart failure",
        "right heart failure, unspecified",
        "left ventricular failure, unspecified",
        "acute on chronic right heart failure",
        "unspecified combined systolic (congestive) and diastolic (congestive) heart failure",
        "chronic combined systolic (congestive) and diastolic (congestive) heart failure",
        "acute combined systolic (congestive) and diastolic (congestive) heart failure",
        "acute systolic (congestive) heart failure",
        "acute diastolic (congestive) heart failure",
        "chronic total occlusion of coronary artery",  # sometimes grouped under severe HF
    ],
    "myocardial infarction": [
        "old myocardial infarction",
        "non-st elevation (nstemi) myocardial infarction",
        "st elevation (stemi) myocardial infarction of unspecified site",
        "st elevation (stemi) myocardial infarction involving other coronary artery of inferior wall",
        "st elevation (stemi) myocardial infarction involving other coronary artery of anterior wall",
        "st elevation (stemi) myocardial infarction involving right coronary artery",
        "myocardial infarction type 2",
        "subsequent non-st elevation (nstemi) myocardial infarction",
        "acute ischemic heart disease, unspecified",
        "other forms of acute ischemic heart disease",
        "acute coronary thrombosis not resulting in myocardial infarction",
    ],
    "atrial fibrillation": [
        "unspecified atrial fibrillation",
        "paroxysmal atrial fibrillation",
        "chronic atrial fibrillation",
        "persistent atrial fibrillation",
    ],
    "atrial flutter": [
        "unspecified atrial flutter",
        "typical atrial flutter",
        "atypical atrial flutter",
    ],
    "cardiomyopathy": [
        "cardiomyopathy, unspecified",
        "ischemic cardiomyopathy",
        "other cardiomyopathies",
        "dilated cardiomyopathy",
        "obstructive hypertrophic cardiomyopathy",
        "alcoholic cardiomyopathy",
        "cardiomyopathy in diseases classified elsewhere",
        "cardiomyopathy due to drug and external agent",
        "other restrictive cardiomyopathy",
    ],
    "pulmonary hypertension": [
        "pulmonary hypertension, unspecified",
        "other secondary pulmonary hypertension",
        "pulmonary hypertension due to left heart disease",
        "secondary pulmonary arterial hypertension",
        "primary pulmonary hypertension",
        "pulmonary hypertension due to lung diseases and hypoxia",
    ],
    "angina": [
        "unstable angina",
        "other forms of angina pectoris",
        "angina pectoris, unspecified",
        "angina pectoris with documented spasm",
        "atherosclerotic heart disease of native coronary artery with unstable angina pectoris",
        "atherosclerotic heart disease of native coronary artery with other forms of angina pectoris",
        "atherosclerosis of autologous vein coronary artery bypass graft(s) with unstable angina pectoris",
        "atherosclerosis of autologous vein coronary artery bypass graft(s) with other forms of angina pectoris",
        "atherosclerotic heart disease of native coronary artery with angina pectoris with documented spasm",
    ],
    "bundle branch block": [
        "left bundle-branch block, unspecified",
        "unspecified right bundle-branch block",
        "bifascicular block",
        "left anterior fascicular block",
    ],
    "atrioventricular block": [
        "atrioventricular block, complete",
        "atrioventricular block, second degree",
        "atrioventricular block, first degree",
        "other atrioventricular block",
        "other specified heart block",
        "unspecified atrioventricular block",
    ],
    "arrhythmias": [
        "ventricular tachycardia",
        "ventricular fibrillation",
        "ventricular premature depolarization",
    ],
    "pericardial disease": [
        "pericardial effusion (noninflammatory)",
        "acute pericarditis, unspecified",
        "cardiac tamponade",
        "infective pericarditis",
        "disease of pericardium, unspecified",
    ],
    "coronary artery disease": [
        "atherosclerotic heart disease of native coronary artery without angina pectoris",
        "atherosclerotic heart disease of native coronary artery with unspecified angina pectoris",
        "atherosclerotic heart disease of native coronary artery with angina pectoris with documented spasm",
        "atherosclerotic heart disease of native coronary artery with unstable angina pectoris",
        "coronary atherosclerosis due to calcified coronary lesion",
        "atherosclerosis of autologous vein coronary artery bypass graft(s) with other forms of angina pectoris",
    ],
    "valvular disease": [
        "nonrheumatic aortic (valve) stenosis",
        "nonrheumatic mitral (valve) insufficiency",
        "nonrheumatic mitral (valve) prolapse",
        "nonrheumatic aortic (valve) insufficiency",
        "nonrheumatic tricuspid (valve) insufficiency",
        "other nonrheumatic aortic valve disorders",
    ],
    "conduction disorder": [
        "other specified conduction disorders",
        "conduction disorder, unspecified",
        "pre-excitation syndrome",
    ],
    "cardiomegaly": [
        "cardiomegaly",
    ],
    "other": [
        "takotsubo syndrome",
        "chronic pulmonary embolism",
        "saddle embolus of pulmonary artery without acute cor pulmonale",
        "other pulmonary embolism with acute cor pulmonale",
        "other pulmonary embolism without acute cor pulmonale",
        "pulmonary hypertension due to left heart disease",
        "secondary pulmonary arterial hypertension",
        "primary pulmonary hypertension",
        "pulmonary hypertension due to lung diseases and hypoxia",
        "cor pulmonale (chronic)",
        "rupture of chordae tendineae, not elsewhere classified",
        "intracardiac thrombosis, not elsewhere classified",
        "endocarditis, valve unspecified",
        "acute and subacute infective endocarditis",
        "acute coronary thrombosis not resulting in myocardial infarction",
        "cardiac arrest, cause unspecified",
        "cardiac arrest due to underlying cardiac condition",
        "cardiac arrest due to other underlying condition",
    ],
}

canonical_map['heart failure'].extend([
    "unspecified diastolic (congestive) heart failure",
    "unspecified systolic (congestive) heart failure",
    "acute on chronic combined systolic (congestive) and diastolic (congestive) heart failure",
    "chronic right heart failure",
    "acute right heart failure",
    "other heart failure"
])

canonical_map['arrhythmias'].extend([
    "supraventricular tachycardia",
    "sick sinus syndrome",
    "atrial premature depolarization",
    "other specified cardiac arrhythmias",
    "cardiac arrhythmia, unspecified",
    "paroxysmal tachycardia, unspecified"
])

canonical_map['pericardial disease'].extend([
    "acute nonspecific idiopathic pericarditis",
    "chronic constrictive pericarditis",
    "acute and subacute endocarditis, unspecified",
    "other forms of acute pericarditis",
    "acute pericarditis, unspecified",
    "chronic adhesive pericarditis",
    "hemopericardium, not elsewhere classified",
    "cardiac tamponade",
    "infective pericarditis",
    "disease of pericardium, unspecified"
])

canonical_map['coronary artery disease'].extend([
    "atherosclerosis of coronary artery bypass graft(s) without angina pectoris",
    "atherosclerosis of autologous vein coronary artery bypass graft(s) with unspecified angina pectoris",
    "atherosclerosis of autologous artery coronary artery bypass graft(s) with unstable angina pectoris"
])

canonical_map['myocardial infarction'].extend([
    "st elevation (stemi) myocardial infarction involving other sites",
    "st elevation (stemi) myocardial infarction involving left anterior descending coronary artery",
    "st elevation (stemi) myocardial infarction involving left circumflex coronary artery",
    "acute myocardial infarction, unspecified",
    "other myocardial infarction type",
    "thrombosis of atrium, auricular appendage, and ventricle as current complications following acute myocardial infarction"
])

canonical_map['angina'].extend([
    "postinfarction angina"
])

canonical_map['valvular disease'].extend([
    "nonrheumatic aortic (valve) stenosis with insufficiency",
    "other nonrheumatic mitral valve disorders"
])

canonical_map['other'].extend([
    "chronic ischemic heart disease, unspecified",
    "aneurysm of heart",
    "coronary artery dissection",
    "other ill-defined heart diseases",
])

# Arrhythmias / conduction disorders
canonical_map['arrhythmias'].extend([
    "long qt syndrome",
    "trifascicular block",
    "other right bundle-branch block",
    "junctional premature depolarization",
    "re-entry ventricular arrhythmia",
    "ventricular flutter",
    "unspecified fascicular block",
    "other premature depolarization"
])

# Cardiomyopathy / myocarditis
canonical_map['cardiomyopathy'].extend([
    "other hypertrophic cardiomyopathy",
    "myocarditis, unspecified",
    "isolated myocarditis",
    "acute myocarditis, unspecified",
    "other acute myocarditis"
])

# Heart failure / structural defects
canonical_map['heart failure'].extend([
    "right heart failure due to left heart failure",
    "high output heart failure"
])
canonical_map['other'].extend([
    "cardiac septal defect, acquired",
    "ventricular septal defect as current complication following acute myocardial infarction"
])

# Pericardial disease
canonical_map['pericardial disease'].extend([
    "other specified diseases of pericardium",
    "pericarditis in diseases classified elsewhere",
    "dressler's syndrome",
    "endocardial fibroelastosis"
])

# Valve disease
canonical_map['valvular disease'].extend([
    "nonrheumatic mitral (valve) stenosis",
    "nonrheumatic pulmonary valve stenosis",
    "nonrheumatic pulmonary valve insufficiency",
    "nonrheumatic tricuspid (valve) stenosis",
    "nonrheumatic aortic valve disorder, unspecified",
    "other nonrheumatic pulmonary valve disorders",
    "nonrheumatic aortic (valve) stenosis with insufficiency"
])

# Coronary artery / ischemic disease
canonical_map['coronary artery disease'].extend([
    "subsequent st elevation (stemi) myocardial infarction of inferior wall",
    "st elevation (stemi) myocardial infarction involving left main coronary artery",
    "subsequent st elevation (stemi) myocardial infarction of unspecified site",
    "subsequent st elevation (stemi) myocardial infarction of anterior wall",
    "coronary artery aneurysm",
    "coronary atherosclerosis due to lipid rich plaque",
    "atherosclerosis of coronary artery bypass graft(s), unspecified, with unstable angina pectoris",
    "atherosclerosis of coronary artery bypass graft(s), unspecified, with unspecified angina pectoris",
    "atherosclerosis of autologous artery coronary artery bypass graft(s) with unspecified angina pectoris",
    "atherosclerosis of coronary artery bypass graft(s), unspecified, with other forms of angina pectoris",
    "atherosclerosis of native coronary artery of transplanted heart without angina pectoris"
])

# Pulmonary vessels / pulmonary hypertension
canonical_map['pulmonary hypertension'].extend([
    "septic pulmonary embolism without acute cor pulmonale",
    "saddle embolus of pulmonary artery with acute cor pulmonale",
    "chronic thromboembolic pulmonary hypertension",
    "arteriovenous fistula of pulmonary vessels",
    "other diseases of pulmonary vessels"
])

# -----------------------------
# 6. Map phrases to canonical labels
# -----------------------------
def map_row_to_labels(phrases):
    if not isinstance(phrases, list):
        return []
    labels = []
    phrases_lower = [p.lower() for p in phrases]
    for label, variants in canonical_map.items():
        if any(v in phrases_lower for v in variants):
            labels.append(label)
    return labels

def final_label_bucket(labels):
    if not labels:
        return ["other rare cardiac"]
    return labels

# -----------------------------
# 7. Main function
# -----------------------------
def run_entity_extraction(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Run full clinical entity extraction and save results.
    """
    df = df.copy()

    # Parse ICD lists
    df['hosp_icd_codes_diagnosis_parsed'] = df['hosp_icd_codes_diagnosis'].apply(parse_icd_list)
    df['ed_icd_codes_diagnosis_parsed'] = df['ed_icd_codes_diagnosis'].apply(parse_icd_list)

    # Filter heart ICD codes
    df['hosp_heart_icd_codes'] = df['hosp_icd_codes_diagnosis_parsed'].apply(filter_heart_codes)
    df['ed_heart_icd_codes'] = df['ed_icd_codes_diagnosis_parsed'].apply(filter_heart_codes)

    # Convert to descriptions
    df['hosp_heart_descriptions'] = df['hosp_heart_icd_codes'].apply(icd_codes_to_descriptions)
    df['ed_heart_descriptions'] = df['ed_heart_icd_codes'].apply(icd_codes_to_descriptions)

    # Map to canonical labels
    df['canonical_labels'] = df['hosp_heart_descriptions'].apply(map_row_to_labels) + df['ed_heart_descriptions'].apply(map_row_to_labels)
    df['canonical_labels'] = df['canonical_labels'].apply(final_label_bucket)
    df['canonical_labels'] = df['canonical_labels'].apply(lambda x: list(set(x)))

    # One-hot encode labels
    for label in canonical_map.keys():
        df[label] = df['canonical_labels'].apply(lambda x: int(label in x))
    df = df.drop(columns=['canonical_labels'])

    # Cardiovascular flag
    df["cardiovascular"] = df.apply(
        lambda row: "cardiovascular" if row["hosp_heart_icd_codes"] or row["ed_heart_icd_codes"] else "not cardiovascular",
        axis=1
    )

    # Save output
    df.to_csv(out_path, index=False)
    print(f"Clinical entity extraction complete! Saved to {out_path}")
    return df
