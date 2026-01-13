# file: entity_extraction.py
import numpy as np
import re
import pandas as pd
from collections import Counter

# --- Cardiovascular keywords ---
HOSP_CV_KEYWORDS = [
    'heart', 'coronary', 'infarction', 'atrial', 'fibrillation', 'myocardial',
    'ventricular', 'vascular','congestive', 'systolic', 'diastolic', 'hypertension',
    'hypertensive', 'artery', 'ischemia', 'block', 'bundle', 'rbbb', 'lbbb',
    'tachycardia', 'bradycardia', 'valve', 'stenosis', 'oxygen'
]

ED_CV_KEYWORDS = {
    "Arrhythmia": [
        "atrial fibrillation", "atrial flutter",
        "unspecified atrial fibrillation", "unspecified atrial flutter",
        "palpitations", "ventricular tachycardia",
        "bradycardia", "dysrhythmia"
    ],

    "Ischemic": [
        "myocardial infarction", "infarction",
        "stemi", "nstemi",
        "acute ischemic heart disease",
        "intermed coronary synd"
    ],

    "Heart Failure": [
        "heart failure", "congestive heart failure", "chf"
    ],

    "Chest Pain / Symptoms": [
        "chest pain", "other chest pain", "chest pain nos", "upper quadrant pain",
        "shortness of breath", "dyspnea",
        "syncope", "collapse",
        "dizziness", "giddiness"
    ],

    "Vascular / Embolic": [
        "pulmonary embolism", "embolism",
        "cerebral infarction", 
        "cerebral art occlus", 
        "infarct"
    ],

    "Cardiac Arrest": [
        "cardiac arrest", "asystole"
    ],

    "Structural / Cardiomyopathy": [
        "cardiomyopathy", "hypertrophy"
    ]
}

# --- Functions ---

def extract_hosp_entities(diagnosis_text, keywords=HOSP_CV_KEYWORDS):
    """
    Extract cardiovascular keywords from hospital diagnosis text.
    Works if diagnosis_text is a string, list, or array.
    """
    # If input is a list/array/Series, join into a single string
    if isinstance(diagnosis_text, (list, np.ndarray, pd.Series)):
        diagnosis_text = " ".join([str(d) for d in diagnosis_text if pd.notna(d)])
    
    # Now check if the scalar/string is NaN or empty
    if diagnosis_text is None or (isinstance(diagnosis_text, float) and np.isnan(diagnosis_text)):
        return []

    diagnosis_lower = diagnosis_text.lower()
    found = [kw for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', diagnosis_lower)]
    return list(set(found))


def extract_ed_entities(title_text, keyword_dict=ED_CV_KEYWORDS):
    """
    Extract structured cardiovascular categories from ED diagnosis titles.
    Works if title_text is a string, list, or array.
    Returns a dict {category: [matches]} or empty dict if nothing matches.
    """
    # If input is a list/array/Series, join into a single string
    if isinstance(title_text, (list, np.ndarray, pd.Series)):
        title_text = " ".join([str(t) for t in title_text if pd.notna(t)])
    
    # Now check if scalar is NaN/None
    if title_text is None or (isinstance(title_text, float) and np.isnan(title_text)):
        return {}

    text = title_text.lower()
    extracted = {k: [] for k in keyword_dict.keys()}
    found_any = False

    for category, kws in keyword_dict.items():
        for kw in kws:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                extracted[category].append(kw)
                found_any = True

    if not found_any:
        return {}

    return {k: list(set(v)) for k, v in extracted.items()}




def get_top_words(series, top_n=100):
    """
    Helper to get top N words in a text series.
    """
    all_text = " ".join(series.dropna().str.lower().tolist())
    tokens = re.findall(r'\b\w+\b', all_text)
    word_counts = Counter(tokens)
    return [word for word, _ in word_counts.most_common(top_n)]


def filter_keywords_from_top(top_words, keywords):
    """
    Filter a keyword list to only those appearing in top_words.
    """
    return [kw for kw in keywords if kw in top_words]


def apply_entity_extraction(df, hosp_col="hosp_diagnosis", ed_col="ed_diagnosis"):
    """
    Apply both hospital and ED entity extraction to a dataframe.
    Adds:
        - diagnosis_entities (hospital)
        - num_diagnosis_entities (count)
        - ed_entities (ED)
    """
    df = df.copy()
    
    df['hosp_diagnosis_entities'] = df[hosp_col].apply(extract_hosp_entities)
    df['num_hosp_diagnosis_entities'] = df['hosp_diagnosis_entities'].apply(len)
    df['ed_entities'] = df[ed_col].apply(extract_ed_entities)
    df['num_ed_diagnosis_entities'] = df['ed_entities'].apply(len)
    
    return df

def run_entity_extraction(df, out_path):
    print("[1/1] Applying clinical entity extraction...")
    df = apply_entity_extraction(df)
    
    print("\n" + "=" * 60)
    print(f"Final dataset shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print("=" * 60)
    
    print(f"\nSaving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print("Clinical entity extraction complete!")
    return df


