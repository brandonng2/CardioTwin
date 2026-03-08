# ClinicalDigitalTwin

An end-to-end pipeline for preprocessing MIMIC-IV clinical and ECG data and training multimodal cardiovascular diagnosis models — from XGBoost and MLP baselines through CardioTwin, a gated fusion deep learning digital twin.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Data Access](#data-access)
- [Data Extraction](#data-extraction)
- [Installation](#installation)
- [CLI Quick Reference](#cli-quick-reference)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [XGBoost Models](#xgboost-models)
  - [MLP Models](#mlp-models)
  - [CardioTwin](#cardiotwin)
- [Models](#models)
  - [XGBoost](#xgboost)
  - [MLP](#mlp)
  - [CardioTwin (Final Model)](#cardiotwin-final-model)
- [Evaluation](#evaluation)
- [Citations](#citations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This repository provides an end-to-end framework for preparing integrated MIMIC-IV datasets for cardiovascular digital twin modeling. It combines data from:

- **MIMIC-IV:** Core hospital and ICU data including admissions, diagnoses, lab events, vital signs, and patient demographics
- **MIMIC-IV-ED:** Emergency department visits, diagnoses, and vitals
- **MIMIC-IV-ECG:** 12-lead electrocardiogram recordings and machine measurements

The pipeline supports both static (EHR/demographic) and temporal (vitals and ECG) preprocessing, enabling streamlined integration for downstream predictive modeling. **This project specifically focuses on the cohort of ED patients with cardiovascular presentations.**

Three tiers of models are included, each building on the last:

- **XGBoost (baseline, weighted, SMOTE, embedding):** Multi-label classifiers predicting 17 cardiovascular ICD-10 diagnosis categories from ECG machine measurements, vital sign statistics, and demographics
- **MLP (baseline, weighted, SMOTE, embedding, embedding_weighted):** Feedforward neural network variants for the same task
- **CardioTwin:** A multimodal gated fusion network combining ECG-FM embeddings (1536-dim), vital sign sequences (LSTM), and EHR features — the primary model of this project

---

## Project Structure

```
.
├── configs/
│   ├── static_preprocessing_params.json    # Config for static/column-based CSV preprocessing
│   ├── ecg_preprocessing_params.json       # Config for ECG signal preprocessing
│   ├── vitals_preprocessing_params.json    # Config for vital signs preprocessing
│   ├── icdcode_extractor_params.json       # Config for ICD code extraction and labeling
│   ├── xgboost_params.json                 # Config for all XGBoost model variants
│   ├── mlp_params.json                     # Config for all MLP model variants
│   ├── cardiotwin_params.json              # Config for CardioTwin full ablation sweep
│   └── CardioTwin_model_params.json        # Config for CardioTwin final model
├── data/
│   ├── raw/                                # Raw MIMIC-IV input files (CSVs, WFDB ECG records)
│   └── processed/                          # Output of all preprocessing scripts
├── model_results/
│   ├── xgboost/
│   │   ├── xgboost_baseline/               # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── xgboost_weighted/               # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── xgboost_smote/                  # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── xgboost_embedding/              # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── xgboost_overall_results_ablation.csv   # One aggregate-metrics row per variant
│   │   └── xgboost_results_ablation.csv           # 17 per-label metric rows per variant
│   ├── mlp/
│   │   ├── mlp_baseline/                   # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── mlp_weighted/                   # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── mlp_smote/                      # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── mlp_embedding/                  # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── mlp_embedding_weighted/         # Plots, per-label CSV, and overall CSV for this variant
│   │   ├── mlp_overall_results_ablation.csv        # One aggregate-metrics row per variant
│   │   └── mlp_results_ablation.csv                # 17 per-label metric rows per variant
│   ├── cardio_digital_twin/                # Full ablation sweep (4 variants x 3 losses x 2 samplers)
│   │   ├── cardio_digital_twin_baseline_bce_none/         # enc_dim=128, gated fusion, plain BCE, no sampler
│   │   ├── cardio_digital_twin_baseline_bce_weighted/     # enc_dim=128, gated fusion, BCE + pos_weight, no sampler
│   │   ├── cardio_digital_twin_baseline_bce_none_weighted/     # enc_dim=128, gated fusion, plain BCE, weighted sampler
│   │   ├── cardio_digital_twin_baseline_bce_weighted_weighted/ # enc_dim=128, gated fusion, BCE + pos_weight, weighted sampler
│   │   ├── cardio_digital_twin_baseline_focal_none/       # enc_dim=128, gated fusion, focal loss, no sampler
│   │   ├── cardio_digital_twin_baseline_focal_weighted/   # enc_dim=128, gated fusion, focal loss, weighted sampler
│   │   ├── cardio_digital_twin_nogate_*/                  # enc_dim=128, mean-pool fusion (gate ablation), same loss/sampler combos
│   │   ├── cardio_digital_twin_medium_*/                  # enc_dim=256, gated fusion, same loss/sampler combos
│   │   ├── cardio_digital_twin_large_*/                   # enc_dim=512, gated fusion, same loss/sampler combos
│   │   ├── cardio_digital_twin_overall_results_ablation.csv   # One aggregate-metrics row per variant/loss/sampler combo
│   │   └── cardio_digital_twin_results_ablation.csv           # 17 per-label metric rows per variant/loss/sampler combo
│   └── CardioTwin/                         # Final model outputs
│       ├── CardioTwin.pt                   # Saved model checkpoint
│       ├── CardioTwin_roc_curves.png       # ROC curves across all 17 labels
│       ├── CardioTwin_pr_curves.png        # Precision-recall curves across all 17 labels
│       ├── CardioTwin_confusion_matrix.png # Aggregated binary confusion matrix
│       ├── CardioTwin_cooccurrence_matrix.png  # Label co-occurrence heatmap
│       ├── CardioTwin_kfold_loss_curves.png    # 3-fold cross-validation loss curves
│       ├── CardioTwin_results.csv          # Per-label metrics
│       ├── CardioTwin_overall_results.csv  # Aggregate metrics
│       └── CardioTwin_trajectories/        # Per-patient vital trajectory simulations
├── notebooks/
│   ├── static_preprocessing.ipynb         # Notebook for static preprocessing development
│   ├── ecg_preprocessing.ipynb            # Notebook for ECG preprocessing development
│   ├── vitals_preprocessing.ipynb         # Notebook for vitals preprocessing development
│   ├── icd_extraction.ipynb               # Notebook for ICD code extraction development
│   ├── xgboost_baseline.ipynb             # Notebook for XGBoost baseline development
│   └── misc/                              # Miscellaneous exploration notebooks
├── src/
│   ├── preprocessing/
│   │   ├── static_preprocessing.py        # Preprocesses static/column-based MIMIC-IV tables
│   │   ├── ecg_preprocessing.py           # Preprocesses raw WFDB ECG signals
│   │   ├── vitals_preprocessing.py        # Preprocesses ED vital sign time series
│   │   ├── icd_entity_extraction.py       # Extracts cardiovascular ICD-10 labels per encounter
│   │   ├── icd_code_labels.py             # ICD-10 to cardiovascular category label mappings
│   │   └── machine_measurements_labels.py # Label mappings for ECG machine measurements
│   └── models/
│       ├── tabular_utils.py               # Shared data loading, feature engineering, evaluation, and ablation CSV writes
│       ├── ecg_fm.py                      # ECG-FM checkpoint loading and batched embedding extraction
│       ├── xgboost.py                     # XGBoost baseline, weighted, and SMOTE variant pipelines
│       ├── xgboost_embedding.py           # XGBoost pipeline with ECG-FM 1536-dim embeddings
│       ├── mlp.py                         # MLP baseline, weighted, SMOTE, and embedding variant pipelines
│       ├── cardio_digital_twin_classes.py # CardioEDDataset, collate_fn, CardioTwinED and NoGate model classes
│       ├── cardio_digital_twin_utils.py   # Training loops, evaluation, trajectory simulation, and feature builders
│       ├── cardio_digital_twin.py         # Variant registry, run_cardiotwin_pipeline, and run_cardiotwin_ablation_pipeline
│       ├── ecgfm_pretrained.pt            # ECG-FM Model
│       └── CardioTwin.py                  # Final model entry point: 128-dim gated fusion, BCE loss, no sampler
├── run.py                                 # CLI runner for all preprocessing and model pipelines
├── environment.yml                        # Conda environment and dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Project overview and setup instructions
```

---

## Prerequisites

- **Python:** 3.9
- **PhysioNet Access:** Credentialed access to all MIMIC-IV datasets
- **Required Training:** CITI "Data or Specimens Only Research" certification
- **Google BigQuery:** Optional but recommended for efficient data extraction

---

## Data Access

This project requires access to multiple MIMIC-IV datasets.

### 🔓 Unrestricted (No Credentialing Required)

- **MIMIC-IV-ECG v1.0:** https://physionet.org/content/mimic-iv-ecg/1.0/
  Download `machine_measurements.csv` and `record_list.csv` (rename to `ecg_record_list.csv`).

MIMIC-IV-ECG contains 800,000+ diagnostic ECGs matched to hospital admissions, including 12-lead waveform data at 500 Hz (WFDB format), automated interval measurements (PR, QRS, QT, RR), axis calculations, machine-generated ECG findings, and timing linked to ED and hospital stay identifiers.

### 🔒 Restricted-Access Datasets

These require PhysioNet credentialing (CITI training + data use agreement):

- **MIMIC-IV v3.1:** https://physionet.org/content/mimiciv/3.1/
- **MIMIC-IV-ED v2.2:** https://physionet.org/content/mimic-iv-ed/2.2/

**Steps to obtain access:**

1. Complete the CITI "Data or Specimens Only Research" course: https://about.citiprogram.org/
2. Create a PhysioNet account: https://physionet.org/register/
3. Request access to each dataset (separate applications required)
4. Sign the PhysioNet Credentialed Health Data Use Agreement for each
5. Once approved, proceed to Data Extraction below

> Approval typically takes a few business days. You must be approved for all three datasets to use this pipeline.

---

## Data Extraction

### Method 1: Google BigQuery (Recommended)

All queries filter for subjects with ECG records in MIMIC-IV-ECG.

#### 1. Hospital Diagnoses

```sql
-- Save as: diagnoses_icd.csv
SELECT di.*, did.long_title
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS di
JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` AS did
  ON di.icd_code = did.icd_code AND di.icd_version = did.icd_version
WHERE di.subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 2. Hospital Admissions

```sql
-- Save as: admissions.csv
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 3. Patient Demographics

```sql
-- Save as: patients.csv
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.patients`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 4. ECG Record List

```sql
-- Save as: ecg_record_list.csv
SELECT * FROM `physionet-data.mimiciv_ecg.record_list`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 5. ICU Stays

```sql
-- Save as: icustays.csv
SELECT * FROM `physionet-data.mimiciv_3_1_icu.icustays`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 6. ED Diagnoses

```sql
-- Save as: ed_diagnosis.csv
SELECT * FROM `physionet-data.mimiciv_ed.diagnosis`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 7. ED Stays

```sql
-- Save as: edstays.csv
SELECT * FROM `physionet-data.mimiciv_ed.edstays`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 8. ED Vitals

```sql
-- Save as: ed_vitals.csv
SELECT subject_id, stay_id, charttime, temperature, heartrate, resprate, o2sat, sbp, dbp
FROM `physionet-data.mimiciv_ed.vitalsign`
WHERE subject_id IN (
  SELECT DISTINCT subject_id FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
ORDER BY charttime
```

Export each result as a CSV from BigQuery and place all files in `data/raw/`. Required filenames:

- `admissions.csv`, `diagnoses_icd.csv`, `patients.csv`, `icustays.csv`
- `edstays.csv`, `ed_diagnosis.csv`, `ed_vitals.csv`
- `machine_measurements.csv`, `ecg_record_list.csv`

### Method 2: Direct Download

Download the complete datasets from PhysioNet, extract the relevant CSVs to `data/raw/`, and the preprocessing pipeline will filter for ECG patients automatically.

### ECG Waveforms

MIMIC-IV-ECG waveform files are required separately from the CSV tables above. Download them via one of the following methods:

**Option A — wget (recommended for servers/DSMLP):**

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-ecg/1.0/
```

**Option B — ZIP download (33.8 GB):** available on the dataset page linked below.

**Option C — Google BigQuery:** request waveform access through PhysioNet's BigQuery integration.

Full dataset page: https://physionet.org/content/mimic-iv-ecg/1.0/

## Waveforms can be stored anywhere on your system however ensure the waveform path in `cardiotwin_params.json`, `CardioTwin_model_params.json`, `mlp_params.json`, and `xgboost_params.json` points to the waveform directory.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/brandonng2/ClinicalDigitalTwin.git
cd ClinicalDigitalTwin
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate ClinicalDigitalTwin
```

### 3. Install ECG-FM

ECG-FM is a foundation model for ECG analysis pre-trained on MIMIC-IV-ECG. It depends on [fairseq-signals](https://github.com/Jwoo5/fairseq-signals), which must be installed from source. Run the following with the `ClinicalDigitalTwin` conda environment activated:

```bash
git clone https://github.com/Jwoo5/fairseq-signals
cd fairseq-signals
pip install --editable ./
```

Then download the pretrained checkpoint from HuggingFace:

https://huggingface.co/wanglab/ecg-fm/tree/main

Download `mimic_iv_ecg_physionet_pretrained.pt`, rename it to `ecgfm_pretrained.pt`, and place it at `src/models/ecgfm_pretrained.pt`.

See the [ECG-FM repository](https://github.com/bowang-lab/ecg-fm) for further implementation details.

---

## CLI Quick Reference

```bash
# ── Full pipeline ────────────────────────────────────────────────────────────
python run.py --all                      # all preprocessing + CardioTwin final model

# ── Preprocessing ────────────────────────────────────────────────────────────
python run.py --preprocess               # all four steps below in sequence
python run.py --static                   # demographic and comorbidity features
python run.py --ecg                      # ECG signal processing and feature extraction
python run.py --vitals                   # ED vital sign time series processing
python run.py --entities                 # cardiovascular ICD-10 label extraction

# ── XGBoost ──────────────────────────────────────────────────────────────────
python run.py --xgboost-baseline         # normalized, no imbalance handling (best)
python run.py --xgboost-weighted         # per-label scale_pos_weight
python run.py --xgboost-smote            # capped SMOTE on rare labels
python run.py --xgboost-embedding        # ECG-FM 1536-dim embeddings
python run.py --xgboost-ablation         # all four variants in sequence

# ── MLP ──────────────────────────────────────────────────────────────────────
python run.py --mlp-baseline             # uniform BCE, no imbalance handling
python run.py --mlp-weighted             # per-label BCEWithLogitsLoss pos_weight
python run.py --mlp-smote                # capped SMOTE on rare labels
python run.py --mlp-embedding            # ECG-FM 1536-dim embeddings
python run.py --mlp-embedding-weighted   # ECG-FM embeddings + pos_weight
python run.py --mlp-ablation             # all five variants in sequence

# ── CardioTwin ───────────────────────────────────────────────────────────────
python run.py --cardiotwin               # final model: 128-dim, gated fusion, BCE, no sampler
python run.py --cardiotwin-ablation      # 4 variants x 3 loss types x 2 samplers = 24 runs
```

---

## Usage

### Preprocessing

Each step reads its configuration from the corresponding JSON in `configs/` and writes processed files to `data/processed/`.

```bash
python run.py --static      # static/column-based MIMIC-IV tables
python run.py --ecg         # WFDB ECG signal processing
python run.py --vitals      # ED vital sign time series
python run.py --entities    # ICD-10 cardiovascular label extraction
python run.py --preprocess  # all four steps above
```

### XGBoost Models

All variants apply StandardScaler normalization fit on the training set only. Running `--xgboost-ablation` sequences all four and upserts results into the top-level ablation CSVs after each run.

| Flag                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `--xgboost-baseline`  | No imbalance handling — best overall results                  |
| `--xgboost-weighted`  | Per-label `scale_pos_weight = n_neg / n_pos`                  |
| `--xgboost-smote`     | Capped SMOTE (max 15% prevalence per label)                   |
| `--xgboost-embedding` | Replaces derived ECG features with ECG-FM 1536-dim embeddings |
| `--xgboost-ablation`  | Runs all four above in sequence                               |

> `--xgboost-weighted` and `--xgboost-smote` are retained for benchmarking. Both underperform the baseline on this dataset — weighted loss over-corrects on severely imbalanced labels, and SMOTE degrades performance due to the high proportion of binary features.

### MLP Models

Feedforward network (256 → 128 → 64) with BatchNorm, Dropout, and BCEWithLogitsLoss. Features are StandardScaler normalized; early stopping uses a 10% validation split.

| Flag                       | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `--mlp-baseline`           | Uniform BCE, no imbalance handling             |
| `--mlp-weighted`           | Per-label `pos_weight`                         |
| `--mlp-smote`              | Capped SMOTE on rare labels                    |
| `--mlp-embedding`          | ECG-FM 1536-dim embeddings as additional input |
| `--mlp-embedding-weighted` | ECG-FM embeddings + per-label `pos_weight`     |
| `--mlp-ablation`           | Runs all five above in sequence                |

### CardioTwin

```bash
# Final model — same as the model step in --all
python run.py --cardiotwin

# Full ablation sweep
# Variants:    baseline (enc=128, gated), nogate (enc=128, mean-pool),
#              medium (enc=256), large (enc=512)
# Loss types:  bce, bce_weighted, focal
# Samplers:    none, weighted (WeightedRandomSampler on rare-label stays)
python run.py --cardiotwin-ablation
```

---

## Models

### XGBoost

Gradient boosted decision trees with a `MultiOutputClassifier` wrapper — one `XGBClassifier` per label. Inputs: ECG machine measurements, vital sign statistics over a 4-hour pre-ECG window, and patient demographics. Max depth 5, learning rate 0.1, 100 estimators per label, 80/20 patient-stratified split.

### MLP

Three hidden layers (256 → 128 → 64) with BatchNorm, Dropout, and BCEWithLogitsLoss. Same input feature set as XGBoost. Embedding variants replace or augment derived ECG features with ECG-FM 1536-dim pooled embeddings (two half-segment encodings concatenated).

### CardioTwin (Final Model)

A multimodal gated fusion network with three parallel input streams:

| Stream                    | Encoder                                           | Output Dim |
| ------------------------- | ------------------------------------------------- | ---------- |
| Vital signs (time series) | LSTM + statistics MLP, fused                      | 128        |
| ECG-FM embeddings         | Attention pool over up to 2 ECGs + projection MLP | 128        |
| EHR / demographics        | Shallow MLP                                       | 128        |

A learned gate network produces per-patient soft weights over the three 128-dim encodings. The weighted sum passes through a fusion MLP (128 → 256 → 128) to a diagnosis head (128 → 17 sigmoid logits). Gate weights are saved per patient and interpretable as modality importance scores.

**Ablation dimensions:**

| Dimension    | Options                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------- |
| Architecture | `baseline` (enc=128, gated), `nogate` (enc=128, mean-pool), `medium` (enc=256), `large` (enc=512) |
| Loss         | `bce`, `bce_weighted` (pos_weight), `focal`                                                       |
| Sampler      | `none`, `weighted` (WeightedRandomSampler on rare-label stays)                                    |

---

## Evaluation

All models produce the following outputs per variant under `model_results/{family}/{variant}/`:

- **Per-label results CSV** — ROC-AUC, PR-AUC, precision, recall, F1, accuracy for each of the 17 diagnosis labels
- **Overall results CSV** — Aggregate metrics across all labels
- **ROC curves** — All 17 labels overlaid, color-coded by performance tier
- **PR curves** — All 17 labels overlaid, color-coded by performance tier
- **Confusion matrix** — Summed binary predictions across all labels
- **Co-occurrence matrix** — Which labels does the model predict together?

After each run, a row is upserted into the top-level ablation CSVs (`{family}_overall_results_ablation.csv` and `{family}_results_ablation.csv`) so all variants can be compared side by side.

---

## Citations

### ECG-FM

```
McKeen, S., Patel, N., Girgis, H., Emam, T., Cianflone, N., Tsang, T., Gibson, W.,
McIntyre, W., Rayner-Kay, H., & Wang, B. (2024). ECG-FM: An Open Electrocardiogram
Foundation Model. arXiv preprint arXiv:2408.05178.
https://github.com/bowang-lab/ecg-fm
```

### MIMIC-IV

```
Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S.,
Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet.
https://doi.org/10.13026/kpb9-mt58
```

### MIMIC-IV-ECG

```
Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C.,
Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Berkowitz, S., Moukheiber, D.,
Chiu, E., Rosman, J., Ghassemi, M. M., Horng, S., Celi, L. A., & Mark, R. (2023).
MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet.
https://doi.org/10.13026/6mm1-ek67
```

### MIMIC-IV-ED

```
Gaichies, E., Jang, J., Aczon, M., Leu, M., Garcia, J., Rodricks, J., Girkar, U.,
Murray, H., Brenner, L., Hamilton, P., Alpern, E., Moody, B., Pollard, T.,
Johnson, A. E. W., Celi, L. A., Mark, R. G., & Badawi, O. (2024). MIMIC-IV-ED
(version 2.2). PhysioNet. https://doi.org/10.13026/77mm-fy28
```

### Original MIMIC-IV Publication

```
Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible
electronic health record dataset. Sci Data 10, 1 (2023).
https://doi.org/10.1038/s41597-022-01899-x
```

### PhysioNet

```
Goldberger, A., Amaral, L., Glass, L., et al. (2000). PhysioBank, PhysioToolkit,
and PhysioNet: Components of a new research resource for complex physiologic signals.
Circulation, 101(23), e215–e220.
```

---

## License

This project uses data from multiple MIMIC-IV datasets, all licensed under the **PhysioNet Credentialed Health Data License v1.5.0**. By using this repository, you agree to:

- Comply with all PhysioNet data use agreements
- Follow institutional or IRB requirements for de-identified patient data
- Use data for research and educational purposes only (no commercial use)
- Not attempt to identify individuals or institutions in the data
- Not share access to the data with unauthorized parties
- Maintain appropriate data security measures

**Data Use Agreement:** https://physionet.org/content/mimiciv/view-dua/3.1/
**Full License:** https://physionet.org/content/mimiciv/view-license/3.1/

---

## Acknowledgments

This project was built using the MIMIC-IV dataset provided by the MIT Laboratory for Computational Physiology.
