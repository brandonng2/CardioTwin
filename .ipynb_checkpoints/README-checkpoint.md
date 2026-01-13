# ClinicalDigitalTwin

A clean, modular pipeline for preprocessing MIMIC-IV data (Hospital, ICU, ED, and ECG) for clinical digital twin modeling.  
**Note:** Temporal preprocessing is still in progress.

## Project Overview

This repository provides an end-to-end framework for preparing integrated MIMIC-IV datasets for clinical digital twin modeling. It combines data from:

- **MIMIC-IV:** Core hospital and ICU data including admissions, diagnoses, procedures, and patient demographics
- **MIMIC-IV-ED:** Emergency department visits and diagnoses
- **MIMIC-IV-ECG:** Electrocardiogram recordings and measurements

The pipeline supports both static (column-based) and temporal (row-based) preprocessing, enabling streamlined integration for downstream analyses and predictive modeling. **This project specifically focuses on the cohort of patients who have ECG records.**

## Project Structure
```
.
├── configs/
│   ├── static_preprocessing.json      # Configuration for static/column-based CSV preprocessing
│   └── temporal_preprocessing.json    # Configuration for temporal/row-based CSV preprocessing
├── data/
│   ├── raw/                           # Raw input data files (e.g., MIMIC-IV CSVs)
│   │   ├── static/                    # Raw static CSVs
│   │   └── temporal/                  # Raw temporal CSVs
│   └── processed/                     # Output of preprocessing scripts
│       ├── static/                    # Processed static CSVs
│       └── temporal/                  # Processed temporal CSVs
├── notebooks/                         # Jupyter notebooks for testing and exploration
│   └── misc/                          # Miscellaneous notebooks
├── src/
│   └── preprocessing/
│       ├── static_preprocessing.py    # Functions to preprocess static/column-based data
        ├── clinical_entity_extraction.py    # Functions to extracted cardiovascular clinical entities from diagnostic data
│       └── temporal_preprocessing.py  # Functions to preprocess temporal/row-based data
├── run.py                             # Main script to execute the preprocessing pipeline
├── requirements.txt                   # Python dependencies
└── README.md

```

## Prerequisites

- **Python:** 3.8 or higher
- **PhysioNet Access:** Credentialed access to all MIMIC-IV datasets
- **Required Training:** CITI "Data or Specimens Only Research" certification
- **Google BigQuery:** Optional but recommended for efficient data extraction

## Data Access

This project requires access to **multiple MIMIC-IV datasets**, all of which are restricted-access resources:

1. **MIMIC-IV v3.1** (Core hospital data): https://physionet.org/content/mimiciv/3.1/
2. **MIMIC-IV-ECG v1.0** (ECG recordings): https://physionet.org/content/mimic-iv-ecg/1.0/
3. **MIMIC-IV-ED v2.2** (Emergency department data): https://physionet.org/content/mimic-iv-ed/2.2/

### Steps to Obtain Access:

1. Complete the CITI "Data or Specimens Only Research" course: https://about.citiprogram.org/
2. Create a PhysioNet account: https://physionet.org/register/
3. Request access to each dataset above (separate applications required)
4. Sign the PhysioNet Credentialed Health Data Use Agreement for each
5. Once approved, proceed to Data Extraction below

**Note:** Approval typically takes a few business days. You must be approved for all three datasets to use this pipeline.

## Data Extraction

### Method 1: Google BigQuery (Recommended)

This project focuses on **patients who have ECG records** in MIMIC-IV. Use these SQL queries in Google BigQuery to extract the relevant subset:

#### 1. Hospital Diagnoses (ICD Codes)
```sql
-- Save as: diagnoses_icd.csv
SELECT
   di.*,
   did.long_title
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS di
JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` AS did
   ON di.icd_code = did.icd_code AND di.icd_version = did.icd_version
WHERE di.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 2. Hospital Admissions
```sql
-- Save as: admissions.csv
SELECT *
FROM `physionet-data.mimiciv_3_1_hosp.admissions` AS a
WHERE a.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 3. Patient Demographics
```sql
-- Save as: patients.csv
SELECT *
FROM `physionet-data.mimiciv_3_1_hosp.patients` AS p
WHERE p.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 4. ECG Record List
```sql
-- Save as: ecg_record_list.csv
SELECT *
FROM `physionet-data.mimiciv_ecg.record_list` AS rl
WHERE rl.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 5. ICU Stays
```sql
-- Save as: icustays.csv
SELECT *
FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
WHERE icu.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 6. Emergency Department Diagnoses
```sql
-- Save as: ed_diagnosis.csv
SELECT *
FROM `physionet-data.mimiciv_ed.diagnosis` AS edd
WHERE edd.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 7. Emergency Department Stays
```sql
-- Save as: edstays.csv
SELECT *
FROM `physionet-data.mimiciv_ed.edstays` AS eds
WHERE eds.subject_id IN (
   SELECT DISTINCT subject_id
   FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

#### 8. Hospital Drgcode
```sql
-- Save as: drgcodes.csv
SELECT *
FROM `physionet-data.mimiciv_3_1_hosp.drgcodes` AS dc
WHERE dc.subject_id IN (
  SELECT DISTINCT subject_id
  FROM `physionet-data.mimiciv_ecg.machine_measurements`
)
```

### Data Placement

After running these queries:
1. Export each result as CSV from BigQuery
2. Place all CSV files in the `data/raw/` directory
3. Ensure filenames match those specified in the SQL comments

**Cohort Size:** All queries filter for patients with ECG records, resulting in a subset of the full MIMIC-IV population.

### Method 2: Direct Download

Alternatively, download complete datasets and filter locally:
1. Download each dataset from PhysioNet (links above)
2. Extract relevant CSV files to `data/raw/`
3. The preprocessing pipeline will automatically filter for ECG patients

**Note:** BigQuery is more efficient for large-scale filtering.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ClinicalDigitalTwin.git
cd ClinicalDigitalTwin
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Data Placement

Ensure all required CSV files are in `data/raw/static` before running preprocessing:
- `admissions.csv`
- `diagnoses_icd.csv`
- `drgcodes.csv`
- `ecg_record_list.csv`
- `ed_diagnosis.csv`
- `edstays.csv`
- `icustays.csv`
- `patients.csv`

## Usage

### Running the Preprocessing Pipeline
```bash
python run.py
```

This executes both static preprocessing, clinical entity extraction, and temporal preprocessing according to configuration files in `configs/`:

- **Static preprocessing:** Demographic, comorbidity, and baseline features from admissions and patient data
- **Clinical Entity Extraction:** Extracting cardiovascular clinical keywords from diagnoses' data (**In Progress**)
- **Temporal preprocessing:** Time-series events including ICU vitals, ED visits, procedures, and ECG measurements (**In Progress**)

### Exploratory Analysis

Use the notebooks in `notebooks/` to explore data distributions and quality and test preprocessing functions before running the full pipeline.

## Citations

When using this project, please cite all relevant MIMIC-IV datasets:

### MIMIC-IV (Core Hospital Data)
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

## License

This project uses data from multiple MIMIC-IV datasets, all licensed under the **PhysioNet Credentialed Health Data License v1.5.0**.

By using this repository, you agree to:

- Comply with all PhysioNet data use agreements
- Follow institutional or IRB requirements for de-identified patient data
- Use data for **research and educational purposes only** (no commercial use)
- Not attempt to identify individuals or institutions in the data
- Not share access to the data with unauthorized parties
- Maintain appropriate data security measures

**All Data Use Agreement:** [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/3.1/)

**Full License:** [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/3.1/)

## Acknowledgments

This project was built using the MIMIC-IV dataset provided by the MIT Laboratory for Computational Physiology.