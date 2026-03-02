# src/models/lstm_baseline.py

import sys
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

# 🔁 Reuse preprocessing from XGBoost pipeline
from src.models.xgboost_baseline import (
    load_config,
    load_data_files,
    filter_ed_encounters,
    filter_ed_ecg_records,
    extract_earliest_ecg_per_stay,
    aggregate_vitals_to_ecg_time,
    create_model_df,
)

# -------------------------------------------------------
# 1️⃣ VITALS SEQUENCE CREATION
# -------------------------------------------------------

def create_vitals_sequence(ed_vitals, earliest_ecgs, window_hours=4, step_mins=30):
    vital_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    num_steps = int((window_hours * 60) / step_mins)

    vitals_df = ed_vitals.copy()
    vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'], errors='coerce')

    ecgs_df = earliest_ecgs[['ed_stay_id', 'ecg_time']].copy()
    ecgs_df['ecg_time'] = pd.to_datetime(ecgs_df['ecg_time'], errors='coerce')

    vitals_df = vitals_df.dropna(subset=['charttime', 'stay_id'])
    ecgs_df = ecgs_df.dropna(subset=['ecg_time', 'ed_stay_id'])

    merged = vitals_df.merge(ecgs_df, left_on='stay_id', right_on='ed_stay_id')

    start_time = merged['ecg_time'] - pd.Timedelta(hours=window_hours)
    mask = (merged['charttime'] > start_time) & (merged['charttime'] <= merged['ecg_time'])
    filt = merged[mask].copy()

    time_deltas = (filt['charttime'] - (filt['ecg_time'] - pd.Timedelta(hours=window_hours)))
    filt['bucket'] = time_deltas.dt.total_seconds() // (step_mins * 60)
    filt = filt.dropna(subset=['bucket'])
    filt['bucket'] = filt['bucket'].astype(int).clip(0, num_steps - 1)

    grouped = filt.groupby(['ed_stay_id', 'bucket'])[vital_cols].mean()

    final_3d_list = []
    for col in vital_cols:
        pivot = grouped[col].unstack(level='bucket')
        pivot = pivot.reindex(columns=range(num_steps))
        pivot = pivot.ffill(axis=1).bfill(axis=1).fillna(0)
        final_3d_list.append(pivot.values)

    vitals_3d = np.stack(final_3d_list, axis=-1)
    return vitals_3d, pivot.index.tolist()


# -------------------------------------------------------
# 2️⃣ MODEL
# -------------------------------------------------------

class CardiovascularDigitalTwin(nn.Module):
    def __init__(self,
                 vital_dim,
                 num_classes,
                 num_race,
                 num_gender,
                 hidden_dim=256):

        super().__init__()

        # -------- LSTM --------
        self.lstm = nn.LSTM(
            input_size=vital_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        

        # -------- Attention --------
        self.attn = nn.Linear(hidden_dim, 1)
        self.post_attn_dropout = nn.Dropout(0.3)

        # -------- Embeddings --------
        self.race_emb = nn.Embedding(num_race, 8)
        self.gender_emb = nn.Embedding(num_gender, 4)

        # -------- Static Network --------
        self.static_net = nn.Sequential(
            nn.Linear(8 + 4 + 1, 64),  # race + gender + age
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        self.ecg_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # -------- Final Classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, vitals, age, race, gender, ecg):

        # ---- LSTM ----
        outputs, _ = self.lstm(vitals)  # (B, T, H)

        # ---- Attention ----
        attn_weights = torch.softmax(self.attn(outputs), dim=1)
        #v_feat = torch.sum(attn_weights * outputs, dim=1)
        v_feat = torch.sum(attn_weights * outputs, dim=1)
        v_feat = self.post_attn_dropout(v_feat)

        # ---- Embeddings ----
        race_feat = self.race_emb(race)
        gender_feat = self.gender_emb(gender)

        age = age.unsqueeze(1)  # (B,1)

        static_combined = torch.cat((race_feat, gender_feat, age), dim=1)
        s_feat = self.static_net(static_combined)
        e_feat = self.ecg_net(ecg)

        combined = torch.cat((v_feat, s_feat, e_feat), dim=1)

        return self.classifier(combined)

# -------------------------------------------------------
# 3️⃣ DATASET
# -------------------------------------------------------

class TwinDataset(Dataset):
    def __init__(self, vitals, age, race, gender, ecg_embeddings, labels):
        self.vitals = torch.FloatTensor(vitals)
        self.age = torch.FloatTensor(age)
        self.race = torch.LongTensor(race)
        self.gender = torch.LongTensor(gender)
        self.ecg_embeddings = torch.FloatTensor(ecg_embeddings)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.vitals[idx],          # (8, 6)
            self.age[idx],             # (1,)
            self.race[idx],            # scalar
            self.gender[idx],          # scalar
            self.ecg_embeddings[idx],  # (512,)
            self.labels[idx]           # (num_classes,)
        )


# -------------------------------------------------------
# 4️⃣ MAIN PIPELINE
# -------------------------------------------------------

def run_lstm_baseline_pipeline(in_dir, config_path, out_path, target_type="labels"):
    from src.models.xgboost_baseline import extract_embeddings_for_pipeline_batch_10s
    print("\nRunning LSTM Baseline model...\n")

    config = load_config(config_path)
    ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)

    ed_encounters = filter_ed_encounters(clinical_encounters)
    ed_ecg_records = filter_ed_ecg_records(ecg_records)
    earliest_ecgs = extract_earliest_ecg_per_stay(ed_ecg_records)
          # ---------------- ECG EMBEDDINGS ----------------
    print("\nExtracting ECG-FM embeddings...\n")
    earliest_ecgs = extract_embeddings_for_pipeline_batch_10s(earliest_ecgs, config, batch_size=64)
    ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs)
    model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)

  

    #ecg_df = extract_embeddings_for_pipeline_batch_10s(
        #earliest_ecgs.copy(),
        #config,
        #batch_size=64
    #)

    #model_df = create_model_df(ed_encounters,
                               #aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs))

    # ---------------- VITAL SEQUENCES ----------------
    #vitals_3d, patient_list = create_vitals_sequence(ed_vitals, earliest_ecgs)
    #ecg_aligned = (
      #  ecg_df
       # .set_index("ed_stay_id")
        #.reindex(patient_list)
    #)

    #emb_cols = [f"emb_{i}" for i in range(512)]
    #ecg_embeddings = ecg_aligned[emb_cols].values.astype(np.float32)
    
    vitals_3d, patient_list = create_vitals_sequence(ed_vitals, earliest_ecgs)
    
    # ---------------- ALIGN ECG EMBEDDINGS ----------------
    ecg_df = earliest_ecgs.set_index("ed_stay_id")
    emb_cols = [f"emb_{i}" for i in range(512)]
    missing_ids = [pid for pid in patient_list if pid not in ecg_df.index]
    if missing_ids:
        print(f"⚠ Warning: {len(missing_ids)} patients missing ECG embeddings. Filling zeros.")
        zero_emb = np.zeros(512, dtype=np.float32)
        ecg_embeddings = np.array([
            ecg_df.loc[pid, emb_cols].values.astype(np.float32) if pid in ecg_df.index else zero_emb
            for pid in patient_list
        ])
    else:
        ecg_embeddings = ecg_df.reindex(patient_list)[emb_cols].values.astype(np.float32)
    print(f"ECG embeddings aligned: {ecg_embeddings.shape}")

    #scaler = StandardScaler()
    #N, T, F = vitals_3d.shape
    #vitals_scaled = scaler.fit_transform(vitals_3d.reshape(-1, F)).reshape(N, T, F)

    # ---------------- LABELS ----------------
    #aligned_df = model_df.set_index('ed_stay_id').reindex(patient_list)
    model_df = model_df.set_index('ed_stay_id')

    common_ids = [pid for pid in patient_list if pid in model_df.index]

    print(f"Patients in vitals: {len(patient_list)}")
    print(f"Patients in model_df: {len(model_df)}")
    print(f"Patients in intersection: {len(common_ids)}")

    # Filter everything to common_ids
    idx_map = [patient_list.index(pid) for pid in common_ids]

    #vitals_scaled = vitals_scaled[idx_map]
    #ecg_embeddings = ecg_embeddings[idx_map]
    #patient_list = common_ids
    vitals_3d = vitals_3d[idx_map]
    ecg_embeddings = ecg_embeddings[idx_map]
    patient_list = common_ids

    aligned_df = model_df.loc[common_ids]
    print("ECG min:", np.nanmin(ecg_embeddings))
    print("ECG max:", np.nanmax(ecg_embeddings))
    print("ECG NaN count:", np.isnan(ecg_embeddings).sum())

    # Replace NaNs if present
    ecg_embeddings = np.nan_to_num(ecg_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(aligned_df['diagnosis_labels'].iloc[0], str):
        aligned_df['diagnosis_labels'] = aligned_df['diagnosis_labels'].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(aligned_df['diagnosis_labels'])

    subject_ids = aligned_df['subject_id'].values
    

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #train_idx, test_idx = next(gss.split(vitals_scaled, Y, groups=subject_ids))
    train_idx, test_idx = next(gss.split(vitals_3d, Y, groups=subject_ids))
    
    N, T, F = vitals_3d.shape

    # Split FIRST (unscaled)
    X_train_raw = vitals_3d[train_idx]
    X_test_raw  = vitals_3d[test_idx]

    # Fit scaler on TRAIN only
    vital_scaler = StandardScaler()
    X_train = vital_scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(-1, T, F)

    X_test = vital_scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(-1, T, F)

    #X_train, X_test = vitals_scaled[train_idx], vitals_scaled[test_idx]
    #ecg_train = ecg_embeddings[train_idx]
    #ecg_test  = ecg_embeddings[test_idx]
    # Fit ECG scaler on TRAIN only
    ecg_scaler = StandardScaler()
    ecg_train = ecg_scaler.fit_transform(ecg_embeddings[train_idx])
    ecg_test  = ecg_scaler.transform(ecg_embeddings[test_idx])
    y_train, y_test = Y[train_idx], Y[test_idx]
    # ----------------------------------------------------
    # Weighted Sampler for Rare Labels
    # ----------------------------------------------------

    label_freq = y_train.sum(axis=0)
    inv_freq = 1.0 / (label_freq + 1e-6)

    sample_weights = (y_train * inv_freq).sum(axis=1)
    sample_weights = sample_weights + 1e-3  # avoid zeros

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Static features
    # ----------------------------------------------------
    # STATIC FEATURE PROCESSING (AGE + RACE + GENDER)
    # ----------------------------------------------------

    # ---- AGE (numeric, normalized) ----
    aligned_df['anchor_age'] = pd.to_numeric(
        aligned_df['anchor_age'], errors='coerce'
    )

    aligned_df['anchor_age'] = aligned_df['anchor_age'].fillna(
        aligned_df['anchor_age'].median()
    )

    # Standardize age (important for NN)
    age_train_vals = aligned_df.iloc[train_idx]['anchor_age']
    age_mean = age_train_vals.mean()
    age_std = age_train_vals.std() + 1e-8

    aligned_df['anchor_age'] = (aligned_df['anchor_age'] - age_mean) / age_std


    # ---- RACE (categorical → embedding index) ----
    aligned_df['race'] = aligned_df['race'].fillna("UNKNOWN")
    aligned_df['race'] = aligned_df['race'].astype('category')
    aligned_df['race_code'] = aligned_df['race'].cat.codes


    # ---- GENDER (categorical → embedding index) ----
    aligned_df['gender'] = aligned_df['gender'].fillna("UNKNOWN")
    aligned_df['gender'] = aligned_df['gender'].astype('category')
    aligned_df['gender_code'] = aligned_df['gender'].cat.codes


    # Number of embedding categories (needed for model init)
    num_race = aligned_df['race_code'].nunique()
    num_gender = aligned_df['gender_code'].nunique()
    print(f"Race categories: {num_race}, Gender categories: {num_gender}")

    # ---- Convert to numpy ----
    age = aligned_df['anchor_age'].values.astype(np.float32)
    race = aligned_df['race_code'].values.astype(np.int64)
    gender = aligned_df['gender_code'].values.astype(np.int64)


    # ---- Split using same indices ----
    age_train, age_test = age[train_idx], age[test_idx]
    race_train, race_test = race[train_idx], race[test_idx]
    gender_train, gender_test = gender[train_idx], gender[test_idx]
    print(f"Shapes => X_train: {X_train.shape}, ECG_train: {ecg_train.shape}, y_train: {y_train.shape}")
    print("Label matrix shape:", Y.shape)
    print("Total positives:", Y.sum())
    print("Min positives per class:", Y.sum(axis=0).min())
    print("Max positives per class:", Y.sum(axis=0).max())
    print("NaNs in vitals:", np.isnan(vitals_3d).sum())
    print("NaNs in ECG:", np.isnan(ecg_embeddings).sum())
    print("NaNs in age:", np.isnan(age).sum())
    print("✅ Preprocessing complete. Ready to train the model.")

    train_loader = DataLoader(
            TwinDataset(
            X_train,
            age_train,
            race_train,
            gender_train,
            ecg_train,
            y_train
        ),
        batch_size=128,
        sampler=sampler
    )

    val_loader = DataLoader(
            TwinDataset(
            X_test,
            age_test,
            race_test,
            gender_test,
            ecg_test,
            y_test
        ),
        batch_size=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    model = CardiovascularDigitalTwin(
        vital_dim=F,
        num_classes=Y.shape[1],
        num_race=num_race,
        num_gender=num_gender,
        hidden_dim=256
    ).to(device)

    pos_counts = torch.tensor(y_train.sum(axis=0))
    neg_counts = len(y_train) - pos_counts
    pos_weights = torch.clamp(neg_counts / (pos_counts + 1e-6), max=30.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,   # was 1e-3
        weight_decay=1e-4
    )

    # ---------------- TRAIN ----------------
# ---------------- TRAIN ----------------
    epochs = 40
    patience = 6
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs):

    # ---------------- TRAINING ----------------
        model.train()
        train_losses = []

        for v_batch, age_batch, race_batch, gender_batch, ecg_batch, y_batch in train_loader:

            v_batch = v_batch.to(device)
            age_batch = age_batch.to(device)
            race_batch = race_batch.to(device)
            gender_batch = gender_batch.to(device)
            ecg_batch = ecg_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(v_batch, age_batch, race_batch, gender_batch, ecg_batch)
            loss = criterion(outputs, y_batch)

            if torch.isnan(loss):
                print("Loss became NaN!")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for v_batch, age_batch, race_batch, gender_batch, ecg_batch, y_batch in val_loader:

                v_batch = v_batch.to(device)
                age_batch = age_batch.to(device)
                race_batch = race_batch.to(device)
                gender_batch = gender_batch.to(device)
                ecg_batch = ecg_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(v_batch, age_batch, race_batch, gender_batch, ecg_batch)
                val_loss = criterion(outputs, y_batch)

                val_losses.append(val_loss.item())

        mean_val_loss = np.mean(val_losses)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Val Loss: {mean_val_loss:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_lstm_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


    # ---------------- LOAD BEST MODEL ----------------
    model.load_state_dict(torch.load("best_lstm_model.pt", map_location=device))

        # ---------------- EVALUATE ----------------
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for v_batch, age_batch, race_batch, gender_batch, ecg_batch, y_batch in val_loader:
            v_batch = v_batch.to(device)
            age_batch = age_batch.to(device)
            race_batch = race_batch.to(device)
            gender_batch = gender_batch.to(device)
            ecg_batch = ecg_batch.to(device)
            outputs = model(v_batch, age_batch, race_batch, gender_batch, ecg_batch)

            preds.append(outputs.cpu().numpy())
            true.append(y_batch.numpy())

    #y_pred = np.vstack(preds)
    y_logits = np.vstack(preds)
    y_pred = 1 / (1 + np.exp(-y_logits))  # sigmoid
    y_true = np.vstack(true)

    print("\n" + "-"*60)
    print(f"{'Diagnosis':<35} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-"*60)

    for i, label in enumerate(mlb.classes_):
        try:
            roc = roc_auc_score(y_true[:, i], y_pred[:, i])
            pr = average_precision_score(y_true[:, i], y_pred[:, i])
            print(f"{label:<35} | {roc:.3f}      | {pr:.3f}")
        except:
            print(f"{label:<35} | No Positive Samples")

    mean_roc = roc_auc_score(y_true, y_pred, average='macro')
    mean_pr = average_precision_score(y_true, y_pred, average='macro')

    print("-"*60)
    print(f"{'AVERAGE (MACRO)':<35} | {mean_roc:.3f}      | {mean_pr:.3f}")
    print("-"*60)

    print("\n✓ LSTM baseline complete!")

    return mean_roc, mean_pr