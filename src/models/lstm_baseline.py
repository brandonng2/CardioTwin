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
    def __init__(self, vital_dim, static_dim, num_classes, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=vital_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, vitals, static):
        _, (hn, _) = self.lstm(vitals)
        v_feat = hn[-1]
        s_feat = self.static_net(static)
        combined = torch.cat((v_feat, s_feat), dim=1)
        return self.classifier(combined)


# -------------------------------------------------------
# 3️⃣ DATASET
# -------------------------------------------------------

class TwinDataset(Dataset):
    def __init__(self, vitals, static, labels):
        self.vitals = torch.FloatTensor(vitals)
        self.static = torch.FloatTensor(static)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vitals[idx], self.static[idx], self.labels[idx]


# -------------------------------------------------------
# 4️⃣ MAIN PIPELINE
# -------------------------------------------------------

def run_lstm_baseline_pipeline(in_dir, config_path, out_path, target_type="labels"):

    print("\nRunning LSTM Baseline model...\n")

    config = load_config(config_path)
    ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)

    ed_encounters = filter_ed_encounters(clinical_encounters)
    ed_ecg_records = filter_ed_ecg_records(ecg_records)
    earliest_ecgs = extract_earliest_ecg_per_stay(ed_ecg_records)

    model_df = create_model_df(ed_encounters,
                               aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs))

    # ---------------- VITAL SEQUENCES ----------------
    vitals_3d, patient_list = create_vitals_sequence(ed_vitals, earliest_ecgs)

    scaler = StandardScaler()
    N, T, F = vitals_3d.shape
    vitals_scaled = scaler.fit_transform(vitals_3d.reshape(-1, F)).reshape(N, T, F)

    # ---------------- LABELS ----------------
    aligned_df = model_df.set_index('ed_stay_id').reindex(patient_list)

    if isinstance(aligned_df['diagnosis_labels'].iloc[0], str):
        aligned_df['diagnosis_labels'] = aligned_df['diagnosis_labels'].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(aligned_df['diagnosis_labels'])

    subject_ids = aligned_df['subject_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(vitals_scaled, Y, groups=subject_ids))

    X_train, X_test = vitals_scaled[train_idx], vitals_scaled[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]

    # Static features
    static_cols = ['anchor_age']

    # Force numeric conversion
    aligned_df[static_cols] = aligned_df[static_cols].apply(
        pd.to_numeric, errors='coerce'
    )

    X_static = aligned_df[static_cols].fillna(0).astype(np.float32).values

    X_static_train = X_static[train_idx]
    X_static_test = X_static[test_idx]

    train_loader = DataLoader(TwinDataset(X_train, X_static_train, y_train),
                              batch_size=128, shuffle=True)
    test_loader = DataLoader(TwinDataset(X_test, X_static_test, y_test),
                             batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CardiovascularDigitalTwin(
        vital_dim=F,
        static_dim=X_static.shape[1],
        num_classes=Y.shape[1]
    ).to(device)

    pos_counts = torch.tensor(y_train.sum(axis=0))
    neg_counts = len(y_train) - pos_counts
    pos_weights = torch.clamp(neg_counts / (pos_counts + 1e-5), max=50.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------------- TRAIN ----------------
    epochs = 5
    for epoch in range(epochs):
        model.train()
        losses = []

        for v_batch, s_batch, y_batch in train_loader:
            v_batch, s_batch, y_batch = v_batch.to(device), s_batch.to(device), y_batch.to(device)

            outputs = model(v_batch, s_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")

    # ---------------- EVALUATE ----------------
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for v_batch, s_batch, y_batch in test_loader:
            v_batch, s_batch = v_batch.to(device), s_batch.to(device)
            outputs = model(v_batch, s_batch)
            preds.append(outputs.cpu().numpy())
            true.append(y_batch.numpy())

    y_pred = np.vstack(preds)
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