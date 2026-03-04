# =============================================================================
# CardioTwin — Full Pipeline Skeleton
# Multimodal Cardiovascular State Monitoring from ED Presentations
# =============================================================================
#
# ARCHITECTURE:
#   ECG-FM embeddings  (B, 2, 512)  → Attention Pool → MLP → (B, 128)  ──┐
#   ED Vital Stats     (B, 30)      → MLP            → (B, 128)          ├─→ Gated Fusion → (B, 128) → 17 dx heads
#   EHR Static         (B, 15)      → MLP            → (B, 128)          ──┘
#
# DATA FLOW:
#   ecg_embeddings.csv   → prepare_ecg()            → ecg_dict
#   ed_vitals.csv        → create_vital_features()  → vital_features_df
#   static + triage      → create_ehr_features()    → ehr_df
#   labels.csv           → directly used            → labels_df
#
# TRAJECTORY (digital twin simulation):
#   For each patient, run inference at each vital timestep
#   Plot evolving diagnosis probabilities over ED stay
#
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from functools import partial
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONSTANTS
# =============================================================================

MAX_T      = 12     # max vital timesteps per ED stay
MAX_N      = 2      # max ECGs per ED stay
VITAL_DIM  = 6      # hr, sbp, dbp, resp_rate, spo2, temperature_f
VITAL_STAT = 30     # 5 stats (mean, min, max, std, delta) × 6 vitals
EHR_DIM    = 15     # age + gender + 5 race + 8 triage features
ECG_DIM    = 512    # ECG-FM embedding dimension (256 × 2 segments)
ENC_DIM    = 128    # shared encoder output dimension
HIDDEN_DIM = 256    # fusion MLP hidden dimension
LSTM_HIDDEN = 64    # LSTM hidden size for vital sequences
N_LABELS   = 17     # cardiovascular diagnosis categories
BATCH_SIZE = 256
LR         = 1e-3
EPOCHS     = 50
DROPOUT    = 0.3

VITAL_COLS = [
    "heart_rate", "sbp", "dbp",
    "resp_rate", "spo2", "temperature_f"
]

EMB_COLS = [f"emb_{i}" for i in range(ECG_DIM)]

LABEL_COLS = [
    # TODO: replace with your actual 17 label column names
    "atrial_fibrillation", "heart_failure", "hypertension",
    "myocardial_infarction", "bundle_branch_block",
    "ventricular_tachycardia", "st_depression", "st_elevation",
    "left_ventricular_hypertrophy", "right_ventricular_hypertrophy",
    "av_block", "sinus_bradycardia", "sinus_tachycardia",
    "prolonged_qt", "atrial_flutter", "pericarditis", "cardiomyopathy"
]

TRIAGE_CONTINUOUS = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "pain", "acuity"
]

STATIC_CONTINUOUS = ["anchor_age"]

# =============================================================================
# 1. LOAD RAW DATA
# =============================================================================

def load_data():
    """Load all preprocessed CSVs."""
    static_df    = pd.read_csv("data/processed/static_features.csv")
    triage_df    = pd.read_csv("data/processed/triage.csv")
    ed_vitals_df = pd.read_csv("data/processed/ed_vitals.csv",
                               parse_dates=["charttime"])
    ecg_df       = pd.read_csv("data/processed/ecg_embeddings.csv")
    labels_df    = pd.read_csv("data/processed/labels.csv")
    return static_df, triage_df, ed_vitals_df, ecg_df, labels_df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

# --- 2a. ECG: attention pool over N ECG-FM embeddings ---

def prepare_ecg(ecg_df):
    """
    Group ECG-FM embeddings by subject_id.
    ECG-FM already handles GAP internally (your extraction code does
    .mean(dim=1) over transformer tokens before concatenating segments).
    So each row is already a 512-dim summary vector — no further pooling needed here.

    Output: dict subject_id -> np.array (N, 512), N <= MAX_N
    """
    ecg_dict = {}
    for sid, group in ecg_df.groupby("subject_id"):
        # sort by ecg_time if available to get chronological order
        if "ecg_time" in group.columns:
            group = group.sort_values("ecg_time")
        embs = group[EMB_COLS].values.astype(np.float32)
        ecg_dict[sid] = embs[:MAX_N]   # truncate to MAX_N
    return ecg_dict


# --- 2b. Vital Signs: statistical summary embedding ---

def create_vital_features(ed_vitals_df, scaler=None, fit_scaler=False):
    """
    Convert long-format ED vitals into per-stay statistical summary.

    For each of 6 vitals, compute 5 statistics:
        mean, min, max, std, delta (last - first)
    → 30 features total (VITAL_STAT = 30)

    This is the "Flattened Approach" — best for Gated Fusion with sparse data.
    Delta captures trajectory: is the patient improving or deteriorating?

    Output: DataFrame (subject_id, ed_stay_id, vital_feat_0...vital_feat_29)
            scaler fitted on train set
    """
    df = ed_vitals_df.copy()
    df = df.sort_values(["subject_id", "stay_id", "charttime"])

    # forward fill within stay, then population median
    df[VITAL_COLS] = (
        df.groupby(["subject_id", "stay_id"])[VITAL_COLS]
        .transform(lambda x: x.ffill().bfill())
    )
    pop_medians = df[VITAL_COLS].median()
    df[VITAL_COLS] = df[VITAL_COLS].fillna(pop_medians)

    records = []
    for (sid, stay_id), group in df.groupby(["subject_id", "stay_id"]):
        vals = group[VITAL_COLS].values.astype(np.float32)  # (T, 6)

        v_mean  = vals.mean(axis=0)                          # (6,)
        v_min   = vals.min(axis=0)                           # (6,)
        v_max   = vals.max(axis=0)                           # (6,)
        v_std   = vals.std(axis=0) if len(vals) > 1 else np.zeros(VITAL_DIM)
        v_delta = vals[-1] - vals[0]                         # (6,) last - first

        features = np.concatenate([v_mean, v_min, v_max, v_std, v_delta])  # (30,)
        records.append({
            "subject_id": sid,
            "ed_stay_id": stay_id,
            **{f"vf_{i}": features[i] for i in range(VITAL_STAT)}
        })

    vital_features_df = pd.DataFrame(records)

    # normalize on train set only
    feat_cols = [f"vf_{i}" for i in range(VITAL_STAT)]
    if fit_scaler:
        scaler = StandardScaler()
        vital_features_df[feat_cols] = scaler.fit_transform(
            vital_features_df[feat_cols]
        )
    elif scaler is not None:
        vital_features_df[feat_cols] = scaler.transform(
            vital_features_df[feat_cols]
        )

    return vital_features_df, scaler


def create_vital_sequences(ed_vitals_df, vital_scaler=None, fit_scaler=False):
    """
    Also keep raw sequences for trajectory simulation.
    Used only during inference/visualization, not training.

    Output: dict (subject_id, stay_id) -> np.array (T, 6), T <= MAX_T
    """
    df = ed_vitals_df.copy()
    df = df.sort_values(["subject_id", "stay_id", "charttime"])

    df[VITAL_COLS] = (
        df.groupby(["subject_id", "stay_id"])[VITAL_COLS]
        .transform(lambda x: x.ffill().bfill())
    )
    df[VITAL_COLS] = df[VITAL_COLS].fillna(df[VITAL_COLS].median())

    if fit_scaler:
        vital_scaler = StandardScaler()
        df[VITAL_COLS] = vital_scaler.fit_transform(df[VITAL_COLS])
    elif vital_scaler is not None:
        df[VITAL_COLS] = vital_scaler.transform(df[VITAL_COLS])

    sequences = {}
    for (sid, stay_id), group in df.groupby(["subject_id", "stay_id"]):
        seq = group[VITAL_COLS].values.astype(np.float32)
        sequences[(sid, stay_id)] = seq[:MAX_T]

    return sequences, vital_scaler


# --- 2c. EHR: demographics + triage snapshot ---

def create_ehr_features(static_df, triage_df):
    """
    Combine static demographics + triage snapshot.
    All features are available at ED arrival — zero leakage.

    Features:
        anchor_age (1) + gender_female (1) + race (5) + triage vitals (8) = 15
        EHR_DIM = 15

    Output: DataFrame (subject_id, ed_stay_id, ehr features...)
    """
    df = static_df.copy()

    # demographics
    df["gender_female"] = (df["gender"] == "F").astype(np.float32)
    df["anchor_age"]    = df["anchor_age"].fillna(df["anchor_age"].median())

    # race one-hot
    def clean_race(x):
        x = str(x).upper()
        if "WHITE"    in x: return "WHITE"
        if "BLACK"    in x: return "BLACK"
        if "ASIAN"    in x: return "ASIAN"
        if "HISPANIC" in x: return "HISPANIC"
        return "OTHER"

    df["race_clean"] = df["race"].apply(clean_race)
    race_dummies = pd.get_dummies(df["race_clean"], prefix="race").astype(np.float32)
    for col in ["race_WHITE", "race_BLACK", "race_ASIAN",
                "race_HISPANIC", "race_OTHER"]:
        if col not in race_dummies.columns:
            race_dummies[col] = 0.0

    # triage
    triage = triage_df.copy()
    triage["pain"]   = pd.to_numeric(triage["pain"],   errors="coerce")
    triage["acuity"] = pd.to_numeric(triage["acuity"], errors="coerce")
    for col in TRIAGE_CONTINUOUS:
        triage[col] = triage[col].fillna(triage[col].median())

    # assemble
    ehr = pd.concat([
        df[["subject_id", "ed_stay_id", "anchor_age", "gender_female"]].reset_index(drop=True),
        race_dummies[["race_WHITE", "race_BLACK", "race_ASIAN",
                      "race_HISPANIC", "race_OTHER"]].reset_index(drop=True)
    ], axis=1)

    ehr = ehr.merge(
        triage[["subject_id", "stay_id"] + TRIAGE_CONTINUOUS],
        left_on=["subject_id", "ed_stay_id"],
        right_on=["subject_id", "stay_id"],
        how="left"
    ).drop(columns="stay_id")

    for col in TRIAGE_CONTINUOUS:
        ehr[col] = ehr[col].fillna(triage[col].median())

    return ehr


def normalize_ehr(ehr_train, ehr_val, ehr_test):
    """Normalize continuous EHR cols on train set only."""
    cont_cols = STATIC_CONTINUOUS + TRIAGE_CONTINUOUS
    scaler = StandardScaler()
    ehr_train = ehr_train.copy()
    ehr_val   = ehr_val.copy()
    ehr_test  = ehr_test.copy()
    ehr_train[cont_cols] = scaler.fit_transform(ehr_train[cont_cols])
    ehr_val[cont_cols]   = scaler.transform(ehr_val[cont_cols])
    ehr_test[cont_cols]  = scaler.transform(ehr_test[cont_cols])
    return ehr_train, ehr_val, ehr_test, scaler

# =============================================================================
# 3. DATASET + DATALOADER
# =============================================================================

VITAL_FEAT_COLS = [f"vf_{i}" for i in range(VITAL_STAT)]
EHR_FEAT_COLS   = [
    "anchor_age", "gender_female",
    "race_WHITE", "race_BLACK", "race_ASIAN", "race_HISPANIC", "race_OTHER"
] + TRIAGE_CONTINUOUS


class CardioEDDataset(Dataset):
    def __init__(self, subject_stay_ids, vital_feat_df,
                 ecg_dict, ehr_df, labels_df, vital_sequences=None):
        """
        subject_stay_ids: list of (subject_id, ed_stay_id) tuples
        vital_feat_df:    DataFrame with VITAL_STAT features per stay
        ecg_dict:         dict subject_id -> np.array (N, 512)
        ehr_df:           DataFrame with EHR_DIM features per stay
        labels_df:        DataFrame with N_LABELS columns per stay
        vital_sequences:  dict (subject_id, ed_stay_id) -> np.array (T, 6)
        """
        self.ids = subject_stay_ids

        # index by (subject_id, ed_stay_id) for fast lookup
        self.vital_feats = vital_feat_df.set_index(
            ["subject_id", "ed_stay_id"]
        )
        self.ehr    = ehr_df.set_index(["subject_id", "ed_stay_id"])
        self.labels = labels_df.set_index(["subject_id", "ed_stay_id"])
        self.ecg_dict = ecg_dict
        self.vital_sequences = vital_sequences or {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid, stay_id = self.ids[idx]

        # vital features (VITAL_STAT,) = (30,)
        vital_feats = (
            self.vital_feats.loc[(sid, stay_id)][VITAL_FEAT_COLS]
            .values.astype(np.float32)
        )

        # ecg (N, 512) — N varies 1-2
        ecg = self.ecg_dict.get(
            sid,
            np.zeros((1, ECG_DIM), dtype=np.float32)  # fallback if missing
        )

        # ehr (EHR_DIM,) = (15,)
        ehr = (
            self.ehr.loc[(sid, stay_id)][EHR_FEAT_COLS]
            .values.astype(np.float32)
        )

        # labels (N_LABELS,) = (17,)
        labels = (
            self.labels.loc[(sid, stay_id)][LABEL_COLS]
            .values.astype(np.float32)
        )

        # vital sequence (T, 6) for LSTM; T may vary
        vital_seq = self.vital_sequences.get(
            (sid, stay_id),
            np.zeros((1, VITAL_DIM), dtype=np.float32)
        )
        vital_len = vital_seq.shape[0]

        return {
            "vital_feats": vital_feats,
            "vital_seq":   vital_seq,
            "vital_len":   vital_len,
            "ecg":         ecg,
            "ehr":         ehr,
            "labels":      labels
        }


def collate_fn(batch, max_N=MAX_N, max_T=MAX_T):
    """
    Pad ECGs to max_N and vital sequences to max_T per batch.
    Vital features and EHR are already fixed-length vectors.
    """
    ecg_padded = []
    ecg_mask   = []

    for b in batch:
        N = b["ecg"].shape[0]

        e = b["ecg"][:max_N]
        N_clip = e.shape[0]
        e_pad  = np.zeros((max_N, ECG_DIM), dtype=np.float32)
        e_pad[:N_clip] = e
        ecg_padded.append(e_pad)
        ecg_mask.append([True] * N_clip + [False] * (max_N - N_clip))

    # Pad vital sequences to (B, max_T, VITAL_DIM) for LSTM
    vital_lens = torch.tensor([b["vital_len"] for b in batch], dtype=torch.long)
    vital_seqs = [torch.tensor(b["vital_seq"], dtype=torch.float32) for b in batch]
    vital_padded = pad_sequence(
        vital_seqs,
        batch_first=True,
        padding_value=0.0,
    )
    # Truncate or pad to exactly max_T
    if vital_padded.size(1) > max_T:
        vital_padded = vital_padded[:, :max_T]
        vital_lens = torch.clamp(vital_lens, max=max_T)
    elif vital_padded.size(1) < max_T:
        pad_right = torch.zeros(
            vital_padded.size(0),
            max_T - vital_padded.size(1),
            VITAL_DIM,
            dtype=vital_padded.dtype,
        )
        vital_padded = torch.cat([vital_padded, pad_right], dim=1)

    return {
        "vital_feats": torch.tensor(
            np.stack([b["vital_feats"] for b in batch])
        ),                                                    # (B, 30)
        "vital_seq":   vital_padded,                          # (B, max_T, 6)
        "vital_lengths": vital_lens,                           # (B,)
        "ecg":         torch.tensor(np.stack(ecg_padded)),   # (B, 2, 512)
        "ecg_mask":    torch.tensor(ecg_mask, dtype=torch.bool),  # (B, 2)
        "ehr":         torch.tensor(
            np.stack([b["ehr"]    for b in batch])
        ),                                                    # (B, 15)
        "labels":      torch.tensor(
            np.stack([b["labels"] for b in batch])
        )                                                     # (B, 17)
    }

# =============================================================================
# 4. MODEL — Gated Fusion
# =============================================================================

class CardioTwinED(nn.Module):
    """
    Multimodal Gated Fusion Network for Cardiovascular Diagnosis Prediction.

    Three modality-specific encoders project into a shared ENC_DIM space.
    Vitals branch uses both: (1) LSTM over raw vital sequences (T, 6), and
    (2) MLP over statistical summary (30). A gating network learns soft
    weights over modalities per patient.

    Returned gates enable interpretability:
    - Which modality did the model trust most per patient?
    - For AFib patients, does ECG dominate?
    - For hemodynamic instability, do vitals dominate?
    """

    def __init__(
        self,
        vital_stat=VITAL_STAT,    # 30: 5 stats × 6 vitals
        vital_dim=VITAL_DIM,      # 6: raw vital channels per timestep
        ehr_dim=EHR_DIM,          # 15: demographics + triage
        ecg_emb_dim=ECG_DIM,      # 512: ECG-FM embedding
        enc_dim=ENC_DIM,          # 128: shared encoder output
        hidden_dim=HIDDEN_DIM,    # 256: fusion MLP hidden
        lstm_hidden=LSTM_HIDDEN,  # 64: LSTM hidden size
        dropout=DROPOUT,
        n_labels=N_LABELS         # 17: diagnosis categories
    ):
        super().__init__()

        # --- Vital sequence LSTM (PyTorch) ---
        # Input: (B, T, 6) padded; use pack_padded_sequence for variable length
        self.vital_lstm = nn.LSTM(
            input_size=vital_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.vital_lstm_proj = nn.Sequential(
            nn.Linear(lstm_hidden, enc_dim),
            nn.GELU(),
        )

        # --- Vital statistics encoder (MLP) ---
        # Input: (B, 30) statistical summary of ED stay vitals
        self.vital_encoder = nn.Sequential(
            nn.Linear(vital_stat, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, enc_dim),
            nn.GELU()
        )

        # --- Combine LSTM + stats into single vital representation ---
        self.vital_fusion = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- ECG-FM Encoder ---
        # Input: (B, N, 512) — N ECG embeddings per stay
        # Attention pool over N ECGs → single cardiac representation
        # Note: ECG-FM already does GAP over transformer tokens internally
        # (your extraction code: .mean(dim=1) over features)
        # This attention layer learns WHICH ECG in the stay is most informative
        self.ecg_attn = nn.Linear(ecg_emb_dim, 1)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, enc_dim),
            nn.GELU()
        )

        # --- EHR Static Encoder ---
        # Input: (B, 15) — age, gender, race, triage vitals, acuity, pain
        # All available at ED arrival — zero leakage
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, enc_dim),
            nn.GELU()
        )

        # --- Gated Fusion ---
        # Learns how much to trust each modality PER PATIENT
        # "For this patient, ECG is most informative (high cardiac signal)"
        # "For that patient, vitals dominate (hemodynamic instability)"
        # Input: concatenation of all 3 encodings (enc_dim * 3 = 384)
        # Output: 3 soft weights summing to 1 via softmax
        self.gate = nn.Sequential(
            nn.Linear(enc_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
            # softmax applied in forward()
        )

        # --- Fusion MLP ---
        # Input: gated weighted sum (enc_dim = 128)
        # Projects to hidden_dim then to output
        self.fusion = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # --- Output Head ---
        # Multi-label: sigmoid not softmax (patient can have multiple diagnoses)
        self.dx_head = nn.Linear(hidden_dim // 2, n_labels)

    def forward(self, vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask=None):
        """
        vital_feats:    (B, 30)       — statistical vital summary
        vital_seq:      (B, T, 6)     — padded vital sequence
        vital_lengths:  (B,) long     — actual lengths for pack_padded_sequence
        ecg_embs:       (B, N, 512)   — ECG-FM embeddings, N=1 or 2
        ehr_x:          (B, 15)       — static EHR features
        ecg_mask:       (B, N) bool   — True=real ECG, False=padding
        """

        # 1a. Encode vital sequence with LSTM (PyTorch)
        B = vital_feats.size(0)
        if vital_lengths is not None and (vital_lengths > 0).any() and vital_seq.size(1) > 0:
            packed = pack_padded_sequence(
                vital_seq,
                vital_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.vital_lstm(packed)
            lstm_enc = self.vital_lstm_proj(h_n.squeeze(0))   # (B, 128)
        else:
            lstm_enc = self.vital_lstm_proj(
                torch.zeros(B, LSTM_HIDDEN, device=vital_feats.device, dtype=vital_feats.dtype)
            )

        # 1b. Encode vital statistics (MLP)
        stats_enc = self.vital_encoder(vital_feats)          # (B, 128)

        # 1c. Fuse LSTM + stats
        vital_enc = self.vital_fusion(
            torch.cat([lstm_enc, stats_enc], dim=1)
        )                                                     # (B, 128)

        # 2. Encode ECG — attention pool over N ECGs
        # learns which ECG in the stay is most diagnostically relevant
        attn_scores = self.ecg_attn(ecg_embs)                # (B, N, 1)
        if ecg_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~ecg_mask.unsqueeze(-1), float("-inf")
            )
        attn_weights = torch.softmax(attn_scores, dim=1)     # (B, N, 1)
        ecg_pooled   = (attn_weights * ecg_embs).sum(dim=1)  # (B, 512)
        ecg_enc      = self.ecg_encoder(ecg_pooled)          # (B, 128)

        # 3. Encode EHR
        ehr_enc = self.ehr_encoder(ehr_x)                    # (B, 128)

        # 4. Gated fusion — learn modality weights per patient
        concat_all = torch.cat(
            [vital_enc, ecg_enc, ehr_enc], dim=1
        )                                                     # (B, 384)
        gates = F.softmax(
            self.gate(concat_all), dim=1
        )                                                     # (B, 3) sums to 1

        # weighted sum of modality encodings
        fused = (
            gates[:, 0:1] * vital_enc +                      # vital weight
            gates[:, 1:2] * ecg_enc   +                      # ECG weight
            gates[:, 2:3] * ehr_enc                          # EHR weight
        )                                                     # (B, 128)

        # 5. Fusion MLP
        shared = self.fusion(fused)                          # (B, 128)

        # 6. Output
        logits = self.dx_head(shared)                        # (B, 17)

        return {
            "logits": logits,
            "probs":  torch.sigmoid(logits),
            "gates":  gates,   # (B, 3) — return for interpretability
            "latent": shared   # (B, 128) — twin state vector for trajectory
        }

# =============================================================================
# 5. TRAINING
# =============================================================================

def compute_class_weights(labels_df):
    """
    Compute pos_weight for BCEWithLogitsLoss.
    Handles class imbalance — cardiovascular labels are rare events.
    """
    labels = labels_df[LABEL_COLS].values
    n_neg  = (labels == 0).sum(axis=0)
    n_pos  = (labels == 1).sum(axis=0).clip(min=1)
    weights = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    return weights


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(
            batch["vital_feats"].to(device),
            batch["vital_seq"].to(device),
            batch["vital_lengths"].to(device),
            batch["ecg"].to(device),
            batch["ehr"].to(device),
            batch["ecg_mask"].to(device),
        )
        loss = criterion(out["logits"], batch["labels"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []

    for batch in loader:
        out = model(
            batch["vital_feats"].to(device),
            batch["vital_seq"].to(device),
            batch["vital_lengths"].to(device),
            batch["ecg"].to(device),
            batch["ehr"].to(device),
            batch["ecg_mask"].to(device),
        )
        loss = criterion(out["logits"], batch["labels"].to(device))
        total_loss += loss.item()
        all_probs.append(out["probs"].cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # per-label AUC — skip labels with no positive examples in this split
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))

    return total_loss / len(loader), np.mean(aucs), aucs

# =============================================================================
# 6. DIGITAL TWIN — TRAJECTORY SIMULATION
# Analogous to MOSAIC Lab opioid paper (Rahman et al., Nature Mental Health)
# Instead of stress/pain/craving trajectories, we track cardiovascular state
# =============================================================================

@torch.no_grad()
def simulate_trajectory(model, patient_vitals_raw, ecg_embs,
                         ehr_x, device, vital_stat_scaler=None):
    """
    Simulate digital twin trajectory for one patient.
    Run inference at each vital timestep to show evolving cardiovascular state.

    This mimics continuous monitoring — each new vital reading updates
    the twin's risk assessment, analogous to the MOSAIC 30-min sliding window.

    patient_vitals_raw: np.array (T, 6) — raw vital sequence for one patient
    ecg_embs:           np.array (N, 512) — ECG embeddings
    ehr_x:              np.array (15,) — static EHR features
    """
    model.eval()
    T = len(patient_vitals_raw)
    trajectory = []

    for t in range(1, T + 1):
        # use only vitals up to current timestep t
        window = patient_vitals_raw[:t]     # (t, 6)

        # compute vital statistics from available window
        v_mean  = window.mean(axis=0)
        v_min   = window.min(axis=0)
        v_max   = window.max(axis=0)
        v_std   = window.std(axis=0) if t > 1 else np.zeros(VITAL_DIM)
        v_delta = window[-1] - window[0]    # trajectory direction
        vital_feats = np.concatenate(
            [v_mean, v_min, v_max, v_std, v_delta]
        ).astype(np.float32)                # (30,)

        # normalize if scaler provided
        if vital_stat_scaler is not None:
            vital_feats = vital_stat_scaler.transform(
                vital_feats.reshape(1, -1)
            ).flatten()

        # prepare vital sequence for LSTM: (1, MAX_T, 6), length t
        v_seq = np.zeros((MAX_T, VITAL_DIM), dtype=np.float32)
        v_seq[:t] = window
        v_len = torch.tensor([t], dtype=torch.long, device=device)

        # prepare ECG batch
        N = min(len(ecg_embs), MAX_N)
        e_pad = np.zeros((MAX_N, ECG_DIM), dtype=np.float32)
        e_pad[:N] = ecg_embs[:N]
        ecg_mask = [True] * N + [False] * (MAX_N - N)

        # run inference
        out = model(
            torch.tensor(vital_feats).unsqueeze(0).to(device),   # (1, 30)
            torch.tensor(v_seq).unsqueeze(0).to(device),         # (1, MAX_T, 6)
            v_len,
            torch.tensor(e_pad).unsqueeze(0).to(device),         # (1, 2, 512)
            torch.tensor(ehr_x).unsqueeze(0).to(device),         # (1, 15)
            torch.tensor([ecg_mask]).to(device),
        )

        trajectory.append({
            "timestep":    t,
            "probs":       out["probs"].squeeze().cpu().numpy(),   # (17,)
            "gates":       out["gates"].squeeze().cpu().numpy(),   # (3,)
            "latent":      out["latent"].squeeze().cpu().numpy()   # (128,) twin state
        })

    return trajectory


def plot_trajectory(trajectory, patient_id, label_names=LABEL_COLS,
                    save_path=None):
    """
    Plot cardiovascular state trajectory over ED stay.
    Analogous to MOSAIC paper's stress/pain/craving trajectory plots.

    Top panel:    diagnosis probability evolution over timesteps
    Bottom panel: gated fusion modality weights over timesteps
    """
    timesteps    = [t["timestep"] for t in trajectory]
    probs        = np.stack([t["probs"] for t in trajectory])   # (T, 17)
    gates        = np.stack([t["gates"] for t in trajectory])   # (T, 3)

    # identify top 5 diagnoses at final timestep
    top5_idx = probs[-1].argsort()[-5:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top: diagnosis probability trajectories ---
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for i, idx in enumerate(top5_idx):
        axes[0].plot(
            timesteps, probs[:, idx],
            label=label_names[idx],
            linewidth=2.5,
            color=colors[i]
        )
    axes[0].set_ylabel("Diagnosis Probability", fontsize=12)
    axes[0].set_title(
        f"Patient {patient_id} — Cardiovascular Digital Twin State Trajectory",
        fontsize=13
    )
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Decision threshold")
    axes[0].grid(True, alpha=0.3)

    # --- Bottom: gate weights over time ---
    axes[1].stackplot(
        timesteps,
        gates[:, 0], gates[:, 1], gates[:, 2],
        labels=["Vitals", "ECG-FM", "EHR/Demographics"],
        colors=["#2196F3", "#F44336", "#4CAF50"],
        alpha=0.75
    )
    axes[1].set_ylabel("Modality Weight", fontsize=12)
    axes[1].set_xlabel("Vital Sign Timestep (ED Stay)", fontsize=12)
    axes[1].set_title(
        "Gated Fusion — Modality Trust Over Time",
        fontsize=13
    )
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

# =============================================================================
# 7. ABLATION STUDY
# Train with one modality missing at a time to show fusion benefit
# =============================================================================

@torch.no_grad()
def run_ablations(model, test_loader, criterion, device):
    """
    Ablation study: zero out each modality to measure its contribution.
    Produces the key results table for your paper.
    """
    results = {}

    ablation_configs = {
        "Full model (all 3)":    {"zero_vital": False, "zero_ecg": False, "zero_ehr": False},
        "No vitals":             {"zero_vital": True,  "zero_ecg": False, "zero_ehr": False},
        "No ECG":                {"zero_vital": False, "zero_ecg": True,  "zero_ehr": False},
        "No EHR":                {"zero_vital": False, "zero_ecg": False, "zero_ehr": True},
        "ECG only":              {"zero_vital": True,  "zero_ecg": False, "zero_ehr": True},
        "Vitals only":           {"zero_vital": False, "zero_ecg": True,  "zero_ehr": True},
        "EHR only":              {"zero_vital": True,  "zero_ecg": True,  "zero_ehr": False},
    }

    for config_name, config in ablation_configs.items():
        model.eval()
        all_probs, all_labels = [], []

        for batch in test_loader:
            vf   = batch["vital_feats"].to(device)
            vseq = batch["vital_seq"].to(device)
            vlen = batch["vital_lengths"].to(device)
            ecg  = batch["ecg"].to(device)
            ehr  = batch["ehr"].to(device)
            msk  = batch["ecg_mask"].to(device)

            # zero out modalities for ablation
            if config["zero_vital"]:
                vf   = torch.zeros_like(vf)
                vseq = torch.zeros_like(vseq)
            if config["zero_ecg"]:
                ecg = torch.zeros_like(ecg)
            if config["zero_ehr"]:
                ehr = torch.zeros_like(ehr)

            out = model(vf, vseq, vlen, ecg, ehr, msk)
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

        probs  = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)

        aucs = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0:
                aucs.append(roc_auc_score(labels[:, i], probs[:, i]))

        results[config_name] = np.mean(aucs)
        print(f"{config_name:30s} | AUC: {np.mean(aucs):.4f}")

    return results

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 8a. Load ---
    static_df, triage_df, ed_vitals_df, ecg_df, labels_df = load_data()

    # --- 8b. Split at PATIENT level (not stay level) ---
    # critical: same patient must not appear in both train and test
    all_sids = labels_df["subject_id"].unique()
    train_sids, test_sids  = train_test_split(
        all_sids, test_size=0.15, random_state=42
    )
    train_sids, val_sids   = train_test_split(
        train_sids, test_size=0.15, random_state=42
    )

    def filter_sids(df, sids):
        return df[df["subject_id"].isin(sids)].reset_index(drop=True)

    # --- 8c. Feature engineering (fit on train only) ---

    # ECG — no fitting needed
    ecg_dict = prepare_ecg(ecg_df)

    # Vital statistics — fit scaler on train
    train_vital_feat, vital_scaler = create_vital_features(
        filter_sids(ed_vitals_df, train_sids), fit_scaler=True
    )
    val_vital_feat, _   = create_vital_features(
        filter_sids(ed_vitals_df, val_sids),   scaler=vital_scaler
    )
    test_vital_feat, _  = create_vital_features(
        filter_sids(ed_vitals_df, test_sids),  scaler=vital_scaler
    )

    # Vital sequences for trajectory simulation (separate from training features)
    train_seqs, seq_scaler = create_vital_sequences(
        filter_sids(ed_vitals_df, train_sids), fit_scaler=True
    )
    val_seqs,   _  = create_vital_sequences(
        filter_sids(ed_vitals_df, val_sids),   vital_scaler=seq_scaler
    )
    test_seqs,  _  = create_vital_sequences(
        filter_sids(ed_vitals_df, test_sids),  vital_scaler=seq_scaler
    )

    # EHR — fit scaler on train
    ehr_df = create_ehr_features(static_df, triage_df)
    train_ehr = filter_sids(ehr_df, train_sids)
    val_ehr   = filter_sids(ehr_df, val_sids)
    test_ehr  = filter_sids(ehr_df, test_sids)
    train_ehr, val_ehr, test_ehr, ehr_scaler = normalize_ehr(
        train_ehr, val_ehr, test_ehr
    )

    # Labels
    train_labels = filter_sids(labels_df, train_sids)
    val_labels   = filter_sids(labels_df, val_sids)
    test_labels  = filter_sids(labels_df, test_sids)

    # --- 8d. Build (subject_id, ed_stay_id) index ---
    def get_ids(df):
        return list(zip(df["subject_id"], df["ed_stay_id"]))

    # --- 8e. Datasets (with vital sequences for LSTM) ---
    train_ds = CardioEDDataset(
        get_ids(train_labels), train_vital_feat,
        ecg_dict, train_ehr, train_labels,
        vital_sequences=train_seqs,
    )
    val_ds   = CardioEDDataset(
        get_ids(val_labels), val_vital_feat,
        ecg_dict, val_ehr, val_labels,
        vital_sequences=val_seqs,
    )
    test_ds  = CardioEDDataset(
        get_ids(test_labels), test_vital_feat,
        ecg_dict, test_ehr, test_labels,
        vital_sequences=test_seqs,
    )

    collate = partial(collate_fn, max_N=MAX_N)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=4, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=4, pin_memory=True
    )

    # --- 8f. Sanity check tensor shapes before training ---
    batch = next(iter(train_loader))
    print("\n--- Tensor Shape Sanity Check ---")
    print(f"vital_feats:   {batch['vital_feats'].shape}")    # (B, 30)
    print(f"vital_seq:     {batch['vital_seq'].shape}")       # (B, MAX_T, 6)
    print(f"vital_lengths: {batch['vital_lengths'].shape}")  # (B,)
    print(f"ecg:           {batch['ecg'].shape}")             # (B, 2, 512)
    print(f"ecg_mask:      {batch['ecg_mask'].shape}")      # (B, 2)
    print(f"ehr:           {batch['ehr'].shape}")            # (B, 15)
    print(f"labels:        {batch['labels'].shape}")         # (B, 17)
    print("---------------------------------\n")

    # --- 8g. Model ---
    model      = CardioTwinED().to(device)
    pos_weight = compute_class_weights(train_labels).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # --- 8h. Training loop ---
    best_val_auc = 0.0
    for epoch in range(EPOCHS):
        train_loss              = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc, _   = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")

    # --- 8i. Final test evaluation ---
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_auc, per_label_aucs = eval_epoch(
        model, test_loader, criterion, device
    )
    print(f"\n{'='*40}")
    print(f"Test AUC (macro): {test_auc:.4f}")
    print(f"{'='*40}")
    for i, label in enumerate(LABEL_COLS):
        if i < len(per_label_aucs):
            print(f"  {label:35s}: {per_label_aucs[i]:.4f}")

    # --- 8j. Ablation study ---
    print("\n--- Ablation Study ---")
    ablation_results = run_ablations(model, test_loader, criterion, device)

    # --- 8k. Digital twin trajectory visualization ---
    # Find an interesting patient: one with rhythm changes during ED stay
    print("\n--- Generating Twin Trajectories ---")

    # example: take first 5 test patients with >= 3 vital timesteps
    trajectory_patients = []
    for sid, stay_id in get_ids(test_labels):
        seq = test_seqs.get((sid, stay_id))
        if seq is not None and len(seq) >= 3:
            trajectory_patients.append((sid, stay_id))
        if len(trajectory_patients) >= 5:
            break

    for sid, stay_id in trajectory_patients:
        raw_seq  = test_seqs[(sid, stay_id)]         # (T, 6)
        ecg_embs = ecg_dict.get(
            sid, np.zeros((1, ECG_DIM), dtype=np.float32)
        )
        ehr_row  = (
            test_ehr[test_ehr["subject_id"] == sid]
            .iloc[0][EHR_FEAT_COLS]
            .values.astype(np.float32)
        )

        traj = simulate_trajectory(
            model, raw_seq, ecg_embs, ehr_row, device
        )

        plot_trajectory(
            traj,
            patient_id=sid,
            save_path=f"trajectory_{sid}.png"
        )

    print("\nDone. Check trajectory_*.png for twin visualizations.")


if __name__ == "__main__":
    main()