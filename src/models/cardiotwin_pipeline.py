# =============================================================================
# CardioTwin — Integrated Full Pipeline
# Multimodal Cardiovascular Diagnosis Prediction from ED Presentations
#
# DATA STREAMS
#
# Stream 1 — ECG-FM Embeddings
#   ed_ecg_records (all ECGs) -> _attach_ecg_embeddings_all()
#     -> emb_0...emb_1535 per row -> prepare_ecg()
#     -> ecg_dict: (subject_id, ed_stay_id) -> (N, 1536)
#
#   extract_earliest_ecg_per_stay() is called separately as a time anchor
#   for aggregate_vitals_to_ecg_time() / create_model_df() only.
#
# Stream 2 — ED Vital Signs (dual representation)
#   ed_vitals -> create_vital_features()  -> (N, 30) flat stats per stay
#             -> create_vital_sequences() -> (N, T, 6) raw sequences per stay
#
#   Flat (30-dim): mean/min/max/std/delta for each of 6 vitals.
#   Sequence (T, 6): padded to MAX_T=12, fed to LSTM branch.
#   Both branches project to enc_dim=128 and are fused into vital_enc.
#
# Stream 3 — EHR Static Features
#   model_df -> prepare_ehr_features() -> (N, ehr_dim)
#   Columns discovered dynamically — no hardcoded race/triage names.
#   Continuous columns normalised on train set only.
#
# GATED FUSION
#   vital_enc (128) + ecg_enc (128) + ehr_enc (128)
#   -> gate network -> softmax weights (3,)
#   -> weighted sum -> fusion MLP -> 17 dx logits
#   Gates returned per patient for interpretability.
#
# DIGITAL TWIN TRAJECTORY
#   At each timestep t: recompute vital_feats from window [0:t], feed LSTM
#   t steps, keep ECG/EHR constant. Yields probs/gates/latent over time.
# =============================================================================

import os
import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.models.ecg_fm import run_pooled_ecg_extraction
from src.models.tabular_utils import (
    aggregate_vitals_to_ecg_time,
    create_model_df,
    extract_earliest_ecg_per_stay,
    filter_ed_ecg_records,
    filter_ed_encounters,
    load_config,
    load_data_files,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ECG-FM config path (always read from ecg_fm_params.json)
ECG_FM_CONFIG_PATH = _REPO_ROOT / "configs" / "ecg_fm_params.json"

# =============================================================================
# 0. CONSTANTS (defaults; overridden by config in run_cardiotwin_pipeline)
# =============================================================================

MAX_T = 12       # max vital timesteps per ED stay
MAX_N = 2        # max ECGs per stay
VITAL_DIM = 6    # hr, sbp, dbp, resp_rate, spo2, temperature
VITAL_STAT = 30  # 5 stats x 6 vitals
ECG_FM_DIM = 1536
ENC_DIM = 128
HIDDEN_DIM = 256
LSTM_HIDDEN = 64
N_LABELS = 17
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
DROPOUT = 0.3

VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

LABEL_COLS = [
    "ami_stemi", "ami_nstemi", "unstable_angina_ac_ischemia",
    "chronic_ischemic_disease", "heart_failure_acute", "heart_failure_chronic",
    "afib_aflutter", "ventricular_arrhythmias_arrest",
    "supraventricular_tachyarrhythmias", "brady_heart_block_conduction",
    "valvular_endocardial_disease", "cardiomyopathy_myocarditis",
    "pericardial_disease_tamponade", "pe_dvt_venous_thromboembolism",
    "aortic_peripheral_vascular", "hypertension_crisis", "stroke_tia",
]


# =============================================================================
# 1. STREAM 1 — ECG-FM EMBEDDINGS
# =============================================================================

def _attach_ecg_embeddings_all(ed_ecg_records: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract ECG-FM embeddings for ALL ECG records (every ECG per stay).
    
    Builds ecg_path column and calls run_pooled_ecg_extraction to attach
    1536-dim embeddings (emb_0 … emb_1535) to ed_ecg_records.
    
    ECG-FM settings always come from ecg_fm_params.json.
    config is only used for base_records_dir path resolution.
    """
    base_path = config["paths"]["base_records_dir"]
    
    paths = []
    for p in ed_ecg_records["path"]:
        p = os.path.splitext(str(p))[0]
        if p.startswith("files/"):
            p = p[len("files/"):]
        paths.append(os.path.join(base_path, p))

    subject_df = ed_ecg_records.copy().reset_index(drop=True)
    subject_df["ecg_path"] = paths

    return run_pooled_ecg_extraction(str(ECG_FM_CONFIG_PATH), subject_df)


def prepare_ecg(ecg_df: pd.DataFrame, max_n: int = MAX_N, ecg_fm_dim: int = ECG_FM_DIM) -> dict:
    """
    Group all ECG-FM embeddings by (subject_id, ed_stay_id).
    Returns dict: (subject_id int, ed_stay_id) -> np.array (N, ecg_fm_dim).

    All ECGs per stay are included up to max_n, sorted chronologically.
    The model's attention layer learns which ECG is most diagnostically
    relevant per stay.
    """
    emb_cols = [f"emb_{i}" for i in range(ecg_fm_dim)]
    stay_col = "ed_stay_id" if "ed_stay_id" in ecg_df.columns else "stay_id"

    ecg_dict = {}
    for (sid, stay_id), group in ecg_df.groupby(["subject_id", stay_col]):
        if "ecg_time" in group.columns:
            group = group.sort_values("ecg_time")
        embs = group[emb_cols].values.astype(np.float32)
        ecg_dict[(int(sid), stay_id)] = embs[:max_n]

    n_stays_with_ecg = len(ecg_dict)
    ecg_counts = [v.shape[0] for v in ecg_dict.values()]
    print(f"\n[ECG] ecg_dict: {n_stays_with_ecg} stays | "
          f"ECGs per stay — min:{min(ecg_counts)} max:{max(ecg_counts)} "
          f"mean:{np.mean(ecg_counts):.1f} | emb_dim={ecg_fm_dim}")

    # Bug 5: NaN ECG embeddings silently corrupt attention pooling
    nan_ecgs = [(k, v) for k, v in ecg_dict.items() if np.isnan(v).any()]
    if nan_ecgs:
        print(f"  WARNING: {len(nan_ecgs)} stays have NaN ECG embeddings — zeroing out")
        for k, v in nan_ecgs:
            ecg_dict[k] = np.nan_to_num(v, nan=0.0)

    return ecg_dict


# =============================================================================
# 2. STREAM 2 — ED VITAL SIGNS
# =============================================================================

def create_vital_features(ed_vitals_df: pd.DataFrame, scaler=None, fit_scaler: bool = False):
    """
    Convert long-format ED vitals to per-stay statistical summary (N, 30).
    For each of 6 vitals: mean, min, max, std, delta (last - first).
    Forward-fills within stay, fills remaining NaNs with population median.
    Returns (vital_feat_df, scaler).
    """
    df = ed_vitals_df.copy()
    stay_col = "stay_id" if "stay_id" in df.columns else "ed_stay_id"

    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values(["subject_id", stay_col, "charttime"])

    present_vitals = [c for c in VITAL_COLS if c in df.columns]
    df[present_vitals] = (
        df.groupby(["subject_id", stay_col])[present_vitals]
        .transform(lambda x: x.ffill().bfill())
    )
    df[present_vitals] = df[present_vitals].fillna(df[present_vitals].median())

    records = []
    for (sid, stay_id), group in df.groupby(["subject_id", stay_col]):
        vals = group[present_vitals].values.astype(np.float32)
        features = np.concatenate([
            vals.mean(axis=0),
            vals.min(axis=0),
            vals.max(axis=0),
            vals.std(axis=0) if len(vals) > 1 else np.zeros(len(present_vitals)),
            vals[-1] - vals[0],
        ])
        records.append({
            "subject_id": int(sid),
            "ed_stay_id": stay_id,
            **{f"vf_{i}": float(features[i]) for i in range(len(features))},
        })

    vital_feat_df = pd.DataFrame(records)
    feat_cols = [c for c in vital_feat_df.columns if c.startswith("vf_")]

    if fit_scaler:
        scaler = StandardScaler()
        vital_feat_df[feat_cols] = scaler.fit_transform(vital_feat_df[feat_cols])
    elif scaler is not None:
        vital_feat_df[feat_cols] = scaler.transform(vital_feat_df[feat_cols])

    print(f"\n[Vitals] create_vital_features: {len(vital_feat_df)} stays | "
          f"{len(feat_cols)} stat features | "
          f"vitals present: {present_vitals} | "
          f"fit_scaler={fit_scaler}")

    return vital_feat_df, scaler


def create_vital_sequences(ed_vitals_df: pd.DataFrame, vital_scaler=None, fit_scaler: bool = False):
    """
    Return raw vital sequences per stay for the LSTM branch and trajectory sim.
    Returns dict: (subject_id, stay_id) -> np.array (T, n_vitals), T <= MAX_T.
    StandardScaler fitted on train set only when fit_scaler=True.
    """
    df = ed_vitals_df.copy()
    stay_col = "stay_id" if "stay_id" in df.columns else "ed_stay_id"
    present_vitals = [c for c in VITAL_COLS if c in df.columns]

    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values(["subject_id", stay_col, "charttime"])

    df[present_vitals] = (
        df.groupby(["subject_id", stay_col])[present_vitals]
        .transform(lambda x: x.ffill().bfill())
    )
    df[present_vitals] = df[present_vitals].fillna(df[present_vitals].median())

    if fit_scaler:
        vital_scaler = StandardScaler()
        df[present_vitals] = vital_scaler.fit_transform(df[present_vitals])
    elif vital_scaler is not None:
        df[present_vitals] = vital_scaler.transform(df[present_vitals])

    sequences = {}
    for (sid, stay_id), group in df.groupby(["subject_id", stay_col]):
        sequences[(int(sid), stay_id)] = group[present_vitals].values.astype(np.float32)[:MAX_T]

    seq_lens = [v.shape[0] for v in sequences.values()]
    print(f"\n[Vitals] create_vital_sequences: {len(sequences)} stays | "
          f"seq len — min:{min(seq_lens)} max:{max(seq_lens)} mean:{np.mean(seq_lens):.1f} | "
          f"vitals: {present_vitals} | fit_scaler={fit_scaler}")

    return sequences, vital_scaler


# =============================================================================
# 3. STREAM 3 — EHR STATIC FEATURES
# =============================================================================

def prepare_ehr_features(model_df: pd.DataFrame, subject_stay_ids: list,
                          scaler=None, fit_scaler: bool = False):
    """
    Build EHR feature matrix from model_df for the given (subject_id, ed_stay_id) pairs.
    Discovers columns dynamically — excludes label_/report_/emb_ prefixes and ID columns.
    Binary columns kept as-is; continuous columns normalised on train set only.
    Returns (X np.float32 (N, ehr_dim), scaler, feature_names list).
    """
    df = model_df.copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df = df.set_index(["subject_id", "ed_stay_id"])

    exclude_prefixes = ("label_", "report_", "emb_", "path", "ecg_time", "charttime")
    exclude_exact = {"stay_id", "subject_id", "ed_stay_id", "hadm_id", "study_id", "split"}

    candidate_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and c not in exclude_exact
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, bool, "bool")
    ]

    if not candidate_cols:
        raise ValueError(
            "No numeric EHR feature columns found in model_df. "
            "Verify that create_model_df() has run and columns are numeric."
        )

    continuous_cols, binary_cols = [], []
    for c in candidate_cols:
        if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(c)
        else:
            continuous_cols.append(c)

    all_feat_cols = continuous_cols + binary_cols
    idx = pd.MultiIndex.from_tuples(
        [(int(s), e) for s, e in subject_stay_ids],
        names=["subject_id", "ed_stay_id"],
    )
    X = df.reindex(idx)[all_feat_cols].fillna(0).astype(np.float32).values

    if fit_scaler and continuous_cols:
        scaler = StandardScaler()
        cont_idx = [all_feat_cols.index(c) for c in continuous_cols]
        X[:, cont_idx] = scaler.fit_transform(X[:, cont_idx])
    elif scaler is not None and continuous_cols:
        cont_idx = [all_feat_cols.index(c) for c in continuous_cols]
        X[:, cont_idx] = scaler.transform(X[:, cont_idx])

    nan_rows = np.isnan(X).any(axis=1).sum()
    print(f"\n[EHR] prepare_ehr_features: shape={X.shape} | "
          f"continuous={len(continuous_cols)} binary={len(binary_cols)} | "
          f"rows with NaN after fillna={nan_rows} | fit_scaler={fit_scaler}")
    if fit_scaler:
        print(f"      feature names: {all_feat_cols[:8]}{'...' if len(all_feat_cols) > 8 else ''}")

    return X, scaler, all_feat_cols


# =============================================================================
# 4. DATASET + DATALOADER
# =============================================================================

class CardioEDDataset(Dataset):
    """
    Per-stay dataset returning all three modality tensors plus labels.

    vital_feat_df   : DataFrame with vf_0...vf_{VITAL_STAT-1} per stay
    ecg_dict        : (subject_id, ed_stay_id) -> np.array (N, ECG_FM_DIM)
    ehr_matrix      : np.float32 (n_stays, ehr_dim) aligned to subject_stay_ids
    labels_df       : DataFrame with LABEL_COLS per stay
    vital_sequences : dict (subject_id, stay_id) -> np.array (T, vital_dim)
    vital_dim       : actual number of vital channels in ed_vitals (<=6)
    """

    def __init__(self, subject_stay_ids, vital_feat_df, ecg_dict, ehr_matrix,
                 labels_df, vital_sequences=None, vital_dim=VITAL_DIM, ecg_fm_dim=ECG_FM_DIM):
        self.ids = subject_stay_ids
        self.ecg_dict = ecg_dict
        self.ehr = ehr_matrix
        self.vital_sequences = vital_sequences or {}
        self.vital_dim = vital_dim
        self.ecg_fm_dim = ecg_fm_dim

        # Bug 4: duplicate (subject_id, ed_stay_id) in vital_feat_df causes .loc to
        # return a DataFrame instead of a Series, producing shape mismatches downstream
        dups = vital_feat_df.duplicated(["subject_id", "ed_stay_id"]).sum()
        if dups > 0:
            raise ValueError(
                f"vital_feat_df has {dups} duplicate (subject_id, ed_stay_id) rows. "
                f"Deduplicate before constructing dataset."
            )

        self.vital_feats = vital_feat_df.set_index(["subject_id", "ed_stay_id"])
        self.labels = labels_df.set_index(["subject_id", "ed_stay_id"])

        # Bug 1: EHR matrix must be row-aligned to subject_stay_ids — misalignment
        # is silent (no crash) but every patient gets the wrong EHR features
        assert len(ehr_matrix) == len(subject_stay_ids), (
            f"EHR matrix rows ({len(ehr_matrix)}) != subject_stay_ids ({len(subject_stay_ids)}). "
            f"Ensure prepare_ehr_features() is called with the same ids list passed here."
        )

        label_cols_in_df = [c for c in LABEL_COLS if c in self.labels.columns]
        ecg_coverage = sum(1 for sid, stay in subject_stay_ids if (int(sid), stay) in ecg_dict)
        vital_coverage = sum(1 for sid, stay in subject_stay_ids if (int(sid), stay) in self.vital_feats.index)
        print(f"\n[Dataset] CardioEDDataset: {len(subject_stay_ids)} stays | "
              f"label cols={len(label_cols_in_df)}/17 | "
              f"ECG coverage={ecg_coverage}/{len(subject_stay_ids)} | "
              f"vital coverage={vital_coverage}/{len(subject_stay_ids)}")
        if len(label_cols_in_df) > 0:
            label_matrix = self.labels[label_cols_in_df].values
            pos_rates = label_matrix.mean(axis=0)
            print(f"      label prevalence (top 5 by rate): "
                  + str({label_cols_in_df[i]: f"{pos_rates[i]:.3f}"
                         for i in np.argsort(pos_rates)[::-1][:5]}))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid, stay_id = self.ids[idx]

        feat_cols = [c for c in self.vital_feats.columns if c.startswith("vf_")]
        vital_feats = self.vital_feats.loc[(sid, stay_id)][feat_cols].values.astype(np.float32)

        ecg = self.ecg_dict.get((int(sid), stay_id), np.zeros((1, self.ecg_fm_dim), dtype=np.float32))

        labels = self.labels.loc[(sid, stay_id)][LABEL_COLS].values.astype(np.float32)
        vital_seq = self.vital_sequences.get(
            (sid, stay_id), np.zeros((1, self.vital_dim), dtype=np.float32)
        )

        return {
            "vital_feats": vital_feats,
            "vital_seq": vital_seq,
            "vital_len": vital_seq.shape[0],
            "ecg": ecg,
            "ehr": self.ehr[idx],
            "labels": labels,
        }


def collate_fn(batch, max_N=MAX_N, max_T=MAX_T, ecg_fm_dim=ECG_FM_DIM):
    vital_dim = batch[0]["vital_seq"].shape[-1]

    ecg_padded, ecg_mask = [], []
    for b in batch:
        e = b["ecg"][:max_N]
        n_clip = e.shape[0]
        pad = np.zeros((max_N, ecg_fm_dim), dtype=np.float32)
        pad[:n_clip] = e
        ecg_padded.append(pad)
        ecg_mask.append([True] * n_clip + [False] * (max_N - n_clip))

    vital_lens = torch.tensor([b["vital_len"] for b in batch], dtype=torch.long)
    # Bug 3: pack_padded_sequence crashes if any length is 0. Clamp to min=1 so
    # zero-vital stays pass through the LSTM as a single zero-padded timestep.
    vital_lens = torch.clamp(vital_lens, min=1)
    vital_seqs = [torch.tensor(b["vital_seq"], dtype=torch.float32) for b in batch]
    vital_padded = pad_sequence(vital_seqs, batch_first=True, padding_value=0.0)

    if vital_padded.size(1) > max_T:
        vital_padded = vital_padded[:, :max_T]
        vital_lens = torch.clamp(vital_lens, max=max_T)
    elif vital_padded.size(1) < max_T:
        pad_right = torch.zeros(vital_padded.size(0), max_T - vital_padded.size(1), vital_dim)
        vital_padded = torch.cat([vital_padded, pad_right], dim=1)

    return {
        "vital_feats": torch.tensor(np.stack([b["vital_feats"] for b in batch])),
        "vital_seq": vital_padded,
        "vital_lengths": vital_lens,
        "ecg": torch.tensor(np.stack(ecg_padded)),
        "ecg_mask": torch.tensor(ecg_mask, dtype=torch.bool),
        "ehr": torch.tensor(np.stack([b["ehr"] for b in batch])),
        "labels": torch.tensor(np.stack([b["labels"] for b in batch])),
    }


# =============================================================================
# 5. MODEL — Gated Fusion (CardioTwinED)
# =============================================================================

class CardioTwinED(nn.Module):
    """
    Multimodal Gated Fusion Network for cardiovascular diagnosis prediction.

    Three modality encoders each project to enc_dim=128:
      - Vital encoder: dual-branch LSTM (sequence) + MLP (stats), fused to vital_enc
      - ECG encoder: attention pool over N ECGs + MLP
      - EHR encoder: shallow MLP (built lazily via set_ehr_dim())

    Gate network learns soft per-patient weights (3,) over the three encodings.
    Outputs: logits, probs, gates (interpretability), latent (twin state).
    """

    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, enc_dim=ENC_DIM, hidden_dim=HIDDEN_DIM,
                 lstm_hidden=LSTM_HIDDEN, dropout=DROPOUT, n_labels=N_LABELS):
        super().__init__()
        self._enc_dim = enc_dim
        self._dropout = dropout
        self._lstm_hidden = lstm_hidden

        self.vital_lstm = nn.LSTM(input_size=vital_dim, hidden_size=lstm_hidden,
                                  num_layers=1, batch_first=True)
        self.vital_lstm_proj = nn.Sequential(nn.Linear(lstm_hidden, enc_dim), nn.GELU())

        self.vital_encoder = nn.Sequential(
            nn.Linear(vital_stat, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, enc_dim), nn.GELU(),
        )
        self.vital_fusion = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim), nn.GELU(), nn.Dropout(dropout),
        )

        self.ecg_attn = nn.Linear(ecg_emb_dim, 1)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_emb_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, enc_dim), nn.GELU(),
        )

        # EHR encoder and gate built lazily once ehr_dim is known
        self.ehr_encoder = None
        self.gate = None

        self.fusion = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.dx_head = nn.Linear(hidden_dim // 2, n_labels)

    def set_ehr_dim(self, ehr_dim: int, device=None):
        """Build EHR encoder and gate once ehr_dim is known."""
        enc_dim = self._enc_dim
        dropout = self._dropout
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, enc_dim), nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Linear(enc_dim * 3, 128), nn.ReLU(), nn.Linear(128, 3))
        if device is not None:
            self.ehr_encoder = self.ehr_encoder.to(device)
            self.gate = self.gate.to(device)

    def forward(self, vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask=None):
        """
        vital_feats    (B, VITAL_STAT)
        vital_seq      (B, T, vital_dim)    padded
        vital_lengths  (B,)                actual lengths
        ecg_embs       (B, N, ECG_FM_DIM)  up to MAX_N ECGs
        ehr_x          (B, ehr_dim)
        ecg_mask       (B, N) bool          True = real ECG
        """
        B = vital_feats.size(0)
        device = vital_feats.device

        if self.ehr_encoder is None:
            self.set_ehr_dim(ehr_x.size(1), device=device)

        # Vital LSTM branch
        if vital_lengths is not None and (vital_lengths > 0).any() and vital_seq.size(1) > 0:
            packed = pack_padded_sequence(
                vital_seq, vital_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.vital_lstm(packed)
            lstm_enc = self.vital_lstm_proj(h_n.squeeze(0))
        else:
            lstm_enc = self.vital_lstm_proj(
                torch.zeros(B, self._lstm_hidden, device=device, dtype=vital_feats.dtype)
            )

        stats_enc = self.vital_encoder(vital_feats)
        vital_enc = self.vital_fusion(torch.cat([lstm_enc, stats_enc], dim=1))

        # ECG attention pool
        attn_scores = self.ecg_attn(ecg_embs)
        if ecg_mask is not None:
            attn_scores = attn_scores.masked_fill(~ecg_mask.unsqueeze(-1), float("-inf"))
        ecg_pooled = (torch.softmax(attn_scores, dim=1) * ecg_embs).sum(dim=1)
        ecg_enc = self.ecg_encoder(ecg_pooled)

        ehr_enc = self.ehr_encoder(ehr_x)

        gates = F.softmax(self.gate(torch.cat([vital_enc, ecg_enc, ehr_enc], dim=1)), dim=1)
        fused = (
            gates[:, 0:1] * vital_enc
            + gates[:, 1:2] * ecg_enc
            + gates[:, 2:3] * ehr_enc
        )
        shared = self.fusion(fused)

        return {
            "logits": self.dx_head(shared),
            "probs": torch.sigmoid(self.dx_head(shared)),
            "gates": gates,   # (B, 3) vital / ECG / EHR weights
            "latent": shared, # (B, hidden_dim/2) twin state vector
        }


# =============================================================================
# 6. TRAINING
# =============================================================================

def compute_class_weights(labels_df: pd.DataFrame) -> torch.Tensor:
    """pos_weight = n_negative / n_positive per label for BCEWithLogitsLoss."""
    cols = [c for c in LABEL_COLS if c in labels_df.columns]
    labels = labels_df[cols].values
    n_pos = (labels == 1).sum(axis=0).clip(min=1)
    n_neg = (labels == 0).sum(axis=0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip_norm: float = 1.0) -> float:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Returns (mean_loss, macro_auc, per_label_aucs) with per_label_aucs[i] = AUC for label i or np.nan if no positives."""
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
        total_loss += criterion(out["logits"], batch["labels"].to(device)).item()
        all_probs.append(out["probs"].cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    n_labels = labels.shape[1]
    per_label_aucs = []
    for i in range(n_labels):
        if labels[:, i].sum() > 0:
            per_label_aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        else:
            per_label_aucs.append(np.nan)
    valid_aucs = [a for a in per_label_aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return total_loss / len(loader), macro_auc, per_label_aucs


# =============================================================================
# 7. ABLATION STUDY
# =============================================================================

@torch.no_grad()
def run_ablations(model, test_loader, criterion, device) -> dict:
    """Zero out each modality in turn to measure individual contribution."""
    configs = {
        "Full model":  dict(zero_vital=False, zero_ecg=False, zero_ehr=False),
        "No vitals":   dict(zero_vital=True,  zero_ecg=False, zero_ehr=False),
        "No ECG":      dict(zero_vital=False, zero_ecg=True,  zero_ehr=False),
        "No EHR":      dict(zero_vital=False, zero_ecg=False, zero_ehr=True),
        "ECG only":    dict(zero_vital=True,  zero_ecg=False, zero_ehr=True),
        "Vitals only": dict(zero_vital=False, zero_ecg=True,  zero_ehr=True),
        "EHR only":    dict(zero_vital=True,  zero_ecg=True,  zero_ehr=False),
    }
    results = {}
    for name, cfg in configs.items():
        model.eval()
        all_probs, all_labels = [], []
        for batch in test_loader:
            vf = batch["vital_feats"].to(device)
            vseq = batch["vital_seq"].to(device)
            vlen = batch["vital_lengths"].to(device)
            ecg = batch["ecg"].to(device)
            ehr = batch["ehr"].to(device)
            msk = batch["ecg_mask"].to(device)
            if cfg["zero_vital"]:
                vf = torch.zeros_like(vf)
                vseq = torch.zeros_like(vseq)
            if cfg["zero_ecg"]:
                ecg = torch.zeros_like(ecg)
            if cfg["zero_ehr"]:
                ehr = torch.zeros_like(ehr)
            out = model(vf, vseq, vlen, ecg, ehr, msk)
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        aucs = [
            roc_auc_score(labels[:, i], probs[:, i])
            for i in range(labels.shape[1]) if labels[:, i].sum() > 0
        ]
        results[name] = float(np.mean(aucs))
    return results


# =============================================================================
# 8. DIGITAL TWIN — TRAJECTORY SIMULATION
# =============================================================================

@torch.no_grad()
def simulate_trajectory(model, patient_vitals_raw, ecg_embs, ehr_x,
                         device, vital_stat_scaler=None,
                         max_t: int = MAX_T, max_n: int = MAX_N,
                         ecg_fm_dim: int = ECG_FM_DIM) -> list:
    """
    Simulate evolving cardiovascular state across an ED stay.

    At each timestep t, recomputes vital_feats from window [0:t] and feeds
    the LSTM only t steps. ECG and EHR stay constant throughout.

    patient_vitals_raw : np.array (T, n_vitals)
    ecg_embs           : np.array (N, ecg_fm_dim)
    ehr_x              : np.array (ehr_dim,)

    Returns list of dicts per timestep:
      {timestep, probs (n_labels,), gates (3,), latent (enc_dim,)}
    """
    model.eval()
    T = len(patient_vitals_raw)
    n_vitals = patient_vitals_raw.shape[1]
    trajectory = []

    for t in range(1, T + 1):
        window = patient_vitals_raw[:t]
        vital_feats = np.concatenate([
            window.mean(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            window.std(axis=0) if t > 1 else np.zeros(n_vitals),
            window[-1] - window[0],
        ]).astype(np.float32)

        if vital_stat_scaler is not None:
            vital_feats = vital_stat_scaler.transform(vital_feats.reshape(1, -1)).flatten()

        v_seq = np.zeros((max_t, n_vitals), dtype=np.float32)
        v_seq[:t] = window

        N = min(len(ecg_embs), max_n)
        e_pad = np.zeros((max_n, ecg_fm_dim), dtype=np.float32)
        e_pad[:N] = ecg_embs[:N]
        ecg_mask = [True] * N + [False] * (max_n - N)

        out = model(
            torch.tensor(vital_feats).unsqueeze(0).to(device),
            torch.tensor(v_seq).unsqueeze(0).to(device),
            torch.tensor([t], dtype=torch.long, device=device),
            torch.tensor(e_pad).unsqueeze(0).to(device),
            torch.tensor(ehr_x).unsqueeze(0).to(device),
            torch.tensor([ecg_mask]).to(device),
        )
        trajectory.append({
            "timestep": t,
            "probs": out["probs"].squeeze().cpu().numpy(),
            "gates": out["gates"].squeeze().cpu().numpy(),
            "latent": out["latent"].squeeze().cpu().numpy(),
        })

    return trajectory


def plot_trajectory(trajectory, patient_id, label_names=LABEL_COLS, save_path=None):
    timesteps = [t["timestep"] for t in trajectory]
    probs = np.stack([t["probs"] for t in trajectory])
    gates = np.stack([t["gates"] for t in trajectory])
    top5 = probs[-1].argsort()[-5:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for i, idx in enumerate(top5):
        axes[0].plot(timesteps, probs[:, idx], label=label_names[idx],
                     linewidth=2.5, color=colors[i])
    axes[0].set_ylabel("Diagnosis Probability")
    axes[0].set_title(f"Patient {patient_id} — Cardiovascular Digital Twin Trajectory")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].stackplot(
        timesteps, gates[:, 0], gates[:, 1], gates[:, 2],
        labels=["Vitals", "ECG-FM", "EHR"],
        colors=["#2196F3", "#F44336", "#4CAF50"], alpha=0.75,
    )
    axes[1].set_ylabel("Modality Weight")
    axes[1].set_xlabel("Vital Sign Timestep (ED Stay)")
    axes[1].set_title("Gated Fusion — Modality Trust Over Time")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# =============================================================================
# 9. EVALUATION PLOTS
# =============================================================================

def plot_evaluation(y_true, y_pred_probs, label_names, out_path):
    """ROC, PR curves, and aggregated confusion matrix."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    valid = [l for l in label_names if len(np.unique(y_true[:, list(label_names).index(l)])) > 1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for label in valid:
        idx = list(label_names).index(label)
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred_probs[:, idx])
        auc = roc_auc_score(y_true[:, idx], y_pred_probs[:, idx])
        color = "#2E5090" if auc >= 0.95 else ("#6B46C1" if auc >= 0.85 else "#D32F2F")
        axes[0].plot(fpr, tpr, linewidth=1.5, alpha=0.5, color=color)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=2)
    mean_auc = np.mean([
        roc_auc_score(y_true[:, list(label_names).index(l)],
                      y_pred_probs[:, list(label_names).index(l)]) for l in valid
    ])
    axes[0].set_title(f"ROC Curves\nMean AUC: {mean_auc:.3f}", fontsize=14)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, alpha=0.3)

    for label in valid:
        idx = list(label_names).index(label)
        prec, rec, _ = precision_recall_curve(y_true[:, idx], y_pred_probs[:, idx])
        ap = average_precision_score(y_true[:, idx], y_pred_probs[:, idx])
        color = "#2E5090" if ap >= 0.5 else ("#6B46C1" if ap >= 0.25 else "#D32F2F")
        axes[1].plot(rec, prec, linewidth=1.5, alpha=0.5, color=color)
    mean_ap = np.mean([
        average_precision_score(y_true[:, list(label_names).index(l)],
                                y_pred_probs[:, list(label_names).index(l)]) for l in valid
    ])
    axes[1].set_title(f"Precision-Recall Curves\nMean PR-AUC: {mean_ap:.3f}", fontsize=14)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)

    y_pred_bin = (y_pred_probs >= 0.5).astype(int)
    total_cm = sum(
        confusion_matrix(y_true[:, list(label_names).index(l)],
                         y_pred_bin[:, list(label_names).index(l)]) for l in valid
    )
    sns.heatmap(total_cm, annot=True, fmt=".0f", cmap="Blues", ax=axes[2],
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], cbar=False)
    axes[2].set_title(f"Aggregated Confusion Matrix\n(Sum Across {len(valid)} Labels)")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 9. MODULAR HELPER FUNCTIONS
# =============================================================================

def _load_and_prepare_cardiotwin(in_dir, config_path, pbar, steps) -> tuple:
    """Load and prepare all data: filters, ECG embeddings, vitals, EHR features."""
    pbar.set_description(steps[0])
    config = load_config(config_path)
    ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)
    pbar.update(1)

    pbar.set_description(steps[1])
    ed_encounters = filter_ed_encounters(clinical_encounters)
    ed_ecg_records = filter_ed_ecg_records(ecg_records)
    pbar.update(1)

    pbar.set_description(steps[2])
    ed_ecg_records["ecg_time"] = pd.to_datetime(ed_ecg_records["ecg_time"])
    ed_ecg_records = ed_ecg_records.sort_values(["subject_id", "ed_stay_id", "ecg_time"])
    ed_ecg_records["qrs_duration"] = ed_ecg_records["qrs_end"] - ed_ecg_records["qrs_onset"]
    ed_ecg_records["pr_interval"] = ed_ecg_records["qrs_onset"] - ed_ecg_records["p_onset"]
    ed_ecg_records["qt_proxy"] = ed_ecg_records["t_end"] - ed_ecg_records["qrs_onset"]
    earliest_ecgs = extract_earliest_ecg_per_stay(ed_ecg_records)
    pbar.update(1)

    pbar.set_description(steps[3])
    all_ecgs_embedded = _attach_ecg_embeddings_all(ed_ecg_records, config)
    pbar.update(1)

    pbar.set_description(steps[4])
    ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs, agg_window_hours=4.0)
    pbar.update(1)

    pbar.set_description(steps[5])
    model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)
    pbar.update(1)

    return config, ed_vitals, model_df, all_ecgs_embedded


def _val_split(model_df, X_train, y_train, val_size=0.1, random_state=0) -> tuple:
    """Carve a patient-aware validation split for early stopping."""
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_groups = model_df.loc[y_train.index, "subject_id"].astype(int).values
    tr_idx, val_idx = next(val_splitter.split(X_train, y_train, groups=train_groups))
    return (
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        X_train.iloc[val_idx], y_train.iloc[val_idx],
    )


def _train_cardiotwin_model(
    model, train_loader, val_loader, test_loader,
    params: dict, out_path: str, device, pbar=None
) -> tuple:
    """Train CardioTwinED with early stopping. Returns (model, test_auc, per_label_aucs)."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(
        train_loader.dataset.labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                  weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    best_val_auc = 0.0
    best_model_pt = os.path.join(out_path, "best_cardiotwin.pt")

    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                grad_clip_norm=params["grad_clip_norm"])
        val_loss, val_auc, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_pt)
        print(f"  Epoch {epoch+1:3d}/{params['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_auc={val_auc:.4f}{' ✓' if improved else ''}")

    model.load_state_dict(torch.load(best_model_pt, map_location=device))
    _, test_auc, per_label_aucs = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[Train] Best val_auc={best_val_auc:.4f} | test_auc={test_auc:.4f}")
    valid_aucs = [(LABEL_COLS[i], auc) for i, auc in enumerate(per_label_aucs) if not np.isnan(auc)]
    valid_aucs.sort(key=lambda x: x[1], reverse=True)
    print(f"[Train] Per-label AUC (non-nan): {valid_aucs}")
    return model, test_auc, per_label_aucs


def plot_kfold_loss_curves_cardiotwin(
    train_loader_fn, val_loader_fn, model_fn, label_cols_present,
    params: dict, out_path: str, device, n_folds: int = 3,
    model_name: str = "cardiotwin"
) -> dict:
    """
    Train CardioTwinED using K-fold cross-validation with early stopping.
    Returns a plot of mean train/val loss across folds.
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    train_loss_folds = np.zeros((n_folds, params["epochs"]))
    val_loss_folds = np.zeros((n_folds, params["epochs"]))
    stopped_epochs = []

    for fold_idx in range(n_folds):
        train_loader = train_loader_fn(fold_idx)
        val_loader = val_loader_fn(fold_idx)
        
        model = model_fn()
        criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(
            train_loader.dataset.labels).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                      weight_decay=params["weight_decay"])

        best_val_loss = float("inf")
        patience_counter = 0
        stopped_at = params["epochs"]

        for epoch in range(params["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                    grad_clip_norm=params["grad_clip_norm"])
            val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
            
            train_loss_folds[fold_idx, epoch] = train_loss
            val_loss_folds[fold_idx, epoch] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:  # early stopping patience
                stopped_at = epoch + 1
                train_loss_folds[fold_idx, epoch + 1:] = train_loss
                val_loss_folds[fold_idx, epoch + 1:] = val_loss
                break

        stopped_epochs.append(stopped_at)

    mean_train = train_loss_folds.mean(axis=0)
    mean_val = val_loss_folds.mean(axis=0)
    best_epoch = int(np.argmin(mean_val))
    mean_stopped = int(np.mean(stopped_epochs))

    # Plot
    epochs_range = np.arange(1, params["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    for fold_idx in range(n_folds):
        ax.plot(epochs_range, train_loss_folds[fold_idx], color="#2E5090", alpha=0.12, linewidth=0.8)
        ax.plot(epochs_range, val_loss_folds[fold_idx], color="#D32F2F", alpha=0.12, linewidth=0.8)

    ax.plot(epochs_range, mean_train, color="#2E5090", linewidth=2.5, label="Train loss (mean)")
    ax.plot(epochs_range, mean_val, color="#D32F2F", linewidth=2.5, label="Val loss (mean)")

    ax.axvline(best_epoch + 1, color="gray", linestyle="--", linewidth=1.5,
               label=f"Best epoch: {best_epoch + 1}  (val={mean_val[best_epoch]:.4f})")

    if mean_stopped < params["epochs"]:
        ax.axvline(mean_stopped, color="#F57C00", linestyle=":", linewidth=1.5,
                   label=f"Mean early stop: epoch {mean_stopped}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(f"{model_name} — Train vs Val Loss\n{n_folds}-Fold CV · patience=10",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = Path(out_path) / model_name / f"{model_name}_kfold_loss_curves.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "mean_train_loss": mean_train,
        "mean_val_loss": mean_val,
        "best_epoch": best_epoch,
        "mean_stopped_epoch": mean_stopped,
    }


def evaluate_and_visualize_cardiotwin(
    model, test_loader, label_cols_present, out_path: str,
    model_name: str = "cardiotwin", device=None
) -> pd.DataFrame:
    """
    Evaluate CardioTwinED and produce:
      1. ROC + PR curves + aggregated confusion matrix
      2. Per-label results CSV
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["vital_feats"].to(device),
                batch["vital_seq"].to(device),
                batch["vital_lengths"].to(device),
                batch["ecg"].to(device),
                batch["ehr"].to(device),
                batch["ecg_mask"].to(device),
            )
            all_probs.append(out["probs"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_probs)

    # Compute per-label AUCs
    results = []
    for i, label in enumerate(label_cols_present):
        n_pos = y_true[:, i].sum()
        if y_true[:, i].nunique() > 1:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        else:
            auc = ap = np.nan
        results.append({
            "target": label,
            "n_test_pos": int(n_pos),
            "pos_rate": y_true[:, i].mean(),
            "roc_auc": auc,
            "pr_auc": ap,
        })

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    valid_labels = [l for l in label_cols_present if y_true[:, label_cols_present.index(l)].nunique() > 1]

    # ROC + PR + Confusion Matrix
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for label in valid_labels:
        idx = label_cols_present.index(label)
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
        auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
        color = "#2E5090" if auc >= 0.85 else ("#6B46C1" if auc >= 0.75 else "#D32F2F")
        axes[0].plot(fpr, tpr, linewidth=1.5, alpha=0.5, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=2)
    mean_auc = results_df["roc_auc"].mean()
    axes[0].set_title(f"ROC Curves\nMean AUC: {mean_auc:.3f}", fontsize=14)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, alpha=0.3)

    for label in valid_labels:
        idx = label_cols_present.index(label)
        prec, rec, _ = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        ap = average_precision_score(y_true[:, idx], y_pred[:, idx])
        color = "#2E5090" if ap >= 0.5 else ("#6B46C1" if ap >= 0.3 else "#D32F2F")
        axes[1].plot(rec, prec, linewidth=1.5, alpha=0.5, color=color)

    mean_ap = results_df["pr_auc"].mean()
    axes[1].set_title(f"Precision-Recall Curves\nMean PR-AUC: {mean_ap:.3f}", fontsize=14)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)

    y_pred_bin = (y_pred >= 0.5).astype(int)
    total_cm = np.zeros((2, 2))
    for label in valid_labels:
        idx = label_cols_present.index(label)
        total_cm += confusion_matrix(y_true[:, idx], y_pred_bin[:, idx])

    sns.heatmap(total_cm, annot=True, fmt=".0f", cmap="Blues", ax=axes[2],
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], cbar=False)
    axes[2].set_title(f"Aggregated Confusion Matrix\n(Sum Across {len(valid_labels)} Labels)")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    plt.tight_layout()
    fig_path = Path(out_path) / model_name / f"{model_name}_evaluation_plots.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = Path(out_path) / model_name / f"{model_name}_results.csv"
    results_df.to_csv(csv_path, index=False)

    return results_df


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def run_cardiotwin_pipeline(in_dir, config_path, out_path):
    """
    Full CardioTwin pipeline: ECG-FM + vitals + EHR with gated fusion.
    
    Features:
      - Multimodal fusion (vitals + ECG + EHR)
      - K-fold cross-validation with loss curves
      - Comprehensive evaluation (ROC, PR, confusion matrix)
      - Digital twin trajectory simulation
    
    Same signature as run_mlp_*_pipeline for consistency.
    """
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECG per stay (vitals time anchor)",
        "Extracting ECG-FM embeddings — ALL ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Building vital features + sequences",
        "Building EHR features",
        "Building ECG dict + datasets",
        "Training CardioTwinED",
        "K-fold validation curves",
        "Evaluating + generating trajectories",
    ]

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        config = load_config(config_path)
        in_dir = str(Path(in_dir).resolve()) if in_dir else str(_REPO_ROOT / config["paths"]["in_dir"])
        out_path = str(Path(out_path).resolve()) if out_path else str(_REPO_ROOT / config["paths"].get("out_dir", "data/model_results/cardiotwin"))
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # Extract configuration parameters
        pl = config.get("pipeline", {})
        max_t = pl.get("max_t", MAX_T)
        max_n = pl.get("max_n", MAX_N)
        test_size = pl.get("test_size", 0.2)
        val_size = pl.get("val_size", 0.1)
        random_state = pl.get("random_state", 42)
        val_random_state = pl.get("val_random_state", 0)
        num_workers = pl.get("num_workers", 4)
        n_trajectory_samples = pl.get("n_trajectory_samples", 5)
        min_trajectory_steps = pl.get("min_trajectory_steps", 3)

        mdl = config.get("model", {})
        enc_dim = mdl.get("enc_dim", ENC_DIM)
        hidden_dim = mdl.get("hidden_dim", HIDDEN_DIM)
        lstm_hidden = mdl.get("lstm_hidden", LSTM_HIDDEN)
        dropout = mdl.get("dropout", DROPOUT)
        ecg_fm_dim = mdl.get("ecg_fm_dim", ECG_FM_DIM)

        trn = config.get("training", {})
        batch_size = trn.get("batch_size", BATCH_SIZE)
        lr = trn.get("learning_rate", LR)
        weight_decay = trn.get("weight_decay", 1e-4)
        epochs = trn.get("epochs", EPOCHS)
        grad_clip_norm = trn.get("grad_clip_norm", 1.0)

        params = {
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "grad_clip_norm": grad_clip_norm,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and prepare data
        config, ed_vitals, model_df, all_ecgs_embedded = _load_and_prepare_cardiotwin(
            in_dir, config_path, pbar, steps
        )

        print(f"\n{'='*60}")
        print(f"[Pipeline] Data loaded")
        print(f"  model_df:          {model_df.shape}")
        print(f"  ed_vitals:         {ed_vitals.shape}")
        print(f"  ecgs embedded:     {all_ecgs_embedded.shape}")
        label_cols_check = [c for c in LABEL_COLS if c in model_df.columns]
        print(f"  label cols in model_df: {len(label_cols_check)}/17 — "
              + (f"{label_cols_check}" if len(label_cols_check) <= 5
                 else f"{label_cols_check[:5]}..."))
        if len(label_cols_check) == 0:
            raise ValueError("CRITICAL: No label columns in model_df. Check create_model_df().")
        pos_counts = model_df[label_cols_check].sum()
        print(f"  positive label counts:\n{pos_counts.to_string()}")
        print(f"{'='*60}")

        # Patient-level split
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(model_df, groups=model_df["subject_id"].astype(int)))
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=val_random_state)
        rel_train_idx, rel_val_idx = next(
            val_splitter.split(model_df.iloc[train_idx],
                               groups=model_df.iloc[train_idx]["subject_id"].astype(int))
        )
        train_df = model_df.iloc[train_idx].iloc[rel_train_idx].reset_index(drop=True)
        val_df = model_df.iloc[train_idx].iloc[rel_val_idx].reset_index(drop=True)
        test_df = model_df.iloc[test_idx].reset_index(drop=True)

        def get_ids(df):
            return list(zip(df["subject_id"].astype(int), df["ed_stay_id"]))

        train_ids = get_ids(train_df)
        val_ids = get_ids(val_df)
        test_ids = get_ids(test_df)

        print(f"\n[Pipeline] Patient-level split")
        print(f"  train: {len(train_df)} stays ({train_df['subject_id'].nunique()} patients)")
        print(f"  val:   {len(val_df)} stays ({val_df['subject_id'].nunique()} patients)")
        print(f"  test:  {len(test_df)} stays ({test_df['subject_id'].nunique()} patients)")
        # Check for patient leakage across splits
        train_pats = set(train_df["subject_id"])
        val_pats   = set(val_df["subject_id"])
        test_pats  = set(test_df["subject_id"])
        tv_overlap = train_pats & val_pats
        tt_overlap = train_pats & test_pats
        if tv_overlap or tt_overlap:
            print(f"  WARNING: patient overlap — train/val:{len(tv_overlap)} train/test:{len(tt_overlap)}")
        else:
            print(f"  ✓ No patient leakage across splits")

        label_cols_present = [c for c in LABEL_COLS if c in model_df.columns]
        id_label_cols = ["subject_id", "ed_stay_id"] + label_cols_present
        train_labels = train_df[id_label_cols]
        val_labels = val_df[id_label_cols]
        test_labels = test_df[id_label_cols]

        print(f"\n[Pipeline] Labels")
        print(f"  label_cols_present: {len(label_cols_present)}/17")
        if label_cols_present:
            print(f"  train positives per label:\n{train_labels[label_cols_present].sum().to_string()}")

        pbar.set_description(steps[6])
        train_sids = set(train_df["subject_id"])
        val_sids = set(val_df["subject_id"])
        test_sids = set(test_df["subject_id"])

        train_vital_feat, vital_scaler = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(train_sids)], fit_scaler=True)
        val_vital_feat, _ = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(val_sids)], scaler=vital_scaler)
        test_vital_feat, _ = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(test_sids)], scaler=vital_scaler)

        train_seqs, seq_scaler = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(train_sids)], fit_scaler=True)
        val_seqs, _ = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(val_sids)], vital_scaler=seq_scaler)
        test_seqs, _ = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(test_sids)], vital_scaler=seq_scaler)

        actual_vital_dim = len([c for c in VITAL_COLS if c in ed_vitals.columns])
        actual_vital_stat = len([c for c in train_vital_feat.columns if c.startswith("vf_")])
        pbar.update(1)

        # Filter IDs to those with vitals coverage
        def filter_to_vitals(ids, vital_feat_df):
            valid = set(zip(vital_feat_df["subject_id"].astype(int), vital_feat_df["ed_stay_id"]))
            filtered = [(sid, stay) for sid, stay in ids if (sid, stay) in valid]
            return filtered

        before = len(train_ids), len(val_ids), len(test_ids)
        train_ids = filter_to_vitals(train_ids, train_vital_feat)
        val_ids   = filter_to_vitals(val_ids,   val_vital_feat)
        test_ids  = filter_to_vitals(test_ids,  test_vital_feat)
        after = len(train_ids), len(val_ids), len(test_ids)
        print(f"\n[Pipeline] Vitals filter")
        print(f"  vital_dim={actual_vital_dim} stat_features={actual_vital_stat}")
        print(f"  train: {before[0]} -> {after[0]} stays retained")
        print(f"  val:   {before[1]} -> {after[1]} stays retained")
        print(f"  test:  {before[2]} -> {after[2]} stays retained")
        dropped = sum(b - a for b, a in zip(before, after))
        if dropped > 0:
            print(f"  WARNING: {dropped} stays dropped (no vitals data)")

        pbar.set_description(steps[7])
        train_ehr, ehr_scaler, ehr_feat_names = prepare_ehr_features(model_df, train_ids, fit_scaler=True)
        val_ehr, _, _ = prepare_ehr_features(model_df, val_ids, scaler=ehr_scaler)
        test_ehr, _, _ = prepare_ehr_features(model_df, test_ids, scaler=ehr_scaler)
        ehr_dim = train_ehr.shape[1]
        print(f"\n[Pipeline] EHR features: ehr_dim={ehr_dim}")
        print(f"  train_ehr={train_ehr.shape} val_ehr={val_ehr.shape} test_ehr={test_ehr.shape}")
        pbar.update(1)

        pbar.set_description(steps[8])
        ecg_dict = prepare_ecg(all_ecgs_embedded, max_n=max_n, ecg_fm_dim=ecg_fm_dim)
        train_ecg_cov = sum(1 for sid, stay in train_ids if (int(sid), stay) in ecg_dict)
        print(f"\n[Pipeline] ECG coverage in train: {train_ecg_cov}/{len(train_ids)} "
              f"({100*train_ecg_cov/max(len(train_ids),1):.1f}%)")
        collate = partial(collate_fn, max_N=max_n, max_T=max_t, ecg_fm_dim=ecg_fm_dim)

        def make_loader(ids, vf, ehr, labels, seqs, shuffle):
            return DataLoader(
                CardioEDDataset(ids, vf, ecg_dict, ehr, labels,
                                vital_sequences=seqs, vital_dim=actual_vital_dim, ecg_fm_dim=ecg_fm_dim),
                batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                num_workers=num_workers, pin_memory=True,
            )

        train_loader = make_loader(train_ids, train_vital_feat, train_ehr, train_labels, train_seqs, True)
        val_loader = make_loader(val_ids, val_vital_feat, val_ehr, val_labels, val_seqs, False)
        test_loader = make_loader(test_ids, test_vital_feat, test_ehr, test_labels, test_seqs, False)
        pbar.update(1)

        pbar.set_description(steps[9])
        model = CardioTwinED(
            vital_stat=actual_vital_stat,
            vital_dim=actual_vital_dim,
            ehr_dim=ehr_dim,
            ecg_emb_dim=ecg_fm_dim,
            enc_dim=enc_dim,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
            n_labels=len(label_cols_present),
        ).to(device)
        model.set_ehr_dim(ehr_dim, device=device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Pipeline] CardioTwinED built")
        print(f"  device={device} | trainable_params={total_params:,}")
        print(f"  vital_stat={actual_vital_stat} vital_dim={actual_vital_dim} "
              f"ehr_dim={ehr_dim} ecg_emb_dim={ecg_fm_dim}")
        print(f"  enc_dim={enc_dim} hidden_dim={hidden_dim} lstm_hidden={lstm_hidden} "
              f"n_labels={len(label_cols_present)}")
        print(f"\n[Pipeline] Starting training: epochs={epochs} batch={batch_size} lr={lr}")

        model, test_auc, per_label_aucs = _train_cardiotwin_model(
            model, train_loader, val_loader, test_loader, params, out_path, device
        )
        pbar.update(1)

        pbar.set_description(steps[10])
        plot_kfold_loss_curves_cardiotwin(
            lambda fold: train_loader,
            lambda fold: val_loader,
            lambda: CardioTwinED(
                vital_stat=actual_vital_stat,
                vital_dim=actual_vital_dim,
                ehr_dim=ehr_dim,
                ecg_emb_dim=ecg_fm_dim,
                enc_dim=enc_dim,
                hidden_dim=hidden_dim,
                lstm_hidden=lstm_hidden,
                dropout=dropout,
                n_labels=len(label_cols_present),
            ).to(device),
            label_cols_present,
            params,
            out_path,
            device,
            n_folds=3,
            model_name="cardiotwin_baseline",
        )
        pbar.update(1)

        pbar.set_description(steps[11])
        evaluate_and_visualize_cardiotwin(
            model, test_loader, label_cols_present, out_path,
            model_name="cardiotwin_baseline", device=device
        )

        # Generate digital twin trajectories
        candidates = [
            (sid, stay_id) for sid, stay_id in test_ids
            if test_seqs.get((sid, stay_id)) is not None
            and len(test_seqs[(sid, stay_id)]) >= min_trajectory_steps
        ][:n_trajectory_samples]

        # Bug 6: test_ids.index() is O(n) per call and returns wrong row if test_ids
        # was filtered after test_ehr was built. Build O(1) lookup once instead.
        test_id_to_row = {(sid, stay): i for i, (sid, stay) in enumerate(test_ids)}

        for sid, stay_id in candidates:
            raw_seq = test_seqs[(sid, stay_id)]
            ecg_embs = ecg_dict.get((int(sid), stay_id), np.zeros((1, ecg_fm_dim), dtype=np.float32))
            row_idx = test_id_to_row.get((sid, stay_id))
            if row_idx is None:
                print(f"  WARNING: ({sid}, {stay_id}) not found in test_id_to_row, skipping trajectory")
                continue
            ehr_row = test_ehr[row_idx]
            traj = simulate_trajectory(
                model, raw_seq, ecg_embs, ehr_row, device,
                vital_stat_scaler=vital_scaler,
                max_t=max_t, max_n=max_n, ecg_fm_dim=ecg_fm_dim,
            )
            plot_trajectory(traj, patient_id=sid, label_names=label_cols_present,
                            save_path=os.path.join(out_path, f"cardiotwin_baseline/trajectory_{sid}_{stay_id}.png"))

        pbar.close()

    except Exception as e:
        pbar.close()
        raise e

    print("\n✓ CardioTwin pipeline complete!")
    return None