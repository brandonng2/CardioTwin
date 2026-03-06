# =============================================================================
# CardioTwin — Modular Multimodal Pipeline
# Multimodal Cardiovascular Diagnosis Prediction from ED Presentations
#
# ARCHITECTURE
#
# Stream 1 — ECG-FM Embeddings (1536-dim pooled per stay)
# Stream 2 — ED Vital Signs: flat stats (N, 30) + raw sequences (N, T, 6)
# Stream 3 — EHR Static Features (dynamically discovered)
#
# MODULAR COMPONENTS (plug-and-play via CardioTwinConfig)
#
#   Fusion modes : "gated"   — softmax gate over 3 modality encodings
#                  "concat"  — concatenate encodings, project to hidden_dim
#                  "mean"    — simple mean of encodings (no learned gate)
#
#   Loss modes   : "bce"          — BCEWithLogitsLoss, uniform weights
#                  "weighted_bce" — BCEWithLogitsLoss with pos_weight
#                  "focal"        — Focal loss (gamma=2 default)
#
#   Sampler modes: None            — standard shuffle
#                  "weighted"      — WeightedRandomSampler on label frequency
#
#   Active modalities: any subset of {"vitals", "ecg", "ehr"}
#
# ABLATION
#   run_modality_ablation() — trains one model per modality combination and
#   produces a comparison CSV + bar chart. Reuses the loaded data so ECG-FM
#   extraction only happens once.
# =============================================================================

import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Set

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
ECG_FM_CONFIG_PATH = _REPO_ROOT / "configs" / "ecg_fm_params.json"

# =============================================================================
# 0. CONSTANTS
# =============================================================================

MAX_T = 12
MAX_N = 2
VITAL_DIM = 6
VITAL_STAT = 30
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
# 1. CARDIOTWIN CONFIG — single object controls all variant axes
# =============================================================================

@dataclass
class CardioTwinConfig:
    """
    Controls every modular axis of the CardioTwin pipeline.

    Parameters
    ----------
    fusion_mode : "gated" | "concat" | "mean"
        How modality encodings are combined before the prediction head.
    loss_mode : "bce" | "weighted_bce" | "focal"
        Training loss. "weighted_bce" uses per-label pos_weight.
        "focal" uses Focal Loss with gamma=focal_gamma.
    focal_gamma : float
        Focusing parameter for Focal Loss (only used when loss_mode="focal").
    sampler_mode : None | "weighted"
        DataLoader sampler. "weighted" uses WeightedRandomSampler on the
        training set so rare-label patients are oversampled.
    active_modalities : set of str
        Subset of {"vitals", "ecg", "ehr"}. Missing modalities are zeroed.
    enc_dim, hidden_dim, lstm_hidden, dropout : architecture hyperparams.
    max_t, max_n : sequence length caps.
    ecg_fm_dim : ECG-FM embedding dimension (1536 with split, 768 without).
    batch_size, learning_rate, weight_decay, epochs, grad_clip_norm : training.
    test_size, val_size, random_state, val_random_state : split config.
    num_workers : DataLoader workers.
    n_trajectory_samples, min_trajectory_steps : trajectory simulation.
    """

    # Modular axes
    fusion_mode: Literal["gated", "concat", "mean"] = "gated"
    loss_mode: Literal["bce", "weighted_bce", "focal"] = "weighted_bce"
    focal_gamma: float = 2.0
    sampler_mode: Optional[Literal["weighted"]] = None
    active_modalities: Set[str] = field(default_factory=lambda: {"vitals", "ecg", "ehr"})

    # Architecture
    enc_dim: int = ENC_DIM
    hidden_dim: int = HIDDEN_DIM
    lstm_hidden: int = LSTM_HIDDEN
    dropout: float = DROPOUT
    ecg_fm_dim: int = ECG_FM_DIM

    # Training
    batch_size: int = BATCH_SIZE
    learning_rate: float = LR
    weight_decay: float = 1e-4
    epochs: int = EPOCHS
    grad_clip_norm: float = 1.0

    # Splits
    max_t: int = MAX_T
    max_n: int = MAX_N
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    val_random_state: int = 0

    # Loader
    num_workers: int = 4

    # Trajectory
    n_trajectory_samples: int = 5
    min_trajectory_steps: int = 3

    @classmethod
    def from_json(cls, config_path: str) -> "CardioTwinConfig":
        """Build a CardioTwinConfig from a cardiotwin_params.json file."""
        cfg = load_config(config_path)
        pl = cfg.get("pipeline", {})
        mdl = cfg.get("model", {})
        trn = cfg.get("training", {})
        return cls(
            max_t=pl.get("max_t", MAX_T),
            max_n=pl.get("max_n", MAX_N),
            test_size=pl.get("test_size", 0.2),
            val_size=pl.get("val_size", 0.1),
            random_state=pl.get("random_state", 42),
            val_random_state=pl.get("val_random_state", 0),
            num_workers=pl.get("num_workers", 4),
            n_trajectory_samples=pl.get("n_trajectory_samples", 5),
            min_trajectory_steps=pl.get("min_trajectory_steps", 3),
            enc_dim=mdl.get("enc_dim", ENC_DIM),
            hidden_dim=mdl.get("hidden_dim", HIDDEN_DIM),
            lstm_hidden=mdl.get("lstm_hidden", LSTM_HIDDEN),
            dropout=mdl.get("dropout", DROPOUT),
            ecg_fm_dim=mdl.get("ecg_fm_dim", ECG_FM_DIM),
            batch_size=trn.get("batch_size", BATCH_SIZE),
            learning_rate=trn.get("learning_rate", LR),
            weight_decay=trn.get("weight_decay", 1e-4),
            epochs=trn.get("epochs", EPOCHS),
            grad_clip_norm=trn.get("grad_clip_norm", 1.0),
        )


# =============================================================================
# 2. LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Multi-label Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Equivalent to BCEWithLogitsLoss when gamma=0.
    pos_weight (optional) mirrors BCEWithLogitsLoss behaviour for class imbalance.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def build_criterion(
    loss_mode: str,
    train_labels_df: pd.DataFrame,
    focal_gamma: float = 2.0,
    device: torch.device = None,
) -> nn.Module:
    """
    Factory for loss functions.

    Parameters
    ----------
    loss_mode : "bce" | "weighted_bce" | "focal"
    train_labels_df : DataFrame with label columns — used to compute pos_weight.
    focal_gamma : gamma for Focal Loss (ignored for bce/weighted_bce).
    device : target device.
    """
    device = device or torch.device("cpu")
    cols = [c for c in LABEL_COLS if c in train_labels_df.columns]
    labels = train_labels_df[cols].values
    n_pos = (labels == 1).sum(axis=0).clip(min=1)
    n_neg = (labels == 0).sum(axis=0)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)

    if loss_mode == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_mode == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_mode == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss_mode '{loss_mode}'. Choose: bce, weighted_bce, focal")


# =============================================================================
# 3. STREAM 1 — ECG-FM EMBEDDINGS
# =============================================================================

def _attach_ecg_embeddings_all(ed_ecg_records: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Extract ECG-FM embeddings for ALL ECG records (every ECG per stay)."""
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
    Group ECG-FM embeddings by (subject_id, ed_stay_id).
    Returns dict: (subject_id int, ed_stay_id) -> np.array (N, ecg_fm_dim).
    """
    emb_cols = [f"emb_{i}" for i in range(ecg_fm_dim)]
    stay_col = "ed_stay_id" if "ed_stay_id" in ecg_df.columns else "stay_id"

    ecg_dict = {}
    for (sid, stay_id), group in ecg_df.groupby(["subject_id", stay_col]):
        if "ecg_time" in group.columns:
            group = group.sort_values("ecg_time")
        embs = group[emb_cols].values.astype(np.float32)
        ecg_dict[(int(sid), stay_id)] = embs[:max_n]
    return ecg_dict


# =============================================================================
# 4. STREAM 2 — ED VITAL SIGNS
# =============================================================================

def create_vital_features(ed_vitals_df: pd.DataFrame, scaler=None, fit_scaler: bool = False):
    """Per-stay statistical summary (N, 30): mean/min/max/std/delta for 6 vitals."""
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

    return vital_feat_df, scaler


def create_vital_sequences(ed_vitals_df: pd.DataFrame, vital_scaler=None, fit_scaler: bool = False):
    """
    Raw vital sequences per stay for the LSTM branch.
    Returns dict: (subject_id, stay_id) -> np.array (T, n_vitals), T <= MAX_T.
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
    return sequences, vital_scaler


# =============================================================================
# 5. STREAM 3 — EHR STATIC FEATURES
# =============================================================================

def prepare_ehr_features(model_df: pd.DataFrame, subject_stay_ids: list,
                          scaler=None, fit_scaler: bool = False):
    """
    EHR feature matrix from model_df. Discovers columns dynamically.
    Returns (X float32 (N, ehr_dim), scaler, feature_names).
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
        raise ValueError("No numeric EHR feature columns found in model_df.")

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

    return X, scaler, all_feat_cols


# =============================================================================
# 6. DATASET + DATALOADER
# =============================================================================

class CardioEDDataset(Dataset):
    def __init__(self, subject_stay_ids, vital_feat_df, ecg_dict, ehr_matrix,
                 labels_df, vital_sequences=None, vital_dim=VITAL_DIM, ecg_fm_dim=ECG_FM_DIM):
        self.ids = subject_stay_ids
        self.ecg_dict = ecg_dict
        self.ehr = ehr_matrix
        self.vital_sequences = vital_sequences or {}
        self.vital_dim = vital_dim
        self.ecg_fm_dim = ecg_fm_dim
        self.vital_feats = vital_feat_df.set_index(["subject_id", "ed_stay_id"])
        self.labels = labels_df.set_index(["subject_id", "ed_stay_id"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid, stay_id = self.ids[idx]
        feat_cols = [c for c in self.vital_feats.columns if c.startswith("vf_")]
        vital_feats = self.vital_feats.loc[(sid, stay_id)][feat_cols].values.astype(np.float32)
        ecg = self.ecg_dict.get((int(sid), stay_id),
                                 np.zeros((1, self.ecg_fm_dim), dtype=np.float32))
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


def build_sampler(dataset: CardioEDDataset) -> WeightedRandomSampler:
    """
    WeightedRandomSampler: patients with rarer positive labels are upsampled.
    Weight per patient = mean positive-label frequency across their active labels.
    """
    label_cols = [c for c in LABEL_COLS if c in dataset.labels.columns]
    label_freqs = dataset.labels[label_cols].mean(axis=0).clip(lower=1e-6)

    weights = []
    for ids in dataset.ids:
        row = dataset.labels.loc[ids][label_cols].values.astype(np.float32)
        active = row > 0
        if active.sum() > 0:
            w = 1.0 / float(label_freqs[active].mean())
        else:
            w = 1.0
        weights.append(w)

    weights_t = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(weights=weights_t, num_samples=len(weights_t), replacement=True)


def make_loader(
    ids, vital_feat, ehr, labels, seqs,
    cfg: CardioTwinConfig,
    actual_vital_dim: int,
    ecg_dict: dict,
    shuffle: bool,
    use_sampler: bool = False,
) -> DataLoader:
    """
    Build a DataLoader. If use_sampler=True and cfg.sampler_mode="weighted",
    replaces shuffle with WeightedRandomSampler.
    """
    dataset = CardioEDDataset(
        ids, vital_feat, ecg_dict, ehr, labels,
        vital_sequences=seqs, vital_dim=actual_vital_dim, ecg_fm_dim=cfg.ecg_fm_dim,
    )
    collate = partial(collate_fn, max_N=cfg.max_n, max_T=cfg.max_t, ecg_fm_dim=cfg.ecg_fm_dim)

    sampler = None
    if use_sampler and cfg.sampler_mode == "weighted":
        sampler = build_sampler(dataset)
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


# =============================================================================
# 7. MODEL — Pluggable Fusion (CardioTwinED)
# =============================================================================

class CardioTwinED(nn.Module):
    """
    Multimodal Cardiovascular Prediction Network.

    fusion_mode controls how the three modality encodings are combined:
      "gated"  — learned softmax gate (original behaviour)
      "concat" — concatenate then project via MLP
      "mean"   — simple unweighted mean (no gate parameters)

    active_modalities controls which streams are live:
      Any subset of {"vitals", "ecg", "ehr"}.
      Inactive streams are zeroed before the fusion step.
    """

    def __init__(
        self,
        vital_stat: int = VITAL_STAT,
        vital_dim: int = VITAL_DIM,
        ehr_dim: Optional[int] = None,
        ecg_emb_dim: int = ECG_FM_DIM,
        enc_dim: int = ENC_DIM,
        hidden_dim: int = HIDDEN_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        dropout: float = DROPOUT,
        n_labels: int = N_LABELS,
        fusion_mode: str = "gated",
        active_modalities: Optional[Set[str]] = None,
    ):
        super().__init__()
        self._enc_dim = enc_dim
        self._dropout = dropout
        self._lstm_hidden = lstm_hidden
        self.fusion_mode = fusion_mode
        self.active_modalities = active_modalities or {"vitals", "ecg", "ehr"}

        # --- Vital encoder (LSTM + stats MLP) ---
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

        # --- ECG encoder ---
        self.ecg_attn = nn.Linear(ecg_emb_dim, 1)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_emb_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, enc_dim), nn.GELU(),
        )

        # --- EHR encoder (lazy, built in set_ehr_dim) ---
        self.ehr_encoder: Optional[nn.Module] = None
        self.gate: Optional[nn.Module] = None

        # --- Fusion head ---
        n_active = len(self.active_modalities)
        n_active = max(n_active, 1)

        if fusion_mode == "gated":
            # Gate and fusion projection built in set_ehr_dim (needs ehr_dim)
            fusion_in = enc_dim
        elif fusion_mode == "concat":
            fusion_in = enc_dim * n_active
        else:  # mean
            fusion_in = enc_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.dx_head = nn.Linear(hidden_dim // 2, n_labels)

    def set_ehr_dim(self, ehr_dim: int, device=None):
        """Build EHR encoder (and gate if fusion_mode='gated') once ehr_dim is known."""
        enc_dim = self._enc_dim
        dropout = self._dropout
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, enc_dim), nn.GELU(),
        )
        if self.fusion_mode == "gated":
            n_active = len(self.active_modalities)
            self.gate = nn.Sequential(
                nn.Linear(enc_dim * n_active, 128), nn.ReLU(),
                nn.Linear(128, n_active),
            )
        if device is not None:
            self.ehr_encoder = self.ehr_encoder.to(device)
            if self.gate is not None:
                self.gate = self.gate.to(device)

    def _encode_modalities(self, vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask, device, B):
        """Encode each modality; zero out inactive ones."""
        encodings = {}

        # Vitals
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
        if "vitals" not in self.active_modalities:
            vital_enc = torch.zeros_like(vital_enc)
        encodings["vitals"] = vital_enc

        # ECG
        attn_scores = self.ecg_attn(ecg_embs)
        if ecg_mask is not None:
            attn_scores = attn_scores.masked_fill(~ecg_mask.unsqueeze(-1), float("-inf"))
        ecg_pooled = (torch.softmax(attn_scores, dim=1) * ecg_embs).sum(dim=1)
        ecg_enc = self.ecg_encoder(ecg_pooled)
        if "ecg" not in self.active_modalities:
            ecg_enc = torch.zeros_like(ecg_enc)
        encodings["ecg"] = ecg_enc

        # EHR
        if self.ehr_encoder is None:
            self.set_ehr_dim(ehr_x.size(1), device=device)
        ehr_enc = self.ehr_encoder(ehr_x)
        if "ehr" not in self.active_modalities:
            ehr_enc = torch.zeros_like(ehr_enc)
        encodings["ehr"] = ehr_enc

        return encodings

    def forward(self, vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask=None):
        B = vital_feats.size(0)
        device = vital_feats.device

        if self.ehr_encoder is None:
            self.set_ehr_dim(ehr_x.size(1), device=device)

        encodings = self._encode_modalities(
            vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask, device, B
        )
        vital_enc = encodings["vitals"]
        ecg_enc = encodings["ecg"]
        ehr_enc = encodings["ehr"]

        # Ordered list of active encoding tensors
        active_encs = [encodings[m] for m in ("vitals", "ecg", "ehr") if m in self.active_modalities]
        if not active_encs:
            active_encs = [vital_enc]  # fallback

        if self.fusion_mode == "gated":
            gate_input = torch.cat(active_encs, dim=1)
            gates = F.softmax(self.gate(gate_input), dim=1)
            fused = sum(gates[:, i:i+1] * enc for i, enc in enumerate(active_encs))
        elif self.fusion_mode == "concat":
            fused = torch.cat(active_encs, dim=1)
        else:  # mean
            fused = torch.stack(active_encs, dim=0).mean(dim=0)
            gates = None

        shared = self.fusion(fused)
        logits = self.dx_head(shared)

        out = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "latent": shared,
        }
        if self.fusion_mode == "gated":
            # Expand to always-3 for downstream trajectory plots
            full_gates = torch.zeros(B, 3, device=device)
            active_order = [i for i, m in enumerate(("vitals", "ecg", "ehr"))
                            if m in self.active_modalities]
            for gate_idx, mod_idx in enumerate(active_order):
                full_gates[:, mod_idx] = gates[:, gate_idx]
            out["gates"] = full_gates
        else:
            out["gates"] = torch.zeros(B, 3, device=device)

        return out


def build_model(
    cfg: CardioTwinConfig,
    vital_stat: int,
    vital_dim: int,
    ehr_dim: int,
    n_labels: int,
    device: torch.device,
) -> CardioTwinED:
    """Instantiate and move CardioTwinED to device using a CardioTwinConfig."""
    model = CardioTwinED(
        vital_stat=vital_stat,
        vital_dim=vital_dim,
        ehr_dim=ehr_dim,
        ecg_emb_dim=cfg.ecg_fm_dim,
        enc_dim=cfg.enc_dim,
        hidden_dim=cfg.hidden_dim,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout,
        n_labels=n_labels,
        fusion_mode=cfg.fusion_mode,
        active_modalities=cfg.active_modalities,
    ).to(device)
    model.set_ehr_dim(ehr_dim, device=device)
    return model


# =============================================================================
# 8. TRAINING UTILITIES
# =============================================================================

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
    """Returns (mean_loss, macro_auc, per_label_aucs)."""
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
    per_label_aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            per_label_aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        else:
            per_label_aucs.append(np.nan)
    valid = [a for a in per_label_aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid)) if valid else 0.0
    return total_loss / len(loader), macro_auc, per_label_aucs


def train_cardiotwin(
    model: CardioTwinED,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: CardioTwinConfig,
    criterion: nn.Module,
    out_path: str,
    device: torch.device,
    model_name: str = "cardiotwin",
) -> tuple:
    """
    Core training loop with early stopping.
    Returns (model, test_auc, per_label_aucs).

    Decoupled from CardioTwinConfig so different criterion / loader combos
    can be passed in freely.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_val_auc = 0.0
    best_model_pt = os.path.join(out_path, f"best_{model_name}.pt")
    patience = 10
    no_improve = 0

    for epoch in range(cfg.epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip_norm)
        _, val_auc, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_pt)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(torch.load(best_model_pt, map_location=device))
    _, test_auc, per_label_aucs = eval_epoch(model, test_loader, criterion, device)
    return model, test_auc, per_label_aucs


# =============================================================================
# 9. MODALITY ABLATION
# =============================================================================

# All 7 combinations of the three modalities
MODALITY_COMBOS = [
    {"vitals", "ecg", "ehr"},   # Full model
    {"vitals", "ecg"},
    {"vitals", "ehr"},
    {"ecg", "ehr"},
    {"vitals"},
    {"ecg"},
    {"ehr"},
]


def _combo_label(combo: set) -> str:
    ordered = ["vitals", "ecg", "ehr"]
    return "+".join(m for m in ordered if m in combo)


def run_modality_ablation(
    cfg: CardioTwinConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vital_stat: int,
    vital_dim: int,
    ehr_dim: int,
    n_labels: int,
    label_cols_present: list,
    out_path: str,
    device: torch.device,
    combos: Optional[list] = None,
) -> pd.DataFrame:
    """
    Train one CardioTwinED per modality combination and compare test AUCs.

    Parameters
    ----------
    combos : list of sets, optional
        Override the default MODALITY_COMBOS. Useful if you only want a subset.
    All loaders are reused across combos — ECG-FM extraction runs only once upstream.

    Returns
    -------
    pd.DataFrame with columns: [modalities, macro_auc, <per_label_aucs...>]
    Also saves ablation_results.csv and ablation_bar.png to out_path/ablation/.
    """
    combos = combos or MODALITY_COMBOS
    Path(out_path).mkdir(parents=True, exist_ok=True)
    ablation_dir = Path(out_path) / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for combo in combos:
        name = _combo_label(combo)
        print(f"  Ablation: {name}")

        # Build a fresh config with only this combo active
        combo_cfg = CardioTwinConfig(
            fusion_mode=cfg.fusion_mode,
            loss_mode=cfg.loss_mode,
            focal_gamma=cfg.focal_gamma,
            sampler_mode=cfg.sampler_mode,
            active_modalities=combo,
            enc_dim=cfg.enc_dim,
            hidden_dim=cfg.hidden_dim,
            lstm_hidden=cfg.lstm_hidden,
            dropout=cfg.dropout,
            ecg_fm_dim=cfg.ecg_fm_dim,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            grad_clip_norm=cfg.grad_clip_norm,
            num_workers=cfg.num_workers,
        )

        model = build_model(combo_cfg, vital_stat, vital_dim, ehr_dim, n_labels, device)
        criterion = build_criterion(
            combo_cfg.loss_mode,
            train_loader.dataset.labels,
            focal_gamma=combo_cfg.focal_gamma,
            device=device,
        )
        model, test_auc, per_label_aucs = train_cardiotwin(
            model, train_loader, val_loader, test_loader,
            combo_cfg, criterion, str(ablation_dir), device,
            model_name=f"ablation_{name.replace('+', '_')}",
        )

        row = {"modalities": name, "macro_auc": test_auc}
        for label, auc in zip(label_cols_present, per_label_aucs):
            row[label] = auc
        records.append(row)
        print(f"    macro AUC = {test_auc:.4f}")

    results_df = pd.DataFrame(records).sort_values("macro_auc", ascending=False)
    csv_path = ablation_dir / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2E5090" if i == 0 else "#6B9AC4" for i in range(len(results_df))]
    ax.barh(results_df["modalities"], results_df["macro_auc"], color=colors)
    ax.set_xlabel("Macro ROC-AUC", fontsize=12)
    ax.set_title("Modality Ablation — Macro AUC by Active Streams", fontsize=14)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig_path = ablation_dir / "ablation_bar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Ablation complete. Results: {csv_path}")
    return results_df


# =============================================================================
# 10. DIGITAL TWIN — TRAJECTORY SIMULATION
# =============================================================================

@torch.no_grad()
def simulate_trajectory(model, patient_vitals_raw, ecg_embs, ehr_x,
                         device, vital_stat_scaler=None,
                         max_t: int = MAX_T, max_n: int = MAX_N,
                         ecg_fm_dim: int = ECG_FM_DIM) -> list:
    """Simulate evolving cardiovascular state across an ED stay."""
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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


# =============================================================================
# 11. EVALUATION
# =============================================================================

def evaluate_and_visualize_cardiotwin(
    model, test_loader, label_cols_present, out_path: str,
    model_name: str = "cardiotwin", device=None,
) -> pd.DataFrame:
    """ROC + PR curves + aggregated confusion matrix + per-label CSV."""
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

    results = []
    for i, label in enumerate(label_cols_present):
        n_pos = y_true[:, i].sum()
        if len(np.unique(y_true[:, i])) > 1:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        else:
            auc = ap = np.nan
        results.append({"target": label, "n_test_pos": int(n_pos),
                         "pos_rate": y_true[:, i].mean(), "roc_auc": auc, "pr_auc": ap})

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    valid_labels = [l for l in label_cols_present
                    if len(np.unique(y_true[:, label_cols_present.index(l)])) > 1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for label in valid_labels:
        idx = label_cols_present.index(label)
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
        auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
        color = "#2E5090" if auc >= 0.85 else ("#6B46C1" if auc >= 0.75 else "#D32F2F")
        axes[0].plot(fpr, tpr, linewidth=1.5, alpha=0.5, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=2)
    axes[0].set_title(f"ROC Curves\nMean AUC: {results_df['roc_auc'].mean():.3f}", fontsize=14)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, alpha=0.3)

    for label in valid_labels:
        idx = label_cols_present.index(label)
        prec, rec, _ = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        ap = average_precision_score(y_true[:, idx], y_pred[:, idx])
        color = "#2E5090" if ap >= 0.5 else ("#6B46C1" if ap >= 0.3 else "#D32F2F")
        axes[1].plot(rec, prec, linewidth=1.5, alpha=0.5, color=color)

    axes[1].set_title(f"PR Curves\nMean PR-AUC: {results_df['pr_auc'].mean():.3f}", fontsize=14)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
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
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")

    plt.tight_layout()
    fig_path = Path(out_path) / model_name / f"{model_name}_evaluation_plots.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = Path(out_path) / model_name / f"{model_name}_results.csv"
    results_df.to_csv(csv_path, index=False)
    return results_df


# =============================================================================
# 12. DATA LOADING HELPERS
# =============================================================================

def _load_and_prepare_cardiotwin(in_dir, config_path, pbar, steps) -> tuple:
    """Load and prepare all data: filters, ECG embeddings, vitals, EHR."""
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


# =============================================================================
# 13. PIPELINE ENTRY POINTS
# =============================================================================

def run_cardiotwin_pipeline(in_dir, config_path, out_path, cfg: Optional[CardioTwinConfig] = None):
    """
    Full CardioTwin pipeline using a CardioTwinConfig.

    If cfg is None, defaults are loaded from config_path via CardioTwinConfig.from_json().
    Pass a custom cfg to override fusion_mode, loss_mode, sampler_mode, etc.

    Example — switch to Focal Loss + WeightedRandomSampler:
        cfg = CardioTwinConfig.from_json("configs/cardiotwin_params.json")
        cfg.loss_mode = "focal"
        cfg.sampler_mode = "weighted"
        run_cardiotwin_pipeline(in_dir, config_path, out_path, cfg=cfg)
    """
    if cfg is None:
        cfg = CardioTwinConfig.from_json(config_path)

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
        "Evaluating + generating trajectories",
    ]

    pbar = tqdm(total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}")

    Path(out_path).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        raw_config, ed_vitals, model_df, all_ecgs_embedded = \
            _load_and_prepare_cardiotwin(in_dir, config_path, pbar, steps)

        # --- Patient-level splits ---
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        train_idx, test_idx = next(splitter.split(model_df, groups=model_df["subject_id"].astype(int)))
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.val_random_state)
        rel_tr, rel_val = next(
            val_splitter.split(model_df.iloc[train_idx],
                               groups=model_df.iloc[train_idx]["subject_id"].astype(int))
        )
        train_df = model_df.iloc[train_idx].iloc[rel_tr].reset_index(drop=True)
        val_df = model_df.iloc[train_idx].iloc[rel_val].reset_index(drop=True)
        test_df = model_df.iloc[test_idx].reset_index(drop=True)

        def get_ids(df):
            return list(zip(df["subject_id"].astype(int), df["ed_stay_id"]))

        train_ids, val_ids, test_ids = get_ids(train_df), get_ids(val_df), get_ids(test_df)
        label_cols_present = [c for c in LABEL_COLS if c in model_df.columns]
        id_label_cols = ["subject_id", "ed_stay_id"] + label_cols_present

        pbar.set_description(steps[6])
        for split_ids, split_sids in [(train_ids, set(train_df["subject_id"])),
                                       (val_ids, set(val_df["subject_id"])),
                                       (test_ids, set(test_df["subject_id"]))]:
            pass  # Just verifying structure

        train_vital_feat, vital_scaler = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(set(train_df["subject_id"]))], fit_scaler=True)
        val_vital_feat, _ = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(set(val_df["subject_id"]))], scaler=vital_scaler)
        test_vital_feat, _ = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(set(test_df["subject_id"]))], scaler=vital_scaler)

        train_seqs, seq_scaler = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(set(train_df["subject_id"]))], fit_scaler=True)
        val_seqs, _ = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(set(val_df["subject_id"]))], vital_scaler=seq_scaler)
        test_seqs, _ = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(set(test_df["subject_id"]))], vital_scaler=seq_scaler)

        actual_vital_dim = len([c for c in VITAL_COLS if c in ed_vitals.columns])
        actual_vital_stat = len([c for c in train_vital_feat.columns if c.startswith("vf_")])
        pbar.update(1)

        pbar.set_description(steps[7])
        train_ehr, ehr_scaler, _ = prepare_ehr_features(model_df, train_ids, fit_scaler=True)
        val_ehr, _, _ = prepare_ehr_features(model_df, val_ids, scaler=ehr_scaler)
        test_ehr, _, _ = prepare_ehr_features(model_df, test_ids, scaler=ehr_scaler)
        ehr_dim = train_ehr.shape[1]
        pbar.update(1)

        pbar.set_description(steps[8])
        ecg_dict = prepare_ecg(all_ecgs_embedded, max_n=cfg.max_n, ecg_fm_dim=cfg.ecg_fm_dim)

        train_loader = make_loader(
            train_ids, train_vital_feat, train_ehr, train_df[id_label_cols],
            train_seqs, cfg, actual_vital_dim, ecg_dict, shuffle=True, use_sampler=True,
        )
        val_loader = make_loader(
            val_ids, val_vital_feat, val_ehr, val_df[id_label_cols],
            val_seqs, cfg, actual_vital_dim, ecg_dict, shuffle=False,
        )
        test_loader = make_loader(
            test_ids, test_vital_feat, test_ehr, test_df[id_label_cols],
            test_seqs, cfg, actual_vital_dim, ecg_dict, shuffle=False,
        )
        pbar.update(1)

        pbar.set_description(steps[9])
        model = build_model(cfg, actual_vital_stat, actual_vital_dim, ehr_dim,
                             len(label_cols_present), device)
        criterion = build_criterion(cfg.loss_mode, train_df[id_label_cols],
                                     focal_gamma=cfg.focal_gamma, device=device)
        model_name = f"cardiotwin_{cfg.fusion_mode}_{cfg.loss_mode}"
        model, test_auc, per_label_aucs = train_cardiotwin(
            model, train_loader, val_loader, test_loader,
            cfg, criterion, out_path, device, model_name=model_name,
        )
        pbar.update(1)

        pbar.set_description(steps[10])
        evaluate_and_visualize_cardiotwin(
            model, test_loader, label_cols_present, out_path,
            model_name=model_name, device=device,
        )

        # Trajectory simulation
        candidates = [
            (sid, stay_id) for sid, stay_id in test_ids
            if test_seqs.get((sid, stay_id)) is not None
            and len(test_seqs[(sid, stay_id)]) >= cfg.min_trajectory_steps
        ][:cfg.n_trajectory_samples]

        for sid, stay_id in candidates:
            raw_seq = test_seqs[(sid, stay_id)]
            ecg_embs = ecg_dict.get((int(sid), stay_id),
                                     np.zeros((1, cfg.ecg_fm_dim), dtype=np.float32))
            row_idx = test_ids.index((sid, stay_id))
            ehr_row = test_ehr[row_idx]
            traj = simulate_trajectory(
                model, raw_seq, ecg_embs, ehr_row, device,
                vital_stat_scaler=vital_scaler,
                max_t=cfg.max_t, max_n=cfg.max_n, ecg_fm_dim=cfg.ecg_fm_dim,
            )
            plot_trajectory(
                traj, patient_id=sid, label_names=label_cols_present,
                save_path=os.path.join(out_path, model_name,
                                        f"trajectory_{sid}_{stay_id}.png"),
            )
        pbar.update(1)
        pbar.close()

    except Exception as e:
        pbar.close()
        raise e

    print(f"\n✓ CardioTwin pipeline complete! (fusion={cfg.fusion_mode}, loss={cfg.loss_mode})")
    return None


def run_cardiotwin_ablation_pipeline(in_dir, config_path, out_path,
                                      cfg: Optional[CardioTwinConfig] = None,
                                      combos: Optional[list] = None):
    """
    Run the full modality ablation study.

    Loads data once, then trains one model per modality combo in `combos`
    (defaults to all 7 combinations). Results saved to out_path/ablation/.

    Parameters
    ----------
    combos : list of sets, optional
        e.g. [{"vitals", "ecg", "ehr"}, {"ecg", "ehr"}, {"ecg"}]
        Defaults to all 7 subsets of the three modalities.

    Example:
        run_cardiotwin_ablation_pipeline(
            in_dir, config_path, out_path,
            cfg=CardioTwinConfig(fusion_mode="gated", loss_mode="focal"),
        )
    """
    if cfg is None:
        cfg = CardioTwinConfig.from_json(config_path)

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
        "Running modality ablation",
    ]

    pbar = tqdm(total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}")

    Path(out_path).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        raw_config, ed_vitals, model_df, all_ecgs_embedded = \
            _load_and_prepare_cardiotwin(in_dir, config_path, pbar, steps)

        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        train_idx, test_idx = next(splitter.split(model_df, groups=model_df["subject_id"].astype(int)))
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.val_random_state)
        rel_tr, rel_val = next(
            val_splitter.split(model_df.iloc[train_idx],
                               groups=model_df.iloc[train_idx]["subject_id"].astype(int))
        )
        train_df = model_df.iloc[train_idx].iloc[rel_tr].reset_index(drop=True)
        val_df = model_df.iloc[train_idx].iloc[rel_val].reset_index(drop=True)
        test_df = model_df.iloc[test_idx].reset_index(drop=True)

        get_ids = lambda df: list(zip(df["subject_id"].astype(int), df["ed_stay_id"]))
        train_ids, val_ids, test_ids = get_ids(train_df), get_ids(val_df), get_ids(test_df)
        label_cols_present = [c for c in LABEL_COLS if c in model_df.columns]
        id_label_cols = ["subject_id", "ed_stay_id"] + label_cols_present

        pbar.set_description(steps[6])
        train_vital_feat, vital_scaler = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(set(train_df["subject_id"]))], fit_scaler=True)
        val_vital_feat, _ = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(set(val_df["subject_id"]))], scaler=vital_scaler)
        test_vital_feat, _ = create_vital_features(
            ed_vitals[ed_vitals["subject_id"].isin(set(test_df["subject_id"]))], scaler=vital_scaler)

        train_seqs, seq_scaler = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(set(train_df["subject_id"]))], fit_scaler=True)
        val_seqs, _ = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(set(val_df["subject_id"]))], vital_scaler=seq_scaler)
        test_seqs, _ = create_vital_sequences(
            ed_vitals[ed_vitals["subject_id"].isin(set(test_df["subject_id"]))], vital_scaler=seq_scaler)

        actual_vital_dim = len([c for c in VITAL_COLS if c in ed_vitals.columns])
        actual_vital_stat = len([c for c in train_vital_feat.columns if c.startswith("vf_")])
        pbar.update(1)

        pbar.set_description(steps[7])
        train_ehr, ehr_scaler, _ = prepare_ehr_features(model_df, train_ids, fit_scaler=True)
        val_ehr, _, _ = prepare_ehr_features(model_df, val_ids, scaler=ehr_scaler)
        test_ehr, _, _ = prepare_ehr_features(model_df, test_ids, scaler=ehr_scaler)
        ehr_dim = train_ehr.shape[1]
        pbar.update(1)

        pbar.set_description(steps[8])
        ecg_dict = prepare_ecg(all_ecgs_embedded, max_n=cfg.max_n, ecg_fm_dim=cfg.ecg_fm_dim)

        train_loader = make_loader(
            train_ids, train_vital_feat, train_ehr, train_df[id_label_cols],
            train_seqs, cfg, actual_vital_dim, ecg_dict, shuffle=True, use_sampler=True,
        )
        val_loader = make_loader(
            val_ids, val_vital_feat, val_ehr, val_df[id_label_cols],
            val_seqs, cfg, actual_vital_dim, ecg_dict, shuffle=False,
        )
        test_loader = make_loader(
            test_ids, test_vital_feat, test_ehr, test_df[id_label_cols],
            test_seqs, cfg, actual_vital_dim, ecg_dict, shuffle=False,
        )
        pbar.update(1)

        pbar.set_description(steps[9])
        ablation_df = run_modality_ablation(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            vital_stat=actual_vital_stat,
            vital_dim=actual_vital_dim,
            ehr_dim=ehr_dim,
            n_labels=len(label_cols_present),
            label_cols_present=label_cols_present,
            out_path=out_path,
            device=device,
            combos=combos,
        )
        pbar.update(1)
        pbar.close()

    except Exception as e:
        pbar.close()
        raise e

    print("\n✓ Modality ablation pipeline complete!")
    return ablation_df