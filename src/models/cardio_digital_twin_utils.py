import logging
import os
import warnings
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

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
from cardio_digital_twin_classes import CardioEDDataset, collate_fn

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ECG_FM_CONFIG_PATH = _REPO_ROOT / "configs" / "ecg_fm_params.json"

# =============================================================================
# CONSTANTS
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
# STREAM 1 — ECG-FM EMBEDDINGS
# =============================================================================

def _attach_ecg_embeddings_all(ed_ecg_records: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract ECG-FM embeddings for ALL ECG records (every ECG per stay).
    Builds ecg_path and calls run_pooled_ecg_extraction to attach
    1536-dim embeddings (emb_0...emb_1535) to ed_ecg_records.
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
    All ECGs per stay included up to max_n, sorted chronologically.
    """
    emb_cols = [f"emb_{i}" for i in range(ecg_fm_dim)]
    stay_col = "ed_stay_id" if "ed_stay_id" in ecg_df.columns else "stay_id"

    ecg_dict = {}
    for (sid, stay_id), group in ecg_df.groupby(["subject_id", stay_col]):
        if "ecg_time" in group.columns:
            group = group.sort_values("ecg_time")
        embs = group[emb_cols].values.astype(np.float32)
        ecg_dict[(int(sid), stay_id)] = embs[:max_n]

    nan_ecgs = [(k, v) for k, v in ecg_dict.items() if np.isnan(v).any()]
    if nan_ecgs:
        log.warning("%d stays have NaN ECG embeddings — zeroing out", len(nan_ecgs))
        for k, v in nan_ecgs:
            ecg_dict[k] = np.nan_to_num(v, nan=0.0)

    return ecg_dict


# =============================================================================
# STREAM 2 — ED VITAL SIGNS
# =============================================================================

def _preprocess_vitals(df: pd.DataFrame, stay_col: str) -> tuple:
    """Sort, forward-fill, and median-fill vitals. Returns (df, present_vitals)."""
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df.sort_values(["subject_id", stay_col, "charttime"])

    present_vitals = [c for c in VITAL_COLS if c in df.columns]
    df[present_vitals] = (
        df.groupby(["subject_id", stay_col])[present_vitals]
        .transform(lambda x: x.ffill().bfill())
    )
    df[present_vitals] = df[present_vitals].fillna(df[present_vitals].median())
    return df, present_vitals


def create_vital_features(ed_vitals_df: pd.DataFrame, scaler=None, fit_scaler: bool = False):
    """
    Convert long-format ED vitals to per-stay statistical summary (N, 30).
    For each of 6 vitals: mean, min, max, std, delta (last - first).
    Returns (vital_feat_df, scaler).
    """
    df = ed_vitals_df.copy()
    stay_col = "stay_id" if "stay_id" in df.columns else "ed_stay_id"
    df, present_vitals = _preprocess_vitals(df, stay_col)

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
    Return raw vital sequences per stay for the LSTM branch and trajectory sim.
    Returns dict: (subject_id, stay_id) -> np.array (T, n_vitals), T <= MAX_T.
    StandardScaler fitted on train set only when fit_scaler=True.
    """
    df = ed_vitals_df.copy()
    stay_col = "stay_id" if "stay_id" in df.columns else "ed_stay_id"
    df, present_vitals = _preprocess_vitals(df, stay_col)

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
# STREAM 3 — EHR STATIC FEATURES
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
    exclude_exact = {
        "stay_id", "subject_id", "ed_stay_id", "hadm_id", "study_id", "split",
        "is_cardiovascular", "file_name", "cart_id",
        *LABEL_COLS,
    }

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

    return X, scaler, all_feat_cols


# =============================================================================
# TRAINING
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
    """Returns (mean_loss, macro_auc, per_label_aucs) with per_label_aucs[i] = AUC or nan."""
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
    per_label_aucs = [
        roc_auc_score(labels[:, i], probs[:, i]) if labels[:, i].sum() > 0 else np.nan
        for i in range(labels.shape[1])
    ]
    valid_aucs = [a for a in per_label_aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return total_loss / len(loader), macro_auc, per_label_aucs


def train_cardiotwin_model(
    model, train_loader, val_loader, test_loader,
    params: dict, out_path: str, device, model_name: str = "cardio_digital_twin_baseline"
) -> tuple:
    """Train CardioTwinED with early stopping. Returns (model, test_auc, per_label_aucs)."""
    # Weighted BCE (re-enable to up-weight rare labels):
    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=compute_class_weights(train_loader.dataset.labels).to(device)
    # )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    best_val_auc = 0.0
    out_dir = Path(out_path) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_pt = str(out_dir / f"{model_name}.pt")

    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 grad_clip_norm=params["grad_clip_norm"])
        val_loss, val_auc, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_pt)
        log.info("Epoch %3d/%d | train=%.4f val=%.4f auc=%.4f%s",
                 epoch + 1, params["epochs"], train_loss, val_loss, val_auc,
                 " ✓" if improved else "")

    model.load_state_dict(torch.load(best_model_pt, map_location=device))
    _, test_auc, per_label_aucs = eval_epoch(model, test_loader, criterion, device)
    log.info("Best val_auc=%.4f | test_auc=%.4f", best_val_auc, test_auc)
    return model, test_auc, per_label_aucs


# =============================================================================
# ABLATION STUDY
# =============================================================================

@torch.no_grad()
def run_ablations(model, test_loader, criterion, device) -> dict:
    """Zero out each modality in turn to measure individual contribution."""
    ablation_configs = {
        "Full model":  dict(zero_vital=False, zero_ecg=False, zero_ehr=False),
        "No vitals":   dict(zero_vital=True,  zero_ecg=False, zero_ehr=False),
        "No ECG":      dict(zero_vital=False, zero_ecg=True,  zero_ehr=False),
        "No EHR":      dict(zero_vital=False, zero_ecg=False, zero_ehr=True),
        "ECG only":    dict(zero_vital=True,  zero_ecg=False, zero_ehr=True),
        "Vitals only": dict(zero_vital=False, zero_ecg=True,  zero_ehr=True),
        "EHR only":    dict(zero_vital=True,  zero_ecg=True,  zero_ehr=False),
    }
    results = {}
    for name, cfg in ablation_configs.items():
        model.eval()
        all_probs, all_labels = [], []
        for batch in test_loader:
            vf   = batch["vital_feats"].to(device)
            vseq = batch["vital_seq"].to(device)
            vlen = batch["vital_lengths"].to(device)
            ecg  = batch["ecg"].to(device)
            ehr  = batch["ehr"].to(device)
            msk  = batch["ecg_mask"].to(device)
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
# TRAJECTORY SIMULATION
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

    Returns list of dicts: {timestep, probs (n_labels,), gates (3,), latent}
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
# EVALUATION
# =============================================================================

def evaluate_and_visualize_cardiotwin(
    model, test_loader, label_cols_present, out_path: str,
    model_name: str = "cardio_digital_twin", device=None
) -> pd.DataFrame:
    """
    Evaluate CardioTwinED and produce:
      1. ROC curves PNG
      2. Precision-Recall curves PNG
      3. Aggregated binary confusion matrix PNG
      4. Label co-occurrence matrix PNG
      5. Per-label results CSV
      6. Overall results CSV
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_path) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

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
    y_pred_bin = (y_pred >= 0.5).astype(int)

    valid_labels = [
        l for l in label_cols_present
        if len(np.unique(y_true[:, label_cols_present.index(l)])) > 1
    ]

    # --- Per-label results CSV ---
    results = []
    for i, label in enumerate(label_cols_present):
        yt, yp, yb = y_true[:, i], y_pred[:, i], y_pred_bin[:, i]
        has_both = len(np.unique(yt)) > 1
        results.append({
            "target":     label,
            "n_test_pos": int(yt.sum()),
            "pos_rate":   yt.mean(),
            "roc_auc":    roc_auc_score(yt, yp) if has_both else np.nan,
            "pr_auc":     average_precision_score(yt, yp) if has_both else np.nan,
            "precision":  precision_score(yt, yb, zero_division=0),
            "recall":     recall_score(yt, yb, zero_division=0),
            "f1":         f1_score(yt, yb, zero_division=0),
            "accuracy":   accuracy_score(yt, yb),
        })

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    results_df.to_csv(out_dir / f"{model_name}_results.csv", index=False)

    # --- Overall results CSV ---
    y_true_flat = y_true.ravel()
    y_pred_bin_flat = y_pred_bin.ravel()
    overall_df = pd.DataFrame([{
        "mean_roc_auc": results_df["roc_auc"].mean(),
        "mean_pr_auc":  results_df["pr_auc"].mean(),
        "accuracy":     accuracy_score(y_true_flat, y_pred_bin_flat),
        "precision":    precision_score(y_true_flat, y_pred_bin_flat, zero_division=0),
        "recall":       recall_score(y_true_flat, y_pred_bin_flat, zero_division=0),
        "f1":           f1_score(y_true_flat, y_pred_bin_flat, zero_division=0),
    }])
    overall_df.to_csv(out_dir / f"{model_name}_overall_results.csv", index=False)

    # --- ROC curves ---
    mean_auc = results_df["roc_auc"].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in valid_labels:
        idx = label_cols_present.index(label)
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
        auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
        color = "#2E5090" if auc >= 0.85 else ("#6B46C1" if auc >= 0.75 else "#D32F2F")
        ax.plot(fpr, tpr, linewidth=1.5, alpha=0.5, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2)
    ax.set_title("ROC Curves — 17 Diagnosis Labels", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.text(0.98, 0.02, f"Mean AUC: {mean_auc:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Precision-Recall curves ---
    mean_ap = results_df["pr_auc"].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in valid_labels:
        idx = label_cols_present.index(label)
        prec, rec, _ = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        ap = average_precision_score(y_true[:, idx], y_pred[:, idx])
        color = "#2E5090" if ap >= 0.5 else ("#6B46C1" if ap >= 0.3 else "#D32F2F")
        ax.plot(rec, prec, linewidth=1.5, alpha=0.5, color=color)
    ax.set_title("PR Curves — 17 Diagnosis Labels", fontsize=13)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.text(0.98, 0.98, f"Mean PR-AUC: {mean_ap:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Aggregated binary confusion matrix ---
    total_cm = np.zeros((2, 2))
    for label in valid_labels:
        idx = label_cols_present.index(label)
        total_cm += confusion_matrix(y_true[:, idx], y_pred_bin[:, idx])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(total_cm, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], cbar=False)
    ax.set_title("Aggregated Diagnosis Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.text(0.5, -0.18, f"Sum across {len(valid_labels)} labels",
            transform=ax.transAxes, ha="center", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Label co-occurrence matrix ---
    n_valid = len(valid_labels)
    co_matrix = np.zeros((n_valid, n_valid))
    for i, true_label in enumerate(valid_labels):
        true_idx = label_cols_present.index(true_label)
        positive_mask = y_true[:, true_idx] == 1
        if positive_mask.sum() > 0:
            for j, pred_label in enumerate(valid_labels):
                pred_idx = label_cols_present.index(pred_label)
                co_matrix[i, j] = y_pred_bin[positive_mask, pred_idx].sum()

    fig_size = max(10, n_valid * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(co_matrix, cmap="Greens", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prediction Count", rotation=270, labelpad=20, fontsize=11)

    short_names = [l.replace("_", " ") for l in valid_labels]
    ax.set_xticks(np.arange(n_valid))
    ax.set_yticks(np.arange(n_valid))
    ax.set_xticklabels(short_names, rotation=90, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    if n_valid <= 30:
        vmax = co_matrix.max()
        for i in range(n_valid):
            for j in range(n_valid):
                count = int(co_matrix[i, j])
                if count > 0:
                    text_color = "white" if co_matrix[i, j] > vmax / 2 else "black"
                    ax.text(j, i, str(count), ha="center", va="center",
                            color=text_color, fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_cooccurrence_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results_df


# =============================================================================
# K-FOLD LOSS CURVES
# =============================================================================

def plot_kfold_loss_curves_cardiotwin(
    train_loader_fn, val_loader_fn, model_fn, label_cols_present,
    params: dict, out_path: str, device, n_folds: int = 3,
    model_name: str = "cardio_digital_twin"
) -> dict:
    """
    Train CardioTwinED using K-fold cross-validation with early stopping.
    Returns a dict with mean train/val loss arrays and best/stopped epoch info.
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)

    train_loss_folds = np.zeros((n_folds, params["epochs"]))
    val_loss_folds = np.zeros((n_folds, params["epochs"]))
    stopped_epochs = []

    for fold_idx in range(n_folds):
        train_loader = train_loader_fn(fold_idx)
        val_loader = val_loader_fn(fold_idx)

        model = model_fn()
        # Weighted BCE (re-enable to up-weight rare labels):
        # criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=compute_class_weights(train_loader.dataset.labels).to(device)
        # )
        criterion = nn.BCEWithLogitsLoss()
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

            if patience_counter >= 10:
                stopped_at = epoch + 1
                train_loss_folds[fold_idx, epoch + 1:] = train_loss
                val_loss_folds[fold_idx, epoch + 1:] = val_loss
                break

        stopped_epochs.append(stopped_at)

    mean_train = train_loss_folds.mean(axis=0)
    mean_val = val_loss_folds.mean(axis=0)
    best_epoch = int(np.argmin(mean_val))
    mean_stopped = int(np.mean(stopped_epochs))

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


# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def _load_and_prepare_data(in_dir, config_path, pbar, steps) -> tuple:
    """Load raw data, filter, compute ECG intervals, extract embeddings, build model_df."""
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


def _build_splits(model_df, test_size, val_size, random_state, val_random_state):
    """Patient-level train/val/test split. Returns (train_df, val_df, test_df)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(model_df, groups=model_df["subject_id"].astype(int)))

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=val_random_state)
    rel_train_idx, rel_val_idx = next(
        val_splitter.split(
            model_df.iloc[train_idx],
            groups=model_df.iloc[train_idx]["subject_id"].astype(int),
        )
    )

    train_df = model_df.iloc[train_idx].iloc[rel_train_idx].reset_index(drop=True)
    val_df = model_df.iloc[train_idx].iloc[rel_val_idx].reset_index(drop=True)
    test_df = model_df.iloc[test_idx].reset_index(drop=True)

    tv_overlap = set(train_df["subject_id"]) & set(val_df["subject_id"])
    tt_overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
    if tv_overlap or tt_overlap:
        log.warning("Patient leakage — train/val: %d  train/test: %d",
                    len(tv_overlap), len(tt_overlap))

    return train_df, val_df, test_df


def _build_vitals(ed_vitals, train_df, val_df, test_df):
    """
    Fit vital feature and sequence scalers on train, transform all splits.
    Returns (train_feat, val_feat, test_feat, train_seqs, val_seqs, test_seqs,
             vital_scaler, seq_scaler, actual_vital_dim, actual_vital_stat).
    """
    def sids(df):
        return set(df["subject_id"])

    train_sids, val_sids, test_sids = sids(train_df), sids(val_df), sids(test_df)

    train_feat, vital_scaler = create_vital_features(
        ed_vitals[ed_vitals["subject_id"].isin(train_sids)], fit_scaler=True
    )
    val_feat, _ = create_vital_features(
        ed_vitals[ed_vitals["subject_id"].isin(val_sids)], scaler=vital_scaler
    )
    test_feat, _ = create_vital_features(
        ed_vitals[ed_vitals["subject_id"].isin(test_sids)], scaler=vital_scaler
    )

    train_seqs, seq_scaler = create_vital_sequences(
        ed_vitals[ed_vitals["subject_id"].isin(train_sids)], fit_scaler=True
    )
    val_seqs, _ = create_vital_sequences(
        ed_vitals[ed_vitals["subject_id"].isin(val_sids)], vital_scaler=seq_scaler
    )
    test_seqs, _ = create_vital_sequences(
        ed_vitals[ed_vitals["subject_id"].isin(test_sids)], vital_scaler=seq_scaler
    )

    actual_vital_dim = len([c for c in VITAL_COLS if c in ed_vitals.columns])
    actual_vital_stat = len([c for c in train_feat.columns if c.startswith("vf_")])

    return (train_feat, val_feat, test_feat,
            train_seqs, val_seqs, test_seqs,
            vital_scaler, seq_scaler,
            actual_vital_dim, actual_vital_stat)


def _filter_to_vitals(ids: list, vital_feat_df: pd.DataFrame) -> list:
    """Drop stays that have no vitals coverage."""
    valid = set(zip(vital_feat_df["subject_id"].astype(int), vital_feat_df["ed_stay_id"]))
    return [(sid, stay) for sid, stay in ids if (sid, stay) in valid]


def _build_ehr(model_df, train_ids, val_ids, test_ids):
    """Fit EHR scaler on train, transform all splits.
    Returns (train_ehr, val_ehr, test_ehr, scaler, ehr_dim, feat_names)."""
    train_ehr, ehr_scaler, ehr_feat_names = prepare_ehr_features(model_df, train_ids, fit_scaler=True)
    val_ehr, _, _ = prepare_ehr_features(model_df, val_ids, scaler=ehr_scaler)
    test_ehr, _, _ = prepare_ehr_features(model_df, test_ids, scaler=ehr_scaler)
    return train_ehr, val_ehr, test_ehr, ehr_scaler, train_ehr.shape[1], ehr_feat_names


def _build_loaders(
    train_ids, val_ids, test_ids,
    train_feat, val_feat, test_feat,
    train_ehr, val_ehr, test_ehr,
    train_labels, val_labels, test_labels,
    train_seqs, val_seqs, test_seqs,
    ecg_dict, actual_vital_dim, ecg_fm_dim,
    batch_size, num_workers, max_n, max_t,
):
    """Construct train/val/test DataLoaders."""
    collate = partial(collate_fn, max_N=max_n, max_T=max_t, ecg_fm_dim=ecg_fm_dim)

    def make_loader(ids, vf, ehr, labels, seqs, shuffle):
        return DataLoader(
            CardioEDDataset(ids, vf, ecg_dict, ehr, labels,
                            vital_sequences=seqs, vital_dim=actual_vital_dim, ecg_fm_dim=ecg_fm_dim),
            batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
            num_workers=num_workers, pin_memory=True,
        )

    return (
        make_loader(train_ids, train_feat, train_ehr, train_labels, train_seqs, True),
        make_loader(val_ids,   val_feat,   val_ehr,   val_labels,   val_seqs,   False),
        make_loader(test_ids,  test_feat,  test_ehr,  test_labels,  test_seqs,  False),
    )


def _run_trajectories(model, test_ids, test_seqs, test_ehr, ecg_dict,
                      label_cols_present, vital_scaler, device,
                      out_path, max_t, max_n, ecg_fm_dim,
                      n_samples, min_steps, model_name):
    """Generate and save digital twin trajectory plots for sample test patients."""
    candidates = [
        (sid, stay_id) for sid, stay_id in test_ids
        if test_seqs.get((sid, stay_id)) is not None
        and len(test_seqs[(sid, stay_id)]) >= min_steps
    ][:n_samples]

    # Bug 6: build O(1) lookup — test_ids.index() is O(n) and returns wrong row
    # if test_ids was filtered after test_ehr was built
    test_id_to_row = {(sid, stay): i for i, (sid, stay) in enumerate(test_ids)}

    for sid, stay_id in candidates:
        row_idx = test_id_to_row.get((sid, stay_id))
        if row_idx is None:
            log.warning("(%s, %s) not in test_id_to_row — skipping trajectory", sid, stay_id)
            continue
        raw_seq = test_seqs[(sid, stay_id)]
        ecg_embs = ecg_dict.get((int(sid), stay_id), np.zeros((1, ecg_fm_dim), dtype=np.float32))
        traj = simulate_trajectory(
            model, raw_seq, ecg_embs, test_ehr[row_idx], device,
            vital_stat_scaler=vital_scaler, max_t=max_t, max_n=max_n, ecg_fm_dim=ecg_fm_dim,
        )
        plot_trajectory(
            traj, patient_id=sid, label_names=label_cols_present,
            save_path=os.path.join(out_path, f"{model_name}/trajectory_{sid}_{stay_id}.png"),
        )