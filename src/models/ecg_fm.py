import warnings
import sys
import os
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Signal I/O
# ---------------------------------------------------------------------------

def _read_ecg_signal(path, target_samples=5000):
    """
    Read a WFDB record and return a (channels, target_samples) float32 array.
    Pads with zeros if the signal is shorter than target_samples.
    Uses max(0, ...) to guard against negative pad widths.
    """
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal.T.astype(np.float32)  # (channels, time)

    if sig.shape[1] < target_samples:
        pad_width = max(0, target_samples - sig.shape[1])
        sig = np.pad(sig, ((0, 0), (0, pad_width)))

    return sig[:, :target_samples]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ecgfm_model(checkpoint_path: str, device: torch.device):
    """
    Load an ECG-FM checkpoint via fairseq-signals.
    Returns the model in eval mode on the requested device.
    """
    try:
        from fairseq_signals.models import build_model_from_checkpoint
    except ImportError as e:
        raise ImportError(
            "fairseq-signals is not installed. "
            "Install it from: https://github.com/Jwoo5/fairseq-signals"
        ) from e

    model = build_model_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Shape probe (run once before batch extraction)
# ---------------------------------------------------------------------------

def probe_model_output(model, device, n_channels=12, target_samples=5000):
    """
    Send a single dummy tensor through the model and print output shape info.
    Useful for confirming hidden_dim before running full extraction.

    Returns:
        hidden_dim (int)
    """
    dummy = torch.randn(1, n_channels, target_samples).to(device)
    with torch.no_grad():
        out = model(source=dummy)

    features = out["features"]  # (batch, seq_len, hidden_dim)
    hidden_dim = features.shape[-1]
    return hidden_dim


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _pool_encoder_out(out: dict) -> torch.Tensor:
    """
    out["features"]: (batch, seq_len, hidden_dim)  — already batch-first
    Mean-pools across seq_len.
    Returns: (batch, hidden_dim)
    """
    return out["features"].mean(dim=1)


# ---------------------------------------------------------------------------
# Batched pooled embedding extraction  (XGBoost / MLP)
# ---------------------------------------------------------------------------

def extract_embeddings_batched(
    model,
    signal_paths: list[str],
    device: torch.device,
    batch_size: int = 32,
    n_channels: int = 12,
    target_samples: int = 5000,
    segment_split: bool = True,
) -> np.ndarray:
    """
    Extract pooled ECG-FM embeddings for a list of WFDB record paths.

    If segment_split=True (default), each record is split into two equal
    halves, both forwarded in a single pass, then concatenated.
    → embedding shape: (N, hidden_dim * 2)  e.g. (N, 1536)

    If segment_split=False, the full signal is used in one forward pass.
    → embedding shape: (N, hidden_dim)  e.g. (N, 768)

    Use for: XGBoost, MLP — any model that needs a flat fixed-size vector.
    For sequential models (LSTM, Transformer), use extract_raw_features_batched().
    """
    half = target_samples // 2
    all_embs = []

    for batch_start in tqdm(range(0, len(signal_paths), batch_size), desc="Extracting ECG embeddings", unit="batch", ncols=80):
        batch_paths = signal_paths[batch_start : batch_start + batch_size]

        segs1_list, segs2_list, full_list = [], [], []
        valid_indices = []

        for i, path in enumerate(batch_paths):
            try:
                sig = _read_ecg_signal(path, target_samples=target_samples)
                if segment_split:
                    segs1_list.append(sig[:, :half])
                    segs2_list.append(sig[:, half:])
                else:
                    full_list.append(sig)
                valid_indices.append(i)
            except Exception as exc:
                warnings.warn(f"Skipping {path}: {exc}")

        if not valid_indices:
            continue

        if segment_split:
            x = torch.from_numpy(
                np.concatenate(
                    [np.stack(segs1_list), np.stack(segs2_list)], axis=0
                )
            ).to(device)  # (2*B, channels, half)

            with torch.no_grad():
                out = model(source=x)
                pooled = _pool_encoder_out(out)   # (2*B, hidden_dim)

            B = len(valid_indices)
            f1 = pooled[:B]   # first-half embeddings
            f2 = pooled[B:]   # second-half embeddings
            batch_embs = torch.cat([f1, f2], dim=1).cpu().numpy()  # (B, hidden_dim*2)

        else:
            x = torch.from_numpy(np.stack(full_list)).to(device)  # (B, channels, T)

            with torch.no_grad():
                out = model(source=x)
                pooled = _pool_encoder_out(out)

            batch_embs = pooled.cpu().numpy()

        all_embs.append(batch_embs)

    if not all_embs:
        raise RuntimeError("No embeddings extracted — check signal paths and model.")

    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Batched raw feature extraction  (LSTM / Transformer / Gated Fusion)
# ---------------------------------------------------------------------------

def extract_raw_features_batched(
    model,
    signal_paths: list[str],
    device: torch.device,
    batch_size: int = 32,
    n_channels: int = 12,
    target_samples: int = 5000,
) -> np.ndarray:
    """
    Extract raw (unpooled) ECG-FM features for a list of WFDB record paths.

    Returns the full transformer sequence without mean pooling.
    → feature shape: (N, seq_len, hidden_dim)  e.g. (N, 312, 768)

    Use for: LSTM, Transformer blocks, Gated Fusion — any model that
    can exploit the temporal structure of the ECG sequence.
    For flat models (XGBoost, MLP), use extract_embeddings_batched() instead.
    """
    all_features = []

    for batch_start in tqdm(range(0, len(signal_paths), batch_size), desc="Extracting ECG features", unit="batch", ncols=80):
        batch_paths = signal_paths[batch_start : batch_start + batch_size]

        sig_list = []
        valid_indices = []

        for i, path in enumerate(batch_paths):
            try:
                sig = _read_ecg_signal(path, target_samples=target_samples)
                sig_list.append(sig)
                valid_indices.append(i)
            except Exception as exc:
                warnings.warn(f"Skipping {path}: {exc}")

        if not valid_indices:
            continue

        x = torch.from_numpy(np.stack(sig_list)).to(device)  # (B, channels, T)

        with torch.no_grad():
            out = model(source=x)
            features = out["features"]  # (B, seq_len, hidden_dim) — no pooling

        all_features.append(features.cpu().numpy())

    if not all_features:
        raise RuntimeError("No features extracted — check signal paths and model.")

    return np.concatenate(all_features, axis=0)  # (N, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Convenience runners
# ---------------------------------------------------------------------------

def run_pooled_ecg_extraction(config_path: str, subject_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline for flat models (XGBoost, MLP):
    load config → load model → extract pooled embeddings → return df.

    subject_df must have columns: ['subject_id', 'study_id', 'ecg_path']
    where ecg_path is the WFDB record base-path (no extension).

    Returns subject_df with added columns emb_0 ... emb_{D-1}
    where D = hidden_dim * 2 = 1536.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ecgfm_model(cfg["model"]["checkpoint_path"], device)

    hidden_dim = probe_model_output(model, device)
    expected_emb_dim = cfg["model"]["embedding_dim"]
    actual_emb_dim = hidden_dim * 2
    if actual_emb_dim != expected_emb_dim:
        warnings.warn(
            f"Config embedding_dim={expected_emb_dim} but model produces "
            f"{actual_emb_dim}. Using actual value."
        )

    paths = subject_df["ecg_path"].tolist()
    embs = extract_embeddings_batched(model, paths, device)

    emb_cols = [f"emb_{i}" for i in range(embs.shape[1])]
    emb_df = pd.DataFrame(embs, columns=emb_cols, index=subject_df.index)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.concat([subject_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)


def run_raw_ecg_extraction(config_path: str, subject_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full pipeline for sequential models (LSTM, Transformer, Gated Fusion):
    load config → load model → extract raw features → return df + feature array.

    subject_df must have columns: ['subject_id', 'study_id', 'ecg_path']
    where ecg_path is the WFDB record base-path (no extension).

    Returns
    -------
    subject_df : pd.DataFrame
        Original subject_df (no embedding columns — features are 3D so
        cannot be stored as flat DataFrame columns).
    features : np.ndarray, shape (N, seq_len, hidden_dim)  e.g. (N, 312, 768)
        Raw transformer features aligned row-by-row with subject_df.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ecgfm_model(cfg["model"]["checkpoint_path"], device)
    probe_model_output(model, device)

    paths = subject_df["ecg_path"].tolist()
    features = extract_raw_features_batched(model, paths, device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return subject_df.reset_index(drop=True), features