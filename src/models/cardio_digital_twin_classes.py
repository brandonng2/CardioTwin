import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

MAX_T = 12
MAX_N = 2
VITAL_DIM = 6
VITAL_STAT = 30
ECG_FM_DIM = 1536
ENC_DIM = 128
HIDDEN_DIM = 256
LSTM_HIDDEN = 64
N_LABELS = 17
DROPOUT = 0.3

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
# DATASET
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

        dups = vital_feat_df.duplicated(["subject_id", "ed_stay_id"]).sum()
        if dups > 0:
            raise ValueError(
                f"vital_feat_df has {dups} duplicate (subject_id, ed_stay_id) rows. "
                f"Deduplicate before constructing dataset."
            )

        self.vital_feats = vital_feat_df.set_index(["subject_id", "ed_stay_id"])
        self.labels = labels_df.set_index(["subject_id", "ed_stay_id"])

        assert len(ehr_matrix) == len(subject_stay_ids), (
            f"EHR matrix rows ({len(ehr_matrix)}) != subject_stay_ids ({len(subject_stay_ids)}). "
            f"Ensure prepare_ehr_features() is called with the same ids list passed here."
        )

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
# BASE MODEL
# =============================================================================

class CardioTwinED(nn.Module):
    """
    Multimodal Gated Fusion Network for cardiovascular diagnosis prediction.

    Three modality encoders each project to enc_dim:
      - Vital encoder  : dual-branch LSTM (sequence) + MLP (stats), fused to vital_enc
      - ECG encoder    : attention pool over N ECGs + MLP
      - EHR encoder    : shallow MLP (built lazily via set_ehr_dim())

    Gate network learns soft per-patient weights (3,) over the three encodings.
    Weighted sum -> fusion MLP -> 17 sigmoid logits.
    Outputs: logits, probs, gates (interpretability), latent (twin state).
    """

    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, enc_dim=ENC_DIM, hidden_dim=HIDDEN_DIM,
                 lstm_hidden=LSTM_HIDDEN, dropout=DROPOUT, n_labels=N_LABELS):
        super().__init__()
        self._enc_dim = enc_dim
        self._dropout = dropout
        self._lstm_hidden = lstm_hidden

        # Vital branch
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

        # ECG branch
        self.ecg_attn = nn.Linear(ecg_emb_dim, 1)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_emb_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, enc_dim), nn.GELU(),
        )

        # EHR encoder and gate built lazily once ehr_dim is known
        self.ehr_encoder = None
        self.gate = None

        # Fusion head
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
        self.gate = nn.Sequential(
            nn.Linear(enc_dim * 3, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        if device is not None:
            self.ehr_encoder = self.ehr_encoder.to(device)
            self.gate = self.gate.to(device)

    def _encode_vitals(self, vital_feats, vital_seq, vital_lengths):
        B = vital_feats.size(0)
        device = vital_feats.device
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
        return self.vital_fusion(torch.cat([lstm_enc, stats_enc], dim=1))

    def _encode_ecg(self, ecg_embs, ecg_mask):
        attn_scores = self.ecg_attn(ecg_embs)
        if ecg_mask is not None:
            attn_scores = attn_scores.masked_fill(~ecg_mask.unsqueeze(-1), float("-inf"))
        ecg_pooled = (torch.softmax(attn_scores, dim=1) * ecg_embs).sum(dim=1)
        return self.ecg_encoder(ecg_pooled)

    def forward(self, vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask=None):
        """
        vital_feats    (B, VITAL_STAT)
        vital_seq      (B, T, vital_dim)    padded
        vital_lengths  (B,)                actual lengths
        ecg_embs       (B, N, ECG_FM_DIM)  up to MAX_N ECGs
        ehr_x          (B, ehr_dim)
        ecg_mask       (B, N) bool          True = real ECG
        """
        device = vital_feats.device
        if self.ehr_encoder is None:
            self.set_ehr_dim(ehr_x.size(1), device=device)

        vital_enc = self._encode_vitals(vital_feats, vital_seq, vital_lengths)
        ecg_enc = self._encode_ecg(ecg_embs, ecg_mask)
        ehr_enc = self.ehr_encoder(ehr_x)

        gates = F.softmax(self.gate(torch.cat([vital_enc, ecg_enc, ehr_enc], dim=1)), dim=1)
        fused = (
            gates[:, 0:1] * vital_enc
            + gates[:, 1:2] * ecg_enc
            + gates[:, 2:3] * ehr_enc
        )
        shared = self.fusion(fused)
        logits = self.dx_head(shared)

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "gates": gates,
            "latent": shared,
        }

# =============================================================================
# NO-GATE BASELINE (ablation: mean-pooled fusion instead of learned gate)
# =============================================================================

class CardioTwinED_NoGate(CardioTwinED):
    """
    Ablation variant — same enc_dim=128 encoders as Baseline but replaces the
    learned gate with a simple mean pooling of the three modality encodings.

    Instead of:
        gates = softmax(Linear(384→128)→Linear(128→3))
        fused = gates[:,0]*vital + gates[:,1]*ecg + gates[:,2]*ehr
    We do:
        fused = (vital_enc + ecg_enc + ehr_enc) / 3

    This isolates the contribution of the gating mechanism. If Baseline > NoGate
    the gate is learning something real; if not, mean pooling is sufficient.
    The gate and ehr_encoder are still built via set_ehr_dim() so the interface
    is identical — gate weights are just not used in forward().
    """
    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, lstm_hidden=LSTM_HIDDEN,
                 dropout=DROPOUT, n_labels=N_LABELS):
        super().__init__(
            vital_stat=vital_stat, vital_dim=vital_dim, ehr_dim=ehr_dim,
            ecg_emb_dim=ecg_emb_dim, enc_dim=128, hidden_dim=256,
            lstm_hidden=lstm_hidden, dropout=dropout, n_labels=n_labels,
        )

    def forward(self, vital_feats, vital_seq, vital_lengths, ecg_embs, ehr_x, ecg_mask=None):
        device = vital_feats.device
        if self.ehr_encoder is None:
            self.set_ehr_dim(ehr_x.size(1), device=device)

        vital_enc = self._encode_vitals(vital_feats, vital_seq, vital_lengths)
        ecg_enc = self._encode_ecg(ecg_embs, ecg_mask)
        ehr_enc = self.ehr_encoder(ehr_x)

        # Mean pool — no gate
        fused = (vital_enc + ecg_enc + ehr_enc) / 3.0
        # Uniform gate weights returned for API compatibility with trajectory plots
        B = vital_feats.size(0)
        gates = torch.full((B, 3), 1.0 / 3.0, device=device, dtype=vital_feats.dtype)

        shared = self.fusion(fused)
        logits = self.dx_head(shared)

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "gates": gates,
            "latent": shared,
        }