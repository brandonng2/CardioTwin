# =============================================================================
# CardioTwinED model variants and the main pipeline runner.
#
# SEQUENTIAL TUNING VARIANTS (active — run by default)
#   CardioTwinED_Baseline  — enc_dim=128, learned gated fusion
#   CardioTwinED_NoGate    — enc_dim=128, mean-pooled fusion (gate ablation)
#
# HELD-OUT SIZE VARIANTS (commented out in VARIANTS — re-enable after tuning)
#   CardioTwinED_Medium    — enc_dim=256  (vital_enc 256 + ecg_enc 256 + ehr_enc 256)
#   CardioTwinED_Large     — enc_dim=512  (vital_enc 512 + ecg_enc 512 + ehr_enc 512)
#
# LOSS FUNCTION VARIANTS (set LOSS_TYPE in run_cardiotwin_pipeline)
#   "bce"          — plain BCEWithLogitsLoss (default)
#   "bce_weighted" — BCE + pos_weight (n_neg/n_pos per label)
#   "focal"        — FocalLoss(alpha=0.25, gamma=2.0)
#
# PIPELINE
#   run_cardiotwin_pipeline() runs all active VARIANTS x LOSS_TYPES sequentially.
#   Outputs land under out_path/{model_name}_{loss_type}/ for each combination.
# =============================================================================

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from src.models.cardio_digital_twin_classes import (
    CardioTwinED,
    CardioTwinED_NoGate,
    VITAL_STAT, VITAL_DIM, ECG_FM_DIM, HIDDEN_DIM, LSTM_HIDDEN, DROPOUT, N_LABELS,
)
from src.models.cardio_digital_twin_utils import (
    LABEL_COLS, VITAL_COLS,
    MAX_T, MAX_N, ENC_DIM, BATCH_SIZE, LR, EPOCHS,
    prepare_ecg,
    build_weighted_sampler,
    train_cardiotwin_model,
    train_cardiotwin_model_bce,
    train_cardiotwin_model_bce_weighted,
    train_cardiotwin_model_focal,
    evaluate_and_visualize_cardiotwin,
    plot_kfold_loss_curves_cardiotwin,
    _load_and_prepare_data,
    _build_splits,
    _build_vitals,
    _filter_to_vitals,
    _build_ehr,
    _build_loaders,
    _run_trajectories,
)

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# VARIANT DEFINITIONS
# =============================================================================

class CardioTwinED_Baseline(CardioTwinED):
    """
    Baseline variant — enc_dim=128.
    Gate input : concat(vital_enc_128, ecg_enc_128, ehr_enc_128) = (B, 384)
    Gate output: softmax(Linear(384→128)→Linear(128→3)) -> (B, 3)
    Fused state: weighted sum -> (B, 128) -> fusion MLP -> latent (B, 64)
    """
    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, lstm_hidden=LSTM_HIDDEN,
                 dropout=DROPOUT, n_labels=N_LABELS):
        super().__init__(
            vital_stat=vital_stat, vital_dim=vital_dim, ehr_dim=ehr_dim,
            ecg_emb_dim=ecg_emb_dim, enc_dim=128, hidden_dim=256,
            lstm_hidden=lstm_hidden, dropout=dropout, n_labels=n_labels,
        )


class CardioTwinED_Medium(CardioTwinED):
    """
    Medium variant — enc_dim=256.
    Gate input : concat(vital_enc_256, ecg_enc_256, ehr_enc_256) = (B, 768)
    Gate output: softmax(Linear(768→256)→Linear(256→3)) -> (B, 3)
    Fused state: weighted sum -> (B, 256) -> fusion MLP -> latent (B, 128)

    Encoder widths scale up proportionally:
      vital stats branch  : Linear(vital_stat→128)→Linear(128→256)
      ECG branch          : Linear(ecg_emb_dim→512)→Linear(512→256)
      EHR branch          : Linear(ehr_dim→128)→Linear(128→256)
    """
    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, lstm_hidden=LSTM_HIDDEN,
                 dropout=DROPOUT, n_labels=N_LABELS):
        # Call nn.Module.__init__ directly; we rebuild all sub-modules below
        nn.Module.__init__(self)
        enc_dim = 256
        hidden_dim = 512
        self._enc_dim = enc_dim
        self._dropout = dropout
        self._lstm_hidden = lstm_hidden

        # Vital branch
        self.vital_lstm = nn.LSTM(input_size=vital_dim, hidden_size=lstm_hidden,
                                  num_layers=1, batch_first=True)
        self.vital_lstm_proj = nn.Sequential(nn.Linear(lstm_hidden, enc_dim), nn.GELU())
        self.vital_encoder = nn.Sequential(
            nn.Linear(vital_stat, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, enc_dim), nn.GELU(),
        )
        self.vital_fusion = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim), nn.GELU(), nn.Dropout(dropout),
        )

        # ECG branch
        self.ecg_attn = nn.Linear(ecg_emb_dim, 1)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_emb_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, enc_dim), nn.GELU(),
        )

        # EHR encoder and gate built lazily
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
        enc_dim = self._enc_dim
        dropout = self._dropout
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, enc_dim), nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(enc_dim * 3, enc_dim), nn.ReLU(), nn.Linear(enc_dim, 3)
        )
        if device is not None:
            self.ehr_encoder = self.ehr_encoder.to(device)
            self.gate = self.gate.to(device)


class CardioTwinED_Large(CardioTwinED):
    """
    Large variant — enc_dim=512.
    Gate input : concat(vital_enc_512, ecg_enc_512, ehr_enc_512) = (B, 1536)
    Gate output: softmax(Linear(1536→512)→Linear(512→3)) -> (B, 3)
    Fused state: weighted sum -> (B, 512) -> fusion MLP -> latent (B, 256)

    Encoder widths scale up proportionally:
      vital stats branch  : Linear(vital_stat→256)→Linear(256→512)
      ECG branch          : Linear(ecg_emb_dim→1024)→Linear(1024→512)
      EHR branch          : Linear(ehr_dim→256)→Linear(256→512)
    """
    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, lstm_hidden=LSTM_HIDDEN,
                 dropout=DROPOUT, n_labels=N_LABELS):
        nn.Module.__init__(self)
        enc_dim = 512
        hidden_dim = 1024
        self._enc_dim = enc_dim
        self._dropout = dropout
        self._lstm_hidden = lstm_hidden

        # Vital branch
        self.vital_lstm = nn.LSTM(input_size=vital_dim, hidden_size=lstm_hidden,
                                  num_layers=1, batch_first=True)
        self.vital_lstm_proj = nn.Sequential(nn.Linear(lstm_hidden, enc_dim), nn.GELU())
        self.vital_encoder = nn.Sequential(
            nn.Linear(vital_stat, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, enc_dim), nn.GELU(),
        )
        self.vital_fusion = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim), nn.GELU(), nn.Dropout(dropout),
        )

        # ECG branch
        self.ecg_attn = nn.Linear(ecg_emb_dim, 1)
        self.ecg_encoder = nn.Sequential(
            nn.Linear(ecg_emb_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, enc_dim), nn.GELU(),
        )

        # EHR encoder and gate built lazily
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
        enc_dim = self._enc_dim
        dropout = self._dropout
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, enc_dim), nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(enc_dim * 3, enc_dim), nn.ReLU(), nn.Linear(enc_dim, 3)
        )
        if device is not None:
            self.ehr_encoder = self.ehr_encoder.to(device)
            self.gate = self.gate.to(device)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

# All known variants — referenced by name in pipeline calls.
_ALL_VARIANTS = {
    "cardio_digital_twin_baseline": (CardioTwinED_Baseline, 128),
    "cardio_digital_twin_nogate":   (CardioTwinED_NoGate,   128),
    "cardio_digital_twin_medium":   (CardioTwinED_Medium,   256),
    "cardio_digital_twin_large":    (CardioTwinED_Large,    512),
}

_LOSS_TRAINERS = {
    "bce":          train_cardiotwin_model_bce,
    "bce_weighted": train_cardiotwin_model_bce_weighted,
    "focal":        train_cardiotwin_model_focal,
}

# Default sets used when callers don't specify
_DEFAULT_VARIANTS      = {"cardio_digital_twin_baseline": _ALL_VARIANTS["cardio_digital_twin_baseline"]}
_DEFAULT_LOSS_TYPES    = ["bce"]
_DEFAULT_SAMPLER_TYPES = ["none"]

# Full ablation sets
_ABLATION_VARIANTS = {
    "cardio_digital_twin_baseline": _ALL_VARIANTS["cardio_digital_twin_baseline"],
    "cardio_digital_twin_nogate":   _ALL_VARIANTS["cardio_digital_twin_nogate"],
    "cardio_digital_twin_medium":   _ALL_VARIANTS["cardio_digital_twin_medium"],
    "cardio_digital_twin_large":    _ALL_VARIANTS["cardio_digital_twin_large"],
}
_ABLATION_LOSS_TYPES    = ["bce", "bce_weighted", "focal"]
_ABLATION_SAMPLER_TYPES = ["none", "weighted"]


def _build_model(actual_vital_stat, actual_vital_dim, ehr_dim, ecg_fm_dim,
                 variant_cls, n_labels, device, lstm_hidden=LSTM_HIDDEN, dropout=DROPOUT):
    """Instantiate a CardioTwinED variant and pre-build the EHR encoder + gate."""
    model = variant_cls(
        vital_stat=actual_vital_stat,
        vital_dim=actual_vital_dim,
        ehr_dim=ehr_dim,
        ecg_emb_dim=ecg_fm_dim,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
        n_labels=n_labels,
    ).to(device)
    model.set_ehr_dim(ehr_dim, device=device)
    return model


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_cardiotwin_pipeline(
    in_dir,
    config_path,
    out_path,
    variants=None,
    loss_types=None,
    sampler_types=None,
):
    """
    Full CardioTwin pipeline: trains the specified VARIANTS x LOSS_TYPES x SAMPLER_TYPES.

    Defaults to the single baseline configuration (enc_dim=128, BCE, no sampler)
    when called with no overrides — i.e. the main pipeline entry point.

    For the full ablation sweep, use run_cardiotwin_ablation_pipeline() which
    passes all variants, all loss types, and both sampler modes.

    Shared data loading (steps 1-9) runs once. Each (variant, loss) pair then
    trains, logs k-fold curves, evaluates, and generates trajectory plots.

    Output layout per combination:
      out_path/{model_name}_{loss_type}/
        {model_name}_{loss_type}.pt
        {model_name}_{loss_type}_roc_curves.png
        {model_name}_{loss_type}_pr_curves.png
        {model_name}_{loss_type}_confusion_matrix.png
        {model_name}_{loss_type}_cooccurrence_matrix.png
        {model_name}_{loss_type}_results.csv
        {model_name}_{loss_type}_overall_results.csv
        {model_name}_{loss_type}_kfold_loss_curves.png
    """
    from src.models.tabular_utils import load_config

    variants      = variants      or _DEFAULT_VARIANTS
    loss_types    = loss_types    or _DEFAULT_LOSS_TYPES
    sampler_types = sampler_types or _DEFAULT_SAMPLER_TYPES

    n_combos = len(variants) * len(loss_types) * len(sampler_types)
    data_steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECG per stay (vitals time anchor)",
        "Extracting ECG-FM embeddings — ALL ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Building vital features + sequences",
        "Building EHR features",
        "Building ECG dict + datasets",
    ]
    train_steps = [
        f"Training {mname} [{lt}] [sampler={st}]"
        for mname in variants
        for lt in loss_types
        for st in sampler_types
    ]
    steps = data_steps + train_steps

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        config = load_config(config_path)
        in_dir = str(Path(in_dir).resolve()) if in_dir else str(_REPO_ROOT / config["paths"]["in_dir"])
        out_path = str(Path(out_path).resolve()) if out_path else str(
            _REPO_ROOT / config["paths"].get("out_dir", "model_results/")
        )
        Path(out_path).mkdir(parents=True, exist_ok=True)

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
        lstm_hidden = mdl.get("lstm_hidden", LSTM_HIDDEN)
        dropout = mdl.get("dropout", DROPOUT)
        ecg_fm_dim = mdl.get("ecg_fm_dim", 1536)

        trn = config.get("training", {})
        params = {
            "batch_size":     trn.get("batch_size", BATCH_SIZE),
            "learning_rate":  trn.get("learning_rate", LR),
            "weight_decay":   trn.get("weight_decay", 1e-4),
            "epochs":         trn.get("epochs", EPOCHS),
            "grad_clip_norm": trn.get("grad_clip_norm", 1.0),
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Steps 1-6: shared data loading
        config, ed_vitals, model_df, all_ecgs_embedded = _load_and_prepare_data(
            in_dir, config_path, pbar, data_steps
        )

        label_cols_present = [c for c in LABEL_COLS if c in model_df.columns]
        if not label_cols_present:
            raise ValueError("No label columns in model_df — check create_model_df().")

        # Step 7: splits + vitals
        train_df, val_df, test_df = _build_splits(
            model_df, test_size, val_size, random_state, val_random_state
        )

        def get_ids(df):
            return list(zip(df["subject_id"].astype(int), df["ed_stay_id"]))

        id_label_cols = ["subject_id", "ed_stay_id"] + label_cols_present
        train_labels = train_df[id_label_cols]
        val_labels = val_df[id_label_cols]
        test_labels = test_df[id_label_cols]

        pbar.set_description(data_steps[6])
        (train_feat, val_feat, test_feat,
         train_seqs, val_seqs, test_seqs,
         vital_scaler, _, actual_vital_dim, actual_vital_stat) = _build_vitals(
            ed_vitals, train_df, val_df, test_df
        )

        train_ids = _filter_to_vitals(get_ids(train_df), train_feat)
        val_ids = _filter_to_vitals(get_ids(val_df), val_feat)
        test_ids = _filter_to_vitals(get_ids(test_df), test_feat)

        # Re-filter labels to match the vitals-filtered id lists.
        # train_labels was built from train_df (pre-filter); build_weighted_sampler
        # and CardioEDDataset both require labels aligned to the filtered ids exactly,
        # otherwise WeightedRandomSampler generates indices beyond len(dataset).
        def _filter_labels(labels_df, ids):
            id_set = set(ids)
            mask = [
                (int(r.subject_id), r.ed_stay_id) in id_set
                for r in labels_df.itertuples(index=False)
            ]
            return labels_df[mask].reset_index(drop=True)

        train_labels = _filter_labels(train_labels, train_ids)
        val_labels = _filter_labels(val_labels, val_ids)
        test_labels = _filter_labels(test_labels, test_ids)
        pbar.update(1)

        # Step 8: EHR
        pbar.set_description(data_steps[7])
        train_ehr, val_ehr, test_ehr, _, ehr_dim, _ = _build_ehr(
            model_df, train_ids, val_ids, test_ids
        )
        pbar.update(1)

        # Step 9: ECG dict (loaders are built per-run inside the combo loop below)
        pbar.set_description(data_steps[8])
        ecg_dict = prepare_ecg(all_ecgs_embedded, max_n=max_n, ecg_fm_dim=ecg_fm_dim)
        pbar.update(1)

        # Train each variant x loss x sampler combination
        for model_name, (variant_cls, enc_dim) in variants.items():
            for loss_type in loss_types:
                for sampler_type in sampler_types:
                    run_name = f"{model_name}_{loss_type}_{sampler_type}"
                    pbar.set_description(f"Training {model_name} [{loss_type}] [sampler={sampler_type}]")

                    # Build sampler once per run (fitted on train labels only)
                    train_sampler = (
                        build_weighted_sampler(train_labels)
                        if sampler_type == "weighted"
                        else None
                    )

                    # Rebuild loaders with/without sampler for this run
                    run_train_loader, run_val_loader, run_test_loader = _build_loaders(
                        train_ids, val_ids, test_ids,
                        train_feat, val_feat, test_feat,
                        train_ehr, val_ehr, test_ehr,
                        train_labels, val_labels, test_labels,
                        train_seqs, val_seqs, test_seqs,
                        ecg_dict, actual_vital_dim, ecg_fm_dim,
                        params["batch_size"], num_workers, max_n, max_t,
                        train_sampler=train_sampler,
                    )

                    model = _build_model(
                        actual_vital_stat, actual_vital_dim, ehr_dim, ecg_fm_dim,
                        variant_cls, len(label_cols_present), device, lstm_hidden, dropout,
                    )
                    log.info(
                        "Run: %s | enc_dim=%d | loss=%s | sampler=%s | params=%s",
                        run_name, enc_dim, loss_type, sampler_type,
                        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
                    )

                    trainer = _LOSS_TRAINERS[loss_type]
                    model, _, _ = trainer(
                        model, run_train_loader, run_val_loader, run_test_loader,
                        params, out_path, device, model_name=run_name,
                    )

                    plot_kfold_loss_curves_cardiotwin(
                        lambda fold: run_train_loader,
                        lambda fold: run_val_loader,
                        lambda: _build_model(
                            actual_vital_stat, actual_vital_dim, ehr_dim, ecg_fm_dim,
                            variant_cls, len(label_cols_present), device, lstm_hidden, dropout,
                        ),
                        label_cols_present, params, out_path, device,
                        n_folds=3, model_name=run_name,
                    )

                    evaluate_and_visualize_cardiotwin(
                        model, run_test_loader, label_cols_present, out_path,
                        model_name=run_name, device=device,
                    )

                    _run_trajectories(
                        model, test_ids, test_seqs, test_ehr, ecg_dict,
                        label_cols_present, vital_scaler, device, out_path,
                        max_t, max_n, ecg_fm_dim, n_trajectory_samples, min_trajectory_steps,
                        model_name=run_name,
                    )

                    pbar.update(1)

        pbar.close()

    except Exception as e:
        pbar.close()
        raise e

    log.info("Cardio Digital Twin pipeline complete.")
    return None

# =============================================================================
# PUBLIC ABLATION ENTRY POINT
# =============================================================================

def run_cardiotwin_ablation_pipeline(in_dir, config_path, out_path):
    """
    Full CardioTwin ablation sweep.

    Runs ALL variants x ALL loss types x ALL sampler types:
      Variants  : baseline (128), nogate (128), medium (256), large (512)
      Loss types: bce, bce_weighted, focal
      Samplers  : none, weighted

    Total combinations: 4 x 3 x 2 = 24 runs.
    Each run outputs under out_path/{model_name}_{loss_type}_{sampler_type}/.
    """
    return run_cardiotwin_pipeline(
        in_dir,
        config_path,
        out_path,
        variants=_ABLATION_VARIANTS,
        loss_types=_ABLATION_LOSS_TYPES,
        sampler_types=_ABLATION_SAMPLER_TYPES,
    )