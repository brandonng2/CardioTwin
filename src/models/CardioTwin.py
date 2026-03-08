# =============================================================================
# Architecture: enc_dim=128, learned gated fusion, BCEWithLogitsLoss, no sampler.
# This is the production entry point. For ablations and variant sweeps, see
# cardio_digital_twin.py.
#
# Usage:
#   from src.models.CardioTwin import run_cardiotwin_final
#   run_cardiotwin_final(in_dir, config_path, out_path)
# =============================================================================

import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.cardio_digital_twin_classes import (
    CardioTwinED,
    VITAL_STAT, VITAL_DIM, ECG_FM_DIM, LSTM_HIDDEN, DROPOUT, N_LABELS,
)
from src.models.cardio_digital_twin_utils import (
    LABEL_COLS, MAX_T, MAX_N, ENC_DIM, BATCH_SIZE, LR, EPOCHS,
    prepare_ecg,
    train_cardiotwin_model_bce,
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

MODEL_NAME = "CardioTwin"


# =============================================================================
# MODEL
# =============================================================================

class CardioTwinBaseline(CardioTwinED):
    """
    Final CardioTwin model.

    enc_dim=128 — each modality encoder (vital, ECG, EHR) projects to 128-dim.
    Gate: softmax(Linear(384→128) → Linear(128→3)) → (B, 3) soft weights.
    Fused: weighted sum → (B, 128) → fusion MLP → latent (B, 64) → 17 logits.
    Loss: BCEWithLogitsLoss (no pos_weight, no sampler).
    """
    def __init__(self, vital_stat=VITAL_STAT, vital_dim=VITAL_DIM, ehr_dim=None,
                 ecg_emb_dim=ECG_FM_DIM, lstm_hidden=LSTM_HIDDEN,
                 dropout=DROPOUT, n_labels=N_LABELS):
        super().__init__(
            vital_stat=vital_stat,
            vital_dim=vital_dim,
            ehr_dim=ehr_dim,
            ecg_emb_dim=ecg_emb_dim,
            enc_dim=128,
            hidden_dim=256,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
            n_labels=n_labels,
        )


# =============================================================================
# BUILD HELPER
# =============================================================================

def _build_model(actual_vital_stat, actual_vital_dim, ehr_dim, ecg_fm_dim,
                 n_labels, device, lstm_hidden=LSTM_HIDDEN, dropout=DROPOUT):
    """Instantiate CardioTwinBaseline and pre-build the EHR encoder + gate."""
    model = CardioTwinBaseline(
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
# PIPELINE
# =============================================================================

def run_cardiotwin_final(in_dir, config_path, out_path):
    """
    Train and evaluate the final CardioTwin model.

    Configuration: enc_dim=128 | gated fusion | BCEWithLogitsLoss | no sampler.
    All hyperparameters are read from config_path (CardioTwin_model_params.json).

    Output layout:
      out_path/CardioTwin/
        CardioTwin.pt
        CardioTwin_roc_curves.png
        CardioTwin_pr_curves.png
        CardioTwin_confusion_matrix.png
        CardioTwin_cooccurrence_matrix.png
        CardioTwin_results.csv
        CardioTwin_overall_results.csv
        CardioTwin_kfold_loss_curves.png
        CardioTwin_trajectories/
      out_path/overall_results_ablation.csv  (upserted)
    """
    from src.models.tabular_utils import load_config

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
        f"Training {MODEL_NAME}",
        "K-fold loss curves",
        f"Evaluating {MODEL_NAME}",
    ]

    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        config = load_config(config_path)
        in_dir = str(Path(in_dir).resolve()) if in_dir else str(_REPO_ROOT / config["paths"]["in_dir"])
        out_path = str(Path(out_path).resolve()) if out_path else str(
            _REPO_ROOT / config["paths"].get("out_dir", "model_results/CardioTwin/")
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
            "batch_size": trn.get("batch_size", BATCH_SIZE),
            "learning_rate": trn.get("learning_rate", LR),
            "weight_decay": trn.get("weight_decay", 1e-4),
            "epochs": trn.get("epochs", EPOCHS),
            "grad_clip_norm": trn.get("grad_clip_norm", 1.0),
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Steps 1-6: shared data loading
        data_steps = steps[:9]
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

        pbar.set_description(steps[6])
        (train_feat, val_feat, test_feat,
         train_seqs, val_seqs, test_seqs,
         vital_scaler, _, actual_vital_dim, actual_vital_stat) = _build_vitals(
            ed_vitals, train_df, val_df, test_df
        )

        train_ids = _filter_to_vitals(get_ids(train_df), train_feat)
        val_ids = _filter_to_vitals(get_ids(val_df), val_feat)
        test_ids = _filter_to_vitals(get_ids(test_df), test_feat)

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
        pbar.set_description(steps[7])
        train_ehr, val_ehr, test_ehr, _, ehr_dim, _ = _build_ehr(
            model_df, train_ids, val_ids, test_ids
        )
        pbar.update(1)

        # Step 9: ECG dict
        pbar.set_description(steps[8])
        ecg_dict = prepare_ecg(all_ecgs_embedded, max_n=max_n, ecg_fm_dim=ecg_fm_dim)
        pbar.update(1)

        # Build loaders (no sampler — standard shuffle)
        train_loader, val_loader, test_loader = _build_loaders(
            train_ids, val_ids, test_ids,
            train_feat, val_feat, test_feat,
            train_ehr, val_ehr, test_ehr,
            train_labels, val_labels, test_labels,
            train_seqs, val_seqs, test_seqs,
            ecg_dict, actual_vital_dim, ecg_fm_dim,
            params["batch_size"], num_workers, max_n, max_t,
            train_sampler=None,
        )

        # Step 10: Train
        pbar.set_description(steps[9])
        model = _build_model(
            actual_vital_stat, actual_vital_dim, ehr_dim, ecg_fm_dim,
            len(label_cols_present), device, lstm_hidden, dropout,
        )
        log.info(
            "CardioTwin | enc_dim=128 | loss=bce | sampler=none | params=%s",
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        )
        model, _, _ = train_cardiotwin_model_bce(
            model, train_loader, val_loader, test_loader,
            params, out_path, device, model_name=MODEL_NAME,
        )
        pbar.update(1)

        # Step 11: K-fold loss curves
        pbar.set_description(steps[10])
        plot_kfold_loss_curves_cardiotwin(
            lambda fold: train_loader,
            lambda fold: val_loader,
            lambda: _build_model(
                actual_vital_stat, actual_vital_dim, ehr_dim, ecg_fm_dim,
                len(label_cols_present), device, lstm_hidden, dropout,
            ),
            label_cols_present, params, out_path, device,
            n_folds=3, model_name=MODEL_NAME,
        )
        pbar.update(1)

        # Step 12: Evaluate
        pbar.set_description(steps[11])
        results_df = evaluate_and_visualize_cardiotwin(
            model, test_loader, label_cols_present, out_path,
            model_name=MODEL_NAME, device=device,
        )
        pbar.update(1)

        # Trajectory simulations
        _run_trajectories(
            model, test_ids, test_seqs, test_ehr, ecg_dict,
            label_cols_present, vital_scaler, device, out_path,
            max_t, max_n, ecg_fm_dim, n_trajectory_samples, min_trajectory_steps,
            model_name=MODEL_NAME,
        )

        pbar.close()

    except Exception as e:
        pbar.close()
        raise e

    log.info("CardioTwin final pipeline complete.")
    print(f"\n✓ CardioTwin complete — results in {out_path}/{MODEL_NAME}/")
    return results_df