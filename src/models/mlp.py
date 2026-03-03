import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models.tabular_utils import (
    load_config,
    load_data_files,
    filter_ed_encounters,
    filter_ed_ecg_records,
    extract_earliest_ecg_per_stay,
    aggregate_vitals_to_ecg_time,
    create_model_df,
    prepare_model_features,
    create_train_test_set,
    scale_features,
)


# =============================================================================
# Model architecture
# =============================================================================

class MultilabelMLP(nn.Module):
    """
    Feedforward MLP for multi-label binary classification.
    Outputs one logit per label — use with BCEWithLogitsLoss.
    Apply torch.sigmoid to logits to get probabilities.
    """

    def __init__(self, in_features: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_labels = num_labels

        self.backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =============================================================================
# Tensor conversion helpers
# =============================================================================

def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array, DataFrame, or existing tensor to a typed tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)
    raise TypeError(f"Cannot convert type {type(x)} to tensor")


def _as_float_tensor(x) -> torch.Tensor:
    """Return a CPU float32 tensor without moving to any particular device."""
    return _to_tensor(x, device="cpu", dtype=torch.float32)


# =============================================================================
# Low-level training utilities
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
    return total_loss / n if n else 0.0


def predict_proba(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return sigmoid probability matrix as a CPU float tensor (n_samples, num_labels)."""
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = _as_float_tensor(X)
    chunks = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            batch = X[i : i + batch_size].to(device)
            chunks.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(chunks, dim=0) # (n_samples, num_labels)


def predict(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return binary predictions as a CPU int64 tensor (n_samples, num_labels)."""
    return (predict_proba(model, X, device, batch_size) >= threshold).to(torch.int64)


def _compute_pos_weights(y_train: pd.DataFrame) -> torch.Tensor:
    """Per-label pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    weights = []
    for col in y_train.columns:
        n_pos = int(y_train[col].sum())
        n_neg = len(y_train) - n_pos
        weights.append(float(n_neg / n_pos) if n_pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32) # CPU tensor; moved in fit_multilabel_mlp


def _make_tensor_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from two tensors already on CPU."""
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def fit_multilabel_mlp(
    X_train,
    y_train,
    dropout: float = 0.3,
    pos_weight=None,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    device: torch.device = None,
    validation_data=None,
    early_stopping_patience: int = 10,
) -> MultilabelMLP:
    """
    Train a MultilabelMLP with BCEWithLogitsLoss and optional early stopping.
    All internal data is kept as torch.Tensor throughout training.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Convert training data to tensors once ---
    X_t = _as_float_tensor(X_train) # (n, features)
    y_t = _as_float_tensor(y_train) # (n, labels)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)

    n_features = X_t.size(1)
    num_labels = y_t.size(1)

    # --- pos_weight ---
    pw_tensor = None
    if pos_weight is not None:
        if isinstance(pos_weight, torch.Tensor):
            pw_tensor = pos_weight.to(device=device, dtype=torch.float32)
        else:
            pw = np.asarray(pos_weight, dtype=np.float32)
            if pw.ndim == 1 and len(pw) == num_labels:
                pw_tensor = torch.from_numpy(pw).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    model = MultilabelMLP(
        in_features=n_features, num_labels=num_labels, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = _make_tensor_loader(X_t, y_t, batch_size=batch_size, shuffle=True)

    # --- Pre-build validation loader if provided ---
    val_loader = None
    if validation_data is not None:
        X_val_t = _as_float_tensor(validation_data[0])
        y_val_t = _as_float_tensor(validation_data[1])
        if y_val_t.dim() == 1:
            y_val_t = y_val_t.unsqueeze(1)
        val_loader = _make_tensor_loader(X_val_t, y_val_t, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is None:
            # No validation set — track training loss for best-state bookkeeping
            if best_state is None or train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            continue

        # --- Validation pass ---
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
                n_val += X_b.size(0)
        val_loss = val_loss / n_val if n_val else float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_mlp(X_train, y_train, X_val=None, y_val=None, use_pos_weight: bool = False):
    """Shared training wrapper used by all three pipeline variants."""
    pos_weight = _compute_pos_weights(y_train) if use_pos_weight else None
    validation_data = (X_val, y_val) if X_val is not None else None

    return fit_multilabel_mlp(
        X_train=X_train,
        y_train=y_train,
        dropout=0.3,
        pos_weight=pos_weight,
        lr=1e-3,
        epochs=100,
        batch_size=64,
        device=None,
        validation_data=validation_data,
        early_stopping_patience=10,
    )


# =============================================================================
# SMOTE with 15% prevalence cap
# =============================================================================

def _cap_smote(X_train, y_train, prevalence_threshold=0.03, max_prevalence=0.15, random_state=42):
    """
    Apply SMOTE per label with a hard cap on synthetic row generation.

    For each label below `prevalence_threshold`, generates synthetic rows until
    that label reaches `max_prevalence`. This prevents over-generation on
    extremely rare labels.
    """
    from imblearn.over_sampling import SMOTE

    n_orig = len(X_train)
    low_prev_labels = [
        col for col in y_train.columns
        if y_train[col].mean() < prevalence_threshold and y_train[col].sum() > 1
    ]

    if not low_prev_labels:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True), 0, []

    X_synthetic_all = []
    y_synthetic_all = []

    for col in low_prev_labels:
        n_pos_orig = int(y_train[col].sum())

        # Solve for how many synthetic positives reach max_prevalence:
        # max_prevalence = (n_pos_orig + n_syn) / (n_orig + n_syn)
        # => n_syn = (max_prevalence * n_orig - n_pos_orig) / (1 - max_prevalence)
        n_syn_cap = int((max_prevalence * n_orig - n_pos_orig) / (1.0 - max_prevalence))
        if n_syn_cap <= 0:
            continue # already at or above the cap

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train[col])

        n_synthetic = len(X_res) - n_orig
        if n_synthetic <= 0:
            continue

        # Only keep up to n_syn_cap synthetic rows for this label
        n_keep = min(n_synthetic, n_syn_cap)
        X_new = pd.DataFrame(X_res[n_orig : n_orig + n_keep], columns=X_train.columns)

        # Synthetic rows: positive for this label, zero for all others
        y_new = pd.DataFrame(0, index=range(n_keep), columns=y_train.columns)
        y_new[col] = 1

        X_synthetic_all.append(X_new)
        y_synthetic_all.append(y_new)

    if X_synthetic_all:
        X_resampled = pd.concat([X_train] + X_synthetic_all, ignore_index=True)
        y_resampled = pd.concat([y_train] + y_synthetic_all, ignore_index=True)
        n_added = len(X_resampled) - n_orig
    else:
        X_resampled = X_train.reset_index(drop=True)
        y_resampled = y_train.reset_index(drop=True)
        n_added = 0

    return X_resampled, y_resampled, n_added, low_prev_labels


# =============================================================================
# Evaluation & visualisation
# =============================================================================

def evaluate_and_visualize_mlp(
    model: MultilabelMLP,
    X_test,
    y_test: pd.DataFrame,
    y_features,
    model_name: str,
    out_path: str = "../data/model_results/",
    label_group_name: str = "Diagnosis Labels",
) -> pd.DataFrame:
    """
    Evaluate MLP and produce:
      1. ROC + PR curves + aggregated confusion matrix ({model_name}_evaluation_plots.png)
      2. Label co-occurrence heatmap ({model_name}_label_confusion_matrix.png)
      3. Per-label results CSV ({model_name}_results.csv)
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert test features to tensor once
    X_tensor = _as_float_tensor(X_test) # CPU tensor; predict_proba batches to device

    proba_tensor = predict_proba(model, X_tensor, device) # (n, num_labels) CPU tensor
    pred_tensor = (proba_tensor >= 0.5).to(torch.int64) # (n, num_labels) CPU tensor

    # Keep as numpy only where sklearn metrics require it
    proba_matrix = proba_tensor.numpy()
    pred_matrix = pred_tensor.numpy()

    results = []
    for i, target in enumerate(y_features):
        n_pos = y_test[target].sum()
        if y_test[target].nunique() > 1:
            auc = roc_auc_score(y_test[target], proba_matrix[:, i])
            ap = average_precision_score(y_test[target], proba_matrix[:, i])
        else:
            auc = ap = np.nan
        results.append({
            "target": target,
            "n_test_pos": int(n_pos),
            "pos_rate": y_test[target].mean(),
            "roc_auc": auc,
            "pr_auc": ap,
        })

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    valid_labels = [l for l in y_features if y_test[l].nunique() > 1]
    print(f"\nPlotting {len(valid_labels)} labels with valid metrics")

    # --- Figure 1: ROC | PR | Confusion matrix ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for label in valid_labels:
        idx = list(y_features).index(label)
        fpr, tpr, _ = roc_curve(y_test[label], proba_matrix[:, idx])
        roc_auc = results_df[results_df["target"] == label].iloc[0]["roc_auc"]
        color = "#2E5090" if roc_auc >= 0.95 else ("#6B46C1" if roc_auc >= 0.85 else "#D32F2F")
        alpha = 0.3 if roc_auc >= 0.95 else (0.4 if roc_auc >= 0.85 else 0.6)
        axes[0].plot(fpr, tpr, linewidth=1.5, alpha=alpha, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=2, label="Random (AUC=0.5)")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title(
        f"ROC Curves — {len(valid_labels)} {label_group_name}\n"
        f"Mean AUC: {results_df['roc_auc'].mean():.3f}", fontsize=14,
    )
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for label in valid_labels:
        idx = list(y_features).index(label)
        prec, rec, _ = precision_recall_curve(y_test[label], proba_matrix[:, idx])
        pr_auc = results_df[results_df["target"] == label].iloc[0]["pr_auc"]
        color = "#2E5090" if pr_auc >= 0.7 else ("#6B46C1" if pr_auc >= 0.3 else "#D32F2F")
        alpha = 0.3 if pr_auc >= 0.7 else (0.4 if pr_auc >= 0.3 else 0.6)
        axes[1].plot(rec, prec, linewidth=1.5, alpha=alpha, color=color)

    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title(
        f"PR Curves — {len(valid_labels)} {label_group_name}\n"
        f"Mean PR-AUC: {results_df['pr_auc'].mean():.3f}", fontsize=14,
    )
    axes[1].grid(True, alpha=0.3)

    total_cm = np.zeros((2, 2))
    y_true_all = []
    y_pred_all = []
    for label in valid_labels:
        idx = list(y_features).index(label)
        total_cm += confusion_matrix(y_test[label], pred_matrix[:, idx])
        y_true_all.extend(y_test[label].values)
        y_pred_all.extend(pred_matrix[:, idx])

    sns.heatmap(
        total_cm, annot=True, fmt=".0f", cmap="Blues", ax=axes[2],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar_kws={"label": "Count"},
    )
    axes[2].set_title(
        f"Aggregated Confusion Matrix\n(Sum Across {len(valid_labels)} {label_group_name})",
        fontsize=14,
    )
    axes[2].set_ylabel("True Label", fontsize=12)
    axes[2].set_xlabel("Predicted Label", fontsize=12)

    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    axes[2].text(
        1.5, -0.5,
        f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(model_name, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = Path(out_path) / f"{model_name}_evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to '{plot_path}'")

    # --- Figure 2: Label co-occurrence matrix ---
    n_labels = len(valid_labels)
    label_cm = np.zeros((n_labels, n_labels))

    # Build prediction matrix for valid labels as a tensor slice
    valid_indices = [list(y_features).index(l) for l in valid_labels]
    pred_valid_tensor = pred_tensor[:, valid_indices] # (n, n_valid_labels) tensor

    for i, true_label in enumerate(valid_labels):
        positive_mask = torch.from_numpy(y_test[true_label].values == 1) # bool tensor
        if positive_mask.sum() > 0:
            label_cm[i] = pred_valid_tensor[positive_mask].float().sum(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(max(12, n_labels * 0.6), max(10, n_labels * 0.5)))
    im = ax.imshow(label_cm, cmap="Greens", aspect="auto")

    shortened = [l.replace("label_", "").replace("report_", "") for l in valid_labels]
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(shortened, rotation=90, ha="right", fontsize=9)
    ax.set_yticklabels(shortened, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prediction Count", rotation=270, labelpad=20, fontsize=11)

    if n_labels <= 30:
        max_val = label_cm.max()
        for i in range(n_labels):
            for j in range(n_labels):
                count = int(label_cm[i, j])
                if count > 0:
                    text_color = "white" if label_cm[i, j] > max_val / 2 else "black"
                    ax.text(j, i, f"{count}", ha="center", va="center",
                            color=text_color, fontsize=7)

    ax.set_title(
        f"Label Co-occurrence Matrix — {label_group_name}\n"
        "(When true label on Y-axis is positive, how often is predicted label on X-axis positive?)",
        fontsize=13, pad=20,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.suptitle(model_name, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    label_cm_path = Path(out_path) / f"{model_name}_label_confusion_matrix.png"
    plt.savefig(label_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Label co-occurrence matrix saved to '{label_cm_path}'")

    tn, fp, fn, tp = total_cm.ravel()
    print(f"\nAggregated Metrics (across all {len(valid_labels)} labels):")
    print(f"  True Negatives:  {int(tn)} ({tn / total_cm.sum() * 100:.1f}%)")
    print(f"  False Positives: {int(fp)} ({fp / total_cm.sum() * 100:.1f}%)")
    print(f"  False Negatives: {int(fn)} ({fn / total_cm.sum() * 100:.1f}%)")
    print(f"  True Positives:  {int(tp)} ({tp / total_cm.sum() * 100:.1f}%)")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

    csv_path = Path(out_path) / f"{model_name}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")

    return results_df


# =============================================================================
# K-Fold loss curve plotting
# =============================================================================

def plot_kfold_loss_curves(
    X,
    y: pd.DataFrame,
    out_path: str,
    model_name: str = "mlp",
    n_splits: int = 5,
    epochs: int = 100,
    early_stopping_patience: int = 10,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: torch.device = None,
) -> dict:
    """
    Train a MultilabelMLP using k-fold CV with early stopping and save a
    single loss curve PNG.

    Each fold trains up to `epochs` but stops early if val loss does not
    improve for `early_stopping_patience` consecutive epochs. Loss arrays are
    padded with the last recorded value after stopping so all folds stay the
    same length for averaging.

    Two lines on one plot: mean train loss and mean val loss across folds.
    A dashed vertical line marks the mean early-stop epoch across folds.

    Returns a dict with 'mean_train_loss', 'mean_val_loss', and 'mean_stopped_epoch'.
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = _as_float_tensor(X)
    y_t = _as_float_tensor(y)
    y_arr = y_t.numpy()

    n_features = X_t.size(1)
    num_labels = y_t.size(1)
    strat_col = y_arr[:, 0] # stratify on first label

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_loss_folds = np.zeros((n_splits, epochs))
    val_loss_folds = np.zeros((n_splits, epochs))
    stopped_epochs = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_t.numpy(), strat_col)):
        print(f"  Fold {fold_idx + 1}/{n_splits}...")

        X_tr, X_val = X_t[train_idx], X_t[val_idx]
        y_tr, y_val = y_t[train_idx], y_t[val_idx]

        pw = _compute_pos_weights(
            pd.DataFrame(y_tr.numpy(), columns=y.columns)
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        fold_model = MultilabelMLP(n_features, num_labels, dropout).to(device)
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=1e-4)
        train_loader = _make_tensor_loader(X_tr, y_tr, batch_size, shuffle=True)
        val_loader = _make_tensor_loader(X_val, y_val, batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        stopped_at = epochs # default: ran all epochs

        for epoch in range(epochs):
            # --- train ---
            fold_model.train()
            t_loss, t_n = 0.0, 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(fold_model(X_b), y_b)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * X_b.size(0)
                t_n += X_b.size(0)
            train_loss_folds[fold_idx, epoch] = t_loss / t_n if t_n else 0.0

            # --- val ---
            fold_model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    loss = criterion(fold_model(X_b), y_b)
                    v_loss += loss.item() * X_b.size(0)
                    v_n += X_b.size(0)
            val_loss_folds[fold_idx, epoch] = v_loss / v_n if v_n else 0.0

            # --- early stopping ---
            if val_loss_folds[fold_idx, epoch] < best_val_loss:
                best_val_loss = val_loss_folds[fold_idx, epoch]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                stopped_at = epoch + 1 # 1-indexed
                # Pad remaining epochs with last recorded values so array is full length
                train_loss_folds[fold_idx, epoch + 1:] = train_loss_folds[fold_idx, epoch]
                val_loss_folds[fold_idx, epoch + 1:] = val_loss_folds[fold_idx, epoch]
                print(f"    Early stop at epoch {stopped_at}")
                break

        stopped_epochs.append(stopped_at)

    mean_train = train_loss_folds.mean(axis=0)
    mean_val = val_loss_folds.mean(axis=0)
    best_epoch = int(np.argmin(mean_val))
    mean_stopped = int(np.mean(stopped_epochs))

    # --- Plot ---
    epochs_range = np.arange(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Faint individual fold lines
    for fold_idx in range(n_splits):
        ax.plot(epochs_range, train_loss_folds[fold_idx], color="#2E5090", alpha=0.12, linewidth=0.8)
        ax.plot(epochs_range, val_loss_folds[fold_idx], color="#D32F2F", alpha=0.12, linewidth=0.8)

    # Mean lines
    ax.plot(epochs_range, mean_train, color="#2E5090", linewidth=2.5, label="Train loss (mean)")
    ax.plot(epochs_range, mean_val, color="#D32F2F", linewidth=2.5, label="Val loss (mean)")

    # Best val epoch
    ax.axvline(
        best_epoch + 1, color="gray", linestyle="--", linewidth=1.5,
        label=f"Best epoch: {best_epoch + 1}  (val={mean_val[best_epoch]:.4f})",
    )

    # Mean early-stop epoch (only annotate if any fold stopped early)
    if mean_stopped < epochs:
        ax.axvline(
            mean_stopped, color="#F57C00", linestyle=":", linewidth=1.5,
            label=f"Mean early stop: epoch {mean_stopped}",
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(
        f"{model_name} — Train vs Val Loss\n"
        f"{n_splits}-Fold CV · patience={early_stopping_patience}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = Path(out_path) / f"{model_name}_kfold_loss_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nK-fold loss curves saved to '{plot_path}'")

    return {
        "mean_train_loss": mean_train,
        "mean_val_loss": mean_val,
        "best_epoch": best_epoch,
        "mean_stopped_epoch": mean_stopped,
    }


# =============================================================================
# Shared data loading / preparation
# =============================================================================

def _load_and_prepare(in_dir, config_path, pbar, steps):
    """Steps 0-6: load data, filter, aggregate vitals, prepare features, split, scale."""
    pbar.set_description(steps[0])
    config = load_config(config_path)
    ed_vitals, clinical_encounters, ecg_records = load_data_files(in_dir, config)
    pbar.update(1)

    pbar.set_description(steps[1])
    ed_encounters = filter_ed_encounters(clinical_encounters)
    ed_ecg_records = filter_ed_ecg_records(ecg_records)
    pbar.update(1)

    pbar.set_description(steps[2])
    earliest_ecgs = extract_earliest_ecg_per_stay(ed_ecg_records)
    pbar.update(1)

    pbar.set_description(steps[3])
    ecg_aggregate_vitals = aggregate_vitals_to_ecg_time(ed_vitals, earliest_ecgs, agg_window_hours=4.0)
    pbar.update(1)

    pbar.set_description(steps[4])
    model_df = create_model_df(ed_encounters, ecg_aggregate_vitals)
    pbar.update(1)

    pbar.set_description(steps[5])
    X, y, y_features, cols_to_scale = prepare_model_features(model_df)
    pbar.update(1)

    pbar.set_description(steps[6])
    X_train, X_test, y_train, y_test = create_train_test_set(model_df, X, y)
    X_train, X_test, _ = scale_features(X_train, X_test, cols_to_scale)
    pbar.update(1)

    return model_df, X_train, X_test, y_train, y_test, y_features


def _val_split(model_df, X_train, y_train):
    """
    Carve a 10% patient-aware validation split for early stopping.
    Always called on pre-SMOTE data so stopping criterion reflects real distribution.
    """
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_groups = model_df.loc[y_train.index, "subject_id"].astype(int).values
    tr_idx, val_idx = next(val_splitter.split(X_train, y_train, groups=train_groups))
    return (
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        X_train.iloc[val_idx], y_train.iloc[val_idx],
    )


# =============================================================================
# Variant 1: Base (normalized, uniform loss)
# =============================================================================

def run_mlp_base_pipeline(in_dir, config_path, out_path):
    """
    MLP baseline: StandardScaler normalization, uniform BCEWithLogitsLoss.

    No SMOTE, no pos_weight. Establishes a clean baseline before applying
    any class-imbalance strategies.
    """
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Train/test split & scaling",
        "Training MLP (base)",
        "K-fold loss curves",
        "Evaluating model",
    ]

    print("Running MLP Base model...")
    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        model_df, X_train, X_test, y_train, y_test, y_features = _load_and_prepare(
            in_dir, config_path, pbar, steps
        )

        pbar.set_description(steps[7])
        X_tr, y_tr, X_val, y_val = _val_split(model_df, X_train, y_train)
        model = _train_mlp(X_tr, y_tr, X_val=X_val, y_val=y_val, use_pos_weight=False)
        pbar.update(1)

        pbar.set_description(steps[8])
        plot_kfold_loss_curves(X_train, y_train, out_path=out_path, model_name="mlp_base")
        pbar.update(1)

        pbar.set_description(steps[9])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_mlp(
            model, X_test, y_test, y_features, "mlp_base",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ MLP base model complete!")
    return results_df


# =============================================================================
# Variant 2: SMOTE (normalized + capped oversampling)
# =============================================================================

def run_mlp_smote_pipeline(in_dir, config_path, out_path):
    """
    MLP + SMOTE: normalize then oversample labels with prevalence < 3%.

    Synthetic rows are capped so each smoted label reaches at most 15%
    prevalence in the combined training set, preventing over-generation on
    extremely rare labels.

    The early-stopping validation split is carved before SMOTE so the stopping
    criterion reflects real-world class distribution, not the synthetic one.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Train/test split & scaling",
        "Applying capped SMOTE",
        "Training MLP (SMOTE)",
        "K-fold loss curves",
        "Evaluating model",
    ]

    print("Running MLP SMOTE model...")
    print()
    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        model_df, X_train, X_test, y_train, y_test, y_features = _load_and_prepare(
            in_dir, config_path, pbar, steps
        )

        # Carve val split BEFORE SMOTE — val must reflect real distribution
        X_tr, y_tr, X_val, y_val = _val_split(model_df, X_train, y_train)

        pbar.set_description(steps[7])
        X_tr_s, y_tr_s, n_added, smoted_labels = _cap_smote(
            X_tr, y_tr,
            prevalence_threshold=0.03,
            max_prevalence=0.15,
        )
        print(
            f"\n  SMOTE: {n_added} synthetic rows added across "
            f"{len(smoted_labels)} labels (capped at 15% prevalence each)"
        )
        pbar.update(1)

        pbar.set_description(steps[8])
        model = _train_mlp(X_tr_s, y_tr_s, X_val=X_val, y_val=y_val, use_pos_weight=False)
        pbar.update(1)

        pbar.set_description(steps[9])
        plot_kfold_loss_curves(X_train, y_train, out_path=out_path, model_name="mlp_smote")
        pbar.update(1)

        pbar.set_description(steps[10])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_mlp(
            model, X_test, y_test, y_features, "mlp_smote",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ MLP SMOTE model complete!")
    return results_df


# =============================================================================
# Variant 3: Weighted loss (normalized + per-label BCEWithLogitsLoss pos_weight)
# =============================================================================

def run_mlp_weighted_pipeline(in_dir, config_path, out_path):
    """
    MLP + weighted loss: normalize then scale the loss by class imbalance.

    Computes pos_weight = n_negative / n_positive per label and passes it to
    BCEWithLogitsLoss, penalising false negatives on rare positive classes
    more heavily during training. No synthetic data generation.
    """
    steps = [
        "Loading configuration & data",
        "Filtering ED encounters & ECG records",
        "Getting earliest ECGs per stay",
        "Aggregating vitals to ECG time",
        "Creating model dataframe",
        "Preparing features",
        "Train/test split & scaling",
        "Training MLP (weighted loss)",
        "K-fold loss curves",
        "Evaluating model",
    ]

    print("Running MLP Weighted model...")
    print()
    pbar = tqdm(
        total=len(steps), desc="Progress", ncols=80, file=sys.stdout,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    try:
        model_df, X_train, X_test, y_train, y_test, y_features = _load_and_prepare(
            in_dir, config_path, pbar, steps
        )

        pbar.set_description(steps[7])
        X_tr, y_tr, X_val, y_val = _val_split(model_df, X_train, y_train)
        model = _train_mlp(X_tr, y_tr, X_val=X_val, y_val=y_val, use_pos_weight=True)
        pbar.update(1)

        pbar.set_description(steps[8])
        plot_kfold_loss_curves(X_train, y_train, out_path=out_path, model_name="mlp_weighted")
        pbar.update(1)

        pbar.set_description(steps[9])
        pbar.update(1)
        pbar.close()

        results_df = evaluate_and_visualize_mlp(
            model, X_test, y_test, y_features, "mlp_weighted",
            out_path=out_path, label_group_name="Diagnosis Labels",
        )

    except Exception as e:
        pbar.close()
        raise e

    print()
    print("✓ MLP weighted model complete!")
    return results_df