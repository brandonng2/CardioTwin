import warnings
import sys
import os
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_data_files(in_dir, config):
    """Load all required data files based on configuration."""
    ed_vitals = pd.read_csv(os.path.join(in_dir, config["sources"]["vitals"]))
    clinical_encounters = pd.read_csv(
        os.path.join(in_dir, config["sources"]["clinical_encounters"]),
        dtype=str,
        low_memory=False,
    )
    ecg_records = pd.read_csv(os.path.join(in_dir, config["sources"]["ecg_records"]))
    return ed_vitals, clinical_encounters, ecg_records


def _read_ecg_signal(path, target_samples=10000):
    """
    Read a WFDB record and return a (channels, target_samples) float32 array.
    Pads with zeros if the signal is shorter than target_samples.
    """
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal.T.astype(np.float32)  # (channels, time) — cast once here

    if sig.shape[1] < target_samples:
        sig = np.pad(sig, ((0, 0), (0, target_samples - sig.shape[1])))

    return sig[:, :target_samples]