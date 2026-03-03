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