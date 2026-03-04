import warnings
import sys
import os
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


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