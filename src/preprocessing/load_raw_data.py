# -*- coding: utf-8 -*-
"""
Utilities for loading raw PTB-XL ECG signals.

Author: Nasim Beigzadeh
"""

import numpy as np
import wfdb


def load_raw_data(df, sampling_rate, data_path):
    """
    Load raw ECG signals from PTB-XL files.

    Parameters
    ----------
    df : pandas.DataFrame
        PTB-XL metadata containing ECG file names.

    sampling_rate : int
        Sampling frequency. Supported values are 100 or 500 Hz.

    data_path : str
        Path to the PTB-XL dataset directory.

    Returns
    -------
    numpy.ndarray
        Raw ECG signals.
    """

    if sampling_rate == 100:
        filenames = df.filename_lr
    else:
        filenames = df.filename_hr

    data = [
        wfdb.rdsamp(data_path + filename)
        for filename in filenames
    ]

    signals = np.array(
        [signal for signal, metadata in data]
    )

    return signals
