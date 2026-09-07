# -*- coding: utf-8 -*-
"""
Sliding-window segmentation for ECG signals.

Author: Nasim Beigzadeh
"""

import numpy as np


def create_sliding_windows(
    x_data,
    y_data,
    window_length_seconds,
    overlap_rate,
    sampling_rate,
):
    """
    Segment ECG signals into overlapping windows.

    Parameters
    ----------
    x_data : array-like
        ECG signals.

    y_data : pandas.Series
        Labels corresponding to the original ECG recordings.

    window_length_seconds : float
        Length of each window in seconds.

    overlap_rate : float
        Step size as a fraction of the window length.
        For example, 0.5 means that consecutive windows
        start 50% of a window length apart.

    sampling_rate : int
        Sampling frequency in Hz.

    Returns
    -------
    x_sliding_array : numpy.ndarray
        ECG signals after sliding-window segmentation.

    y_sliding_array : numpy.ndarray
        Labels corresponding to each generated window.
    """

    # Convert window length from seconds to samples
    window_length_samples = int(
        window_length_seconds * sampling_rate
    )

    # Calculate step size between consecutive windows
    step_samples = int(
        window_length_samples * overlap_rate
    )

    if step_samples <= 0:
        raise ValueError(
            "The resulting step size must be greater than zero."
        )

    if window_length_samples <= 0:
        raise ValueError(
            "Window length must be greater than zero."
        )

    x_sliding = []
    y_sliding = []

    # Generate sliding windows for each ECG recording
    for i in range(len(x_data)):

        for j in range(
            0,
            len(x_data[i]) - window_length_samples + 1,
            step_samples,
        ):
            x_sliding.append(
                x_data[i][j:j + window_length_samples]
            )

            y_sliding.append(
                y_data.iloc[i]
            )

    # Convert to NumPy arrays
    x_sliding_array = np.asarray(x_sliding)
    y_sliding_array = np.asarray(y_sliding)

    return x_sliding_array, y_sliding_array
