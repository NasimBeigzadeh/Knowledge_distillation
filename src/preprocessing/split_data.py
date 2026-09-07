# -*- coding: utf-8 -*-
"""
Train/validation/test split utilities for PTB-XL.

Author: Nasim Beigzadeh
"""


def split_data_by_strat_fold(
    x_data,
    y_data,
    test_fold=10,
    validation_fold=9,
):
    """
    Split ECG data according to the PTB-XL stratification folds.

    PTB-XL folds:
        - folds 1-8: training
        - fold 9: validation
        - fold 10: test

    Parameters
    ----------
    x_data : numpy.ndarray
        ECG signal data.

    y_data : pandas.DataFrame
        PTB-XL metadata containing the ``strat_fold`` column
        and ``diagnostic_superclass`` labels.

    test_fold : int, default=10
        Fold used for testing.

    validation_fold : int, default=9
        Fold used for validation.

    Returns
    -------
    x_train : numpy.ndarray
    y_train : pandas.Series
    x_valid : numpy.ndarray
    y_valid : pandas.Series
    x_test : numpy.ndarray
    y_test : pandas.Series
    """

    train_mask = y_data["strat_fold"] <= 8
    valid_mask = y_data["strat_fold"] == validation_fold
    test_mask = y_data["strat_fold"] == test_fold

    x_train = x_data[train_mask.to_numpy()]
    y_train = y_data.loc[train_mask, "diagnostic_superclass"]

    x_valid = x_data[valid_mask.to_numpy()]
    y_valid = y_data.loc[valid_mask, "diagnostic_superclass"]

    x_test = x_data[test_mask.to_numpy()]
    y_test = y_data.loc[test_mask, "diagnostic_superclass"]

    return (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )
