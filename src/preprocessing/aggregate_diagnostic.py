# -*- coding: utf-8 -*-
"""
Diagnostic superclass aggregation for the PTB-XL dataset.

Author: Nasim Beigzadeh
"""

import pandas as pd


def load_aggregation_table(data_path):
    """
    Load the PTB-XL SCP statement aggregation table.

    Parameters
    ----------
    data_path : str
        Path to the PTB-XL dataset directory.

    Returns
    -------
    pandas.DataFrame
        Diagnostic SCP statement table.
    """

    agg_df = pd.read_csv(
        data_path + "scp_statements.csv",
        index_col=0,
    )

    agg_df = agg_df[agg_df.diagnostic == 1]

    return agg_df


def aggregate_diagnostic(y_dic, agg_df):
    """
    Map SCP codes to diagnostic superclass labels.

    Parameters
    ----------
    y_dic : dict
        SCP codes associated with an ECG recording.

    agg_df : pandas.DataFrame
        Diagnostic SCP statement aggregation table.

    Returns
    -------
    list
        Diagnostic superclass labels.
    """

    tmp = []

    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(
                agg_df.loc[key].diagnostic_class
            )

    return list(set(tmp))
