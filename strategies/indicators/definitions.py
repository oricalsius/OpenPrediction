"""
This module is defining all the functions that are necessary to create indicators.
Input for each function is a Pandas Dataframe or directly the Ticker or Quotation object
"""

from typing import List, Union, Any
import pandas as pd
import numpy as np


def _get_new_columns_info(columns: Union[List[str], str], prefix: str = ""):
    if isinstance(columns, list):
        source_columns_name = columns
    elif isinstance(columns, str):
        source_columns_name = [columns]
    else:
        raise Exception("Parameter columns should be an str or list of str.")

    target_columns_label = [prefix + "_" + str(col) for col in source_columns_name]

    return source_columns_name, target_columns_label


def moving_window_functions(data: pd.DataFrame, columns: Union[List[str], str],
                            functions: List[str] = ["mean"],
                            window: int = 14, quantile: float = 0.05, add_to_data: bool = True) -> pd.DataFrame:
    """
    Provide rolling window calculations.

    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

    :param data: Pandas data type.
    :param columns: Columns for which we should calculate the ewm.
    :param functions: Functions to apply: mean or std or var.
    :param window: Length of the window.
    :param quantile: quantile in ]0,1[ when we want to compute a quantile of the window.
    :param add_to_data: Specify whether to add the result to DataFrame or not.
    :return: pandas.DataFrame object.
    """

    unknown_functions = set(functions) - {"sum", "mean", "median", "min", "max", "std", "var", "skew", "kurt", \
                                          "quantile"}
    if unknown_functions:
        raise Exception(f"Functions {list(unknown_functions)} are not part of known pandas.DataFrame.rolling methods.")

    src_columns, _ = _get_new_columns_info(columns)
    rolling_window = data[src_columns].rolling(window=window)   # Generate a rolling window object
    result_df = pd.DataFrame()

    for prefix in functions:
        columns_prefix = "Moving_" + str(prefix) + "_" + str(window)
        columns_prefix = columns_prefix + "_" + str(quantile) if prefix == "quantile" else columns_prefix
        src_columns, columns_label = _get_new_columns_info(columns, columns_prefix)

        df = rolling_window.agg(prefix) if prefix != "quantile" else rolling_window.quantile(quantile=quantile)
        df.rename(columns=dict(zip(src_columns, columns_label)), inplace=True)

        if add_to_data:
            data[columns_label] = df
        else:
            result_df = pd.concat([result_df, df], axis=1)

    if add_to_data:
        return data
    else:
        return result_df


def moving_average(data: pd.DataFrame, columns: Union[List[str], str], window: int = 14,
                   add_to_data: bool = True) -> pd.DataFrame:
    return moving_window_functions(data, columns, ["mean"], window, add_to_data)


def moving_std(data: pd.DataFrame, columns: Union[List[str], str], window: int = 14,
               add_to_data: bool = True) -> pd.DataFrame:
    return moving_window_functions(data, columns, ["std"], window, add_to_data)


def moving_var(data: pd.DataFrame, columns: Union[List[str], str], window: int = 14,
               add_to_data: bool = True) -> pd.DataFrame:
    return moving_window_functions(data, columns, ["var"], window, add_to_data)


def true_range(data: pd.DataFrame, close_name: str = "close", high_name: str = "high", low_name: str = "low",
               window: int = 1, columns_label: str = "", add_to_data: bool = True) -> pd.DataFrame:
    if columns_label == "":
        columns_label = "TR_" + str(window)

    df_close = data[close_name].shift(1)
    df = pd.DataFrame(pd.concat([data[high_name], df_close], axis=1).max(axis=1)
                      - pd.concat([data[low_name], df_close], axis=1).min(axis=1), columns=[columns_label])

    if window > 1:
        df = df.rolling(window=window).mean()

    if add_to_data:
        data[columns_label] = df
        return data
    else:
        return df


def average_true_range(data: pd.DataFrame, close_name: str = "close", high_name: str = "high", low_name: str = "low",
                       window: int = 1, columns_label: str = "", add_to_data: bool = True) -> pd.DataFrame:
    if columns_label == "":
        columns_label = "ATR_" + str(window)

    df_tr = true_range(data, close_name=close_name, high_name=high_name, low_name=low_name, window=window,
                       add_to_data=False)

    df_atr = exponential_weighted_functions(df_tr, "TR_" + str(window), functions=["mean"], alpha=1.0/window,
                                            add_to_data=False)
    df_atr.columns = [columns_label]

    if add_to_data:
        data[columns_label] = df_atr
        return data
    else:
        return df_atr


def average_directional_index(data: pd.DataFrame, close_name: str = "close", high_name: str = "high",
                              low_name: str = "low", window: int = 14, columns_label: str = "",
                              add_to_data: bool = True) -> pd.DataFrame:

    """
    Implements the Average Directional Index using the formula from https://www.investopedia.com/terms/a/adx.asp and
    https://fr.wikipedia.org/wiki/Directional_Movement_Index and
    https://www.investopedia.com/ask/answers/112814/how-average-directional-index-adx-calculated-and-what-formula.asp.
    """

    if columns_label == "":
        columns_label = "ADX_" + str(window)

    m_positive = data[high_name] - data[high_name].shift(1)
    m_negative = data[low_name].shift(1) - data[low_name]

    dm_positive = pd.DataFrame(m_positive.copy())
    dm_positive.columns = ["DMp"]
    dm_positive[m_positive < m_negative] = 0
    dm_positive[m_positive < 0] = 0

    dm_negative = pd.DataFrame(m_negative.copy())
    dm_negative.columns = ["DMn"]
    dm_negative[m_negative < m_positive] = 0
    dm_negative[m_negative < 0] = 0

    smoothed_dm_positive = exponential_weighted_functions(dm_positive, "DMp", functions=["mean"],
                                                          alpha=1.0 / window, add_to_data=False)
    smoothed_dm_positive.columns = ["SDMp"]

    smoothed_dm_negative = exponential_weighted_functions(dm_negative, "DMn", functions=["mean"],
                                                          alpha=1.0 / window, add_to_data=False)
    smoothed_dm_negative.columns = ["SDMn"]

    # DI+ and DI-
    atr = average_true_range(data, close_name, high_name, low_name, window, "ATR", False)
    directional_index_positive = pd.DataFrame(smoothed_dm_positive["SDMp"].div(atr["ATR"]) * 100,
                                              columns=["DIp_" + str(window)])
    directional_index_negative = pd.DataFrame(smoothed_dm_negative["SDMn"].div(atr["ATR"]) * 100,
                                              columns=["DIn_" + str(window)])

    # DXI
    dxi = pd.DataFrame(100 * abs(directional_index_positive["DIp_" + str(window)]
                                 - directional_index_negative["DIn_" + str(window)])
                       / abs(directional_index_positive["DIp_" + str(window)]
                             + directional_index_negative["DIn_" + str(window)]), columns=["DXI_" + str(window)])

    # ADX
    adx = exponential_weighted_functions(dxi, "DXI_" + str(window), functions=["mean"], alpha=1.0 / window, add_to_data=False)
    adx.columns = [columns_label]

    df = pd.concat([directional_index_positive, directional_index_negative, dxi, adx], axis=1)

    if add_to_data:
        data[df.columns] = df
        return data
    else:
        return df


def moving_average_channel(data: pd.DataFrame) -> pd.DataFrame:
    return data


def exponential_weighted_functions(data: pd.DataFrame, columns: Union[List[str], str],
                                   functions: List[str] = ["mean"],
                                   span: Any = None, com: Any = None, halflife: Any = None,
                                   alpha: Any = None, add_to_data: bool = True, adjust: bool = True) -> pd.DataFrame:
    """
    Provide exponential weighted functions.

    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

    :param data: Pandas data type.
    :param columns: Columns for which we should calculate the ewm.
    :param functions: Functions to apply: mean or std or var.
    :param span: Specify decay in terms of span, α=2/(span+1) = 1/(1+com), for span≥1.
    :param com: Specify decay in terms of center of mass, α=1/(1+com) = (span-1)/2, for com≥0.
    :param halflife: Specify decay in terms of half-life, α=1−exp(log(0.5)/halflife), for halflife>0
    :param alpha: Specify smoothing factor α directly, 0<α≤1.
    :param adjust: Divide by decaying adjustment factor in beginning periods to account for imbalance.
    :param add_to_data: Specify whether to add the result to DataFrame or not.
    :return: pandas.DataFrame object.
    """

    unknown_functions = set(functions) - {"mean", "std", "var"}
    if unknown_functions:
        raise Exception(f"Functions {list(unknown_functions)} are not part of known pandas.DataFrame.ewm methods.")

    non_null_parameters = [(key, value) for key, value in
                           {"span": span, "com": com, "halflife": halflife, "alpha": alpha}.items()
                           if value is not None]

    if not non_null_parameters:
        raise Exception("Either center of mass, span, halflife or alpha must be specified")

    src_columns, _ = _get_new_columns_info(columns)

    # Generate ewm object
    ewf = data[src_columns].ewm(com=com, span=span, halflife=halflife, alpha=alpha, adjust=adjust)
    result_df = pd.DataFrame()

    key, length = non_null_parameters.pop()

    for prefix in functions:
        columns_prefix = "EWM_" + str(prefix) + "_" + str(key) + "_" + str(round(length,2))
        src_columns, columns_label = _get_new_columns_info(columns, columns_prefix)

        df = ewf.agg(prefix)
        df.rename(columns=dict(zip(src_columns, columns_label)), inplace=True)

        if add_to_data:
            data[columns_label] = df
        else:
            result_df = pd.concat([result_df, df], axis=1)

    if add_to_data:
        return data
    else:
        return result_df


def exponential_weighted_moving_average(data: pd.DataFrame, columns: Union[List[str], str],
                                        span: Any = None, com: Any = None, halflife: Any = None,
                                        alpha: Any = None, add_to_data: bool = True) -> pd.DataFrame:
    return exponential_weighted_functions(data, columns, ["mean"], span, com, halflife, alpha, add_to_data)


def exponential_weighted_moving_std(data: pd.DataFrame, columns: Union[List[str], str],
                                    span: Any = None, com: Any = None, halflife: Any = None,
                                    alpha: Any = None, add_to_data: bool = True) -> pd.DataFrame:
    return exponential_weighted_functions(data, columns, ["std"], span, com, halflife, alpha, add_to_data)


def exponential_weighted_moving_var(data: pd.DataFrame, columns: Union[List[str], str],
                                    span: Any = None, com: Any = None, halflife: Any = None,
                                    alpha: Any = None, add_to_data: bool = True) -> pd.DataFrame:
    return exponential_weighted_functions(data, columns, ["var"], span, com, halflife, alpha, add_to_data)


def modified_moving_average(data: pd.DataFrame, columns: Union[List[str], str], window: int = 1,
                            add_to_data: bool = True) -> pd.DataFrame:
    return exponential_weighted_functions(data, columns, ["mean"],  alpha=1.0/window, add_to_data=add_to_data,
                                          adjust=False)


def hull_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data


def smoothed_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data


def arnaud_legoux_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data


def least_squares_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data

