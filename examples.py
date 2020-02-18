"""
Example implementation.
"""

from exchangeapi.huobi.rest import Rest
from exchangeapi.huobi.models.enumerations import TickerPeriod
from strategies.indicators import Indicators
import pandas as pd
import numpy as np


# Setting the environment
api_huobi_rest = Rest("", "", False)


def get_data_example() -> pd.DataFrame:
    """
    Function to compute and preprocess DataFrame indicators, then transform prices to returns

    This example shows how to use Pandas built-in functions for the DataFrame.

    :param data: All data imported from exchange in form of DataFrame.
    :return: The computed and then preprocessed indicators DataFrame.
    """

    # Import data from the exchange and initialize Indicators object
    indicator_object = Indicators(api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 2000, pd.DataFrame))

    # Compute indicators
    compute_indicators(indicator_object)

    # Preprocess indicators using indicator_object.preprocessing namespace
    preprocess_indicators(indicator_object)

    # Preprocess indicators using Pandas built-in functions
    data = indicator_object.data

    # Columns for which returns computation have sense (X_(i+1) - X_(i)) / X_(i)
    src_columns = ["open", "close", "high", "low"]
    returns_columns = [x + "_log_returns_" + str(i + 1) for x in src_columns for i in range(14)]
    returns_columns.extend([x + "_pct_changes_" + str(i + 1) for x in src_columns for i in range(14)])
    returns_columns.extend([x + "_avg_14" for x in returns_columns])
    target_columns_names = [x + "_square" for x in returns_columns]

    data[target_columns_names] = data[returns_columns].apply(lambda x: np.square(x))

    # Raise Velocity
    data["high_low_raise_velocity"] = data[["high", "low", "amount"]].apply(lambda x: (x["high"] - x["low"]) / x["amount"],
                                                                            axis=1)
    data["tr_raise_velocity"] = data[["tr", "amount"]].apply(lambda x: x["tr"] / x["amount"], axis=1)
    data["tr_avg_14_raise_velocity"] = data[["tr_avg_14", "amount"]].apply(lambda x: x["tr_avg_14"] / x["amount"],
                                                                           axis=1)

    return data


# Defining the list of routines to apply as a strategy
def preprocess_indicators(indicator_object: Indicators):
    """
    This function shows how to use the preprocessing object.
    :param indicator_object:
    :return:
    """

    preprocessing_object = indicator_object.preprocessing
    src_columns = []
    drop_columns = []

    # Compute log returns and normal returns for i+1 to i+14
    src_columns = ["open", "close", "high", "low"]
    for i in range(14):
        preprocessing_object.log_returns(src_columns, [x + "_log_returns_" + str(i+1) for x in src_columns], window=i+1)
        preprocessing_object.pct_change(src_columns, [x + "_pct_changes_" + str(i+1) for x in src_columns], window=i+1)

    # Compute the Simple and Exponential average of returns
    returns_columns = [x + "_log_returns_" + str(i + 1) for x in src_columns for i in range(14)]
    returns_columns.extend([x + "_pct_changes_" + str(i + 1) for x in src_columns for i in range(14)])
    target_columns_names = [x + "_avg_14" for x in returns_columns]
    indicator_object.moving_average(returns_columns, window=14, result_names=target_columns_names)
    indicator_object.exponential_weighted_moving_average(returns_columns, span=14,
                                                         result_names=["ex_" + x for x in target_columns_names])

    # Compute the Simple and Exponential var of returns
    target_columns_names = [x + "_var_14" for x in returns_columns]
    indicator_object.moving_var(returns_columns, window=14, result_names=target_columns_names)
    indicator_object.exponential_weighted_moving_var(returns_columns, span=14,
                                                     result_names=["ex_" + x for x in target_columns_names])

    # Compute the Simple and Exponential std of returns
    target_columns_names = [x + "_std_14" for x in returns_columns]
    indicator_object.moving_std(returns_columns, window=14, result_names=target_columns_names)
    indicator_object.exponential_weighted_moving_std(returns_columns, span=14,
                                                     result_names=["ex_" + x for x in target_columns_names])

    # Compute log division for all columns for which it is meaningful to know how much they deviate from source prices.
    right_columns = []
    for x in ["open", "close", "high", "low"]:
        src_columns.extend(
            [prefix + "machannel_14_3_" + x + suffix for suffix in {"_AVG", "_UP", "_DOWN"} for prefix in {"", "ex_"}])
        right_columns.extend([x] * 6)

    drop_columns.extend(src_columns)
    target_columns_names = [x + "_log_div" for x in src_columns]
    df = pd.DataFrame(np.log(preprocessing_object.data[src_columns].values
                             / preprocessing_object.data[right_columns].values), columns=target_columns_names)
    df.index = preprocessing_object.data.index
    preprocessing_object.data[target_columns_names] = df

    if drop_columns:
        preprocessing_object.data.drop(columns=drop_columns, inplace=True)


# Defining the list of routines to apply as a strategy
def compute_indicators(indicator_object: Indicators):
    indicator_object.set_index(index_column_name="id", new_index_name="timestamp", ascending_order=True,
                               is_timestamp=True, unit='s')

    # Simple and Exponential Moving average Channel
    src_columns = ["open", "close", "high", "low"]
    target_columns_names = ["machannel_14_3_" + x for x in src_columns]
    indicator_object.moving_average_channel(columns=src_columns, window=14, nb_of_deviations=3, add_to_data=True,
                                            result_names=target_columns_names)
    indicator_object.exponential_weighted_moving_average_channel(columns=src_columns, nb_of_deviations=3, span=14,
                                                                 result_names=["ex_" + x for x in target_columns_names])

    # Hull moving average
    indicator_object.hull_moving_average(columns=src_columns, result_names=["hull_14_" + x for x in src_columns],
                                         window=14)

    # True Range and its average
    indicator_object.true_range(window=1, result_names="tr")
    indicator_object.average_true_range(result_names="tr_avg_14", window=14)

    # Average Directional Index
    indicator_object.average_directional_index(result_names="adx_14", window=14)

    # Simple and Exponential moving average, std and var
    indicator_object.moving_average(columns=src_columns, window=14, result_names=[x + "_avg_14" for x in src_columns])
    indicator_object.moving_std(columns=src_columns, window=14, result_names=[x + "_std_14" for x in src_columns])
    indicator_object.moving_var(columns=src_columns, window=14, result_names=[x + "_var_14" for x in src_columns])
    indicator_object.exponential_weighted_moving_average(columns=src_columns, span=14,
                                                         result_names=["ex_" + x + "_avg_14" for x in src_columns])

    indicator_object.exponential_weighted_moving_var(columns=src_columns, span=14,
                                                     result_names=["ex_" + x + "_var_14" for x in src_columns])

    indicator_object.exponential_weighted_moving_std(columns=src_columns, span=14,
                                                     result_names=["ex_" + x + "_std_14" for x in src_columns])

    # Quantile 0.05 and 0.95
    indicator_object.moving_window_functions(indicator_object.data, columns=src_columns, functions=["quantile"],
                                             quantile=0.05, window=14,
                                             result_names={"quantile": [x + "_q_5" for x in src_columns]})

    indicator_object.moving_window_functions(indicator_object.data, columns=src_columns, functions=["quantile"],
                                             quantile=0.95, window=14,
                                             result_names={"quantile": [x + "_q_95" for x in src_columns]})





