"""
This module is defining all the functions that are necessary to create indicators.
"""

from typing import List, Union, Any, Dict
from .utils import _get_columns
from .processing import ProcessingIndicators
import pandas as pd
import numpy as np


class Indicators(ProcessingIndicators):
    """
    Class containing all the definitions of built-in indicators.
    """

    def __init__(self, data: pd.DataFrame):
        if data is None:
            raise Exception("Indicators object must be initialized with a valid pandas.DataFrame object.")

        self._data = data
        self._preprocessing = ProcessingIndicators(self._data)

    @property
    def data(self):
        return self._data

    @property
    def preprocessing(self):
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, value):
        raise Exception("Read only objects cannot be assigned a variable.")

    def set_index(self, index_column_name: str, new_index_name: str = None, ascending_order: bool = True,
                  is_timestamp: bool = False, unit: str = 's') -> pd.DataFrame:
        """
        Take a DataFrame as entry and set the column timestamp_column_name as index.

        :param pandas.DataFrame data: The data in form of pandas.DataFrame.
        :param str index_column_name: Name of the index column.
        :param str new_index_name: The new name of the index. Used to raplace index_column_name.
        :param bool ascending_order: Sort the DataFrame in ascending order.
        :param bool is_timestamp: Is the index column in form of timestamp.
        :param str unit: Used only when is_timestamp is true.
        """

        if index_column_name not in set(self.data.columns):
            raise Exception(f"Column {index_column_name} is not in the DataFrame.")

        column_name = index_column_name
        if new_index_name is not None:
            self.data.rename(columns={index_column_name: new_index_name}, inplace=True)
            column_name = new_index_name

        if is_timestamp:
            self.data[column_name] = pd.to_datetime(self.data[column_name], infer_datetime_format=True, unit=unit)

        self.data.set_index(column_name, inplace=True)
        self.data.sort_index(ascending=ascending_order, inplace=True)

        return self.data

    @staticmethod
    def moving_window_functions(data: pd.DataFrame, columns: Union[List[str], str],
                                functions: List[str] = ["mean"],
                                window: int = 14, quantile: float = 0.05,
                                result_names: Dict[str, List[str]] = {"mean": ["SMA"]},
                                add_to_data: bool = True) -> pd.DataFrame:
        """
        Provide rolling window calculations.

        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

        :param data: Pandas data type.
        :param columns: Columns for which we should calculate the ewm.
        :param functions: Functions to apply: mean or std or var.
        :param window: Length of the window.
        :param quantile: quantile in ]0,1[ when we want to compute a quantile of the window.
        :param result_names: Prefix for the name of the column.
        :param add_to_data: Specify whether to add the result to DataFrame or not.
        :return: pandas.DataFrame object.
        """

        unknown_functions = set(functions) - {"sum", "mean", "median", "min", "max", "std", "var", "skew", "kurt",
                                              "quantile"}
        if unknown_functions:
            raise Exception(
                f"Functions {list(unknown_functions)} are not part of known pandas.DataFrame.rolling methods.")

        src_columns = _get_columns(columns)
        rolling_window = data[src_columns].rolling(window=window)  # Generate a rolling window object
        result_df = pd.DataFrame()

        for i, func in enumerate(functions):
            if func not in result_names:
                raise Exception(f"No name has been given for the result of {func} operation")

            target_columns_names = result_names[func]
            if not isinstance(target_columns_names, list):
                raise Exception(f"{target_columns_names} should be a list")

            if len(src_columns) != len(target_columns_names):
                raise Exception(f"{func} List of result names should have the same length as list of functions")

            df = rolling_window.agg(func) if func != "quantile" else rolling_window.quantile(quantile=quantile)
            df.rename(columns=dict(zip(src_columns, target_columns_names)), inplace=True)

            if add_to_data:
                data[target_columns_names] = df
            else:
                result_df = pd.concat([result_df, df], axis=1)

        if add_to_data:
            return data
        else:
            return result_df

    def _moving_window_functions(self, columns: Union[List[str], str],
                                 functions: List[str] = ["mean"],
                                 window: int = 14, quantile: float = 0.05,
                                 result_names: Dict[str, List[str]] = {"mean": ["SMA"]},
                                 add_to_data: bool = True) -> pd.DataFrame:

        return self.moving_window_functions(self._data, columns, functions, window, quantile, result_names, add_to_data)

    def moving_average(self, columns: Union[List[str], str], window: int = 14,
                       result_names: List[str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["SMA" + str(window) + "_" + col for col in columns]

        return self._moving_window_functions(columns, ["mean"], window, result_names={"mean": result_names},
                                             add_to_data=add_to_data)

    def moving_std(self, columns: Union[List[str], str], window: int = 14,
                   result_names: List[str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["Std" + str(window) + "_" + col for col in columns]

        return self._moving_window_functions(columns, ["std"], window, result_names={"std": result_names},
                                             add_to_data=add_to_data)

    def moving_var(self, columns: Union[List[str], str], window: int = 14,
                   result_names: List[str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["VAR" + str(window) + "_" + col for col in columns]

        return self._moving_window_functions(columns, ["var"], window, result_names={"var": result_names},
                                             add_to_data=add_to_data)

    def true_range(self, close_name: str = "close", high_name: str = "high", low_name: str = "low",
                   window: int = 1, result_names: str = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = "TR_" + str(window)

        df_close = self.data[close_name].shift(1)
        df = pd.DataFrame(pd.concat([self.data[high_name], df_close], axis=1).max(axis=1)
                          - pd.concat([self.data[low_name], df_close], axis=1).min(axis=1), columns=[result_names])

        if window > 1:
            df = df.rolling(window=window).mean()

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def average_true_range(self, close_name: str = "close", high_name: str = "high", low_name: str = "low",
                           window: int = 1, result_names: str = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = "ATR_" + str(window)

        df_tr = self.true_range(close_name=close_name, high_name=high_name, low_name=low_name, window=1,
                                result_names="TR", add_to_data=False)

        df_atr = self.exponential_weighted_functions(df_tr, "TR", functions=["mean"], alpha=1.0 / window,
                                                     add_to_data=False)
        df_atr.columns = [result_names]

        if add_to_data:
            self.data[result_names] = df_atr
            return self.data
        else:
            return df_atr

    def average_directional_index(self, close_name: str = "close", high_name: str = "high",
                                  low_name: str = "low", window: int = 14, result_names: str = None,
                                  add_to_data: bool = True) -> pd.DataFrame:

        """
        Implements the Average Directional Index using the formula from https://www.investopedia.com/terms/a/adx.asp and
        https://fr.wikipedia.org/wiki/Directional_Movement_Index and
        https://www.investopedia.com/ask/answers/112814/how-average-directional-index-adx-calculated-and-what-formula.asp.
        """

        if result_names is None:
            result_names = "ADX_" + str(window)

        m_positive = self.data[high_name] - self.data[high_name].shift(1)
        m_negative = self.data[low_name].shift(1) - self.data[low_name]

        dm_positive = pd.DataFrame(m_positive.copy())
        dm_positive.columns = ["DMp"]
        dm_positive[m_positive < m_negative] = 0
        dm_positive[m_positive < 0] = 0

        dm_negative = pd.DataFrame(m_negative.copy())
        dm_negative.columns = ["DMn"]
        dm_negative[m_negative < m_positive] = 0
        dm_negative[m_negative < 0] = 0

        smoothed_dm_positive = self.exponential_weighted_functions(dm_positive, "DMp", functions=["mean"],
                                                                   alpha=1.0 / window, add_to_data=False)
        smoothed_dm_positive.columns = ["SDMp"]

        smoothed_dm_negative = self.exponential_weighted_functions(dm_negative, "DMn", functions=["mean"],
                                                                   alpha=1.0 / window, add_to_data=False)
        smoothed_dm_negative.columns = ["SDMn"]

        # DI+ and DI-
        atr = self.average_true_range(close_name, high_name, low_name, window, "ATR", False)
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
        adx = self.exponential_weighted_functions(dxi, "DXI_" + str(window), functions=["mean"], alpha=1.0 / window,
                                                  add_to_data=False)
        adx.columns = [result_names]

        df = pd.concat([directional_index_positive, directional_index_negative, dxi, adx], axis=1)

        if add_to_data:
            self.data[df.columns] = df
            return self.data
        else:
            return df

    def moving_average_channel(self, columns: Union[List[str], str], window: int = 1,
                               nb_of_deviations: int = 3, result_names: List[str] = None,
                               add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["MAChannel" + str(window) + "_" + str(nb_of_deviations) + "_" + col for col in columns]

        # Compute average and initialize the result
        average_names = [col + "_AVG" for col in result_names]
        result_df = self.moving_average(columns=columns, window=window, result_names=average_names, add_to_data=False)

        # Compute Std
        std_names = [col + "_STD" for col in result_names]
        std = self.moving_std(columns=columns, window=window, result_names=std_names, add_to_data=False)

        # Initialize UP and DOWN columns names
        upper_channel_names = [col + "_UP" for col in result_names]
        down_channel_names = [col + "_DOWN" for col in result_names]

        for i, col in enumerate(average_names):
            result_df = pd.concat([result_df,
                                   pd.DataFrame(result_df[col] + nb_of_deviations * std[std_names[i]],
                                                columns=[upper_channel_names[i]]),
                                   pd.DataFrame(result_df[col] - nb_of_deviations * std[std_names[i]],
                                                columns=[down_channel_names[i]])
                                   ], axis=1)

        if add_to_data:
            self.data[result_df.columns] = result_df
            return self.data
        else:
            return result_df

    def _exponential_weighted_functions(self, columns: Union[List[str], str],
                                        functions: List[str] = ["mean"],
                                        span: Any = None, com: Any = None, halflife: Any = None,
                                        alpha: Any = None, adjust: bool = True,
                                        result_names: Dict[str, List[str]] = {"mean": ["EMA"]},
                                        add_to_data: bool = True) -> pd.DataFrame:

        return self.exponential_weighted_functions(self.data, columns, functions, span, com, halflife, alpha, adjust,
                                                   result_names, add_to_data)

    @staticmethod
    def exponential_weighted_functions(data: pd.DataFrame, columns: Union[List[str], str],
                                       functions: List[str] = ["mean"],
                                       span: Any = None, com: Any = None, halflife: Any = None,
                                       alpha: Any = None, adjust: bool = True,
                                       result_names: Dict[str, List[str]] = {"mean": ["EMA"]},
                                       add_to_data: bool = True) -> pd.DataFrame:
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
        :param result_names: Prefix for the name of the column.
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

        src_columns = _get_columns(columns)

        # Generate ewm object
        ewf = data[src_columns].ewm(com=com, span=span, halflife=halflife, alpha=alpha, adjust=adjust)
        result_df = pd.DataFrame()

        for i, func in enumerate(functions):
            if func not in result_names:
                raise Exception(f"No name has been given for the result of {func} operation")

            target_columns_names = result_names[func]
            if not isinstance(target_columns_names, list):
                raise Exception(f"{target_columns_names} should be a list")

            if len(src_columns) != len(target_columns_names):
                raise Exception(f"{func} List of result names should have the same length as list of functions")

            df = ewf.agg(func)
            df.rename(columns=dict(zip(src_columns, target_columns_names)), inplace=True)

            if add_to_data:
                data[target_columns_names] = df
            else:
                result_df = pd.concat([result_df, df], axis=1)

        if add_to_data:
            return data
        else:
            return result_df

    def exponential_weighted_moving_average(self, columns: Union[List[str], str],
                                            span: Any = None, com: Any = None, halflife: Any = None,
                                            alpha: Any = None, result_names: List[str] = None,
                                            add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMA" + "_" + col for col in columns]

        return self._exponential_weighted_functions(columns, ["mean"], span, com, halflife, alpha,
                                                    result_names={"mean": result_names}, add_to_data=add_to_data)

    def exponential_weighted_moving_std(self, columns: Union[List[str], str],
                                        span: Any = None, com: Any = None, halflife: Any = None,
                                        alpha: Any = None, result_names: List[str] = None,
                                        add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMStd" + "_" + col for col in columns]

        return self._exponential_weighted_functions(columns, ["std"], span, com, halflife, alpha,
                                                    result_names={"std": result_names}, add_to_data=add_to_data)

    def exponential_weighted_moving_var(self, columns: Union[List[str], str],
                                        span: Any = None, com: Any = None, halflife: Any = None,
                                        alpha: Any = None, result_names: List[str] = None,
                                        add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMVar" + "_" + col for col in columns]

        return self._exponential_weighted_functions(columns, ["var"], span, com, halflife, alpha,
                                                    result_names={"var": result_names}, add_to_data=add_to_data)

    def modified_moving_average(self, columns: Union[List[str], str], window: int = 1,
                                result_names: List[str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["ModAverage" + str(window) + "_" + col for col in columns]

        return self._exponential_weighted_functions(columns, ["mean"], alpha=1.0 / window, adjust=False,
                                                    result_names={"mean": result_names}, add_to_data=add_to_data)

    def exponential_weighted_moving_average_channel(self, columns: Union[List[str], str], span: Any = None,
                                                    com: Any = None, halflife: Any = None, alpha: Any = None,
                                                    nb_of_deviations: int = 3, result_names: List[str] = None,
                                                    add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMAChannel" + "_" + str(nb_of_deviations) + "_" + col for col in columns]

        # Compute average and initialize the result
        average_names = [col + "_AVG" for col in result_names]
        result_df = self.exponential_weighted_moving_average(columns=columns, span=span, com=com, halflife=halflife,
                                                             alpha=alpha, result_names=average_names, add_to_data=False)

        # Compute Std
        std_names = [col + "_STD" for col in result_names]
        std = self.exponential_weighted_moving_std(columns=columns, span=span, com=com, halflife=halflife,
                                                   alpha=alpha, result_names=std_names, add_to_data=False)

        # Initialize UP and DOWN columns names
        upper_channel_names = [col + "_UP" for col in result_names]
        down_channel_names = [col + "_DOWN" for col in result_names]

        for i, col in enumerate(average_names):
            result_df = pd.concat([result_df,
                                   pd.DataFrame(result_df[col] + nb_of_deviations * std[std_names[i]],
                                                columns=[upper_channel_names[i]]),
                                   pd.DataFrame(result_df[col] - nb_of_deviations * std[std_names[i]],
                                                columns=[down_channel_names[i]])
                                   ], axis=1)

        if add_to_data:
            self.data[result_df.columns] = result_df
            return self.data
        else:
            return result_df

    def hull_moving_average(self, columns: Union[List[str], str], window: int = 1, result_names: List[str] = None,
                            add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["HullMA" + "_" + col for col in columns]

        df_n = self.exponential_weighted_functions(self.data, columns=columns, functions=["mean"], span=window,
                                                   result_names={"mean": result_names}, add_to_data=False)

        df_n2 = 2 * self.exponential_weighted_functions(self.data, columns=columns, functions=["mean"],
                                                        span=int(window / 2),
                                                        result_names={"mean": result_names}, add_to_data=False)

        df_nsqrt = self.exponential_weighted_functions(df_n2 - df_n, columns=result_names, functions=["mean"],
                                                       span=int(np.sqrt(window)),
                                                       result_names={"mean": result_names}, add_to_data=False)

        if add_to_data:
            self.data[df_nsqrt.columns] = df_nsqrt
            return self.data
        else:
            return df_nsqrt



