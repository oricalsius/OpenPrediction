"""
This module is defining all the functions that are necessary to create indicators.
"""

from typing import List, Union, Any, Dict, Tuple, Callable
from scipy.stats import norminvgauss, norm
from .optim import *

import numpy as np
import pandas as pd

from .processing import ProcessingIndicators
from .utils import _get_columns


class Indicators:
    """
    Class containing all the definitions of built-in indicators.
    """

    def __init__(self, data: pd.DataFrame, is_parallel: bool = False):
        if data is None:
            raise Exception("Indicators object must be initialized with a valid pandas.DataFrame object.")

        self._data = data
        self._preprocessing = ProcessingIndicators(self._data)
        self._is_parallel_computing = is_parallel

    # region property
    @property
    def data(self):
        return self._data

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def is_parallel_computing(self):
        return self._is_parallel_computing

    @preprocessing.setter
    def preprocessing(self, value):
        raise Exception("Read only objects cannot be assigned a variable.")
    # endregion

    # region other functions
    @staticmethod
    def nig_variates(a: float, b: float, loc: float = 0, scale: float = 1, return_callable=False, size: int = 1,
                     random_state: int = None) -> np.ndarray:
        """
        A Normal Inverse Gaussian continuous random variable.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html
        """

        if return_callable:
            return norminvgauss(a=a, b=b, loc=loc, scale=scale).rvs
        else:
            return norminvgauss.rvs(a=a, b=b, loc=loc, scale=scale, size=size, random_state=random_state)

    @staticmethod
    def nig_fit(data: List, floc: float = None, fscale: float = None) -> Tuple[float, float, float, float]:
        """
        Fit data to a Normal Inverse Gaussian continuous random distribution.
        """

        kwargs = {}
        if floc is not None:
            kwargs['floc'] = floc

        if fscale is not None:
            kwargs['fscale'] = fscale

        a, b, loc, scale = norminvgauss.fit(data, **kwargs)

        return a, b, loc, scale

    @staticmethod
    def normal_variates(mean: float = 0, std: float = 1, return_callable=False, size: int = 1,
                        random_state: int = None) -> np.ndarray:
        """
        A normal continuous random variable.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        """

        if return_callable:
            return norm(loc=mean, scale=std).rvs
        else:
            return norm.rvs(loc=mean, scale=std, size=size, random_state=random_state)

    @staticmethod
    def normal_fit(data: List) -> Tuple[float, float]:
        """
        Fit data to a normal continuous random distribution.
        """

        mean, std = norm.fit(data)

        return mean, std
    # endregion

    # region indicators
    @staticmethod
    def moving_window_functions(data: pd.DataFrame, columns: Union[List[str], str],
                                functions: List[str] = ["mean"],
                                window: int = 14, quantile: float = 0.05,
                                result_names: Dict[str, List[str]] = {"mean": ["SMA"]},
                                is_parallel_computing: bool = False, add_to_data: bool = True,
                                ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
        """
        Provide rolling window calculations.

        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

        :param data: Pandas data type.
        :param columns: Columns for which we should calculate the ewm.
        :param functions: Functions to apply: mean or std or var.
        :param window: Length of the window.
        :param quantile: quantile in ]0,1[ when we want to compute a quantile of the window.
        :param result_names: Prefix for the name of the column.
        :param is_parallel_computing: Use parallel computing with numba.
        :param add_to_data: Specify whether to add the result to DataFrame or not.
        :param ml_format: Specify a callable to apply to the result or True to apply the default one.
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

            if ml_format is not None:
                if callable(ml_format):
                    df[target_columns_names] = ml_format(df[target_columns_names], *args, **kwargs)

                elif isinstance(ml_format, bool):
                    if ml_format:
                        df = pd.DataFrame(data=numpy_log(is_parallel_computing,
                                                         numpy_div(is_parallel_computing,
                                                                   df[target_columns_names].values,
                                                                   data[src_columns].values)),
                                          columns=target_columns_names, index=df.index)

                else:
                    raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

            if add_to_data:
                data[target_columns_names] = df
            else:
                result_df = pd.concat([result_df, df], axis=1)

        if add_to_data:
            return data
        else:
            return result_df

    def _moving_window_functions(self, columns: Union[List[str], str], functions: List[str] = ["mean"],
                                 window: int = 14, quantile: float = 0.05,
                                 result_names: Dict[str, List[str]] = {"mean": ["SMA"]},
                                 add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                 *args, **kwargs) -> pd.DataFrame:

        return self.moving_window_functions(self._data, columns, functions, window, quantile,
                                            result_names, self.is_parallel_computing, add_to_data, ml_format,
                                            *args, **kwargs)

    def moving_average(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                       add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                       *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["SMA" + str(window) + "_" + col for col in columns]

        df = self._moving_window_functions(columns, ["mean"], window, result_names={"mean": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_std(self, columns: Union[List[str], str], window: int = 14,
                   result_names: List[str] = None, add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                   *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["Std" + str(window) + "_" + col for col in columns]

        df = self._moving_window_functions(columns, ["std"], window, result_names={"std": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_var(self, columns: Union[List[str], str], window: int = 14,
                   result_names: List[str] = None, add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                   *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["VAR" + str(window) + "_" + col for col in columns]

        df = self._moving_window_functions(columns, ["var"], window, result_names={"var": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_quantile(self, columns: Union[List[str], str], window: int = 14, quantile: float = 0.05,
                        result_names: List[str] = None, add_to_data: bool = True,
                        ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_q_" + str(window) + "_" + str(quantile) for col in columns]

        df = self._moving_window_functions(columns, ["quantile"], window, quantile,
                                           result_names={"quantile": result_names}, add_to_data=add_to_data,
                                           ml_format=ml_format, *args, **kwargs)

        return df

    def moving_sum(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                   add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                   *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_SSum_" + str(window) for col in columns]

        df = self._moving_window_functions(columns, ["sum"], window, result_names={"sum": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_median(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                      add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                      *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_SMedian_" + str(window) for col in columns]

        df = self._moving_window_functions(columns, ["median"], window, result_names={"median": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_min(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                   add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                      *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_SMin_" + str(window) for col in columns]

        df = self._moving_window_functions(columns, ["min"], window, result_names={"min": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_max(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                   add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                      *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_SMax_" + str(window) for col in columns]

        df = self._moving_window_functions(columns, ["max"], window, result_names={"max": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_skew(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                    add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                    *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_SSkew_" + str(window) for col in columns]

        df = self._moving_window_functions(columns, ["skew"], window, result_names={"skew": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def moving_kurt(self, columns: Union[List[str], str], window: int = 14, result_names: List[str] = None,
                    add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                    *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_SKurt_" + str(window) for col in columns]

        df = self._moving_window_functions(columns, ["kurt"], window, result_names={"kurt": result_names},
                                           add_to_data=add_to_data, ml_format=ml_format, *args, **kwargs)

        return df

    def true_range(self, close_name: str = "close", high_name: str = "high", low_name: str = "low",
                   window: int = 1, result_names: str = None, is_parallel: bool = False,
                   add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                   *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = "TR_" + str(window)

        df_close = self.data[close_name].shift(1)
        d1 = pd.concat([self.data[high_name], df_close], axis=1).max(axis=1)
        d2 = pd.concat([self.data[low_name], df_close], axis=1).min(axis=1)
        df = pd.DataFrame(numpy_sub(is_parallel, d1.values, d2.values), columns=[result_names], index=self.data.index)

        if window > 1:
            df = df.rolling(window=window).mean()

        if ml_format is not None:
            if callable(ml_format):
                df[[result_names]] = ml_format(df[[result_names]], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    columns = ["open", "close", "high", "low", "vol", "amount"]
                    columns = list(set(columns).intersection(set(self.data.columns)))
                    new_columns = [result_names + "_velocity_" + x for x in columns]

                    df = pd.DataFrame(numpy_log(is_parallel, numpy_div(is_parallel,
                                                                       df[[result_names]*len(columns)].values,
                                                                       self.data[columns].values)),
                                      columns=new_columns, index=df.index)

                    result_names = new_columns

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def average_true_range(self, close_name: str = "close", high_name: str = "high", low_name: str = "low",
                           window: int = 1, result_names: str = None, is_parallel: bool = False,
                           add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                           *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = "ATR_" + str(window)

        df_tr = self.true_range(close_name=close_name, high_name=high_name, low_name=low_name, window=1,
                                result_names="TR", add_to_data=False)

        df_atr = self.exponential_weighted_functions(df_tr, "TR", functions=["mean"], alpha=1.0 / window,
                                                     add_to_data=False)
        df_atr.columns = [result_names]

        if ml_format is not None:
            if callable(ml_format):
                df_atr[[result_names]] = ml_format(df_atr[[result_names]], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    columns = ["open", "close", "high", "low", "vol", "amount"]
                    columns = list(set(columns).intersection(set(self.data.columns)))
                    new_columns = [result_names + "_velocity_" + x for x in columns]

                    df_atr = pd.DataFrame(numpy_log(is_parallel,
                                                    numpy_div(is_parallel,
                                                              df_atr[[result_names]*len(columns)].values,
                                                              self.data[columns].values)),
                                          columns=new_columns, index=df_atr.index)

                    result_names = new_columns

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[result_names] = df_atr
            return self.data
        else:
            return df_atr

    def returns_dix(self, columns: Union[List[str], str], window: int = 1, result_names: Union[List[str], str] = None,
                    add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "RDIX_" + str(window) for col in columns]

        src_columns = _get_columns(columns)
        is_parallel = self.is_parallel_computing

        # Compute returns
        d1 = numpy_div(is_parallel, self.data[src_columns].values, self.data[src_columns].shift(1).values)
        df = pd.DataFrame(numpy_log(is_parallel, d1), columns=src_columns, index=self.data.index)

        # Rolling window
        rol = df.rolling(window=window)
        abs_rol = df.abs().rolling(window=window)

        df = pd.DataFrame(numpy_div(is_parallel, rol.sum().values, abs_rol.sum().values, True),
                          columns=result_names, index=df.index)

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def returns_dix_average(self, columns: Union[List[str], str], window: int = 1,
                            result_names: Union[List[str], str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["RDIX_avg_" + str(window) + col for col in columns]

        src_columns = _get_columns(columns)

        target_name = ["RDIX_temp_" + x for x in src_columns]
        df_rdix = self.returns_dix(columns=columns, window=window, result_names=target_name, add_to_data=False)

        df_rdix_avg = self.exponential_weighted_functions(df_rdix, target_name, functions=["mean"], alpha=1.0 / window,
                                                          result_names={'mean': result_names}, add_to_data=False)

        if add_to_data:
            self.data[result_names] = df_rdix_avg
            return self.data
        else:
            return df_rdix_avg

    def returns_square_dix(self, columns: Union[List[str], str], window: int = 1,
                           result_names: Union[List[str], str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "Square_RDIX_" + str(window) for col in columns]

        src_columns = _get_columns(columns)
        is_parallel = self.is_parallel_computing

        # Compute returns
        d1 = numpy_div(is_parallel, self.data[src_columns].values, self.data[src_columns].shift(1).values)
        df = pd.DataFrame(numpy_log(is_parallel, d1), columns=src_columns, index=self.data.index)

        # Rolling window
        rol_2 = (df**2).rolling(window=window)
        rol = df.rolling(window=window)

        df = pd.DataFrame(numpy_div(is_parallel, numpy_mul(is_parallel, rol.sum().values, rol.sum().values),
                                    rol_2.sum().values, True), columns=result_names, index=df.index)

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def returns_square_dix_average(self, columns: Union[List[str], str], window: int = 1,
                                   result_names: Union[List[str], str] = None,
                                   add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["Square_RDIX_avg_" + str(window) + col for col in columns]

        src_columns = _get_columns(columns)

        target_name = ["Square_RDIX_temp_" + x for x in src_columns]
        sq_rdix = self.returns_square_dix(columns=columns, window=window, result_names=target_name, add_to_data=False)

        df_rdix_avg = self.exponential_weighted_functions(sq_rdix, target_name, functions=["mean"], alpha=1.0 / window,
                                                          result_names={'mean': result_names}, add_to_data=False)

        if add_to_data:
            self.data[result_names] = df_rdix_avg
            return self.data
        else:
            return df_rdix_avg

    def returns_norm_dix(self, columns: Union[List[str], str], window: int = 1,
                         result_names: Union[List[str], str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "Norm_RDIX_" + str(window) for col in columns]

        src_columns = _get_columns(columns)
        is_parallel = self.is_parallel_computing

        # Compute returns
        d1 = numpy_div(is_parallel, self.data[src_columns].values, self.data[src_columns].shift(1).values)
        dr_1 = pd.DataFrame(numpy_log(is_parallel, d1), columns=src_columns, index=self.data.index)

        d1 = numpy_div(is_parallel, self.data[src_columns].values, self.data[src_columns].shift(window).values)
        dr_w = pd.DataFrame(numpy_log(is_parallel, d1), columns=src_columns, index=self.data.index)

        # Rolling window
        rol = pd.DataFrame(numpy_mul(is_parallel, dr_1.values, dr_1.values),
                           columns=src_columns, index=dr_1.index).rolling(window=window)

        df = pd.DataFrame(numpy_div(is_parallel, dr_w.values, np.sqrt(rol.sum().values), True),
                          columns=result_names, index=dr_w.index)

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def returns_norm_dix_average(self, columns: Union[List[str], str], window: int = 1,
                                 result_names: Union[List[str], str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = ["Norm_RDIX_avg_" + str(window) + col for col in columns]

        src_columns = _get_columns(columns)

        target_name = ["Norm_RDIX_temp_" + x for x in src_columns]
        nr_rdix = self.returns_norm_dix(columns=columns, window=window, result_names=target_name, add_to_data=False)

        df_rdix_avg = self.exponential_weighted_functions(nr_rdix, target_name, functions=["mean"], alpha=1.0 / window,
                                                          result_names={'mean': result_names}, add_to_data=False)

        if add_to_data:
            self.data[result_names] = df_rdix_avg
            return self.data
        else:
            return df_rdix_avg

    def price_velocity(self, columns: Union[List[str], str], window: int = 1,
                       result_names: Union[List[str], str] = None, add_to_data: bool = True,
                       ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_PriceVelocity_" + str(window) for col in columns]

        src_columns = _get_columns(columns)
        is_parallel = self.is_parallel_computing

        df = self.data[src_columns].rolling(window=window)
        id_max = df.apply(np.argmax, raw=True)
        id_min = df.apply(np.argmin, raw=True)
        diff_time = numpy_sub(is_parallel, id_max.values, id_min.values)
        diff_val = numpy_sub(is_parallel, df.max().values, df.min().values)

        with np.errstate(invalid='ignore'):
            vel = numpy_mul(is_parallel, np.sign(diff_time),
                            numpy_div(is_parallel, diff_val, numpy_abs(is_parallel, diff_time) + 1))

        df = pd.DataFrame(vel, columns=result_names, index=id_max.index)
        if ml_format is not None:
            if callable(ml_format):
                df[[result_names]] = ml_format(df[[result_names]], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    df = pd.DataFrame(numpy_div(is_parallel, df[result_names].values,
                                                self.data[src_columns].values),
                                      columns=df.columns, index=df.index)

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def price_velocity_average(self, columns: Union[List[str], str], window: int = 1,
                               result_names: Union[List[str], str] = None, add_to_data: bool = True,
                               ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_PriceVelocity_avg_" + str(window) for col in columns]

        src_columns = _get_columns(columns)
        is_parallel = self.is_parallel_computing

        target_name = [x + "_PriceVelocity_temp" for x in src_columns]
        pv = self.price_velocity(columns=src_columns, window=window, result_names=target_name, add_to_data=False)

        pv_avg = self.exponential_weighted_functions(pv, target_name, functions=["mean"], alpha=1.0 / window,
                                                     result_names={'mean': result_names}, add_to_data=False)

        if ml_format is not None:
            if callable(ml_format):
                pv_avg[[result_names]] = ml_format(pv_avg[[result_names]], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    pv_avg = pd.DataFrame(numpy_div(is_parallel, pv_avg[result_names].values,
                                                    self.data[src_columns].values),
                                          columns=pv_avg.columns, index=pv_avg.index)

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[result_names] = pv_avg
            return self.data
        else:
            return pv_avg

    def returns_velocity(self, columns: Union[List[str], str], window: int = 1,
                         result_names: Union[List[str], str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_ReturnsVelocity_" + str(window) for col in columns]

        src_columns = _get_columns(columns)
        is_parallel = self.is_parallel_computing

        # Compute returns
        d1 = numpy_div(is_parallel, self.data[src_columns].values, self.data[src_columns].shift(1).values)
        ret = pd.DataFrame(numpy_log(is_parallel, d1), columns=src_columns, index=self.data.index)

        df = ret.rolling(window=window)
        id_max = df.apply(np.argmax, raw=True)
        id_min = df.apply(np.argmin, raw=True)
        diff_time = numpy_sub(is_parallel, id_max.values, id_min.values)
        diff_val = numpy_sub(is_parallel, df.max().values, df.min().values)

        with np.errstate(invalid='ignore'):
            vel = numpy_mul(is_parallel, np.sign(diff_time),
                            numpy_div(is_parallel, diff_val, numpy_abs(is_parallel, diff_time) + 1))

        df = pd.DataFrame(vel, columns=result_names, index=id_max.index)

        if add_to_data:
            self.data[result_names] = df
            return self.data
        else:
            return df

    def returns_velocity_average(self, columns: Union[List[str], str], window: int = 1,
                                 result_names: Union[List[str], str] = None, add_to_data: bool = True) -> pd.DataFrame:
        if result_names is None:
            result_names = [col + "_ReturnsVelocity_avg_" + str(window) for col in columns]

        src_columns = _get_columns(columns)

        target_name = [x + "_ReturnsVelocity_temp" for x in src_columns]
        rv = self.returns_velocity(columns=src_columns, window=window, result_names=target_name, add_to_data=False)

        rv_avg = self.exponential_weighted_functions(rv, target_name, functions=["mean"], alpha=1.0 / window,
                                                     result_names={'mean': result_names}, add_to_data=False)

        if add_to_data:
            self.data[result_names] = rv_avg
            return self.data
        else:
            return rv_avg

    def rsi(self, columns: Union[List[str], str], window: int = 1, result_names: Union[List[str], str] = None,
            add_to_data: bool = True) -> pd.DataFrame:

        if result_names is None:
            result_names = [col + "_rsi_" + str(window) for col in columns]

        src_columns, result_names = _get_columns(columns), _get_columns(result_names)
        is_parallel = self.is_parallel_computing

        # Check length
        if window > self.data.shape[0]:
            raise Exception(f"Parameter window greater than the length of the DataFrame ({self.data.shape[0]}).")

        # Compute returns
        diff = self.data[src_columns].diff(1).values

        up_vec = diff.copy()
        with np.errstate(invalid='ignore'):
            up_vec[diff < 0] = 0
        col_up = [x + "_UP" for x in src_columns]
        df_up = pd.DataFrame(up_vec, columns=col_up, index=self.data.index)
        avg_up = self.exponential_weighted_functions(df_up, col_up, functions=["mean"], span=window,
                                                     result_names={'mean': col_up}, add_to_data=False)

        down_vec = diff.copy()
        with np.errstate(invalid='ignore'):
            down_vec[diff > 0] = 0
            down_vec = np.abs(down_vec)
        col_down = [x + "_DOWN" for x in src_columns]
        df_down = pd.DataFrame(down_vec, columns=col_down, index=self.data.index)
        avg_down = self.exponential_weighted_functions(df_down, col_down, functions=["mean"], span=window,
                                                       result_names={'mean': col_down}, add_to_data=False)

        df_rsi = numpy_div(is_parallel, avg_up.values, avg_down.values, True)
        df_rsi = 100 - (100 / numpy_add(is_parallel, df_rsi, 1))
        df_rsi = pd.DataFrame(df_rsi, columns=result_names, index=self.data.index)

        if add_to_data:
            self.data[result_names] = df_rsi
            return self.data
        else:
            return df_rsi

    def relative_slope(self, high: str = 'high', low: str = 'low', close: str = 'close', typical_window: int = 1,
                       rels_window: int = 1, result_names: Union[List[str], str] = None,
                       add_to_data: bool = True) -> pd.DataFrame:

        if result_names is None:
            result_names = ["RELS_" + str(typical_window) + "_" + str(rels_window)]

        result_names = _get_columns(result_names)

        # Typical price
        typical_price = pd.DataFrame((self.data[high].values + self.data[low].values + self.data[close].values)/3,
                                     columns=['typical_price'], index=self.data.index)

        mv_ty_price = self.exponential_weighted_functions(typical_price, ['typical_price'], functions=["mean"],
                                                          alpha=1.0 / typical_window,
                                                          result_names={'mean': ['typical_price']}, add_to_data=False)
        mv_ty_price_1 = mv_ty_price.shift(1)

        df = 2*(mv_ty_price-mv_ty_price_1)/(mv_ty_price+mv_ty_price_1)
        rels = self.exponential_weighted_functions(df, ['typical_price'], functions=["mean"],
                                                   alpha=1.0 / rels_window, result_names={'mean': result_names},
                                                   add_to_data=False)
        rels = rels.mul(100)

        if add_to_data:
            self.data[result_names] = rels
            return self.data
        else:
            return rels

    def average_directional_index(self, close_name: str = "close", high_name: str = "high",
                                  low_name: str = "low", window: int = 14, result_name: str = None,
                                  is_parallel: bool = False, add_to_data: bool = True) -> pd.DataFrame:

        """
        Implements the Average Directional Index using the formula from https://www.investopedia.com/terms/a/adx.asp and
        https://fr.wikipedia.org/wiki/Directional_Movement_Index and
        https://www.investopedia.com/ask/answers/112814/how-average-directional-index-adx-calculated-and-what-formula.asp.
        """

        if result_name is None:
            result_name = "ADX_" + str(window)

        if not isinstance(result_name, str):
            raise Exception("Parameter result_name should be a string.")

        m_positive = numpy_sub(is_parallel, self.data[high_name].values, self.data[high_name].shift(1).values)
        m_negative = numpy_sub(is_parallel, self.data[low_name].shift(1).values, self.data[low_name].values)

        dm_positive = m_positive.copy()
        with np.errstate(invalid='ignore'):
            dm_positive[(m_positive < m_negative) | (m_positive < 0)] = 0
        df = pd.DataFrame(dm_positive, columns=["DMp"], index=self.data.index)
        smoothed_dm_positive = self.exponential_weighted_functions(df, "DMp", functions=["mean"],
                                                                   result_names={'mean': ["SDMp"]},
                                                                   alpha=1.0 / window, add_to_data=False)

        dm_negative = m_negative.copy()
        with np.errstate(invalid='ignore'):
            dm_negative[(m_negative < m_positive) | (m_negative < 0)] = 0
        df = pd.DataFrame(dm_negative, columns=["DMn"], index=self.data.index)
        smoothed_dm_negative = self.exponential_weighted_functions(df, "DMn", functions=["mean"],
                                                                   result_names={'mean': ["SDMn"]},
                                                                   alpha=1.0 / window, add_to_data=False)

        # DI+ and DI-
        atr = self.average_true_range(close_name, high_name, low_name, window, "ATR",
                                      is_parallel=is_parallel, add_to_data=False)

        d1 = numpy_mul(is_parallel, numpy_div(is_parallel, smoothed_dm_positive["SDMp"].values, atr["ATR"].values), 100)
        directional_index_positive = pd.DataFrame(d1, columns=["DIp_" + str(window)], index=atr.index)

        d1 = numpy_mul(is_parallel, numpy_div(is_parallel, smoothed_dm_negative["SDMn"].values, atr["ATR"].values), 100)
        directional_index_negative = pd.DataFrame(d1, columns=["DIn_" + str(window)], index=atr.index)

        # DXI
        d1 = numpy_abs(is_parallel, numpy_sub(is_parallel, directional_index_positive["DIp_" + str(window)].values,
                                              directional_index_negative["DIn_" + str(window)].values))
        d2 = numpy_abs(is_parallel, numpy_add(is_parallel, directional_index_positive["DIp_" + str(window)].values,
                                              directional_index_negative["DIn_" + str(window)].values))

        dxi = pd.DataFrame(numpy_mul(is_parallel, numpy_div(is_parallel, d1, d2), 100),
                           columns=["DXI_" + str(window)], index=directional_index_positive.index)

        # ADX
        adx = self.exponential_weighted_functions(dxi, "DXI_" + str(window), functions=["mean"], alpha=1.0 / window,
                                                  result_names={'mean': [result_name]}, add_to_data=False)

        df = pd.concat([directional_index_positive, directional_index_negative, dxi, adx], axis=1)

        if add_to_data:
            self.data[df.columns] = df
            return self.data
        else:
            return df

    def moving_average_channel(self, columns: Union[List[str], str], window: int = 1, nb_of_deviations: int = 3,
                               result_names: List[str] = None, add_to_data: bool = True,
                               ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["MAChannel_" + str(window) + "_" + str(nb_of_deviations) + "_" + col for col in columns]

        is_parallel = self.is_parallel_computing

        src_columns, result_names = _get_columns(columns), _get_columns(result_names)

        average_names = [col + "_AVG" for col in result_names]
        std_names = [col + "_STD" for col in result_names]

        result_df = self.moving_average(columns=columns, window=window, result_names=average_names, add_to_data=False)
        std = self.moving_std(columns=columns, window=window, result_names=std_names, add_to_data=False)

        # Initialize UP and DOWN columns names
        upper_channel_names = [col + "_UP" for col in result_names]
        down_channel_names = [col + "_DOWN" for col in result_names]

        d1 = pd.DataFrame(numpy_add(is_parallel, result_df.values,
                                    numpy_mul(is_parallel, std.values, nb_of_deviations)),
                          columns=upper_channel_names, index=result_df.index)

        d2 = pd.DataFrame(np.maximum(numpy_sub(is_parallel, result_df.values,
                                     numpy_mul(is_parallel, std.values, nb_of_deviations)), 1),
                          columns=down_channel_names, index=result_df.index)

        result_df = pd.concat([result_df, d1, d2], axis=1)

        if ml_format is not None:
            if callable(ml_format):
                for col_list in (average_names, upper_channel_names, down_channel_names):
                    result_df[col_list] = ml_format(result_df[col_list], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    col_list = average_names + upper_channel_names + down_channel_names

                    d1 = pd.DataFrame(numpy_log(is_parallel,
                                                numpy_div(is_parallel, result_df[col_list].values,
                                                          self.data[columns*3].values)),
                                      columns=col_list, index=result_df.index)

                    result_df[col_list] = d1

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[result_df.columns] = result_df
            return self.data
        else:
            return result_df

    def commodity_channel_index(self, high: str = 'high', low: str = 'low', close: str = 'close',
                                window: int = 1, result_name: str = None, add_to_data: bool = True,
                                exponential_avg: bool = True) -> pd.DataFrame:
        if result_name is None:
            result_name = ["CCI_" + str(window)]

        is_parallel = self.is_parallel_computing
        result_name = _get_columns(result_name)

        typical_price = (self.data[high].values + self.data[low].values + self.data[close].values)/3
        typical_price = pd.DataFrame(typical_price, columns=['TP'], index=self.data.index)

        if exponential_avg:
            df = self.exponential_weighted_functions(typical_price, ['TP'], ["mean", "std"], add_to_data=False,
                                                     result_names={"mean": ['TP_avg'], "std": ['TP_std']}, span=window)

        else:
            df = self.moving_window_functions(typical_price, ['TP'], ["mean", "std"], add_to_data=False,
                                              result_names={"mean": ['TP_avg'], "std": ['TP_std']}, window=window)

        res = pd.DataFrame((typical_price.values - df[['TP_avg']].values) / (0.015*df[['TP_std']].values),
                           columns=result_name, index=self.data.index)

        if add_to_data:
            self.data[res.columns] = res
            return self.data
        else:
            return res

    def bollinger_bands(self, columns: Union[List[str], str], window: int = 1, nb_of_deviations: int = 2,
                        result_names: Union[List[str], str] = None, add_to_data: bool = True,
                        ml_format: Union[Callable, bool] = None,
                        *args, **kwargs) -> pd.DataFrame:

        if result_names is None:
            result_names = ["bb_" + str(window) + "_" + str(nb_of_deviations) + "_" + col for col in columns]

        df = self.moving_average_channel(columns, window, nb_of_deviations, result_names,
                                         add_to_data, ml_format, *args, **kwargs)
        return df

    def bollinger_bands_percentage(self, columns: Union[List[str], str], window: int = 1, nb_of_deviations: int = 2,
                                   result_names: Union[List[str], str] = None,
                                   add_to_data: bool = True) -> pd.DataFrame:

        if result_names is None:
            result_names = ["bb_per_" + str(window) + "_" + str(nb_of_deviations) + "_" + col for col in columns]

        src_columns, result_names = _get_columns(columns), _get_columns(result_names)

        is_parallel = self.is_parallel_computing

        df = self.bollinger_bands(columns, window, nb_of_deviations, result_names, add_to_data=False)

        col_up = [x + "_UP" for x in result_names]
        col_avg = [x + "_AVG" for x in result_names]
        col_down = [x + "_DOWN" for x in result_names]

        df = self._compute_bollinger_bands_percentage(df, self.data[src_columns], col_up, col_avg, col_down,
                                                      is_parallel, result_names)

        if add_to_data:
            self.data[df.columns] = df
            return self.data
        else:
            return df

    def pivot_points_fibonacci(self, previous_high: str = 'high', previous_low: str = 'low', previous_close: str = None,
                               previous_open: str = None, current_open: str = None, moving_window: int = 1,
                               fibonacci_levels: list = None, result_name: str = None, add_to_data: bool = True,
                               ml_format: Union[Callable, bool] = None, ref_columns: str = 'close',
                               *args, **kwargs) -> pd.DataFrame:

        if previous_close is None and previous_open is None and current_open is None:
            raise Exception('Either previous_close or previous_open or current_open parameters should be non empty.')

        if result_name is None:
            result_name = "PP_" + str(moving_window)

        if fibonacci_levels is None:
            fibonacci_rates = [0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618]#, 3.618, 4.236]
        else:
            fibonacci_rates = fibonacci_levels

        is_parallel = self.is_parallel_computing

        col_prev = [previous_high, previous_low]
        div_count = len(col_prev)

        pp_prev = self.data[col_prev].shift(1).rolling(window=moving_window)

        # High and Low
        high_prev, low_prev = pp_prev[[previous_high]].max(),  pp_prev[[previous_low]].min()
        pp_prev = None

        # PP
        pp = high_prev.values + low_prev.values
        diff = high_prev.values - low_prev.values
        high_prev, low_prev = None, None

        if previous_close is not None:
            pp += self.data[[previous_close]].shift(1).values
            div_count += 1

        if previous_open is not None:
            pp += self.data[[previous_open]].shift(moving_window).values
            div_count += 1

        if current_open is not None:
            pp += self.data[[current_open]].values
            div_count += 1

        pp /= div_count

        df = pd.DataFrame(pp, columns=[result_name], index=self.data.index)
        for i, lvl in enumerate(fibonacci_rates):
            df = pd.concat([df, pd.DataFrame(pp + lvl * diff,
                                             columns=[result_name + '_R' + str(i+1)], index=self.data.index),
                            pd.DataFrame(np.maximum(pp - lvl * diff, 1),
                                         columns=[result_name + '_S' + str(i + 1)], index=self.data.index)], axis=1)

        if ml_format is not None:
            if callable(ml_format):
                df = ml_format(df, *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    d1 = numpy_div(is_parallel, df.values, self.data[[ref_columns]].values, True)
                    df = pd.DataFrame(numpy_log(is_parallel, d1), columns=df.columns, index=df.index)

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[df.columns] = df
            return self.data
        else:
            return df

    def date_info(self, halving_dates: List = None, add_to_data: bool = True) -> pd.DataFrame:
        df_index = self.data.index

        # day, dayofweek, month
        df = pd.concat([pd.DataFrame(df_index.day.values.astype(np.int32), columns=['day'], index=df_index),
                        pd.DataFrame(df_index.dayofweek.values.astype(np.int32), columns=['dayofweek'], index=df_index),
                        pd.DataFrame(df_index.month.values.astype(np.int32), columns=['month'], index=df_index)],
                       axis=1)

        if halving_dates is not None:
            dx = (np.array(halving_dates, dtype='datetime64[D]')
                  - df_index.values.reshape((-1,1)).astype('datetime64[D]')).astype(np.float64)
            dx[dx < 0] = np.inf

            df = pd.concat([df, pd.DataFrame(dx.min(axis=1).astype(np.int64),
                                             columns=['halving_delta'], index=df_index)], axis=1)

        if add_to_data:
            self.data[df.columns] = df
            return self.data
        else:
            return df

    def _exponential_weighted_functions(self, columns: Union[List[str], str], functions: List[str] = ["mean"],
                                        span: Any = None, com: Any = None, halflife: Any = None, alpha: Any = None,
                                        adjust: bool = True, result_names: Dict[str, List[str]] = {"mean": ["EMA"]},
                                        add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                        *args, **kwargs) -> pd.DataFrame:

        return self.exponential_weighted_functions(self.data, columns, functions, span, com, halflife, alpha, adjust,
                                                   result_names, self.is_parallel_computing, add_to_data,
                                                   ml_format, *args, **kwargs)

    @staticmethod
    def exponential_weighted_functions(data: pd.DataFrame, columns: Union[List[str], str],
                                       functions: List[str] = ["mean"], span: Any = None, com: Any = None,
                                       halflife: Any = None, alpha: Any = None, adjust: bool = True,
                                       result_names: Dict[str, List[str]] = {"mean": ["EMA"]},
                                       is_parallel: bool = False, add_to_data: bool = True,
                                       ml_format: Union[Callable, bool] = None,
                                       *args, **kwargs) -> pd.DataFrame:
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
        :param is_parallel: Use parallel computing with numba.
        :param add_to_data: Specify whether to add the result to DataFrame or not.
        :param ml_format: Specify a callable to apply to the result or True to apply the default one.
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

            if ml_format is not None:
                if callable(ml_format):
                    df[target_columns_names] = ml_format(df[target_columns_names], *args, **kwargs)

                elif isinstance(ml_format, bool):
                    if ml_format:
                        d1 = numpy_div(is_parallel, df[target_columns_names].values, data[src_columns].values)
                        d1 = numpy_log(is_parallel, d1)

                        df = pd.DataFrame(d1, columns=target_columns_names, index=df.index)

                else:
                    raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

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
                                            add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                            *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMA" + "_" + col for col in columns]

        df = self._exponential_weighted_functions(columns, ["mean"], span, com, halflife, alpha,
                                                  result_names={"mean": result_names}, add_to_data=add_to_data,
                                                  ml_format=ml_format, *args, **kwargs)

        return df

    def exponential_weighted_moving_std(self, columns: Union[List[str], str],
                                        span: Any = None, com: Any = None, halflife: Any = None,
                                        alpha: Any = None, result_names: List[str] = None,
                                        add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                        *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMStd" + "_" + col for col in columns]

        df = self._exponential_weighted_functions(columns, ["std"], span, com, halflife, alpha,
                                                  result_names={"std": result_names}, add_to_data=add_to_data,
                                                  ml_format=ml_format, *args, **kwargs)

        return df

    def exponential_weighted_moving_var(self, columns: Union[List[str], str],
                                        span: Any = None, com: Any = None, halflife: Any = None,
                                        alpha: Any = None, result_names: List[str] = None,
                                        add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                        *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMVar" + "_" + col for col in columns]

        df = self._exponential_weighted_functions(columns, ["var"], span, com, halflife, alpha,
                                                  result_names={"var": result_names}, add_to_data=add_to_data,
                                                  ml_format=ml_format, *args, **kwargs)

        return df

    def modified_moving_average(self, columns: Union[List[str], str], window: int = 1,
                                result_names: List[str] = None, add_to_data: bool = True,
                                ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["ModAverage" + str(window) + "_" + col for col in columns]

        df = self._exponential_weighted_functions(columns, ["mean"], alpha=1.0 / window, adjust=False,
                                                  result_names={"mean": result_names}, add_to_data=add_to_data,
                                                  ml_format=ml_format, *args, **kwargs)

        return df

    def exponential_weighted_moving_average_channel(self, columns: Union[List[str], str], span: Any = None,
                                                    com: Any = None, halflife: Any = None, alpha: Any = None,
                                                    nb_of_deviations: int = 3, result_names: List[str] = None,
                                                    add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                                    *args, **kwargs) -> pd.DataFrame:
        if result_names is None:
            result_names = ["EMAChannel" + "_" + str(nb_of_deviations) + "_" + col for col in columns]

        is_parallel = self.is_parallel_computing

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

        d1 = pd.DataFrame(numpy_add(is_parallel, result_df.values,
                                    numpy_mul(is_parallel, std.values, nb_of_deviations)),
                          columns=upper_channel_names, index=result_df.index)

        d2 = pd.DataFrame(np.maximum(numpy_sub(is_parallel, result_df.values,
                                     numpy_mul(is_parallel, std.values, nb_of_deviations)), 1),
                          columns=down_channel_names, index=result_df.index)

        result_df = pd.concat([result_df, d1, d2], axis=1)

        if ml_format is not None:
            col_list = average_names + upper_channel_names + down_channel_names

            if callable(ml_format):
                result_df[col_list] = ml_format(result_df[col_list], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    d1 = numpy_div(is_parallel, result_df[col_list].values, self.data[columns*3].values)
                    d1 = numpy_log(is_parallel, d1)

                    result_df[col_list] = pd.DataFrame(d1, columns=col_list, index=result_df.index)

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[result_df.columns] = result_df
            return self.data
        else:
            return result_df

    def exponential_weighted_bollinger_bands(self, columns: Union[List[str], str], span: Any = None,
                                             com: Any = None, halflife: Any = None, alpha: Any = None,
                                             nb_of_deviations: int = 2, result_names: Union[List[str], str] = None,
                                             add_to_data: bool = True, ml_format: Union[Callable, bool] = None,
                                             *args, **kwargs) -> pd.DataFrame:

        if result_names is None:
            result_names = [col + "_ex_bb_" + str(nb_of_deviations) for col in columns]

        df = self.exponential_weighted_moving_average_channel(columns, span, com, halflife, alpha, nb_of_deviations,
                                                              result_names, add_to_data, ml_format, *args, **kwargs)
        return df

    def exponential_weighted_bollinger_bands_percentage(self, columns: Union[List[str], str], span: Any = None,
                                                        com: Any = None, halflife: Any = None, alpha: Any = None,
                                                        nb_of_deviations: int = 2,
                                                        result_names: Union[List[str], str] = None,
                                                        add_to_data: bool = True) -> pd.DataFrame:

        if result_names is None:
            result_names = [col + "_bb_per_" + str(nb_of_deviations) for col in columns]

        src_columns, result_names = _get_columns(columns), _get_columns(result_names)

        is_parallel = self.is_parallel_computing

        df = self.exponential_weighted_bollinger_bands(columns, span, com, halflife, alpha, nb_of_deviations,
                                                       result_names, add_to_data=False)

        col_up = [x + "_UP" for x in result_names]
        col_avg = [x + "_AVG" for x in result_names]
        col_down = [x + "_DOWN" for x in result_names]

        df = self._compute_bollinger_bands_percentage(df, self.data[src_columns], col_up, col_avg, col_down,
                                                      is_parallel, result_names)

        if add_to_data:
            self.data[df.columns] = df
            return self.data
        else:
            return df

    @staticmethod
    def _compute_bollinger_bands_percentage(df, original_data, col_up: List[str] = None, col_avg: List[str] = None,
                                            col_down: List[str] = None, is_parallel: bool = False,
                                            result_names: Union[List[str], str] = None) -> pd.DataFrame:

        per_diff = df[col_up].values - df[col_down].values

        per_rng = numpy_div(is_parallel, per_diff, df[col_avg].values)
        per_pr_down = numpy_div(is_parallel, original_data.values - df[col_down].values, per_diff)
        per_pr_avg = numpy_div(is_parallel, original_data.values - df[col_avg].values, per_diff)
        per_pr_up = numpy_div(is_parallel, df[col_up].values - original_data.values, per_diff)

        per_rng = pd.DataFrame(per_rng, columns=[x + "_RNG" for x in result_names], index=original_data.index)
        per_pr_down = pd.DataFrame(per_pr_down, columns=[x + "_DOWN" for x in result_names], index=original_data.index)
        per_pr_avg = pd.DataFrame(per_pr_avg, columns=[x + "_AVG" for x in result_names], index=original_data.index)
        per_pr_up = pd.DataFrame(per_pr_up, columns=[x + "_UP" for x in result_names], index=original_data.index)

        return pd.concat([per_rng, per_pr_down, per_pr_avg, per_pr_up], axis=1)

    def hull_moving_average(self, columns: Union[List[str], str], window: int = 1, result_names: List[str] = None,
                            is_parallel: bool = False, add_to_data: bool = True,
                            ml_format: Union[Callable, bool] = None, *args, **kwargs) -> pd.DataFrame:
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

        if ml_format is not None:
            if callable(ml_format):
                df_nsqrt[result_names] = ml_format(df_nsqrt[result_names], *args, **kwargs)

            elif isinstance(ml_format, bool):
                if ml_format:
                    d1 = numpy_div(is_parallel, df_nsqrt[result_names].values, self.data[columns].values)
                    d1 = numpy_log(is_parallel, d1)

                    df_nsqrt[result_names] = pd.DataFrame(d1, columns=result_names, index=df_nsqrt.index)

            else:
                raise Exception("ml_format parameter not recognized as a callable object neither as boolean")

        if add_to_data:
            self.data[df_nsqrt.columns] = df_nsqrt
            return self.data
        else:
            return df_nsqrt
    # endregion

    # region utils
    def set_index(self, index_column_name: str, new_index_name: str = None, ascending_order: bool = True,
                  is_timestamp: bool = False, unit: str = 's', drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Take a DataFrame as entry and set the column timestamp_column_name as index.

        It also removes duplicate lines.

        :param pandas.DataFrame data: The data in form of pandas.DataFrame.
        :param str index_column_name: Name of the index column.
        :param str new_index_name: The new name of the index. Used to raplace index_column_name.
        :param bool ascending_order: Sort the DataFrame in ascending order.
        :param bool is_timestamp: Is the index column in form of timestamp.
        :param str unit: Used only when is_timestamp is true.
        :param bool drop_duplicates: Whether to drop duplicates or not.
        """

        if self.data.index.name == index_column_name:
            self.data.reset_index(inplace=True)

        index_column_name = 'index' if index_column_name is None else index_column_name

        if index_column_name not in set(self.data.columns):
            raise Exception(f"Column {index_column_name} is not in the DataFrame.")

        column_name = index_column_name
        if new_index_name is not None:
            self.data.rename(columns={index_column_name: new_index_name}, inplace=True)
            column_name = new_index_name

        if drop_duplicates:
            self.data.drop_duplicates(subset=column_name, keep="first", inplace=True)

        if is_timestamp:
            self.data[column_name] = pd.to_datetime(self.data[column_name], infer_datetime_format=True, unit=unit)

        if self.data.index.name != column_name:
            self.data.set_index(column_name, inplace=True)

        self.data.sort_index(ascending=ascending_order, inplace=True)

        return self.data
    # endregion


