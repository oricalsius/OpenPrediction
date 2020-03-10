"""
This package includes the main high level class to use.
"""

from learning.transforms import Normalization
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from joblib import dump, load
from strategies.indicators import Indicators
from typing import Any, Callable, Tuple, Union, List, Dict, Optional, Generator, Iterable
from strategies.indicators.optim import *
import numpy as np
import pandas as pd
import os
import sys


class MLArchitect:
    def __init__(self, x_data: Any, y_data: Any = None, save_x_path: str = None, is_x_flat: bool = False,
                 save_indicators_path: str = None, save_y_path: str = None,
                 index_col: str = None, index_infer_datetime_format: bool = True, index_unit_datetime: str = None,
                 learning_indicators_callable: Any = "default_gen_learning_indicators",
                 display_indicators_callable: Any = "default_gen_display_indicators",
                 y_restoration_routine: Any = "default",
                 normalize_x_callable: Any = "default_normalizer", save_normalize_x_model: str = None,
                 normalize_y_callable: Any = "default_normalizer", save_normalize_y_model: str = None,
                 window_prediction: int = 4, test_size: float = 0.1, random_state: int = 0, add_pca: bool = False,
                 is_kernel_pca: bool = False, pca_n_components: Union[float, None] = 0.95, ml_model: Any = None,
                 save_ml_path: str = None, is_parallel: bool = False):

        # Variable initializing
        self._is_parallel_computing = is_parallel
        default_src_columns = ["open", "close", "high", "low"]
        columns_to_clean = ["open", "close", "high", "low", "count", "vol", "amount"]
        self._window_prediction = window_prediction

        # Load x
        self._x = self._load_data(x_data, index_col, index_infer_datetime_format, index_unit_datetime)

        # Get default indicators or user defined ones with learning_indicators_callable
        self._data_with_indicators = pd.DataFrame()
        self._get_display_indicators(display_indicators_callable, is_x_flat)

        # Get default learning indicators or user defined ones with learning_indicators_callable
        self._get_learning_indicators(learning_indicators_callable, is_x_flat)

        # Get y
        self._y = pd.DataFrame()
        self._get_y(y_data, index_col, index_infer_datetime_format, index_unit_datetime)

        # Clean data
        self.clean_data(self._x, columns_to_clean, dropna=True, inplace=True)
        self.clean_data(self._data_with_indicators, [], dropna=True, inplace=True)
        self.clean_data(self._y, [], dropna=True, inplace=True)

        # Save data if requested
        self._save_data(self._x, save_x_path)
        self._save_data(self._data_with_indicators, save_indicators_path)
        self._save_data(self._y, save_y_path)

        self._index_min, self._index_max = self.get_x_y_index_intersection(self._x, self._y)

        # y restoration routine
        self._y_restoration_routine = None
        self.set_y_restoration_routine(y_restoration_routine)

        # Split test and train data
        self._x_train, self._x_test, self._y_train, self._y_test = None, None, None, None
        self._only_indices = True
        self.split_train_test(test_size, random_state, only_indices=True)

        # Normalize train and test x
        self._x_norm_model = None
        self._add_pca = add_pca
        self._is_kernel_pca = is_kernel_pca
        self._pca_n_components = pca_n_components
        self.norm_input_fit(normalize_x_callable, save_normalize_x_model)

        # Normalize train and test y
        self._y_norm_model = None
        self.norm_output_fit(normalize_y_callable, save_normalize_y_model)

        # ML model
        self._ml_model = None
        self._save_ml_path = save_ml_path
        self.ml_init_model(ml_model)

    # region machine learning
    def ml_init_model(self, ml_model: Any = None):
        if ml_model is not None:
            if isinstance(ml_model, str):
                self._ml_load_model(ml_model)

            elif callable(ml_model):
                self._ml_model = ml_model()

            else:
                self._ml_model = ml_model

    def gridsearchcv(self, param_grid: Union[Dict, List[Dict]], x: object = None, y: object = None,
                     scoring: Union[str, Callable, List, Tuple, Dict, None] = None, n_jobs: Optional[int] = None,
                     cv: Union[int, Generator, Iterable, None] = None, verbose: int = 0, **kwargs):
        """
        Exhaustive search over specified parameter values for an estimator.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        """

        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        GridS = GridSearchCV(self.ml_model, param_grid=param_grid, scoring=scoring, n_jobs=n_jobs, cv=cv,
                             verbose=verbose, **kwargs)

        x_train, y_train = self.get_normalized_data(x, y, is_train=True)
        GridS.fit(x_train, y_train)

        return GridS

    def cross_validate(self, x, y=None, scoring: Union[str, Callable, List, Tuple, Dict, None] = None,
                       cv: Union[int, Generator, Iterable, None] = None, n_jobs: Optional[int] = None,
                       verbose: int = 0, **kwargs):
        """
        Evaluate metric(s) by cross-validation and also record fit/score times.

        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
        """

        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        x_train, y_train = self.get_normalized_data(x, y, is_train=True)

        cvl = cross_validate(self.ml_model, X=x_train, y=y_train, cv=cv, scoring=scoring, n_jobs=n_jobs,
                             verbose=verbose, **kwargs)
        return cvl

    def fit(self, x: object = None, y: object = None):
        self.ml_fit(x, y)

    def ml_fit(self, x: object = None, y: object = None):
        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if not callable(getattr(self._ml_model, "fit", None)):
            raise Exception("No fit method found for this machine learning model.")

        x_train, y_train = self.get_normalized_data(x, y, is_train=True)
        self._ml_model.fit(x_train, y_train)

        # Save models
        if self.ml_path is not None and self.ml_path != '':
            self.ml_save_model(model_path=self.ml_path)

    def ml_predict(self, x: object = None, columns_for_dataframe: List = None, index_for_dataframe: object = None,
                   sort_index: bool = True, restore_y: bool = False, **kwargs_restoration):
        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if not callable(getattr(self._ml_model, "predict", None)):
            raise Exception("No fit method found for this machine learning model.")

        x_to_predict = x
        columns = columns_for_dataframe
        index = index_for_dataframe
        if x is None or len(x) == 0:
            x_to_predict = self.norm_input_transform(self._x)

        columns = list(self._y.columns) if not columns else columns
        index = list(x_to_predict.index) if not index else index

        y_predicted = self._ml_model.predict(x_to_predict)
        y_predicted = self.norm_output_inverse_transform(y_predicted)

        y_predicted = pd.DataFrame(y_predicted, columns=columns, index=index)
        if sort_index: y_predicted.sort_index()

        if restore_y:
            if callable(self._y_restoration_routine):
                return self._y_restoration_routine(y_predicted, **kwargs_restoration)
            else:
                raise Exception(f"y_restoration_routine: {self._y_restoration_routine} is not recognized as a "
                                f"callable object.")
        else:
            return y_predicted

    def ml_score(self, x: object = None, y: object = None):
        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if not callable(getattr(self._ml_model, "score", None)):
            raise Exception("No score method found for this machine learning model.")

        x_train = x
        if x is None or len(x) == 0:
            x_train = self.x_train
            x_train = self.norm_input_transform(x_train)

        y_train = y
        if y is None or len(y) == 0:
            y_train = self.y_train
            y_train = self.norm_output_transform(y_train)

        return self._ml_model.score(x_train, y_train)

    def ml_r2_score(self, x: object = None, y: object = None):

        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        x_train = x
        if x is None or len(x) == 0:
            x_train = self.x_train
            x_train = self.norm_input_transform(x_train)

        y_train = y
        if y is None or len(y) == 0:
            y_train = self.y_train

        y_predicted = self.ml_predict(x_train, restore_y=False)
        return r2_score(y_train, y_predicted)

    def ml_score_train(self):
        x_train = self.norm_input_transform(self.x_train)
        y_train = self.norm_output_transform(self.y_train)

        return self.ml_score(x_train, y_train)

    def ml_score_test(self):
        x_test = self.norm_input_transform(self.x_test)
        y_test = self.norm_output_transform(self.y_test)

        return self.ml_score(x_test, y_test)

    def ml_r2_score_train(self):
        x_train = self.norm_input_transform(self.x_train)
        y_train = self.norm_output_transform(self.y_train)

        return self.ml_r2_score(x_train, y_train)

    def ml_r2_score_test(self):
        x_test = self.norm_input_transform(self.x_test)
        y_test = self.norm_output_transform(self.y_test)

        return self.ml_r2_score(x_test, y_test)

    # endregion

    # region properties
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data_with_indicators(self):
        return self._data_with_indicators

    @property
    def x_train(self):
        return self._x.loc[self._x_train] if self._only_indices else self._x_train

    @property
    def x_test(self):
        return self._x.loc[self._x_test] if self._only_indices else self._x_test

    @property
    def y_train(self):
        return self._y.loc[self._y_train] if self._only_indices else self._y_train

    @property
    def y_test(self):
        return self._y.loc[self._y_test] if self._only_indices else self._y_test

    @property
    def x_norm_model(self):
        return self._x_norm_model

    @property
    def y_norm_model(self):
        return self._y_norm_model

    @property
    def ml_model(self):
        return self._ml_model

    @property
    def ml_path(self):
        return self._save_ml_path
    # endregion

    # region indicators
    def _get_learning_indicators(self, indicators_callable: Any = "default_gen_learning_indicators",
                                 is_x_flat: bool = False):
        if indicators_callable is not None:
            if isinstance(indicators_callable, str) and indicators_callable == "default_gen_learning_indicators":
                if is_x_flat:
                    _ = self.default_compute_learning_indicators(self._x, inplace=True)

            elif callable(indicators_callable):
                self._x = indicators_callable(self._x)

            else:
                raise Exception("'indicators_callable' not recognized as a callable object.")

    @staticmethod
    def default_compute_learning_indicators(original_data: pd.DataFrame, ascending_order: bool = True,
                                            drop_duplicates: bool = True, inplace: bool = True):
        """
        Function to generate defaults indicators and preprocess them.

        :param pd.DataFrame original_data: Original x data.
        :param bool ascending_order: Order data in ascending order.
        :param bool drop_duplicates: Drop duplicated lines with the same index.
        :param bool inplace: If True, do operation inplace and return None.
        :return: None if inplace=True else data
        """

        data = original_data if inplace else original_data.copy()
        ind_obj = Indicators(data)
        proc_object = ind_obj.preprocessing

        src_columns = ["open", "close", "high", "low"]

        # Sorting and Dropping duplicates from index
        ind_obj.set_index(index_column_name=data.index.name, ascending_order=ascending_order,
                          drop_duplicates=drop_duplicates)

        # Simple moving average, std
        ind_obj.moving_std(columns=src_columns, window=14, result_names=[x + "_std_14" for x in src_columns],
                           ml_format=True)

        # Quantile 0.05 and 0.95
        ind_obj.moving_quantile(columns=src_columns, quantile=0.05, window=14, ml_format=True,
                                result_names=[x + "_q_05" for x in src_columns])

        ind_obj.moving_quantile(columns=src_columns, quantile=0.95, window=14, ml_format=True,
                                result_names=[x + "_q_95" for x in src_columns])

        # Median
        ind_obj.moving_median(columns=src_columns, window=14, result_names=[x + "_median_14" for x in src_columns],
                              ml_format=True)

        # Sum
        ind_obj.moving_sum(columns=src_columns, window=5, result_names=[x + "_sum_5" for x in src_columns],
                           ml_format=True)
        ind_obj.moving_sum(columns=src_columns, window=10, result_names=[x + "_sum_10" for x in src_columns])
        ind_obj.moving_sum(columns=src_columns, window=10, result_names=[x + "_sum_15" for x in src_columns],
                           ml_format=True)

        # Compute log divisions of Sum 10
        columns = [x + "_sum_10" for x in src_columns]
        proc_object.log_returns(columns, [x + "_log_div" for x in columns], window=1, delete_columns=True)

        # Min, Max
        ind_obj.moving_min(columns=src_columns, window=15, result_names=[x + "_min_15" for x in src_columns],
                           ml_format=True)
        ind_obj.moving_max(columns=src_columns, window=15, result_names=[x + "_max_15" for x in src_columns],
                           ml_format=True)

        # Skew, Kurt
        ind_obj.moving_skew(columns=src_columns, window=14, result_names=[x + "_skew_14" for x in src_columns])
        ind_obj.moving_kurt(columns=src_columns, window=14, result_names=[x + "_kurt_14" for x in src_columns])

        # True Range and its average
        ind_obj.true_range(window=1, result_names="tr", ml_format=True)
        ind_obj.average_true_range(result_names="tr_avg_14", window=14, ml_format=True)

        # Average Directional Index
        ind_obj.average_directional_index(result_name="adx_14", window=14)

        # Custom Returns Directional Index
        ind_obj.returns_dix(columns=src_columns, result_names=[x + "_rdix_5" for x in src_columns], window=5)
        ind_obj.returns_dix_average(columns=src_columns, result_names=[x + "_rdix_avg_14" for x in src_columns],
                                    window=14)

        ind_obj.returns_square_dix(columns=src_columns, result_names=[x + "_square_rdix_5" for x in src_columns],
                                   window=5)
        ind_obj.returns_square_dix_average(columns=src_columns, window=14,
                                           result_names=[x + "_square_rdix_avg_14" for x in src_columns])

        ind_obj.returns_norm_dix(columns=src_columns, window=5,
                                 result_names=[x + "_norm_rdix_5" for x in src_columns])

        ind_obj.returns_norm_dix_average(columns=src_columns, window=14,
                                         result_names=[x + "_norm_rdix_avg_14" for x in src_columns])

        ind_obj.price_velocity(columns=src_columns, window=5,
                               result_names=[x + "_PriceVelo_5" for x in src_columns], ml_format=True)
        ind_obj.price_velocity_average(columns=src_columns, window=14,
                                       result_names=[x + "_PriceVelo_avg_14" for x in src_columns], ml_format=True)

        ind_obj.returns_velocity(columns=src_columns, window=5,
                                 result_names=[x + "_ReturnsVelo_5" for x in src_columns])
        ind_obj.returns_velocity_average(columns=src_columns, window=14,
                                         result_names=[x + "_ReturnsVelo_avg_14" for x in src_columns])

        # RSI
        ind_obj.rsi(high='high', low='low', window=14, result_names=["RSI_14"])

        # Exponential Moving average
        ind_obj.exponential_weighted_moving_average(columns=src_columns, span=14, ml_format=True,
                                                    result_names=["ex_" + x + "_avg_14" for x in src_columns])

        ind_obj.exponential_weighted_moving_std(columns=src_columns, span=14,
                                                result_names=["ex_" + x + "_std_14" for x in src_columns],
                                                ml_format=True)

        # Simple and Exponential Moving average Channel
        target_columns_names = [x + "_machannel_14_3" for x in src_columns]
        ind_obj.exponential_weighted_moving_average_channel(columns=src_columns, nb_of_deviations=3, span=14,
                                                            result_names=["ex_" + x for x in target_columns_names],
                                                            ml_format=True)

        # Hull moving average
        ind_obj.hull_moving_average(columns=src_columns, result_names=[x + "_hull_14" for x in src_columns], window=14,
                                    ml_format=True)

        # Compute log returns and normal returns for i+1 to i+14
        for i in range(5):
            proc_object.log_returns(src_columns, [x + "_log_returns_" + str(i + 1) for x in src_columns], window=i + 1)

        # Compute the Simple and Exponential average of returns
        returns_columns = [x + "_log_returns_" + str(i + 1) for x in src_columns for i in range(5)]
        target_columns_names = ["ex_" + x + "_avg_14" for x in returns_columns]
        ind_obj.exponential_weighted_moving_average(returns_columns, span=14, result_names=target_columns_names)

        # Compute the Simple and Exponential std of returns
        target_columns_names = ["ex_" + x + "_std_14" for x in returns_columns]
        ind_obj.exponential_weighted_moving_std(returns_columns, span=14, result_names=target_columns_names)

        # Columns for which we want to compute the square
        returns_columns.extend(["ex_" + x + "_avg_14" for x in returns_columns])
        target_columns_names = [x + "_square" for x in returns_columns]

        # Compute square values
        data[target_columns_names] = pd.DataFrame(np.square(data[returns_columns].values),
                                                  columns=target_columns_names, index=data.index)

        # Raise Velocity = (high-low)/amount
        data["high_low_raise_velocity"] = pd.DataFrame((data["high"].values - data["low"].values)
                                                       / data["amount"].values,
                                                       columns=["high_low_raise_velocity"], index=data.index)

        if inplace:
            return None
        else:
            return ind_obj.data

    def _get_display_indicators(self, indicators_callable: Any = "default_gen_display_indicators",
                                is_x_flat: bool = False):
        if indicators_callable is not None:
            if isinstance(indicators_callable, str) and indicators_callable == "default_gen_display_indicators":
                if is_x_flat:
                    self._data_with_indicators = self.compute_indicators(self._x, inplace=False)

            elif callable(indicators_callable):
                self._data_with_indicators = indicators_callable(self._x)

            else:
                raise Exception("'indicators_callable' not recognized as a callable object.")

    @staticmethod
    def compute_indicators(original_data: pd.DataFrame, ascending_order: bool = True,
                           drop_duplicates: bool = True, inplace: bool = True):
        data = original_data if inplace else original_data.copy()
        ind_obj = Indicators(data)
        proc_object = ind_obj.preprocessing

        src_columns = ["open", "close", "high", "low"]

        # Sorting and Dropping duplicates from index
        ind_obj.set_index(index_column_name=data.index.name, ascending_order=ascending_order,
                          drop_duplicates=drop_duplicates)

        # Quantile 0.05 and 0.95
        ind_obj.moving_quantile(columns=src_columns, quantile=0.01, window=14,
                                result_names=[x + "_q_1" for x in src_columns])

        ind_obj.moving_quantile(columns=src_columns, quantile=0.99, window=14,
                                result_names=[x + "_q_99" for x in src_columns])

        # Median
        ind_obj.moving_median(columns=src_columns, window=14, result_names=[x + "_median_14" for x in src_columns],
                              ml_format=False)

        # Sum
        ind_obj.moving_sum(columns=src_columns, window=5, result_names=[x + "_sum_5" for x in src_columns],
                           ml_format=True)

        # Skew, Kurt
        ind_obj.moving_skew(columns=src_columns, window=14, result_names=[x + "_skew_14" for x in src_columns])
        ind_obj.moving_kurt(columns=src_columns, window=14, result_names=[x + "_kurt_14" for x in src_columns])

        # True Range and its average
        ind_obj.true_range(window=1, result_names="tr")
        ind_obj.average_true_range(result_names="tr_avg_14", window=14)

        # Average Directional Index
        ind_obj.average_directional_index(result_name="adx_14", window=14)

        # Exponential Moving average
        ind_obj.exponential_weighted_moving_average(columns=src_columns, span=14,
                                                    result_names=["ex_" + x + "_avg_14" for x in src_columns])

        # Simple and Exponential Moving average Channel
        target_columns_names = [x + "_machannel_14_3" for x in src_columns]
        ind_obj.exponential_weighted_moving_average_channel(columns=src_columns, nb_of_deviations=3, span=14,
                                                            result_names=["ex_" + x for x in target_columns_names],
                                                            ml_format=False)

        # Hull moving average
        ind_obj.hull_moving_average(columns=src_columns, result_names=[x + "_hull_14" for x in src_columns], window=14,
                                    ml_format=False)

        # Custom Returns Directional Index
        ind_obj.returns_dix(columns=src_columns, result_names=[x + "_rdix_5" for x in src_columns], window=5)
        ind_obj.returns_dix_average(columns=src_columns, result_names=[x + "_rdix_avg_14" for x in src_columns],
                                    window=14)

        ind_obj.returns_square_dix(columns=src_columns, result_names=[x + "_square_rdix_5" for x in src_columns],
                                   window=5)
        ind_obj.returns_square_dix_average(columns=src_columns, window=14,
                                           result_names=[x + "_square_rdix_avg_14" for x in src_columns])

        ind_obj.returns_norm_dix(columns=src_columns, window=5,
                                 result_names=[x + "_norm_rdix_5" for x in src_columns])

        ind_obj.returns_norm_dix_average(columns=src_columns, window=14,
                                         result_names=[x + "_norm_rdix_avg_14" for x in src_columns])

        ind_obj.price_velocity(columns=src_columns, window=5, ml_format=True,
                               result_names=[x + "_PriceVelo_5" for x in src_columns])
        ind_obj.price_velocity_average(columns=src_columns, window=14,
                                       result_names=[x + "_PriceVelo_avg_14" for x in src_columns], ml_format=True)

        ind_obj.returns_velocity(columns=src_columns, window=5,
                                 result_names=[x + "_ReturnsVelo_5" for x in src_columns])
        ind_obj.returns_velocity_average(columns=src_columns, window=14,
                                         result_names=[x + "_ReturnsVelo_avg_14" for x in src_columns])

        # RSI
        ind_obj.rsi(high='high', low='low', window=14, result_names=["RSI_14"])

        # RELS
        ind_obj.relative_slope(high='high', low='low', close='close', typical_window=5, rels_window=5,
                               result_names=["RELS_5_5"])

        # Raise Velocity = (high-low)/amount
        data["high_low_raise_velocity"] = pd.DataFrame((data["high"].values - data["low"].values)
                                                       / data["amount"].values,
                                                       columns=["high_low_raise_velocity"], index=data.index)

        if inplace:
            return None
        else:
            return ind_obj.data

    # endregion

    # region normalization
    def split_train_test(self, test_size: float = 0.1, random_state: int = 0, only_indices: bool = False):
        self._only_indices = only_indices

        if only_indices:
            x = self._x.loc[self._index_min:self._index_max].index.to_numpy()
            y = self._y.loc[self._index_min:self._index_max].index.to_numpy()
        else:
            x = self._x.loc[self._index_min:self._index_max]
            y = self._y.loc[self._index_min:self._index_max]

        self._x_train, self._x_test, self._y_train, self._y_test = Normalization.train_test_split(x, y,
                                                                                              test_size=test_size,
                                                                                              random_state=random_state)

    def norm_input_fit(self, normalize_x_callable: Any = "default_normalizer",
                       save_normalize_x_model: str = None):
        if normalize_x_callable is not None:
            if isinstance(normalize_x_callable, str):
                if normalize_x_callable == "default_normalizer":
                    self._x_norm_model = self._default_normalizer(self.x_train, add_standard_scaling=True,
                                                                  add_power_transform=True,
                                                                  min_max_range=(-1, 1), add_pca=self._add_pca,
                                                                  is_kernel_pca=self._is_kernel_pca,
                                                                  pca_n_components=self._pca_n_components,
                                                                  save_normalize_x_model=save_normalize_x_model)

                elif normalize_x_callable != '' and os.path.isfile(normalize_x_callable):
                    self.norm_load_model(is_input=True, model_to_load=normalize_x_callable)

            elif callable(normalize_x_callable):
                self._x_norm_model = normalize_x_callable(self.x_train)

                # Save models
                if save_normalize_x_model is not None and save_normalize_x_model != '':
                    self.norm_save_model(is_input=True, model_path=save_normalize_x_model)

            else:
                raise Exception(f"'normalize_x_callable' not recognized as a callable object neither as a path for"
                                f" a normalization model.")

    def norm_output_fit(self, normalize_y_callable: Any = "default_normalizer",
                        save_normalize_y_model: str = None):
        if normalize_y_callable is not None:
            if isinstance(normalize_y_callable, str):
                if normalize_y_callable == "default_normalizer":
                    self._y_norm_model = self._default_normalizer(self.y_train, add_standard_scaling=True,
                                                                  min_max_range=(-1, 1),
                                                                  add_power_transform=True,
                                                                  save_normalize_x_model=save_normalize_y_model)

                elif normalize_y_callable != '' and os.path.isfile(normalize_y_callable):
                    self.norm_load_model(is_input=False, model_to_load=normalize_y_callable)

            elif callable(normalize_y_callable):
                self._y_norm_model = normalize_y_callable(self.y_train)

                # Save models
                if save_normalize_y_model is not None and save_normalize_y_model != '':
                    self.norm_save_model(is_input=False, model_path=save_normalize_y_model)

            else:
                raise Exception(f"'normalize_y_callable' not recognized as a callable object neither as a path for"
                                f" a normalization model.")

    def norm_input_transform(self, data_to_transform: pd.DataFrame) -> object:
        if not self._x_norm_model:
            raise Exception("Normalization model for inputs not found. You should call norm_input_fit to define "
                            "a model and fit it.")

        if not callable(getattr(self._x_norm_model, "transform", None)):
            raise Exception("No transform method found for inputs normalization model.")

        return self._x_norm_model.transform(data_to_transform)

    def norm_output_transform(self, data_to_transform: pd.DataFrame) -> object:
        if not self._y_norm_model:
            raise Exception("Normalization model for outputs not found. You should call norm_output_fit "
                            "to define a model and fit it.")

        if not callable(getattr(self._y_norm_model, "transform", None)):
            raise Exception("No transform method found for outputs normalization model.")

        return self._y_norm_model.transform(data_to_transform)

    def norm_input_inverse_transform(self, data_to_inverse: pd.DataFrame) -> object:
        if not self._x_norm_model:
            raise Exception("Normalization model for inputs not found. You should call norm_input_fit to define "
                            "a model and fit it.")

        if not callable(getattr(self._x_norm_model, "inverse_transform", None)):
            raise Exception("No inverse_transform method found for inputs normalization model.")

        return self._x_norm_model.inverse_transform(data_to_inverse)

    def norm_output_inverse_transform(self, data_to_inverse: pd.DataFrame) -> object:
        if not self._y_norm_model:
            raise Exception("Normalization model for outputs not found. You should call norm_output_fit "
                            "to define a model and fit it.")

        if not callable(getattr(self._y_norm_model, "inverse_transform", None)):
            raise Exception("No inverse_transform method found for outputs normalization model.")

        return self._y_norm_model.inverse_transform(data_to_inverse)

    def get_normalized_data(self, x: object = None, y: object = None, is_train: bool = True) -> Tuple[object, object]:
        """
        Normalize x and y.

        If x is None, train input data are normalized when is_train=True, otherwise, test input data are normalized.
        If y is None, train output data are normalized when is_train=True, otherwise, test output data are normalized.
        """

        if x is None or len(x) == 0:
            x_norm = self.x_train if is_train else self.x_test
        else:
            x_norm = x

        if y is None or len(y) == 0:
            y_norm = self.y_train if is_train else self.y_test
        else:
            y_norm = y

        x_norm = self.norm_input_transform(x_norm)
        y_norm = self.norm_output_transform(y_norm)

        return x_norm, y_norm

    @staticmethod
    def _default_normalizer(data_to_fit: pd.DataFrame, min_max_range: Union[None, tuple] = None,
                            add_standard_scaling: bool = False, add_power_transform: bool = False,
                            is_kernel_pca: bool = False, add_pca: bool = False, pca_n_components: float = None,
                            save_normalize_x_model: str = None):

        norm_object = Normalization()
        if add_standard_scaling:
            norm_object.add_standard_scaling(data_to_fit)

        if min_max_range:
            norm_object.add_min_max_scaling(data_to_fit, min_max_range=min_max_range)

        if add_power_transform:
            norm_object.add_power_transform_scaling(data_to_fit)

        #if min_max_range:
            #norm_object.add_min_max_scaling(data_to_fit, min_max_range=min_max_range)

        # Apply PCA reduction to data_to_fit
        if add_pca:
            if is_kernel_pca:
                norm_object.add_kernel_pca_reduction(pca_n_components, kernel='linear', n_jobs=-1)
            else:
                norm_object.add_pca_reduction(pca_n_components, svd_solver="full")

        norm_object.fit(data_to_fit)

        # Save models
        if save_normalize_x_model is not None and save_normalize_x_model != '':
            norm_object.save_model(save_normalize_x_model)

        return norm_object
    # endregion

    # region y processing
    def _get_y(self, y_data: Any, index_col: str = None, index_infer_datetime_format: bool = True,
               index_unit_datetime: str = None):
        if y_data is not None:
            self._y = self._load_data(y_data, index_col, index_infer_datetime_format, index_unit_datetime)

        else:
            Y = pd.DataFrame()
            src_columns = ["open", "close", "high", "low"]

            # Compute y returns for window = self._window_prediction
            for i in range(self._window_prediction):
                returns = self.default_get_y_to_predict(self._x, ["open", "close", "high", "low"],
                                                        [x + "_log_returns_" + str(i+1) for x in src_columns],
                                                        window=i+1, is_returns=True)

                Y = pd.concat([Y, returns], axis=1)

            self._y = Y

    @staticmethod
    def default_get_y_to_predict(original_data: pd.DataFrame, src_columns: List = None,
                                 target_columns_names: List = None, window: int = 1, is_returns: bool = True):
        """
        Returns the expected columns for which we need to predict future values.

        :param pd.DataFrame original_data: Original x data.
        :param List src_columns: Source columns for which we need to predict future values.
        :param List target_columns_names: Target names.
        :param int window: The window time we want to predict. Window future values to predict.
        :param bool is_returns: Compute returns of src_columns.
        :return: data to predict
        """

        # Get data object
        data = original_data
        if not src_columns:
            src_columns = ["open", "close", "high", "low"]

        if not target_columns_names:
            target_columns_names = [x + "_log_returns_" + str(window) for x in src_columns]

        # Compute log returns and normal returns for i+1 to i+14
        if is_returns:
            indicator_object = Indicators(data)
            preprocessing_object = indicator_object.preprocessing
            Y = preprocessing_object.log_returns(src_columns, target_columns_names, window=window, add_to_data=False)
        else:
            Y = data[src_columns]

        Y.columns = target_columns_names
        return Y.shift(-window)

    def set_y_restoration_routine(self, y_restoration_routine: Any = "default"):
        if y_restoration_routine is not None:
            if isinstance(y_restoration_routine, str):
                if y_restoration_routine == "default":
                    self._y_restoration_routine = self.default_restore_y

                else:
                    raise Exception('y_restoration_routine parameter could only be default or a callable.')

            elif callable(y_restoration_routine):
                self._y_restoration_routine = y_restoration_routine

            else:
                raise Exception('y_restoration_routine parameter could only be default or a callable.')

    @staticmethod
    def default_restore_y(y_predicted: pd.DataFrame, original_data: pd.DataFrame, src_columns: List = None,
                          windows: int = 1, is_returns: bool = True) -> pd.DataFrame:
        """
        Restore y using the default algorithm.

        To use only if y returns used to train the model have been generated using the default algorithm
        default_get_y_to_predict. If it is not the case, you should define your proper routine to restore y and
        bind it to architect object using set_y_restoration_routine.

        :param y_predicted: Output from ML model.
        :param original_data: Original Dataframe in which we can find src_columns.
        :param src_columns: Source columns used to compute y in the correct order. Ex: ["open", "close", "high", "low"]
        :param windows: The considered prediction window.
        :param is_returns: Predicted y represents returns.
        :return:
        """

        if not src_columns:
            src_columns = ["open", "close", "high", "low"]

        # Get source data object
        src_data = original_data[src_columns]

        res = pd.DataFrame()
        for i in range(windows):
            df = MLArchitect._restore_unique_window(y_predicted.iloc[:, i:i+len(src_columns)],
                                                    src_data, window=i+1, is_returns=is_returns)
            res = pd.concat([res, df], axis=1)

        return res

    @staticmethod
    def _restore_unique_window(y_predicted: pd.DataFrame, flat_data: pd.DataFrame, target_columns_names: List = [],
                               window: int = 1, is_returns: bool = True, is_parallel: bool = False):
        """
        Restore flat values for y.

        :param flat_data: Flat previous prices.
        :param y_predicted: Future returns.
        :param target_columns_names: Target names.
        :param int window: The window time we have predicted.
        :param bool is_returns: Compute returns of src_columns.
        :return: Restored flat values for y.
        """

        if not target_columns_names:
            target_columns_names = [x + "_predicted_" + str(window) for x in list(flat_data.columns)]

        # Add nb window new lines to y_predicted
        sorted_index = sorted(y_predicted.index)
        delta = sorted_index[1] - sorted_index[0]

        df = pd.DataFrame([[]], index=pd.date_range(start=y_predicted.index.max()+delta,
                                                    end=y_predicted.index.max()+(window*delta),
                                                    periods=window))
        predicted = pd.concat([y_predicted, df], axis=0)

        if is_returns:
            exp_values = numpy_exp(is_parallel, predicted.values)
            nu_df = numpy_mul(is_parallel, flat_data.reindex(index=predicted.index).values, exp_values)
            df = pd.DataFrame(nu_df, columns=flat_data.columns, index=predicted.index)
        else:
            df = predicted

        df.columns = target_columns_names

        return df.shift(window)
    # endregion

    # region utils
    @staticmethod
    def get_x_y_index_intersection(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[int, int]:
        intersection = set(x.index) & set(y.index)

        return min(intersection), max(intersection)

    def _load_data(self, data_path_or_object: Any, index_col: str = None, index_infer_datetime_format: bool = True,
                   index_unit_datetime: str = None):
        if isinstance(data_path_or_object, str):
            if os.path.isfile(data_path_or_object):
                self._check_str("index_col", index_col)

                df = pd.read_csv(data_path_or_object)
                unit = None
                if index_unit_datetime is not None and index_unit_datetime.strip() != '':
                    unit = index_unit_datetime

                if index_infer_datetime_format or unit:
                    df[index_col] = pd.to_datetime(df[index_col], infer_datetime_format=index_infer_datetime_format,
                                                   unit=unit)

                df.set_index(index_col, inplace=True)
                return df

            else:
                raise Exception(f"{data_path_or_object} is not a file to load.")

        elif isinstance(data_path_or_object, pd.DataFrame):
            return data_path_or_object.copy()

        else:
            raise Exception(f"Only file path or objects of type pandas.DataFrame are accepted.")

    @staticmethod
    def _save_data(data: pd.DataFrame, path: str):
        if not isinstance(data, pd.DataFrame):
            raise Exception(f"{sys._getframe(1).f_code.co_name}: data should be of type pandas.DataFrame. "
                            f"Received {type(data)}")

        if path is not None:
            if not isinstance(path, str):
                raise Exception(f"{sys._getframe(1).f_code.co_name}: path variable should be a string.")

            if not data.empty:
                data.to_csv(path)

    @staticmethod
    def clean_data(original_data: pd.DataFrame, columns_to_remove: List = [], dropna: bool = True, inplace=True):
        data = original_data if inplace else original_data.copy()

        intersection = set(columns_to_remove) & set(data.columns)
        if intersection:
            data.drop(columns=list(intersection), inplace=True)

        if dropna:
            data.dropna(inplace=True)

        if inplace:
            return None
        else:
            return data

    def _check_str(self, field_name: str, value: str):
        if value is None:
            raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")

        if (value is not None) and (value.strip() == ""):
            raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")

    def norm_load_model(self, is_input: bool, model_to_load: str):
        self._check_str("model_to_load", model_to_load)

        if not os.path.isfile(model_to_load):
            raise Exception(f"No normalization model file found at location {model_to_load}.")

        if is_input:
            self._x_norm_model = load(model_to_load)
        else:
            self._y_norm_model = load(model_to_load)

    def norm_save_model(self, is_input: bool, model_path: str):
        self._check_str("model_path", model_path)
        if is_input:
            dump(self._x_norm_model, model_path)
        else:
            dump(self._y_norm_model, model_path)

    def ml_load_model(self, model_to_load: str):
        self._check_str("model_to_load", model_to_load)

        if not os.path.isfile(model_to_load):
            raise Exception(f"No model file found at location {model_to_load}.")

        self._ml_model = load(model_to_load)

    def ml_save_model(self, model_path: str):
        self._check_str("model_path", model_path)
        dump(self._ml_model, model_path)
        self._save_ml_path = model_path

    # endregion

