"""
This package includes the main high level class to use.
"""

from learning.transforms import Normalization
from graphs import display_graph
from joblib import dump, load
import numpy as np
import pandas as pd
from typing import Any, Callable, Tuple, Union, List
import os
import sys
from strategies.indicators import Indicators


class MLArchitect:
    def __init__(self, x_data: Any, y_data: Any = None, save_x_path: str = None, is_x_flat: bool = False,
                 save_indicators_path: str = None, save_y_path: str = None,
                 index_col: str = None, index_infer_datetime_format: bool = True, index_unit_datetime: str = None,
                 indicators_callable: Any = "default_gen_indicators", y_restoration_routine: Any = "default",
                 normalize_x_callable: Any = "default_normalizer", save_normalize_x_model: str = None,
                 normalize_y_callable: Any = "default_normalizer", save_normalize_y_model: str = None,
                 window_prediction: int = 4, test_size: float = 0.1, random_state: int = 0,
                 ml_model: Any = None, save_ml_path: str = None):

        # Variable initializing
        self._window_prediction = window_prediction

        # Load x
        self._x = self._load_data(x_data, index_col, index_infer_datetime_format, index_unit_datetime)
        self._original_prices = None
        if set(["open", "close", "high", "low"]) in set(self._x.columns):
            self._original_prices = self._x["open", "close", "high", "low"]

        # Get default indicators or user defined indicators with indicators_callable
        self._data_with_indicators = pd.DataFrame()
        self._get_indicators(indicators_callable, is_x_flat)
        if not self._data_with_indicators.empty and self._original_prices.empty \
                and set(["open", "close", "high", "low"]) in set(self._data_with_indicators.columns):
            self._original_prices = self._data_with_indicators["open", "close", "high", "low"]

        # Get y
        self._y = pd.DataFrame()
        self._get_y(y_data, index_col, index_infer_datetime_format, index_unit_datetime)

        # Clean data
        self.clean_data(self._x, ["open", "close", "high", "low"], dropna=True, inplace=True)
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

    def ml_fit(self, x: object = None, y: object = None):
        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if not callable(getattr(self._ml_model, "fit", None)):
            raise Exception("No fit method found for this machine learning model.")

        x_train = x
        if x is None or len(x) == 0:
            x_train = self.x_train
            x_train = self.norm_input_transform(x_train)

        y_train = y
        if y is None or len(y) == 0:
            y_train = self.y_train
            y_train = self.norm_output_transform(y_train)

        self._ml_model = self._ml_model.fit(x_train, y_train)

    def ml_predict(self, x: object = None, columns_for_dataframe: List = [], index_for_dataframe: object = None,
                   sort_index: bool = False, **kwargs_restoration):
        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if not callable(getattr(self._ml_model, "predict", None)):
            raise Exception("No fit method found for this machine learning model.")

        x_to_predict = x
        columns = columns_for_dataframe
        index = index_for_dataframe
        if x is None or len(x) == 0:
            x_to_predict = self.norm_input_transform(self._x)
            columns = self._y.columns if not columns else columns
            index = self._x.index if not index else index

        y_predicted = self._ml_model.predict(x_to_predict, columns, index, sort_index)
        y_predicted = self.norm_output_inverse_transform(y_predicted)

        if callable(self._y_restoration_routine):
            return self._y_restoration_routine(y_predicted, **kwargs_restoration)
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

    def ml_score_train(self):
        x_train = self.x_train
        x_train = self.norm_input_transform(x_train)

        y_train = self.y_train
        y_train = self.norm_output_transform(y_train)

        return self.ml_score(x_train, y_train)

    def ml_score_test(self):
        x_test = self._x[self._x_test] if self._only_indices else self._x_test
        x_test = self.norm_input_transform(x_test)

        y_test = self._y[self._y_test] if self._only_indices else self._y_test
        y_test = self.norm_output_transform(y_test)

        return self.ml_score(x_test, y_test)
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
    def _get_indicators(self, indicators_callable: Any = "default_gen_indicators", is_x_flat: bool = False):
        if indicators_callable is not None:
            if isinstance(indicators_callable, str) and indicators_callable == "default_gen_indicators":
                if is_x_flat:
                    _, self._data_with_indicators = self.default_compute_process_indicators(self._x, inplace=True,
                                                                                    return_data_with_indicators=True)

            elif callable(indicators_callable):
                self._x = indicators_callable(self._x)

            else:
                raise Exception(f"'indicators_callable' not recognized as a callable object.")

    @staticmethod
    def default_compute_process_indicators(original_data: pd.DataFrame, ascending_order: bool = True,
                                           drop_duplicates: bool = True, process_indicators: bool = True,
                                           return_data_with_indicators: bool = False, inplace: bool = True):
        """
        Function to generate defaults indicators and preprocess them.

        It is a capsule calling compute_indicators and preprocess_indicators.

        :param pd.DataFrame original_data: Original x data.
        :param bool ascending_order: Order data in ascending order.
        :param bool drop_duplicates: Drop duplicated lines with the same index.
        :param bool process_indicators: If yes, indicators are processed to respect the correct format.
        :param bool return_data_with_indicators: If yes, returns data with flat indicators.
        :param bool inplace: If True, do operation inplace and return None.
        :return: None if inplace=True else data
        """

        data = original_data if inplace else original_data.copy()
        indicator_object = Indicators(data)

        # Sorting and Dropping duplicates from index
        indicator_object.set_index(index_column_name=data.index.name, ascending_order=ascending_order,
                                   drop_duplicates=drop_duplicates)

        # Compute indicators
        MLArchitect._compute_indicators(data, inplace=True)

        # Process indicators
        if process_indicators:
            # All prices with indicators before preprocessing them
            data_with_indicators = data.copy() if return_data_with_indicators else None

            MLArchitect._process_indicators(data, inplace=True)
        else:
            data_with_indicators = data if return_data_with_indicators else None

        # Example of how to use Pandas builtin methods
        MLArchitect._use_pandas_builtin_functions(data, inplace=True)

        if inplace:
            return None, data_with_indicators
        else:
            return data, data_with_indicators

    @staticmethod
    def _compute_indicators(original_data: pd.DataFrame, inplace: bool = True):
        data = original_data if inplace else original_data.copy()
        indicator_object = Indicators(data)

        # Simple and Exponential Moving average Channel
        src_columns = ["open", "close", "high", "low"]
        target_columns_names = ["machannel_14_3_" + x for x in src_columns]
        indicator_object.moving_average_channel(columns=src_columns, window=14, nb_of_deviations=3, add_to_data=True,
                                                result_names=target_columns_names)
        indicator_object.exponential_weighted_moving_average_channel(columns=src_columns, nb_of_deviations=3, span=14,
                                                                     result_names=["ex_" + x for x in
                                                                                   target_columns_names])

        # Hull moving average
        indicator_object.hull_moving_average(columns=src_columns, result_names=["hull_14_" + x for x in src_columns],
                                             window=14)

        # True Range and its average
        indicator_object.true_range(window=1, result_names="tr")
        indicator_object.average_true_range(result_names="tr_avg_14", window=14)

        # Average Directional Index
        indicator_object.average_directional_index(result_names="adx_14", window=14)

        # Simple and Exponential moving average, std and var
        indicator_object.moving_average(columns=src_columns, window=14,
                                        result_names=[x + "_avg_14" for x in src_columns])
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

        if inplace:
            return None,
        else:
            return indicator_object.data

    @staticmethod
    def _process_indicators(original_data: pd.DataFrame, inplace: bool = True):
        """
        This function shows how to use the preprocessing object to do some computation on indicators.

        We need to preprocess indicators so they can have a meaning when predicting future.

        :param original_data: original data with generated indicators.
        :param bool inplace: If True, do operation inplace and return None.
        :return: None if inplace=True else data
        """

        data = original_data if inplace else original_data.copy()
        indicator_object = Indicators(data)

        preprocessing_object = indicator_object.preprocessing
        drop_columns = []

        # Compute log returns and normal returns for i+1 to i+14
        src_columns = ["open", "close", "high", "low"]
        for i in range(14):
            preprocessing_object.log_returns(src_columns, [x + "_log_returns_" + str(i + 1) for x in src_columns],
                                             window=i + 1)
            preprocessing_object.pct_change(src_columns, [x + "_pct_changes_" + str(i + 1) for x in src_columns],
                                            window=i + 1)

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

        # Compute log division for all columns for which it is meaningful to know how much they deviate from
        # source prices.
        right_columns, src_columns = [], []
        for x in ["open", "close", "high", "low"]:
            src_columns.extend(
                [prefix + "machannel_14_3_" + x + suffix for suffix in {"_AVG", "_UP", "_DOWN"} for prefix in
                 {"", "ex_"}])
            right_columns.extend([x] * 6)

        drop_columns.extend(src_columns)
        target_columns_names = [x + "_log_div" for x in src_columns]
        df = pd.DataFrame(np.log(preprocessing_object.data[src_columns].values
                                 / preprocessing_object.data[right_columns].values), columns=target_columns_names)
        df.index = preprocessing_object.data.index
        preprocessing_object.data[target_columns_names] = df

        if drop_columns:
            preprocessing_object.data.drop(columns=drop_columns, inplace=True)

        if inplace:
            return None,
        else:
            return preprocessing_object.data

    @staticmethod
    def _use_pandas_builtin_functions(original_data: pd.DataFrame, inplace: bool = True):
        """
        This example shows how to use Pandas built-in functions to Process indicators.

        :param original_data: original data with generated indicators.
        :param bool inplace: If True, do operation inplace and return None.
        :return: None if inplace=True else data
        """

        # Get data object
        data = original_data if inplace else original_data.copy()

        # Columns for which returns computation have sense (X_(i+1) - X_(i)) / X_(i)
        src_columns = ["open", "close", "high", "low"]
        returns_columns = [x + "_log_returns_" + str(i + 1) for x in src_columns for i in range(14)]
        returns_columns.extend([x + "_pct_changes_" + str(i + 1) for x in src_columns for i in range(14)])
        returns_columns.extend([x + "_avg_14" for x in returns_columns])

        target_columns_names = [x + "_square" for x in returns_columns]

        # Pandas apply method
        data[target_columns_names] = data[returns_columns].apply(lambda x: np.square(x))

        # Raise Velocity = (high-low)/amount
        data["high_low_raise_velocity"] = data[["high", "low", "amount"]].apply(
            lambda x: (x["high"] - x["low"]) / x["amount"], axis=1)
        data["tr_raise_velocity"] = data[["tr", "amount"]].apply(lambda x: x["tr"] / x["amount"], axis=1)
        data["tr_avg_14_raise_velocity"] = data[["tr_avg_14", "amount"]].apply(lambda x: x["tr_avg_14"] / x["amount"],
                                                                               axis=1)
        if inplace:
            return None,
        else:
            return data
    # endregion

    # region normalization
    def split_train_test(self, test_size: int = 0.1, random_state: int = 0, only_indices: bool = False):
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
                                                                  add_power_transform=True, pca_n_components=0.95,
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

    @staticmethod
    def _default_normalizer(data_to_fit: pd.DataFrame, min_max_range: Union[None, tuple] = None,
                            add_standard_scaling: bool = False, add_power_transform: bool = False,
                            pca_n_components: float = -1, save_normalize_x_model: str = None):

        norm_object = Normalization()
        if add_standard_scaling:
            norm_object.add_standard_scaling(data_to_fit)

        if min_max_range:
            norm_object.add_min_max_scaling(data_to_fit, min_max_range=min_max_range)

        if add_power_transform:
            norm_object.add_power_transform_scaling(data_to_fit)

        if min_max_range:
            norm_object.add_min_max_scaling(data_to_fit, min_max_range=min_max_range)

        # Apply PCA reduction to data_to_fit
        if pca_n_components > 0:
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
    def default_restore_y(flat_data: pd.DataFrame, y_predicted: pd.DataFrame, target_columns_names: List = [],
                          window: int = 1, is_returns: bool = True):
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
        delta = sorted_index.index[1] - sorted_index.index[0]

        df = pd.DataFrame([[]], index=pd.date_range(start=y_predicted.index.max()+delta,
                                                    end=y_predicted.index.max()+(window*delta),
                                                    periods=window))
        predicted = pd.concat([y_predicted, df], axis=0)

        if is_returns:
            df = pd.DataFrame(flat_data.loc[predicted.index] * predicted.apply(lambda x: np.exp(x)).values)
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

