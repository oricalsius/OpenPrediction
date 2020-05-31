"""
This package includes the main high level class to use.
"""

from learning.transforms import Normalization
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as sklmod
from sklearn import neural_network
from joblib import dump, load
from strategies.indicators import Indicators
from typing import Any, Callable, Tuple, Union, List, Dict, Optional, Generator, Iterable
from strategies.indicators.optim import *
from keras.models import load_model, save_model, Sequential, Model
from itertools import zip_longest

import tensorflow as tf
from keras import backend as K
import keras
#from tensorflow import keras
import keras.layers as kl
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
import pandas as pd
import random
import os
import sys
import time

eps = np.finfo(float).eps


class MLArchitect:
    def __init__(self, x_data: Any, y_data: Any = None, save_x_path: str = None, is_x_flat: bool = True,
                 data_enhancement: Union[bool, Callable] = False, columns_equivalences: Dict[str, str] = None,
                 save_indicators_path: str = None, save_y_path: str = None,
                 index_infer_datetime_format: bool = True, index_unit_datetime: str = None,
                 learning_indicators_callable: Any = "default_gen_learning_indicators",
                 display_indicators_callable: Union[bool, Callable] = False,
                 y_restoration_routine: Any = "default",
                 normalize_x_callable: Any = "default_normalizer", save_normalize_x_model: str = None,
                 normalize_y_callable: Any = "default_normalizer", save_normalize_y_model: str = None,
                 columns_to_predict: List = None, window_prediction: int = 4, columns_indicators: List = None,
                 test_size: float = 0.1, random_state: int = 0,
                 pca_reductions: Union[List[Tuple], None] = [('linear', 0.95), ('kernel', None, 'linear')],
                 ml_model: Any = None, save_ml_path: str = None, is_parallel: bool = False, disgard_last: bool = True,
                 window_tuple: Tuple[int, int, int, int] = (6, 14, 22, 35)):

        # Variable initializing
        self._is_parallel_computing = is_parallel
        self._disgard_last = disgard_last
        self._window_tuple = window_tuple
        self._window_prediction = window_prediction

        # Columns initialization
        default_columns = ["index", "open", "close", "high", "low", "vol", "amount"]
        self._columns_equivalences = dict(zip(default_columns, default_columns))

        if columns_equivalences is not None:
            self._columns_equivalences.update(columns_equivalences)

        if not columns_indicators:
            columns_indicators = ['open', 'close', 'high', 'low']

        if not columns_to_predict:
            columns_to_predict = ['open', 'close', 'high', 'low']

        columns_to_clean = [self._columns_equivalences.get(x, x)
                            for x in ["open", "close", "high", "low", "count", "vol", "amount", "simulated"]]

        self._columns_to_predict = columns_to_predict
        self._columns_indicators = columns_indicators

        # Load x
        self._x = self._load_data(x_data, index_infer_datetime_format, index_unit_datetime)

        # Data correction
        self._do_data_enhancement(data_enhancement)

        # Get default indicators or user defined ones with learning_indicators_callable
        self._data_with_indicators = pd.DataFrame()
        self._get_display_indicators(display_indicators_callable, is_x_flat)

        # Get default learning indicators or user defined ones with learning_indicators_callable
        self._get_learning_indicators(learning_indicators_callable, is_x_flat)

        # Get y
        self._y = pd.DataFrame()
        self._get_y(y_data, index_infer_datetime_format, index_unit_datetime)

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
        self._pca_reductions = pca_reductions
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

            elif isinstance(ml_model, Sequential) or isinstance(ml_model, Model):
                self._ml_model = ml_model

            elif callable(ml_model):
                self._ml_model = ml_model()

            else:
                self._ml_model = ml_model

    def sklearn_gridsearchcv(self, param_grid: Union[Dict, List[Dict]], x=None, y=None,
                             scoring: Union[str, Callable, List, Tuple, Dict, None] = None,
                             n_jobs: Optional[int] = None, cv: Union[int, Generator, Iterable, None] = None,
                             verbose: int = 0, n: Union[float, pd.Index] = None, start_ind=None,
                             end_ind=None, **kwargs):
        """
        Exhaustive search over specified parameter values for an estimator.
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        """

        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        GridS = GridSearchCV(self.ml_model, param_grid=param_grid, scoring=scoring, n_jobs=n_jobs, cv=cv,
                             verbose=verbose, **kwargs)

        if x is None:
            x_train = self.x_train_normalized(n, start_ind, end_ind)
        else:
            x_train = self._get_sub_elt(x, n, start_ind, end_ind)

        if y is None:
            y_train = self.y_train_normalized(n, start_ind, end_ind)
        else:
            y_train = self._get_sub_elt(y, n, start_ind, end_ind)

        GridS.fit(x_train, y_train)

        return GridS

    def cross_validate(self, x, y=None, scoring: Union[str, Callable, List, Tuple, Dict, None] = None,
                       cv: Union[int, Generator, Iterable, None] = None, n_jobs: Optional[int] = None,
                       verbose: int = 0, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None, **kwargs):
        """
        Evaluate metric(s) by cross-validation and also record fit/score times.

        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
        """

        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if x is None:
            x_train = self.x_train_normalized(n, start_ind, end_ind)
        else:
            x_train = self._get_sub_elt(x, n, start_ind, end_ind)

        if y is None:
            y_train = self.y_train_normalized(n, start_ind, end_ind)
        else:
            y_train = self._get_sub_elt(y, n, start_ind, end_ind)

        cvl = sklmod.cross_validate(self.ml_model, X=x_train, y=y_train, cv=cv, scoring=scoring, n_jobs=n_jobs,
                                    verbose=verbose, **kwargs)
        return cvl

    def fit(self, x: object = None, y: object = None, start_ind=None, end_ind=None,
            prefit_simulation_data=None, prefit_sample_data=None, prefit_simulation_size=None,
            prefit_loops=1, **kwargs):

        return self.ml_fit(x, y, start_ind=start_ind, end_ind=end_ind, prefit_simulation_data=prefit_simulation_data,
                           prefit_sample_data=prefit_sample_data, prefit_simulation_size=prefit_simulation_size,
                           prefit_loops=prefit_loops, **kwargs)

    def ml_fit(self, x=None, y=None, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None,
               prefit_simulation_data=None, prefit_sample_data=None, prefit_simulation_size=None,
               prefit_loops=1, **kwargs):

        if not self._ml_model:
            raise Exception("Machine learning model not found. You should call ml_init_model.")

        if not callable(getattr(self._ml_model, "fit", None)):
            raise Exception("No fit method found for this machine learning model.")

        self._run_prefit_simulation(model=self._ml_model, prefit_simulation_data=prefit_simulation_data,
                                    prefit_sample_data=prefit_sample_data,
                                    prefit_simulation_size=prefit_simulation_size, prefit_loops=prefit_loops,
                                    kwargs_fit=kwargs)

        if x is None:
            x_train = self.x_train_normalized(n, start_ind, end_ind)
        else:
            x_train = self._get_sub_elt(x, n, start_ind, end_ind)

        if y is None:
            y_train = self.y_train_normalized(n, start_ind, end_ind)
        else:
            y_train = self._get_sub_elt(y, n, start_ind, end_ind)

        fit = self._ml_model.fit(x_train, y_train, **kwargs)

        # Save models
        if self.ml_path is not None and self.ml_path != '':
            self.ml_save_model(model_path=self.ml_path)

        return fit

    def _run_prefit_simulation(self, model, prefit_simulation_data=None, prefit_sample_data=None,
                               prefit_simulation_size=-1, prefit_loops=-1, kwargs_fit=None):

        columns_equivalences = self._columns_equivalences

        # Get the exchange data
        def get_data():
            if prefit_simulation_data is not None:
                if isinstance(prefit_simulation_data, list):
                    list_prefit_simulation_data = prefit_simulation_data
                else:
                    list_prefit_simulation_data = [prefit_simulation_data]

                for elt in list_prefit_simulation_data:
                    yield elt

            elif prefit_sample_data is not None and prefit_simulation_size > 100 and prefit_loops > 0:
                ind = prefit_sample_data.index.sort_values()
                start_ind, end_ind = ind[-1], ind[-1] + prefit_simulation_size * (ind[-1] - ind[-2])
                high_name, low_name = columns_equivalences.get('high', 'high'), columns_equivalences.get('low', 'low')

                for loop_number in range(prefit_loops):
                    yield MLArchitect.simulate_prices(prefit_sample_data, start_ind=start_ind, end_ind=end_ind,
                                                      high_name=high_name, low_name=low_name, is_nig=True)

            elif prefit_sample_data is not None and (prefit_simulation_size <= 100 or prefit_loops <= 0):
                raise Exception("'prefit_simulation_size' should be > 100 and prefit_loops > 0")

            else:
                return list()

        pca_reductions, i = model.input_shape[1], 1
        for fit_data in get_data():
            arc = MLArchitect(x_data=fit_data, columns_equivalences=columns_equivalences,
                              index_infer_datetime_format=True, pca_reductions=[('linear', pca_reductions)],
                              columns_to_predict=self._columns_to_predict, window_prediction=self._window_prediction,
                              columns_indicators=self._columns_indicators, test_size=0.10, ml_model=model,
                              disgard_last=False, window_tuple=self._window_tuple)

            x_train, y_train = arc.x_train_normalized(), arc.y_train_normalized()
            x_test, y_test = arc.x_test_normalized(), arc.y_test_normalized()

            # Fit the model
            print("*" * 50 + f"Prefit simulation {i}" + "*" * 50)
            i += 1
            arc.fit(x=x_train.values, y=y_train.values, **kwargs_fit)

            loss, accuracy = model.evaluate(x_test.values, y_test.values)
            y_pred = arc.ml_predict(x_test)
            mde = arc.mean_directional_accuracy(y_test.values, y_pred.values)
            mae = arc.mean_absolute_error(y_test.values, y_pred.values)
            print(
                f"loss: {loss} - accuracy: {accuracy} - mean_directional_accuracy: {mde} - mean_absolute_error: {mae}")

        return model

    def ml_predict(self, x: object = None, columns_for_dataframe: List = None, index_for_dataframe: object = None,
                   sort_index: bool = False, restore_y: bool = False, **kwargs_restoration):
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
        if hasattr(x_to_predict, 'index'):
            index = list(x_to_predict.index)

        y_predicted = pd.DataFrame(np.float64(self._ml_model.predict(x_to_predict)), columns=columns, index=index)
        y_predicted = self.norm_output_inverse_transform(y_predicted)

        y_predicted = pd.DataFrame(y_predicted, columns=columns, index=index)

        if sort_index:
            y_predicted.sort_index()

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

    @staticmethod
    def mean_directional_accuracy(actual: np.ndarray, predicted: np.ndarray):
        """ Mean Directional Accuracy """
        return np.mean((np.sign(actual[1:] - actual[:-1]).astype(int) ==
                        np.sign(predicted[1:] - predicted[:-1])).astype(int), dtype=np.float64)

    @staticmethod
    def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray):
        """ Mean Directional Accuracy """
        return np.mean(np.abs(actual - predicted), dtype=np.float64)

    @staticmethod
    def keras_r2_score(y_true, y_pred):
        u = K.sum(K.square(y_true - y_pred))
        v = K.sum(K.square(y_true - K.mean(y_true, axis=0)))
        return 1 - u / (v + K.epsilon())

    @staticmethod
    def keras_mean_absolute_percentage_error(y_true, y_pred):
        f_error = y_true - y_pred

        return 100 * K.mean(K.abs(f_error / y_true))

    @staticmethod
    def swish(x, beta=1):
        return x * keras.activations.sigmoid(beta * x)

    @staticmethod
    def sklearn_build_model(hidden_layer_sizes=(350, 100), activation='logistic', random_state=0,
                            validation_fraction=0.2, max_iter=1000, batch_size=50):
        ml_model = neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                               solver="adam", random_state=random_state, verbose=True,
                                               learning_rate_init=1e-3, early_stopping=True,
                                               validation_fraction=validation_fraction,
                                               tol=1e-6, alpha=0.001, learning_rate="adaptive", max_iter=max_iter,
                                               batch_size=batch_size, n_iter_no_change=10, warm_start=False)

        return ml_model

    @staticmethod
    def keras_build_model(os_seed, np_seed, backend_seed, n_input, n_output, n_steps_in, n_steps_out,
                          n_features, n_init_neurons=100):
        random.seed(os_seed)
        np.random.seed(np_seed)
        tf.random.set_seed(backend_seed)

        keras.utils.generic_utils.get_custom_objects().update({'swish': kl.Activation(MLArchitect.swish)})

        #n = arc.x_train_normalized(2).shape
        #n_output = arc.y_train_normalized(2).shape[1]

        r_l1, r_l2 = 1e-3, 1e-3
        drop, fact1 = 0.2, 1.1

        model = Sequential()

        # model.add(kl.TimeDistributed(kl.Conv1D(filters=n_init_neurons * 3, kernel_size=1, activation='relu'),
        #                              input_shape=(None, n_steps_in, n_features)))
        # model.add(kl.TimeDistributed(kl.MaxPooling1D(pool_size=2)))
        # model.add(kl.TimeDistributed(kl.Flatten()))

        model.add(kl.LSTM(n_init_neurons, activation='relu', input_shape=(n_steps_in, n_features),
                          activity_regularizer=keras.regularizers.l2(r_l2), return_sequences=True))

        model.add(kl.LSTM(n_init_neurons, activation='relu'))
        model.add(kl.Dropout(rate=0.1))

        model.add(kl.Dense(n_init_neurons*2, activity_regularizer=keras.regularizers.l2(r_l2)))
        model.add(kl.Activation(MLArchitect.swish))
        model.add(kl.BatchNormalization())
        model.add(kl.Dropout(rate=drop, seed=backend_seed))

        model.add(kl.Dense(n_init_neurons))
        model.add(kl.Activation(keras.activations.softsign))

        model.add(kl.Dense(n_output))
        #model.add(kl.Activation(keras.activations.softsign))
        model.add(kl.Activation('linear'))


        # model.add(kl.Dense(n_init_neurons*3, input_dim=n_input, activity_regularizer=keras.regularizers.l2(r_l2)))
        # model.add(kl.PReLU())
        # model.add(kl.BatchNormalization())
        # model.add(kl.Dropout(rate=drop, seed=backend_seed))
        #
        # #model.add(kl.Dense(n_init_neurons * 2, activity_regularizer=keras.regularizers.l2(r_l2)))
        # model.add(kl.Activation(keras.activations.softsign))
        # model.add(kl.BatchNormalization())
        # #model.add(kl.Dropout(rate=drop, seed=backend_seed))
        #
        # # model.add(kl.Dense(n_init_neurons * 2))
        # # model.add(kl.Activation(keras.activations.softsign))
        # #
        # # model.add(kl.Dense(n_init_neurons))
        # # model.add(kl.Activation(keras.activations.softsign))
        #
        # model.add(kl.Dense(n_output))
        # model.add(kl.Activation(keras.activations.softsign))

        optim = keras.optimizers.Adadelta()
        #optim = keras.optimizers.adam(learning_rate=1e-3, clipnorm=1.0)
        #optim = keras.optimizers.Nadam()
        model.compile(optimizer=optim, loss='mse', #loss='mean_absolute_error',
                      metrics=[MLArchitect.keras_r2_score,
                               MLArchitect.keras_mean_absolute_percentage_error])

        return model

    @staticmethod
    def _get_activation_function(activation_function):
        dict_switch = {
            'softsign': kl.Activation('softsign'),
            'prelu': kl.PReLU(),
            'relu': kl.Activation('relu'),
            'elu': kl.Activation('elu'),
            'sigmoid': kl.Activation('sigmoid'),
            'tanh': kl.Activation('tanh')
        }

        return dict_switch.get(activation_function, kl.Activation('sigmoid'))

    def keras_tuner_build_model(self, hp):
        #from kerastuner.engine.hyperparameters import HyperParameters
        #hpp = HyperParameters()

        os_seed = hp.Choice('os_seed', [0, 500, 1000])
        np_seed = hp.Choice('np_seed', [0, 500, 1000])
        backend_seed = hp.Choice('backend_seed', [0, 500, 1000])

        random.seed(os_seed)
        np.random.seed(np_seed)
        tf.random.set_seed(backend_seed)

        n = self.x_train_normalized(10).shape
        r_l1, r_l2 = 1e-2, 1e-2
        n_layers, n_output = 3, 16
        drop = 0.5
        fact1 = 1.1

        model = Sequential()
        model.add(kl.Dense(hp.Int('input_units', int(n[1]*0.5), n[1]*5, step=int(n[1]*0.5),
                                  default=n[1]*3), input_dim=n[1]))

        activation_str = hp.Choice('activation_1', ['softsign', 'prelu', 'relu', 'elu', 'sigmoid', 'tanh'])
        model.add(self._get_activation_function(activation_str))

        for i in range(2, n_layers):
            model.add(kl.Dense(hp.Int('hidden_layer_' + str(i), int(n[1]*0.5), n[1]*5,
                                      step=int(n[1]*0.5), default=n[1]*3)))

            activation_str = hp.Choice('hidden_activation_' + str(i), ['softsign', 'prelu', 'relu', 'elu',
                                                                       'sigmoid', 'tanh'])
            model.add(self._get_activation_function(activation_str))

            if i % 2 == 0:
                #model.add(kl.Dropout(0.5))
                pass

        model.add(kl.Dense(n_output))
        activation_str = hp.Choice('final_activation', ['softsign', 'prelu', 'relu', 'elu', 'sigmoid', 'tanh'])
        model.add(self._get_activation_function(activation_str))

        model.compile(optimizer=keras.optimizers.adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='mse', metrics=['accuracy'])

        return model

    @staticmethod
    def keras_tuner(model_func, seed):
        tuner = RandomSearch(
            hypermodel=model_func,
            objective='val_accuracy',
            max_trials=1,
            seed=seed,
            executions_per_trial=1,
            directory=f"models/"
        )

        return tuner

    def run_tunner(self, x, y):
        n = 100

        x_train = self.x_train_normalized(n)
        y_train = self.y_train_normalized(n)

        x_test = self.x_test_normalized(n)
        y_test = self.y_test_normalized(n)

        tuner = self.keras_tuner(self.keras_tuner_build_model, 0)
        tuner.search_space_summary()

        tuner.search(x=x_train, y=y_train, epochs=1, batch_size=75, validation_data=(x_test, y_test),
                     callbacks=[keras.callbacks.EarlyStopping('val_accuracy',
                                                              min_delta=1e-5,
                                                              patience=10,
                                                              verbose=1,
                                                              restore_best_weights=True)])
        models = tuner.get_best_models(num_models=2)
        tuner.results_summary()

        return models
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
    def index_min(self):
        return self._index_min

    @property
    def index_max(self):
        return self._index_max

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
                    _ = self.default_compute_learning_indicators(self._x, inplace=True,
                                                                 disgard_last=self._disgard_last,
                                                                 columns_equivalences=self._columns_equivalences,
                                                                 columns_indicators=self._columns_indicators,
                                                                 window_tuple=self._window_tuple)

            elif callable(indicators_callable):
                self._x = indicators_callable(self._x)

            else:
                raise Exception("'indicators_callable' not recognized as a callable object.")

    @staticmethod
    def default_compute_learning_indicators(original_data: pd.DataFrame, ascending_order: bool = True,
                                            drop_duplicates: bool = True, inplace: bool = True,
                                            disgard_last: bool = True, columns_equivalences: Dict = None,
                                            columns_indicators: List = None,
                                            window_tuple: Tuple[int, int, int, int] = (5, 14, 22, 35)):
        """
        Function to generate defaults indicators and preprocess them.

        :param pd.DataFrame original_data: Original x data.
        :param bool ascending_order: Order data in ascending order.
        :param bool drop_duplicates: Drop duplicated lines with the same index.
        :param bool inplace: If True, do operation inplace and return None.
        :param bool disgard_last: Discard the last elt.
        :param bool columns_equivalences: Equivalence between internal column names and effective data column names.
                                          internal column names are: open, close, high, low, vol, amount.
        :param bool columns_indicators: Columns for which we want to create indicators.
                                        Default are: open, close, high, low.
        :param bool window_tuple: window short, medium, long and return window.
        :return: None if inplace=True else data
        """

        data = original_data if inplace else original_data.copy()
        ind_obj = Indicators(data)
        proc_object = ind_obj.preprocessing

        if columns_equivalences is None:
            columns_equivalences = {'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'vol': 'vol',
                                    'amount': 'amount'}

        src = columns_indicators
        col_high, col_low, col_close, col_open = MLArchitect._get_columns_equivalence(['high', 'low', 'close', 'open'],
                                                                                      columns_equivalences, data)

        # Sorting and Dropping duplicates from index
        ind_obj.set_index(index_column_name=ind_obj.data.index.name, ascending_order=ascending_order,
                          drop_duplicates=drop_duplicates)

        # Discard the last index
        if disgard_last:
            ind_obj.data.drop(data.index[-1], inplace=True)

        win_s, win_m, win_l = window_tuple[0], window_tuple[1], window_tuple[2]
        ret_window = window_tuple[3]

        # date info
        halving_dates = [np.datetime64("2012-11-28"), np.datetime64("2016-07-09")]
        halving_dates += [np.datetime64("2020-05-13"), np.datetime64("2024-02-13")]
        ind_obj.date_info(halving_dates=halving_dates)

        # Quantile 0.05 and 0.99
        ind_obj.moving_quantile(columns=src, quantile=0.05, window=win_m, ml_format=True,
                                result_names=[x + "_q_05_" + str(win_m) for x in src])

        ind_obj.moving_quantile(columns=src, quantile=0.99, window=win_m, ml_format=True,
                                result_names=[x + "_q_99_" + str(win_m) for x in src])

        # Median
        ind_obj.moving_median(columns=src, window=win_l, ml_format=True,
                              result_names=[x + "_median_" + str(win_l) for x in src])

        # Sum
        #ind_obj.moving_sum(columns=src, window=win_l, ml_format=True,
        #                   result_names=[x + "_sum_" + str(win_l) for x in src])

        # Compute log divisions of Sum 10
        #columns = [x + "_sum_" + str(win_l) for x in src]
        #proc_object.log_returns(columns, [x + "_log_div" for x in columns], window=1, delete_columns=True)

        # Min, Max
        #ind_obj.moving_min(columns=src, window=win_l, ml_format=True,
        #                   result_names=[x + "_min_" + str(win_l) for x in src])
        #ind_obj.moving_max(columns=src, window=win_l, ml_format=True,
        #                   result_names=[x + "_max_" + str(win_l) for x in src])

        # Skew, Kurt
        ind_obj.moving_skew(columns=src, window=win_m,
                            result_names=[x + "_skew_" + str(win_m) for x in src])
        ind_obj.moving_kurt(columns=src, window=win_m,
                            result_names=[x + "_kurt_" + str(win_m) for x in src])

        # True Range and its average
        ind_obj.true_range(close_name=col_close, high_name=col_high, low_name=col_low, window=1, result_names="tr",
                           ml_format=True)
        ind_obj.average_true_range(close_name=col_close, high_name=col_high, low_name=col_low,
                                   result_names="tr_avg_" + str(win_m), window=win_m, ml_format=True)

        # Average Directional Index
        ind_obj.average_directional_index(close_name=col_close, high_name=col_high, low_name=col_low,
                                          result_name="adx_" + str(win_m), window=win_m)

        # Custom Returns Directional Index
        ind_obj.returns_dix(columns=src, result_names=[x + "_rdix_" + str(win_l) for x in src],
                            window=win_l)
        #ind_obj.returns_dix_average(columns=src, window=win_l,
        #                            result_names=[x + "_rdix_avg_" + str(win_l) for x in src])

        ind_obj.returns_square_dix(columns=src, window=win_l,
                                   result_names=[x + "_square_rdix_" + str(win_l) for x in src])
        #ind_obj.returns_square_dix_average(columns=src, window=win_l,
        #                                   result_names=[x + "_square_rdix_avg_" + str(win_l) for x in src])

        ind_obj.returns_norm_dix(columns=src, window=win_l,
                                 result_names=[x + "_norm_dix_" + str(win_l) for x in src])

        #ind_obj.returns_norm_dix_average(columns=src, window=win_l,
        #                                 result_names=[x + "_norm_dix_avg_" + str(win_l) for x in src])

        ind_obj.price_velocity(columns=src, window=win_l, ml_format=True,
                               result_names=[x + "_pricevelo_" + str(win_l) for x in src])
        #ind_obj.price_velocity_average(columns=src, window=win_l, ml_format=True,
        #                               result_names=[x + "_PriceVelo_avg_" + str(win_l) for x in src])

        ind_obj.returns_velocity(columns=src, window=win_l,
                                 result_names=[x + "_returnsvelo_" + str(win_l) for x in src])
        #ind_obj.returns_velocity_average(columns=src, window=win_l,
        #                                 result_names=[x + "_ReturnsVelo_avg_" + str(win_l) for x in src])

        # RSI
        ind_obj.rsi(columns=src, window=win_l, result_names=[x + "_RSI_" + str(win_l) for x in src])

        # CCI
        ind_obj.commodity_channel_index(high=col_high, low=col_low, close=col_close, window=win_m,
                                        result_name="CCI_" + str(win_m))

        # RELS
        ind_obj.relative_slope(high=col_high, low=col_low, close=col_close, typical_window=win_l, rels_window=win_l,
                               result_names=["RELS_" + str(win_l) + "_" + str(win_l)])

        # Pivot Points
        ind_obj.pivot_points_fibonacci(previous_high=col_high, previous_low=col_low, previous_close=col_close,
                                       previous_open=col_open, current_open=col_open,
                                       moving_window=win_s, result_name='pp_' + str(win_s), ml_format=True)

        # Exponential Moving average
        ind_obj.exponential_weighted_moving_average(columns=src, span=win_l, ml_format=True,
                                                    result_names=["ex_" + x + "_avg_" + str(win_l) for x in src])

        ind_obj.exponential_weighted_moving_average(columns=src, span=win_m, ml_format=True,
                                                    result_names=["ex_" + x + "_avg_" + str(win_m) for x in src])

        ind_obj.exponential_weighted_moving_average(columns=src, span=win_s, ml_format=True,
                                                    result_names=["ex_" + x + "_avg_" + str(win_s) for x in src])

        # Bollinger Bands
        ind_obj.exponential_weighted_bollinger_bands_percentage(columns=src, nb_of_deviations=3, span=win_m,
                                                    result_names=[x + "_bb_per_" + str(win_m) for x in src])

        # Hull moving average
        ind_obj.hull_moving_average(columns=src, ml_format=True, window=win_l,
                                    result_names=[x + "_hull_" + str(win_l) for x in src])

        # Raise Velocity = (high-low)/amount
        col_amount = columns_equivalences.get("amount", None)
        # if col_amount in set(ind_obj.data.index):
        #     ind_obj.data["high_low_raise_velocity"] = pd.DataFrame((ind_obj.data[col_high].values -
        #                                                             ind_obj.data[col_low].values)
        #                                                            / ind_obj.data[col_amount].values,
        #                                                            columns=["high_low_raise_velocity"],
        #                                                            index=ind_obj.data.index)

        # Compute log returns and normal returns for i+1 to i+10
        for i in range(ret_window):
            proc_object.log_returns(src, [x + "_log_returns_" + str(i + 1) for x in src], window=i+1)

        # Compute the Simple and Exponential average of returns
        returns_columns = [x + "_log_returns_" + str(i + 1) for x in src for i in range(ret_window)]
        target_columns_names = ["ex_" + x + "_avg_" + str(win_l) for x in returns_columns]
        if returns_columns:
            ind_obj.exponential_weighted_moving_average(returns_columns, span=win_l, result_names=target_columns_names)

        if inplace:
            return None
        else:
            return ind_obj.data

    def _get_display_indicators(self, indicators_callable: Union[bool, Callable] = False, is_x_flat: bool = False):
        if indicators_callable is not None:
            if isinstance(indicators_callable, bool):
                if indicators_callable and is_x_flat:
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
        ind_obj.set_index(index_column_name=ind_obj.data.index.name, ascending_order=ascending_order,
                          drop_duplicates=drop_duplicates)

        # Quantile 0.05 and 0.95
        ind_obj.moving_quantile(columns=src_columns, quantile=0.01, window=14,
                                result_names=[x + "_q_05" for x in src_columns])

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

        # Bollinger Bands
        ind_obj.exponential_weighted_bollinger_bands(columns=src_columns, nb_of_deviations=3, span=14,
                                                     result_names=["ex_" + x + "_bb_14" for x in src_columns])
        ind_obj.exponential_weighted_bollinger_bands_percentage(columns=src_columns, nb_of_deviations=3, span=14,
                                                                result_names=[x + "_bb_per_14" for x in src_columns])

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
                               result_names=[x + "_pricevelo_5" for x in src_columns])
        ind_obj.price_velocity_average(columns=src_columns, window=14,
                                       result_names=[x + "_pricevelo_avg_14" for x in src_columns], ml_format=True)

        ind_obj.returns_velocity(columns=src_columns, window=5,
                                 result_names=[x + "_returnsvelo_5" for x in src_columns])
        ind_obj.returns_velocity_average(columns=src_columns, window=14,
                                         result_names=[x + "_returnsvelo_avg_14" for x in src_columns])

        # RSI
        ind_obj.rsi(columns=src_columns, window=14, result_names=[x + "_rsi_14" for x in src_columns])

        # CCI
        ind_obj.commodity_channel_index(high='high', low='low', close='close', window=14, result_name="cci_14")

        # RELS
        ind_obj.relative_slope(high='high', low='low', close='close', typical_window=5, rels_window=5,
                               result_names=["rels_5_5"])

        # Pivot Points
        ind_obj.pivot_points_fibonacci(previous_open='open', current_open='open', moving_window=5, result_name='pp_5')

        # Raise Velocity = (high-low)/amount
        if "amount" in set(ind_obj.data.index):
            ind_obj.data["high_low_raise_velocity"] = pd.DataFrame((ind_obj.data["high"].values
                                                                    - ind_obj.data["low"].values)
                                                                   / ind_obj.data["amount"].values,
                                                                   columns=["high_low_raise_velocity"],
                                                                   index=ind_obj.data.index)

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
                                                                  min_max_range=(-1, 1),
                                                                  pca_reductions=self._pca_reductions,
                                                                  save_normalize_x_model=save_normalize_x_model)

                elif normalize_x_callable != '' and os.path.isfile(normalize_x_callable):
                    self.norm_load_model(is_input=True, model_to_load=normalize_x_callable)

            elif callable(normalize_x_callable):
                self._x_norm_model = normalize_x_callable

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
                self._y_norm_model = normalize_y_callable

                # Save models
                if save_normalize_y_model is not None and save_normalize_y_model != '':
                    self.norm_save_model(is_input=False, model_path=save_normalize_y_model)

            else:
                raise Exception(f"'normalize_y_callable' not recognized as a callable object neither as a path for"
                                f" a normalization model.")

    def norm_input_transform(self, data_to_transform: pd.DataFrame) -> Any:
        if not self._x_norm_model:
            raise Exception("Normalization model for inputs not found. You should call norm_input_fit to define "
                            "a model and fit it.")

        if not callable(getattr(self._x_norm_model, "transform", None)):
            raise Exception("No transform method found for inputs normalization model.")

        return self._x_norm_model.transform(data_to_transform)

    def norm_output_transform(self, data_to_transform: pd.DataFrame) -> Any:
        if not self._y_norm_model:
            raise Exception("Normalization model for outputs not found. You should call norm_output_fit "
                            "to define a model and fit it.")

        if not callable(getattr(self._y_norm_model, "transform", None)):
            raise Exception("No transform method found for outputs normalization model.")

        return self._y_norm_model.transform(data_to_transform)

    def norm_input_inverse_transform(self, data_to_inverse: pd.DataFrame) -> Any:
        if not self._x_norm_model:
            raise Exception("Normalization model for inputs not found. You should call norm_input_fit to define "
                            "a model and fit it.")

        if not callable(getattr(self._x_norm_model, "inverse_transform", None)):
            raise Exception("No inverse_transform method found for inputs normalization model.")

        return self._x_norm_model.inverse_transform(data_to_inverse)

    def norm_output_inverse_transform(self, data_to_inverse: pd.DataFrame) -> Any:
        if not self._y_norm_model:
            raise Exception("Normalization model for outputs not found. You should call norm_output_fit "
                            "to define a model and fit it.")

        if not callable(getattr(self._y_norm_model, "inverse_transform", None)):
            raise Exception("No inverse_transform method found for outputs normalization model.")

        return self._y_norm_model.inverse_transform(data_to_inverse)

    @staticmethod
    def _get_sub_elt(data=None, n: Union[float, range, list, int] = None, start_ind=None, end_ind=None):
        """
        Get sub elements of data.

        Only n or (start_ind AND end_ind) should be defined.

        :param data: original data.
        :param n: Number of elements to extract.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: sub elements of data.
        """

        res = None
        if data is not None:
            if n is not None:
                if isinstance(n, float) or isinstance(n, int):
                    length = n if n > 1.0 else int(n * data.shape[0])
                    res = data.iloc[:length]
                else:
                    res = data.iloc[n]

            elif start_ind is not None or end_ind is not None:
                start_ind = data.index[0] if start_ind is None else start_ind
                end_ind = data.index[-1] if end_ind is None else end_ind

                res = data.loc[start_ind:end_ind]
            else:
                res = data

        return res

    def get_normalized_data(self, x=None, y=None, n: Union[float, pd.Index] = None,
                            start_ind=None, end_ind=None) -> Tuple[Any, Any]:
        """
        Normalize x and y.
        """

        x_norm = None
        if x is not None:
            x_norm = self.norm_input_transform(self._get_sub_elt(x, n, start_ind, end_ind))

        y_norm = None
        if y is not None:
            y_norm = self.norm_output_transform(self._get_sub_elt(y, n, start_ind, end_ind))

        return x_norm, y_norm

    def _x_normalized(self, is_train: bool = True, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None):
        """
        Normalize x and return it.
        :param is_train: If is train.
        :param n: Fraction to get.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: Normalized x_train
        """

        x = self.x_train if is_train else self.x_test
        x_res, _ = self.get_normalized_data(x, None, n, start_ind, end_ind)

        return x_res

    def x_train_normalized(self, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None):
        """
        Normalize x_train and return it.
        :param n: Fraction to get.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: Normalized x_train
        """

        return self._x_normalized(True, n, start_ind, end_ind)

    def x_test_normalized(self, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None):
        """
        Normalize x_test and return it.
        :param n: Fraction to get.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: Normalized x_test
        """

        return self._x_normalized(False, n, start_ind, end_ind)

    def _y_normalized(self, is_train: bool = True, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None):
        """
        Normalize y and return it.
        :param is_train: If is train.
        :param n: Fraction to get.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: Normalized x_train
        """

        y = self.y_train if is_train else self.y_test
        _, y_res = self.get_normalized_data(None, y, n, start_ind, end_ind)

        return y_res

    def y_train_normalized(self, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None):
        """
        Normalize y_train and return it.
        :param n: Fraction to get.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: Normalized y_train
        """

        return self._y_normalized(True, n, start_ind, end_ind)

    def y_test_normalized(self, n: Union[float, pd.Index] = None, start_ind=None, end_ind=None):
        """
        Normalize y_test and return it.
        :param n: Fraction to get.
        :param start_ind: First index for sub elements.
        :param end_ind: Last index for sub elements.
        :return: Normalized y_test
        """

        return self._y_normalized(False, n, start_ind, end_ind)

    @staticmethod
    def _default_normalizer(data_to_fit: pd.DataFrame, min_max_range: Union[None, tuple] = None,
                            add_standard_scaling: bool = False, add_power_transform: bool = False,
                            pca_reductions: Union[List[Tuple], None] = None,
                            save_normalize_x_model: str = None):

        norm_object = Normalization()
        if add_standard_scaling:
            norm_object.add_standard_scaling(data_to_fit)

        if min_max_range:
            norm_object.add_min_max_scaling(data_to_fit, min_max_range=min_max_range)

        if add_power_transform:
            norm_object.add_power_transform_scaling(data_to_fit)

        # Apply PCA reduction to data_to_fit
        if pca_reductions:
            for reduction in pca_reductions:
                if reduction[0].lower().strip() == 'kernel':
                    norm_object.add_kernel_pca_reduction(n_components=reduction[1], kernel=reduction[2], n_jobs=-1)
                elif reduction[0].lower().strip() == 'linear':
                    norm_object.add_pca_reduction(n_components=reduction[1], svd_solver="full")
                else:
                    raise Exception(f"{reduction} method is not available. Possible values are "
                                    f"'kernel' and 'linear'.")

        norm_object.fit(data_to_fit)

        # Save models
        if save_normalize_x_model is not None and save_normalize_x_model != '':
            norm_object.save_model(save_normalize_x_model)

        return norm_object
    # endregion

    # region y processing
    def _get_y(self, y_data: Any, index_infer_datetime_format: bool = True,
               index_unit_datetime: str = None):
        if y_data is not None:
            self._y = self._load_data(y_data, index_infer_datetime_format, index_unit_datetime)

        else:
            Y = pd.DataFrame()
            src_columns = self._columns_to_predict

            # Compute y returns for window = self._window_prediction
            for i in range(self._window_prediction):
                returns = self.default_get_y_to_predict(self._x, src_columns,
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
            df = MLArchitect._restore_unique_window(y_predicted.iloc[:, len(src_columns)*i:len(src_columns)*(i+1)],
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
        sorted_index = sorted(flat_data.index)
        delta = sorted_index[1] - sorted_index[0]

        if is_returns:
            exp_values = numpy_exp(is_parallel, y_predicted.values)
            nu_df = numpy_mul(is_parallel, flat_data.reindex(index=y_predicted.index).values, exp_values)
            df = pd.DataFrame(nu_df, columns=flat_data.columns, index=y_predicted.index)
        else:
            df = y_predicted.copy()

        df.columns = target_columns_names
        df.index = df.index + window*delta

        return df
    # endregion

    # region utils
    @staticmethod
    def get_x_y_index_intersection(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[int, int]:
        intersection = set(x.index) & set(y.index)

        return min(intersection), max(intersection)

    def _load_data(self, data_path_or_object: Any, index_infer_datetime_format: bool = True,
                   index_unit_datetime: str = None):
        if data_path_or_object is not None:
            if isinstance(data_path_or_object, str):
                if os.path.isfile(data_path_or_object):
                    df = pd.read_csv(data_path_or_object)
                    unit = None
                    if index_unit_datetime is not None and index_unit_datetime.strip() != '':
                        unit = index_unit_datetime

                    index_col, = self._get_columns_equivalence(['index'], self._columns_equivalences, df)

                    if index_infer_datetime_format or unit:
                        if index_col not in df:
                            raise Exception(f"index equivalence column '{index_col}' has not been found in the loaded "
                                            f"DataFrame")

                        df[index_col] = pd.to_datetime(df[index_col],
                                                       infer_datetime_format=index_infer_datetime_format, unit=unit)

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

    def _do_data_enhancement(self, data_enhancement: Union[bool, Callable] = False):
        if data_enhancement is not None:
            if isinstance(data_enhancement, bool):
                if data_enhancement:
                    self._x = self.enhance_price_data(self._x, columns_equivalences=self._columns_equivalences)
                    self.enhance_volume_data(self._x, columns_equivalences=self._columns_equivalences, inplace=True)

            elif callable(data_enhancement):
                self._x = data_enhancement(self._x)

            else:
                raise Exception(f"'data_enhancement' parameter is not recognized as a callable object neither as a "
                                f" boolean.")

    @staticmethod
    def enhance_price_data(original_data: pd.DataFrame, columns_equivalences: Dict = None, index_delta=None):
        data = original_data.copy()
        use_inf_as_na = pd.options.mode.use_inf_as_na
        pd.options.mode.use_inf_as_na = True

        if columns_equivalences is None:
            columns_equivalences = {'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'vol': 'vol',
                                    'amount': 'amount'}

        price_col = {x: columns_equivalences[x] for x in ['open', 'close', 'high', 'low']
                     if x in columns_equivalences and columns_equivalences[x] in data}
        oc_col = {x: price_col[x] for x in ['open', 'close'] if x in price_col}
        hl_col = {x: price_col[x] for x in ['high', 'low'] if x in price_col}

        # Generate all theoric dates
        sorted_index = data.index.sort_values()
        delta = sorted_index[1]-sorted_index[0] if index_delta is None else index_delta
        generated_index = pd.date_range(start=sorted_index[0], end=sorted_index[-1], freq=delta)

        src_col = list(price_col.values())

        # Replace 0 values by NA
        data[src_col] = data[src_col].where(data[src_col] > 100 + eps, np.nan)
        data = pd.concat([data, pd.DataFrame(False, index=data.index, columns=['simulated'])], axis=1)

        # Put NA when low==high. Incorrect data
        if 'low' in price_col and 'high' in price_col:
            equ_high = np.isclose(data[price_col['low']].values, data[price_col['high']].values, atol=1e-06)
            if equ_high.shape[0] > 0:
                data.loc[equ_high, price_col['high']] = np.nan

                if 'open' in price_col:
                    equ = np.isclose(data[price_col['low']].values, data[price_col['open']].values, atol=1e-06)
                    if equ.shape[0] > 0:
                        data.loc[equ & equ_high, price_col['open']] = np.nan

                if 'close' in price_col:
                    equ = np.isclose(data[price_col['low']].values, data[price_col['close']].values, atol=1e-06)
                    if equ.shape[0] > 0:
                        data.loc[equ & equ_high, price_col['close']] = np.nan

                data.loc[equ_high, price_col['low']] = np.nan

        if oc_col and hl_col:
            null_oc_ind = data.loc[data[oc_col.values()].isnull().max(axis=1), hl_col.values()].isnull().min(axis=1)
            data.loc[null_oc_ind[~null_oc_ind].index, hl_col.values()] = np.nan

        # Add missing index to data
        missing = list(set(generated_index) - set(sorted_index))
        if missing:
            data = pd.concat([data, pd.DataFrame(False, index=missing, columns=['simulated'])], axis=0)

        data.loc[:, 'simulated'] = False

        # Compute indices
        wrong_ind = data.index[data[src_col].isnull().max(axis=1)]
        correct_ind = generated_index.drop(wrong_ind).sort_values(ascending=True)

        # Missing indices
        if wrong_ind.shape[0] > 0:
            # Flag simulated data
            data.loc[wrong_ind, 'simulated'] = True

            # Keep first correct line
            first_correct_line = data.loc[correct_ind[0], src_col]

            # Compute returns
            index_n1 = sorted_index - delta
            data.loc[sorted_index, src_col] = np.log(data.loc[sorted_index, src_col].values
                                                     / data[src_col].reindex(index_n1).values)

            nb_elt = np.minimum(wrong_ind.shape[0], 10000)
            not_null = data.index[~data.isnull().max(axis=1)]
            data.sort_index(ascending=True, inplace=True)

            # Compute NIG law parameters
            info_dict = MLArchitect._generate_nig_parameters(data.loc[not_null, src_col],
                                                             data.loc[not_null, src_col].mean(),
                                                             data.loc[not_null, src_col].std(), sample_max_size=5000)

            def get_column_index_it(columns):
                col = {x: price_col[x] for x in columns if x in price_col}
                null_ind = [data.index[data[x].isnull()].copy() for x in col.values()]
                ite = [zip_longest(*[iter(x)] * nb_elt) for x in null_ind]

                return col, null_ind, ite

            def reconstruct_data(df_tmp, cumsums, col):
                df_tmp.loc[df_tmp.index[1:], col] = first_correct_line[col].values * np.exp(cumsums)

            # if 'close' or 'open' are present in data
            col_dict, col_null_ind, it = get_column_index_it(['open', 'close'])
            if col_dict:
                for i, column in enumerate(col_dict.values()):
                    for miss in it[i]:
                        ind = pd.DatetimeIndex(miss).dropna()
                        variates = info_dict[column]['variates'](size=(ind.shape[0], 100))
                        variates = variates.mean(axis=1)
                        data.loc[ind, column] = variates

                reconstruct_data(data, np.cumsum(data[col_dict.values()].iloc[1:], axis=0), list(col_dict.values()))
                data.loc[correct_ind, col_dict.values()] = original_data.loc[correct_ind, col_dict.values()].values

            if 'close' in col_dict and 'open' in col_dict:
                index_n1 = (pd.DatetimeIndex(set(data.index + delta) - set(correct_ind)) & data.index).sort_values()
                data.loc[index_n1, col_dict['open']] = data.loc[index_n1 - delta, col_dict['close']].values

                index_n1 = (pd.DatetimeIndex(set(correct_ind - delta) - set(correct_ind)) & data.index).sort_values()
                data.loc[index_n1, col_dict['close']] = data.loc[index_n1 + delta, col_dict['open']].values

            # if 'low' or 'high' are present in data
            col_dict, col_null_ind, it = get_column_index_it(['low', 'high'])
            if col_dict:
                for i, column in enumerate(col_dict.values()):
                    res = pd.DataFrame()
                    for miss in it[i]:
                        ind = pd.DatetimeIndex(miss).dropna()
                        df = data[[column]*1000].copy()
                        df.loc[ind] = info_dict[column]['variates'](size=(ind.shape[0], 1000))
                        df.iloc[1:] = first_correct_line[column] * np.exp(np.cumsum(df.iloc[1:], axis=0))
                        df.iloc[0] = first_correct_line[column]

                        if column == 'high':
                            cond = df.loc[ind].values >= data.loc[ind, oc_col.values()].max(axis=1).values.reshape(-1,1)
                            quantile = 0.1
                        else:
                            cond = df.loc[ind].values <= data.loc[ind, oc_col.values()].min(axis=1).values.reshape(-1,1)
                            quantile = 0.9

                        # Compute the mean of highest values or the mean of lowest values
                        data.loc[ind, column] = np.nanquantile(np.where(cond, df.loc[ind], np.nan), quantile, axis=1)
                        res = pd.concat([res, data.loc[ind, [column]]], axis=0)

                        # Compute returns and save it instead of direct value
                        index_n1 = (ind - delta) & data.index
                        df = original_data.loc[correct_ind & index_n1, column]

                        tmp_index = pd.DatetimeIndex(set(res.index) - set(correct_ind)) & index_n1
                        df = pd.concat([df, res.loc[tmp_index, column]], axis=0)
                        df.sort_index(ascending=True, inplace=True)

                        data.loc[index_n1+delta, column] = np.log(data.loc[index_n1+delta, column].values
                                                                  / df.loc[index_n1].values)

                    data.loc[res.index, column] = res.values

                data.loc[correct_ind, col_dict.values()] = original_data.loc[correct_ind, col_dict.values()].values

            data.loc[wrong_ind, src_col] = data.loc[wrong_ind, src_col].round(2)

        pd.options.mode.use_inf_as_na = use_inf_as_na
        return data

    @staticmethod
    def _generate_nig_parameters(df: pd.DataFrame, mean: pd.DataFrame, std: pd.DataFrame,
                                 sample_max_size: int = -1) -> dict:
        ind_obj = Indicators(pd.DataFrame())
        info_dict = dict()

        # Compute NIG law parameters
        for col in df:
            sample = df[col].to_numpy()
            nb_elt = np.minimum(sample.shape[0], sample_max_size) if sample_max_size > 0 else sample.shape[0]
            a, b, _, _ = ind_obj.nig_fit(sample[-nb_elt:], floc=mean[col], fscale=std[col])
            variates = ind_obj.nig_variates(a=a, b=b, loc=mean[col], scale=std[col], return_callable=True)
            info_dict[col] = {'a': a, 'b': b, 'mean': mean[col], 'std': std[col], 'variates': variates}

        return info_dict

    @staticmethod
    def _generate_gaussian_parameters(df: pd.DataFrame, sample_max_size: int = -1) -> dict:
        ind_obj = Indicators(pd.DataFrame())
        info_dict = dict()

        # Compute NIG law parameters
        for col in df:
            sample = df[col].to_numpy()
            nb_elt = np.minimum(sample.shape[0], sample_max_size) if sample_max_size > 0 else sample.shape[0]
            mean, std = ind_obj.normal_fit(sample[-nb_elt:])
            variates = ind_obj.normal_variates(mean=mean, std=std, return_callable=True)
            info_dict[col] = {'mean': mean, 'std': std, 'variates': variates}

        return info_dict

    @staticmethod
    def enhance_volume_data(original_data: pd.DataFrame, columns_equivalences: Dict = None, index_delta=None,
                            inplace: bool = False):
        data = original_data if inplace else original_data.copy()
        use_inf_as_na = pd.options.mode.use_inf_as_na
        pd.options.mode.use_inf_as_na = True

        if columns_equivalences is None:
            columns_equivalences = dict(zip(['close', 'open', 'high', 'low'], ['close', 'open', 'high', 'low']))

        price_col = dict()
        for x in ['close', 'open', 'high', 'low']:
            if x in columns_equivalences and columns_equivalences[x] in data:
                price_col[x] = columns_equivalences[x]
                break

        src_col = price_col.values()
        vol_columns = {x: columns_equivalences[x] for x in ['vol', 'amount'] if x in columns_equivalences
                       and columns_equivalences[x] in data}
        tgt_vol_col = list(vol_columns.values())

        if vol_columns:
            def get_values(line, df, target_column):
                elt_idx = np.where(df[list(src_col)].values >= line.values)[0]
                if elt_idx.shape[0] > 0:
                    return df.loc[df.index[elt_idx[0]], target_column]

            if index_delta is None:
                sorted_index = data.index.sort_values()
                delta = sorted_index[1] - sorted_index[0]
            else:
                delta = index_delta

            # Replace 0 values by NA
            data[tgt_vol_col] = data[tgt_vol_col].where(data[tgt_vol_col] > 0 + eps, np.nan)

            for tgt_vol_col in [('vol', 'amount'), ('amount', 'vol')]:

                init, comp = tgt_vol_col[0], tgt_vol_col[1]
                if init in vol_columns:
                    wrong_ind = data.index[data[vol_columns[init]].isnull()].sort_values(ascending=True)
                    if wrong_ind.shape[0]:
                        wrong_vol = pd.DataFrame(np.log(data.loc[wrong_ind, src_col].values
                                                        / data[src_col].reindex(wrong_ind-delta).values),
                                                 columns=list(src_col), index=wrong_ind)

                        correct_ind = data.index.drop(wrong_ind).sort_values(ascending=True)
                        if not correct_ind.shape[0]:
                            raise Exception(f"There is no initial data  in column '{init}'")

                        correct_vol = data.loc[correct_ind, list(src_col) + [vol_columns[init]]].copy()
                        correct_vol[list(src_col)] = pd.DataFrame(np.log(correct_vol[src_col].values
                                                                         / data[src_col].reindex(correct_ind-delta).values),
                                                                  columns=list(src_col), index=correct_ind)

                        correct_vol.sort_values(list(src_col), inplace=True)
                        correct_vol.dropna(inplace=True)
                        data.loc[wrong_ind, vol_columns[init]] = wrong_vol.apply(get_values,
                                                                                 args=(correct_vol, vol_columns[init]),
                                                                                 axis=1)

                        if comp in vol_columns:
                            wrg_col_2 = data.index[data[vol_columns[comp]].isnull()].sort_values(ascending=True)
                            if comp == 'amount':
                                data.loc[wrg_col_2, vol_columns[comp]] = data.loc[wrg_col_2, [vol_columns[init]]].values \
                                                                         / data.loc[wrg_col_2, list(src_col)].values
                            else:
                                data.loc[wrg_col_2, vol_columns[comp]] = data.loc[wrg_col_2, [vol_columns[init]]].values \
                                                                         * data.loc[wrg_col_2, list(src_col)].values

        pd.options.mode.use_inf_as_na = use_inf_as_na
        if inplace:
            return None
        else:
            return data

    @staticmethod
    def simulate_prices(sample: pd.DataFrame, start_ind, end_ind, delta_ind=None, high_name='high', low_name='low',
                        initial_price=None, is_nig=False):
        # Variables initializing
        start, end = pd.DatetimeIndex([start_ind]), pd.DatetimeIndex([end_ind])

        if not delta_ind:
            sorted_index = sample.index.sort_values()
            delta = sorted_index[1]-sorted_index[0]
        else:
            delta = pd.Timedelta(delta_ind)

        init = initial_price if initial_price else sample.iloc[0, 0]

        # Generate date range
        date_range = np.arange(start.values[0], (end + delta).values[0], delta)
        nb_elt = 10000

        res = pd.DataFrame()
        if date_range.shape[0]:
            ret = pd.DataFrame(np.log(sample.values / sample.shift(1).values),
                               columns=sample.columns, index=sample.index).dropna()

            # NIG fit and get variates callable
            if is_nig:
                info_dict = MLArchitect._generate_nig_parameters(ret, ret.mean(), ret.std())
            else:
                info_dict = MLArchitect._generate_gaussian_parameters(ret)

            columns = list(set(sample.columns) - {high_name, low_name})

            it = zip_longest(*[iter(date_range)] * nb_elt)
            res = pd.DataFrame()
            for miss in it:
                ind = pd.DatetimeIndex(miss, name='id').dropna()
                data = pd.DataFrame()
                for col in columns:
                    p_ini = init if col not in set(res.columns) else res.loc[res.index[-1], col]

                    variates = info_dict[col]['variates'](size=(len(ind), 100)).mean(axis=1)
                    df = pd.DataFrame(p_ini * np.exp(np.cumsum(variates, axis=0)), columns=[col], index=ind)
                    data = pd.concat([data, df], axis=1)

                res = pd.concat([res, data], axis=0)

            it = zip_longest(*[iter(date_range)] * nb_elt)
            high_low = pd.DataFrame()
            for miss in it:
                ind = pd.DatetimeIndex(miss, name='id').dropna()
                data = pd.DataFrame()
                for col in {high_name, low_name} & set(sample.columns):
                    p_ini = init if col not in set(high_low.columns) else high_low.loc[high_low.index[-1], col]

                    variates = info_dict[high_name]['variates'](size=(len(ind), 100))
                    df = pd.DataFrame(p_ini * np.exp(np.cumsum(variates, axis=0)), columns=[col]*100, index=ind)

                    if col == high_name:
                        cond = df.values >= res.loc[ind, columns].max(axis=1).values.reshape(-1, 1)
                        quantile = 0.1
                    else:
                        cond = df.values <= res.loc[ind, columns].min(axis=1).values.reshape(-1, 1)
                        quantile = 0.9

                    # Compute the mean of highest values or the mean of lowest values
                    df = pd.DataFrame(np.nanquantile(np.where(cond, df, np.nan), quantile, axis=1),
                                      columns=[col], index=ind)
                    data = pd.concat([data, df], axis=1)

                high_low = pd.concat([high_low, data], axis=0)

            res = pd.concat([res, high_low], axis=1)
            res.sort_index(ascending=True, inplace=True)
            res.dropna(inplace=True)
            res = res.round(2)

        return res

    def _check_str(self, field_name: str, value: str):
        if value is None:
            raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")

        if (value is not None) and (value.strip() == ""):
            raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")

    @staticmethod
    def _get_columns_equivalence(columns_to_get: List, columns_equivalences: Dict, data: pd.DataFrame = None) -> tuple:
        res, err = list(), list()

        for col in columns_to_get:
            equi = columns_equivalences.get(col, None)
            res.append(equi)

            if (equi is None) or (equi is not None and equi.strip() == ""):
                err += f"No equivalence for column '{col}' has been provided."

            if data is not None and not data.empty and equi not in data:
                err += f"Equivalence '{equi}' for column '{col}' has not been found in data."

        if err:
            raise Exception('\n'.join(err))

        return tuple(res)

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

        self._ml_model = keras.models.load_model(model_to_load)

    def ml_save_model(self, model_path: str):
        self._check_str("model_path", model_path)

        if callable(getattr(self._ml_model, "save", None)):
            self._ml_model.save(model_path)
        else:
            dump(self._ml_model, model_path)

        self._save_ml_path = model_path

    # endregion

