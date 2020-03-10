from sklearn import tree, neural_network
from sklearn.metrics import accuracy_score
from joblib import dump, load
from typing import List
import pandas as pd


class DecisionTreeRegressor:
    def __init__(self):
        self._model = tree.DecisionTreeRegressor()

    @staticmethod
    def load_model(model_to_load: str):
        if model_to_load == '':
            raise Exception("model_to_load parameter should not be empty.")

        model = load(model_to_load)
        return model

    def save_model(self, model_path: str):
        dump(self, model_path)

    def score(self, x_test, y_test):
        return self._model.score(x_test, y_test)

    def fit(self, train_x: object, train_y: object):
        self._model = self._model.fit(train_x, train_y)

    def predict(self, x, columns_for_dataframe: List = [], index_for_dataframe: object = None, sort_index: bool = False):
        y = self._model.predict(x)

        if columns_for_dataframe:
            y = pd.DataFrame(y, columns=columns_for_dataframe)

            if index_for_dataframe:
                y.index = index_for_dataframe

            if sort_index:
                y = y.sort_index()

        return y


class MultiLayerPerceptronNN:
    def __init__(self, hidden_layer_sizes: tuple, batch_size: int,
                 max_iter: int = 10000,
                 activation: str = "logistic",
                 solver: str = "adam", random_state: int = 0, verbose: bool = False,
                 learning_rate_init=1e-6, early_stopping=False, validation_fraction=0.1,
                 tol=1e-4, alpha=0.0001, learning_rate="adaptive", n_iter_no_change=10):

        self._model = neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, batch_size=batch_size,
                                                  max_iter=max_iter, tol=tol, alpha=alpha, learning_rate=learning_rate,
                                                  validation_fraction=validation_fraction, activation=activation,
                                                  solver=solver, random_state=random_state, verbose=verbose,
                                                  learning_rate_init=learning_rate_init, early_stopping=early_stopping,
                                                  n_iter_no_change=n_iter_no_change)

    @staticmethod
    def load_model(model_to_load: str):
        if model_to_load == '':
            raise Exception("model_to_load parameter should not be empty.")

        model = load(model_to_load)
        return model

    def save_model(self, model_path: str):
        dump(self, model_path)

    def score(self, x_test, y_test):
        return self._model.score(x_test, y_test)

    def fit(self, train_x: object, train_y: object):
        self._model = self._model.fit(train_x, train_y)

    def get_params(self, deep=True):
        return self._model.get_params(deep)

    def set_params(self, **params):
        return self._model.set_params(**params)

    def predict(self, x, columns_for_dataframe: List = [], index_for_dataframe: object = None, sort_index: bool = False):
        y = self._model.predict(x)

        if columns_for_dataframe is not None and len(columns_for_dataframe) > 0:
            y = pd.DataFrame(y, columns=columns_for_dataframe)

            if index_for_dataframe is not None and len(index_for_dataframe) > 0:
                y.index = index_for_dataframe

            if sort_index:
                y = y.sort_index()

        return y


























