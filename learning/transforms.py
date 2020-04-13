from sklearn import preprocessing, decomposition
from sklearn import model_selection
from joblib import dump, load
from typing import List, Union
import pandas as pd
import numpy as np
#import numpy as np


class Normalization:
    """
    Class to preprocess data before input them in the machine learning algorithm
    """

    def __init__(self, model_to_load: str = ""):
        """
        Initialize Normalization object.
        :param model_to_load: A saved file path to load to initialize the model.
        """

        self._normalization_steps = []
        self._pca = []

        if model_to_load != '':
            model = load(model_to_load)
            self._normalization_steps = model.get_normalization_steps
            self._pca = model.get_pca_model

    @staticmethod
    def train_test_split(*arrays, **options):
        """
        Split arrays or matrices into random train and test subsets.

        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        :param arrays: Arrays to split.
        :param options: Same options as for sklearn.model_selection.train_test_split
        :return:
        """

        return model_selection.train_test_split(*arrays, **options)

    @property
    def get_normalization_steps(self):
        return self._normalization_steps

    @property
    def get_pca_models(self):
        return self._pca

    def save_model(self, model_path: str):
        dump(self, model_path)

    def add_min_max_scaling(self, data: object, min_max_scale_columns: List[str] = None, min_max_range: tuple = (0, 1)):
        """
        Transform features by scaling each feature to a given range.
        This estimator scales and translates each feature individually such that it is in the given range on
        the training set, e.g. between zero and one.

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

        :param data: Input data to fit the model.
        :param min_max_scale_columns: Columns to which we apply the scaling.
        :param min_max_range: Desired range of transformed data.
        :return:
        """

        scale = preprocessing.MinMaxScaler(feature_range=min_max_range)

        self._normalization_steps.append((scale, min_max_scale_columns))

    def add_standard_scaling(self, data: object, standard_scale_columns: List[str] = None):
        """
        Standardize features by removing the mean and scaling to unit variance.

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        :param data: Input data to fit the model.
        :param standard_scale_columns: Columns to which we apply the scaling.
        :return:
        """

        scale = preprocessing.StandardScaler()

        self._normalization_steps.append((scale, standard_scale_columns))

    def add_power_transform_scaling(self, data: object, gaussian_like_scale_columns: List[str] = None,
                                    gaussian_like_method: str = "yeo-johnson", standardize=True):
        """
        Apply a power transform featurewise to make data more Gaussian-like.

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html

        :param data: Input data to fit the model.
        :param gaussian_like_scale_columns: Columns to which we apply the scaling.
        :param gaussian_like_method: yeo-johnson or box-cox
        :param standardize: Set to True to apply zero-mean, unit-variance normalization to the output.
        :return:
        """
        scale = preprocessing.PowerTransformer(method=gaussian_like_method, standardize=standardize)

        self._normalization_steps.append((scale, gaussian_like_scale_columns))

    def fit(self, data: object):
        """
        Fit each scaling step from self._normalization_steps to data.

        Fit pca reduction also if defined.

        :param data: Input data to fit the model.
        :return:
        """

        transformed_data = data.copy()

        for scale, columns_to_scale in self._normalization_steps:
            if columns_to_scale:
                scale = scale.fit(transformed_data[columns_to_scale])
                df = pd.DataFrame(scale.transform(transformed_data[columns_to_scale]), columns=columns_to_scale)
                df.index = transformed_data.index
                transformed_data[columns_to_scale] = df

            else:
                scale = scale.fit(transformed_data)
                df = pd.DataFrame(scale.transform(transformed_data), columns=data.columns)
                df.index = data.index
                transformed_data = df

        self.pca_reduction_fit(transformed_data)

    def transform(self, data: object):
        """
        Apply each scaling step from self._normalization_steps to data.

        Apply pca reduction also if defined.

        :param data: Input data to transform.
        :return:
        """

        transformed_data = data.copy()

        for scale, columns_to_scale in self._normalization_steps:
            if columns_to_scale:
                df = pd.DataFrame(scale.transform(transformed_data[columns_to_scale]), columns=columns_to_scale,
                                  index=transformed_data.index)
                transformed_data[columns_to_scale] = df

            else:
                df = pd.DataFrame(scale.transform(transformed_data), columns=data.columns, index=data.index)
                transformed_data = df

        return self.pca_reduction_transform(transformed_data)

    def inverse_transform(self, transformed_data: object):
        """
        Apply the inverse of each scaling step from self._normalization_steps to data.

        Apply the inverse of pca reduction also if defined.

        :param transformed_data: Input transform to inverse.
        :return:
        """

        inversed_data = self.pca_reduction_inverse_transform(transformed_data.copy())

        for scale, columns_to_scale in reversed(self._normalization_steps):
            if columns_to_scale:
                df = pd.DataFrame(scale.inverse_transform(inversed_data[columns_to_scale]), columns=columns_to_scale,
                                  index=inversed_data.index)
                inversed_data[columns_to_scale] = df

            else:
                df = pd.DataFrame(scale.inverse_transform(inversed_data), columns=transformed_data.columns,
                                  index=transformed_data.index)
                inversed_data = df

        return inversed_data

    def add_pca_reduction(self, n_components: int = None, **kwargs):
        # n_components can also be 'mle' or a number in [0,1]
        self._pca.append(decomposition.PCA(n_components=n_components, **kwargs))

    def add_kernel_pca_reduction(self, n_components: int = None,
                                 kernel: str = Union['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'],
                                 gamma: float = None, degree: int = 3, coef0: float = 1, n_jobs: int = -1,
                                 remove_zero_eig: bool = False, fit_inverse_transform: bool = False, **kwargs):
        # n_components can also be 'mle' or a number in [0,1]
        self._pca.append(decomposition.KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree,
                                                 coef0=coef0, fit_inverse_transform=fit_inverse_transform, n_jobs=-1,
                                                 remove_zero_eig=remove_zero_eig, **kwargs))

    def pca_reduction_fit(self, data: object):
        if self._pca:
            transformed_data = data.copy()

            for reduction in self._pca:
                reduction.fit(transformed_data)
                transformed_data = reduction.transform(transformed_data)

    def pca_reduction_transform(self, data: object):
        if self._pca:
            transformed_data = data.copy()

            for reduction in self._pca:
                transformed_data = reduction.transform(transformed_data)

            return pd.DataFrame(transformed_data, index=data.index)

        else:
            return data

    def pca_reduction_fit_transform(self, data: object):
        self.pca_reduction_fit(data)
        return self.pca_reduction_transform(data)

    def pca_reduction_inverse_transform(self, transformed_data: object):
        if self._pca:
            inversed_data = transformed_data.copy()

            for reduction in self._pca:
                inversed_data = reduction.inverse_transform(inversed_data)

            return pd.DataFrame(inversed_data, index=transformed_data.index)
        else:
            return transformed_data




