from sklearn import preprocessing, decomposition
from typing import List, Any
#import pandas as pd
#import numpy as np


class DataPreProcessing:
    """
    Class to preprocess data before input them in the machine learning algorithm
    """

    def __init__(self, data: object, min_max_scale_columns: List[str] = None, min_max_range: tuple = (0, 1),
                 standard_scale_columns: List[str] = None, gaussian_like_method: str = "yeo-johnson",
                 gaussian_like_scale_columns: List[str] = None, reduce_dimension: bool = True,
                 n_components: str = None, inplace=False):

        self._transformed_data = data
        self._inplace = inplace

        # Min Max Scale
        self._min_max_scale_columns = min_max_scale_columns
        self._min_max_range = min_max_range
        self._MinMaxScaler = None
        self._scale_min_max()

        # Standard Scale
        self._standard_scale_columns = standard_scale_columns
        self._StandardScalar = None
        self._scale_standard()

        # Gaussian like Scale
        self._gaussian_like_method = gaussian_like_method
        self._gaussian_like_scale_columns = gaussian_like_scale_columns
        self._PowerTransform = None
        self._scale_gaussian_like()

        # Reduce dimension with PCA decomposition method
        self._reduce_dimension = reduce_dimension
        self._n_components = n_components
        self._pca = None
        self._pca_transform_data = None
        self._pca_decompose()

    def _scale_min_max(self):
        if self._min_max_scale_columns:
            self._MinMaxScaler = preprocessing.MinMaxScaler(feature_range=self._min_max_range, copy=not self._inplace)
            self._MinMaxScaler.fit(self._transformed_data[self._min_max_scale_columns])
            self._transformed_data[self._min_max_scale_columns] = self._MinMaxScaler.transform(
                self._transformed_data[self._min_max_scale_columns])

    def _scale_standard(self):
        if self._standard_scale_columns:
            self._StandardScalar = preprocessing.StandardScaler(copy=not self._inplace)
            self._StandardScalar.fit(self._transformed_data[self._standard_scale_columns])
            self._transformed_data[self._standard_scale_columns] = self._StandardScalar.transform(
                self._transformed_data[self._standard_scale_columns])

    def _scale_gaussian_like(self):
        if self._gaussian_like_scale_columns:
            self._PowerTransform = preprocessing.PowerTransformer(method=self._gaussian_like_method, standardize=True,
                                                                  copy=not self._inplace)
            self._PowerTransform.fit(self._transformed_data[self._gaussian_like_scale_columns])
            self._transformed_data[self._gaussian_like_scale_columns] = self._PowerTransform.transform(
                self._transformed_data[self._gaussian_like_scale_columns])

    def _pca_decompose(self):
        # n_components can also be 'mle' or a number in [0,1]
        if self._reduce_dimension:
            self._pca = decomposition.PCA(n_components=self._n_components, svd_solver="full", copy=not self._inplace)
            self._pca.fit(self._transformed_data)
            self._pca_transform_data = self._pca.transform(self._transformed_data)
