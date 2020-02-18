"""
This module is defining all the functions that are necessary to preprocess indicators.
Input for each function is a Pandas Dataframe.
"""

from typing import List, Union
from .utils import _get_columns
import pandas as pd
import numpy as np


class ProcessingIndicators:
    """
    Class containing all the definitions of built-in function to preprocess indicators before sending them to the
    machine learning algorithm.
    """

    def __init__(self, data: pd.DataFrame):
        if data is None:
            raise Exception("ProcessingIndicators object must be initialized with a valid pandas.DataFrame object.")

        self._data = data

    @property
    def data(self):
        return self._data

    def substract(self, left_columns: Union[List[str], str], right_columns: Union[List[str], str],
                  result_names: Union[List[str], str], add_to_data: bool = True, delete_left_from_data: bool = False,
                  delete_right_from_data: bool = False) -> pd.DataFrame:

        left_col, right_col, res_names = _get_columns(left_columns), _get_columns(right_columns), _get_columns(result_names)

        if len(left_col) != len(right_col) or len(result_names) != len(right_col):
            raise Exception("Parameters left_columns and right_columns should have the same length.")

        df = pd.DataFrame(self.data[left_col].values - self.data[right_col].values, columns=res_names)
        df.index = self.data.index

        drop_columns = []
        if delete_left_from_data:
            drop_columns.extend(left_col)

        if delete_right_from_data:
            drop_columns.extend(right_col)

        if add_to_data:
            self.data[res_names] = df

            if drop_columns:
                self.data.drop(columns=drop_columns, inplace=True)

            return self.data
        else:
            return df

    def log_returns(self, columns: Union[List[str], str], result_names: Union[List[str], str], window: int = 1,
                    add_to_data: bool = True, delete_columns: bool = False) -> pd.DataFrame:

        src_columns, target_columns_names = _get_columns(columns), _get_columns(result_names)

        if len(src_columns) != len(target_columns_names):
            raise Exception("Parameters columns and result_names should have the same length.")

        df = np.log(self.data[src_columns] / self.data[src_columns].shift(periods=window))

        if add_to_data:
            self.data[result_names] = df

            if delete_columns:
                self.data.drop(columns=src_columns, inplace=True)

            return self.data
        else:
            return df

    def pct_change(self, columns: Union[List[str], str], result_names: Union[List[str], str], window: int = 1,
                   add_to_data: bool = True, delete_columns: bool = False) -> pd.DataFrame:

        src_columns, target_columns_names = _get_columns(columns), _get_columns(result_names)

        if len(src_columns) != len(target_columns_names):
            raise Exception("Parameters columns and result_names should have the same length.")

        df = self.data[src_columns].pct_change(periods=window)

        if add_to_data:
            self.data[result_names] = df

            if delete_columns:
                self.data.drop(columns=src_columns, inplace=True)

            return self.data
        else:
            return df


