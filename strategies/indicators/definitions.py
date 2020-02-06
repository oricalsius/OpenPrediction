"""
This module is defining all the functions that are necessary to create indicators.
Input for each function is a Pandas Dataframe or directly the Ticker or Quotation object
"""

from typing import List
import pandas as pd
import numpy as np


def average_directional_index(data: pd.DataFrame) -> pd.DataFrame:
    return data


def average_true_range(data: pd.DataFrame) -> pd.DataFrame:
    return data


def moving_average(data: pd.DataFrame, column: str, length: int = 20, name="", add_to_data=False) -> pd.DataFrame:
    return data


def moving_average_channel(data: pd.DataFrame) -> pd.DataFrame:
    return data


def moving_average_weighted(data: pd.DataFrame) -> pd.DataFrame:
    return data


def moving_average_exponential(data: pd.DataFrame) -> pd.DataFrame:
    return data


def hull_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data


def smoothed_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data


def arnaud_legoux_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data


def least_squares_moving_average(data: pd.DataFrame) -> pd.DataFrame:
    return data

