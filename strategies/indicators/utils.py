"""Module to define utilities"""
from typing import List, Union


def _get_columns(columns: Union[List[str], str]):
    if isinstance(columns, list):
        source_columns_name = columns
    elif isinstance(columns, str):
        source_columns_name = [columns]
    else:
        raise Exception("Parameter columns should be an str or list of str.")

    return source_columns_name
