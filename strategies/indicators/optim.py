"""
Module defining optimized function with numba.
"""

from numba import jit, njit
from numba.misc.special import prange
from typing import Union

import numpy as np

eps = np.finfo(float).eps


@njit(parallel=True)
def parallel_numpy_add(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    return d1 + d2


@njit
def noparallel_numpy_add(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    return d1 + d2


def numpy_add(is_parallel: bool, d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_add(d1, d2)
    else:
        return noparallel_numpy_add(d1, d2)


@njit(parallel=True)
def parallel_numpy_sub(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    return d1 - d2


@njit
def noparallel_numpy_sub(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    return d1 - d2


def numpy_sub(is_parallel: bool, d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_sub(d1, d2)
    else:
        return noparallel_numpy_sub(d1, d2)


@njit(parallel=True)
def parallel_numpy_mul(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    return d1 * d2


@njit
def noparallel_numpy_mul(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    return d1 * d2


def numpy_mul(is_parallel: bool, d1: Union[np.ndarray, object], d2: Union[np.ndarray, object]) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_mul(d1, d2)
    else:
        return noparallel_numpy_mul(d1, d2)


@njit(parallel=True)
def parallel_numpy_div(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object],
                       with_eps: bool = False) -> np.ndarray:
    if with_eps:
        return d1 / (d2 + eps)
    else:
        return d1 / d2


@njit
def noparallel_numpy_div(d1: Union[np.ndarray, object], d2: Union[np.ndarray, object],
                         with_eps: bool = False) -> np.ndarray:
    if with_eps:
        return d1 / (d2 + eps)
    else:
        return d1 / d2


def numpy_div(is_parallel: bool, d1: Union[np.ndarray, object], d2: Union[np.ndarray, object],
              with_eps: bool = False) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_div(d1, d2, with_eps)
    else:
        return noparallel_numpy_div(d1, d2, with_eps)


@njit(parallel=True)
def parallel_numpy_log(df: Union[np.ndarray, object]) -> np.ndarray:
    return np.log(df)


@njit
def noparallel_numpy_log(df: Union[np.ndarray, object]) -> np.ndarray:
    return np.log(df)


def numpy_log(is_parallel: bool, df: Union[np.ndarray, object]) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_log(df)
    else:
        return noparallel_numpy_log(df)


@njit(parallel=True)
def parallel_numpy_abs(df: Union[np.ndarray, object]) -> np.ndarray:
    return np.abs(df)


@njit
def noparallel_numpy_abs(df: Union[np.ndarray, object]) -> np.ndarray:
    return np.abs(df)


def numpy_abs(is_parallel: bool, df: Union[np.ndarray, object]) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_abs(df)
    else:
        return noparallel_numpy_abs(df)


@njit(parallel=True)
def parallel_numpy_exp(df: Union[np.ndarray, object]) -> np.ndarray:
    return np.exp(df)


@njit
def noparallel_numpy_exp(df: Union[np.ndarray, object]) -> np.ndarray:
    return np.exp(df)


def numpy_exp(is_parallel: bool, df: Union[np.ndarray, object]) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_exp(df)
    else:
        return noparallel_numpy_exp(df)

