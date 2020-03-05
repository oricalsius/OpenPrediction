"""
Module defining optimized function with numba.
"""

from numba import jit, njit
from numba.special import prange

import numpy as np


@njit(parallel=True)
def parallel_numpy_add(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 + d2


@njit
def noparallel_numpy_add(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 + d2


def numpy_add(is_parallel: bool, d1: np.ndarray, d2: object) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_add(d1, d2)
    else:
        return noparallel_numpy_add(d1, d2)


@njit(parallel=True)
def parallel_numpy_sub(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 - d2


@njit
def noparallel_numpy_sub(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 - d2


def numpy_sub(is_parallel: bool, d1: np.ndarray, d2: object) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_sub(d1, d2)
    else:
        return noparallel_numpy_sub(d1, d2)


@njit(parallel=True)
def parallel_numpy_mul(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 * d2


@njit
def noparallel_numpy_mul(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 * d2


def numpy_mul(is_parallel: bool, d1: np.ndarray, d2: object) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_mul(d1, d2)
    else:
        return noparallel_numpy_mul(d1, d2)


@njit(parallel=True)
def parallel_numpy_div(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 / d2


@njit
def noparallel_numpy_div(d1: np.ndarray, d2: object) -> np.ndarray:
    return d1 / d2


def numpy_div(is_parallel: bool, d1: np.ndarray, d2: object) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_div(d1, d2)
    else:
        return noparallel_numpy_div(d1, d2)


@njit(parallel=True)
def parallel_numpy_log(df: np.ndarray) -> np.ndarray:
    return np.log(df)


@njit
def noparallel_numpy_log(df: np.ndarray) -> np.ndarray:
    return np.log(df)


def numpy_log(is_parallel: bool, df: np.ndarray) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_log(df)
    else:
        return noparallel_numpy_log(df)


@njit(parallel=True)
def parallel_numpy_abs(df: np.ndarray) -> np.ndarray:
    return np.abs(df)


@njit
def noparallel_numpy_abs(df: np.ndarray) -> np.ndarray:
    return np.abs(df)


def numpy_abs(is_parallel: bool, df: np.ndarray) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_abs(df)
    else:
        return noparallel_numpy_abs(df)


@njit(parallel=True)
def parallel_numpy_exp(df: np.ndarray) -> np.ndarray:
    return np.exp(df)


@njit
def noparallel_numpy_exp(df: np.ndarray) -> np.ndarray:
    return np.exp(df)


def numpy_exp(is_parallel: bool, df: np.ndarray) -> np.ndarray:
    if is_parallel:
        return parallel_numpy_exp(df)
    else:
        return noparallel_numpy_exp(df)

