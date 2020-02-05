"""
This module is defining all the strategies.
_strategies module variable is storing all the different strategies.

What is a strategy?
A strategy is a list of functions (indicators) to be executed and applied to Ticker object or Quotation object.
"""

from typing import List, Dict, Callable
from .indicators.definitions import *

_strategies = {"0": []}


def get_all_strategies() -> Dict[str, List[Callable]]:
    return _strategies


def get_strategy(strategy_name: str) -> List[Callable]:
    if strategy_name in _strategies:
        return _strategies[strategy_name]
    else:
        return []


