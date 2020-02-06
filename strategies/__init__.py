"""
This module is defining all the strategies.
_strategies class variable is storing all the different strategies.

What is a strategy?
A strategy is a list of routines (indicators) to be executed and applied to Ticker object or Quotation object.
"""

from typing import List, Dict, Callable


class Strategy:
    _strategies = {}

    class Routine:
        def __init__(self, func, *args, **kwargs):
            self._routine = func
            self._args = args
            self._kwargs = kwargs

        @property
        def routine(self):
            return self._routine

        @routine.setter
        def routine(self, func):
            raise Exception(
                "Routine is immutable, it cannot be set to another value. Please create a new routine instead")

        @property
        def routine_args(self):
            return self._args

        @routine_args.setter
        def routine_args(self, *args):
            self._args = args

        @property
        def routine_kwargs(self):
            return self._kwargs

        @routine_kwargs.setter
        def routine_kwargs(self, **kwargs):
            self._kwargs = kwargs

    def __init__(self, strategy_name: str, list_routines: List[Routine] = []):
        self.name = strategy_name
        self._list_routines = list_routines

        Strategy._strategies[self.name] = self

    def __del__(self):
        if self.name in Strategy._strategies:
            del Strategy._strategies[self.name]

    @property
    def list_routines(self) -> List[Routine]:
        return self._list_routines

    @list_routines.setter
    def list_routines(self, list_routines: List[Routine]):
        self._list_routines = list_routines
        Strategy._strategies[self.name] = self._list_routines

    @list_routines.deleter
    def list_routines(self):
        self._list_routines = []

    @staticmethod
    def get_all_strategies() -> Dict[str, List[Callable]]:
        return Strategy._strategies

    @staticmethod
    def reset_all_strategies():
        Strategy._strategies = {}


def apply_strategy(strategy_to_apply: Strategy):
    """
    Decorator to apply a strategy to extracted quotations from func(*args, **kwargs).

    :param strategy_to_apply: Object of type Strategy storing the routines to apply.
    :return: Resulted data object after applying all routines in the strategy.
    """
    def maker(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)

            for routine_object in strategy_to_apply.list_routines:
                args_routines = routine_object.routine_args
                kwargs_routines = routine_object.routine_kwargs

                data = routine_object.routine(data, *args_routines, **kwargs_routines)

            return data
        return wrapper
    return maker


