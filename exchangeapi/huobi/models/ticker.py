from typing import (List, Union, Dict, Any)
from marshmallow import (Schema, fields, post_load, EXCLUDE)

import pandas as pd
import numpy as np


class Ticker:
    """Represents one ticker information"""

    def __init__(self, timestamp: int, open: float, close: float, low: float, high: float,
                 vol: float):
        self.timestamp = timestamp
        self.open = open
        self.close = close
        self.low = low
        self.high = high
        self.vol = vol


class TickerSchema(Schema):
    """
    Define marshmallow schema for Ticker object
    This schema will serialize from received json to Ticker type
    """

    timestamp = fields.Int(data_key="id")
    amount = fields.Float()
    count = fields.Integer()
    open = fields.Float()
    close = fields.Float()
    low = fields.Float()
    high = fields.Float()
    vol = fields.Float()

    @post_load
    def deserialize_ticker(self, data, **kwargs):
        return Ticker(**data)

    class Meta:
        exclude = ["amount", "count"]


class PandasNestedSchema(Schema):
    """
    Define marshmallow schema for Pandas DataFrame
    This schema will serialize from received json to Ticker type
    """

    id = fields.Int()
    time = fields.Int()

    amount = fields.Float()
    count = fields.Integer()
    vol = fields.Float()

    open = fields.Float()
    close = fields.Float()
    low = fields.Float()
    high = fields.Float()

    volumefrom = fields.Float(allow_none=True)
    volumeto = fields.Float(allow_none=True)

    @post_load
    def deserialize_ticker(self, data, **kwargs):
        id_col = 'id'
        df = pd.DataFrame([data])

        if not df.empty:
            if 'time' in df:
                df.rename(columns={'time': 'id', 'volumefrom': 'amount', 'volumeto': 'vol'}, inplace=True)

            df[id_col] = pd.to_datetime(df[id_col], infer_datetime_format=True, unit='s')
            df.set_index(id_col, inplace=True)

        return df

    class Meta:
        exclude = ["count"]


class PandasGlobalSchema(Schema):
    """
    Define marshmallow schema for Ticker object
    This schema will serialize from received json to Ticker type
    """

    # Specify that we want to have as input the overall list
    _global_schema_key = 'data'

    Data = fields.List(fields.Dict, data_key=_global_schema_key)

    @property
    def global_schema_key(self):
        return self._global_schema_key

    @post_load
    def deserialize_ticker(self, data, **kwargs):
        id_col = 'id'
        col_to_drop = ['conversionType', 'conversionSymbol']
        df = pd.DataFrame(data['Data'])

        if not df.empty:
            if 'time' in df:
                df.rename(columns={'time': 'id', 'volumefrom': 'amount', 'volumeto': 'vol'}, inplace=True)
                df.drop(columns=list(df.columns & col_to_drop), inplace=True)

            df.drop_duplicates(subset=id_col, keep="first", inplace=True)
            df[id_col] = pd.to_datetime(df[id_col], infer_datetime_format=True, unit='s')
            df.set_index(id_col, inplace=True)
            df.sort_index(ascending=True, inplace=True)

            with np.errstate(invalid='ignore'):
                df = df.loc[df.index[np.greater(df.values, 0).max(axis=1)]]

        return df

    class Meta:
        exclude = []


class TickerAggregated(Ticker):
    """Represents an aggregated data for tickers with some important 24h aggregated market data"""

    def __init__(self, timestamp: int, open_price: float, close: float, low: float, high: float,
                 vol: float, id_num: int, bid: List[float], ask: List[float]):
        super().__init__(timestamp, open_price, close, low, high, vol)
        self.id = id_num
        self.bid = bid
        self.ask = ask


class TickerAggregatedSchema(Schema):
    """
    Define marshmallow schema for TickerAggregated object
    This schema will serialize from received json to TickerAggregated type
    """

    timestamp = fields.Int(data_key="ts")
    open = fields.Float()
    close = fields.Float()
    low = fields.Float()
    high = fields.Float()
    vol = fields.Float()

    id = fields.Int()
    bid = fields.List(fields.Float())
    ask = fields.List(fields.Float())

    @post_load
    def deserialize_ticker_aggregated(self, data, **kwargs):
        return TickerAggregated(**data)

    class Meta:
        exclude = ["amount", "count", "version"]