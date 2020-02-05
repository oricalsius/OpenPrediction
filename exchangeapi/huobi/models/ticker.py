from typing import (List, Union, Dict, Any)
from marshmallow import (Schema, fields, post_load)


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