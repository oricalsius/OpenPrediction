from sqlalchemy import (Column, BigInteger, Float, String, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from marshmallow import (Schema, fields, post_load)
from typing import List

# Declare declarative base so we can create easily our tables
_Base = declarative_base()


def get_base():
    return _Base


class Indicator(_Base):
    """
    Create and map table indicators.
    We will have 1 line per indicator with the corresponding timestamp, indicator_name and value.
    We choose this architecture so we can have multiple strategies with different indicators.
    Furthermore, each strategie could be defined as a list of function to apply to a quotation object
    """

    __tablename__ = "indicators"

    timestamp = Column(BigInteger, ForeignKey("quotations.timestamp"), primary_key=True)
    indicator_name = Column(String(60))
    value = Column(Float)

    quotation = relationship("Quotation", uselist=False, back_populates="indicators")

    def __init__(self, timestamp: int, indicator_name: str, value: float):
        self.timestamp = timestamp
        self.indicator_name = indicator_name
        self.value = value


class Quotation(_Base):
    """
    Create and map table quotations
    """

    __tablename__ = "quotations"

    timestamp = Column(BigInteger, primary_key=True)
    amount = Column(Float)
    open = Column(Float)
    close = Column(Float)
    low = Column(Float)
    high = Column(Float)
    vol = Column(Float)

    indicators = relationship("Indicator", uselist=True, back_populates="quotation")

    def __init__(self, timestamp: int, amount: float, open: float, close: float, low: float, high: float,
                 vol: float, c_indicators: List[Indicator] = list()):
        self.timestamp = timestamp
        self.amount = amount
        self.open = open
        self.close = close
        self.low = low
        self.high = high
        self.vol = vol
        self.indicators = c_indicators


class QuotationsSchema(Schema):
    """
    Define marshmallow schema for quotations table
    This schema will serialize from received json to Quotation type
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
        return Quotation(**data)

    class Meta:
        exclude = ["count"]



