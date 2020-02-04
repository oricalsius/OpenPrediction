from sqlalchemy import (Column, BigInteger, Float, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from marshmallow import (Schema, fields, post_load)

# Declare declarative base so we can create easily our tables
_Base = declarative_base()


def get_base():
    return _Base


# Create and map table indicators
class Indicators(_Base):
    __tablename__ = "indicators"

    timestamp = Column(BigInteger, ForeignKey("quotations.timestamp"), primary_key=True)
    avg20 = Column(Float)

    quotation = relationship("Quotation", uselist=False, back_populates="indicators")

    def __init__(self, timestamp: int, avg20: float):
        self.timestamp = timestamp
        self.avg20 = avg20


class Quotation(_Base):
    """
    Create and map table quotations
    """
    __tablename__ = "quotations"

    timestamp = Column(BigInteger, primary_key=True)
    open = Column(Float)
    close = Column(Float)
    low = Column(Float)
    high = Column(Float)
    vol = Column(Float)

    indicators = relationship("Indicators", uselist=False, back_populates="quotation")

    def __init__(self, timestamp: int, open: float, close: float, low: float, high: float,
                 vol: float, c_indicators: Indicators = None):
        self.timestamp = timestamp
        self.open = open
        self.close = close
        self.low = low
        self.high = high
        self.vol = vol
        self.indicators = c_indicators


class QuotationsSchema(Schema):
    """
    Define marshmallow schema for quotations table
    This schema will serialize from received json to Quotations type
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
        exclude = ["amount", "count"]



