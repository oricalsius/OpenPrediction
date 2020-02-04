from ..rest import Rest
from ..models.enumerations import TickerPeriod
from ..models.ticker import Ticker, TickerSchema
from marshmallow import EXCLUDE

import asyncio

api_rest = Rest("", "", False, False)


def test_get_ticker_history():
    result = asyncio.run(api_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 10, TickerSchema(many=True,
                                                                                                      unknown=EXCLUDE)))

    assert len(result) > 0
    assert isinstance(result[0], Ticker)
    assert result[0].close > 0

