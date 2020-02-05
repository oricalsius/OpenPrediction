from exchangeapi.huobi.rest import Rest
from exchangeapi.huobi.models.enumerations import TickerPeriod
from database.connexion import DbConnexion
from database.model import (Quotation, Indicator)
from exchangeapi.huobi.models.ticker import TickerSchema
from database.model import QuotationsSchema
from marshmallow import EXCLUDE
import asyncio
import pprint
import aiohttp
import requests
import sys

if __name__ == "__main__":
    api_huobi_rest = Rest("", "", False)
    task = api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 2000, QuotationsSchema(many=True,
                                                                                                      unknown=EXCLUDE))

    res = asyncio.run(task)
    sys.exit(0)



