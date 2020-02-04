from exchangeapi.huobi.rest import Rest
from exchangeapi.huobi.models.enumerations import TickerPeriod
from database.connexion import DbConnexion
from database.model import (Quotation, Indicators)
from exchangeapi.huobi.models.ticker import TickerSchema
from database.model import QuotationsSchema
from marshmallow import EXCLUDE
import asyncio
import pprint
import aiohttp
import requests


if __name__ == "__main__":
    api_huobi_rest = Rest("", "", False)
    task = api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 10, QuotationsSchema(many=True,
                                                                                                      unknown=EXCLUDE))

    res = asyncio.run(task)

    db = DbConnexion(None,False)
    db_session = db.get_session()
    all_quotes = db_session.query(Quotation).all()

    quote1 = Quotation(12222023, 46.282, 63.5, 25, 15.2, 1545.0)
    quote2 = Quotation(12222054, 5.884, 2115, 355, 351, 26658)

    ind1 = Indicators(12222023, 225.0)
    ind2 = Indicators(12222054, 136.0)

    db_session.add_all(res)
    db_session.commit()

    print("")




