from exchangeapi.huobi.rest import Rest
from exchangeapi.huobi.models.enumerations import TickerPeriod
from database.connexion import DbConnexion
from database.model import QuotationsSchema
from strategies import Strategy, apply_strategy
import strategies.indicators.definitions as ind
from marshmallow import EXCLUDE
import pandas as pd
import asyncio
import sys


# Setting the environment
api_huobi_rest = Rest("", "", False)
db_connexion = DbConnexion()
algo_strategy = Strategy("first_strategy")


@apply_strategy(algo_strategy)
def get_process_quotations(symbol: str, period: TickerPeriod, size: int = 1):
    task = api_huobi_rest.get_ticker_history(symbol, period, size)
    result = asyncio.run(task)

    # Transform the json dict to pandas.DataFrame so we can apply our standard routines in the strategy
    return pd.DataFrame(result)


if __name__ == "__main__":
    # Defining the list of routines to apply as a strategy
    list_routines = []
    list_routines.append(Strategy.Routine(ind.moving_average, "close", 20, name="MA (20, Close)", add_to_data=True))
    list_routines.append(Strategy.Routine(ind.average_true_range))

    algo_strategy.list_routines.extend(list_routines)

    del algo_strategy

    #QuotationsSchema(many=True, unknown=EXCLUDE)
    res = get_process_quotations("btcusdt", TickerPeriod.Hour4, 2000)
    sys.exit(0)



