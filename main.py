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
from graphs import display_graph


# Setting the environment
api_huobi_rest = Rest("", "", False)
db_connexion = DbConnexion()
algo_strategy = Strategy("first_strategy")


@apply_strategy(algo_strategy)
def get_process_quotations(symbol: str, period: TickerPeriod, size: int = 1):
    result = asyncio.run(api_huobi_rest.get_ticker_history(symbol, period, size))

    # Transform the json dict to pandas.DataFrame so we can apply our standard routines in the strategy
    df = pd.DataFrame(result)
    df.rename(columns={"id": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, unit='s')
    df.set_index("timestamp", inplace=True)
    df = df[::-1]
    return df


# Defining the list of routines to apply as a strategy
def set_list_routines(list_routines):
    list_routines.append(Strategy.Routine(ind.true_range, window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.moving_average, columns=["open", "close"],
                                          window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.moving_window_functions, columns=["open", "close", "high"],
                                          functions=["mean", "std", "var"], window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.moving_window_functions, columns=["low", "vol", "count"],
                                          functions=["mean", "std", "var"], window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.exponential_weighted_functions, columns=["open", "close"],
                                          functions=["mean", "std", "var"], span=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.exponential_weighted_functions, columns=["high", "low"],
                                          functions=["mean", "std", "var"], span=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.moving_window_functions,
                                          columns=["open", "close", "high", "low", "vol", "count"],
                                          functions=["quantile"], quantile=0.01, window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.moving_window_functions,
                                          columns=["open", "close", "high", "low", "vol", "count"],
                                          functions=["quantile"], quantile=0.95, window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.average_true_range, close_name="close", high_name="high", low_name="low",
                                          window=14, add_to_data=True))

    list_routines.append(Strategy.Routine(ind.average_directional_index, close_name="close", high_name="high",
                                          low_name="low", window=14, columns_label="ADX_14", add_to_data=True))


if __name__ == "__main__":
    set_list_routines(algo_strategy.list_routines)
    res = get_process_quotations("btcusdt", TickerPeriod.Hour4, 2000)

    # display_graph(res.iloc[-600:, :], data_index=res.index, open='open', close='close', high='high', low='low',
    #               QClose1="Moving_quantile_14_0.01_close", QHigh1="Moving_quantile_14_0.01_high",
    #               QLow1="Moving_quantile_14_0.01_low", QClose95="Moving_quantile_14_0.95_close",
    #               QHigh95="Moving_quantile_14_0.95_high", QLow95="Moving_quantile_14_0.95_low")

    scatters_plot = {"QClose1": "Moving_quantile_14_0.01_close"}
    sub_plots = [{"ATR": "ATR_14"}, {"DIp_14": "DIp_14", "DIn_14": "DIn_14", "DXI_14": "DXI_14", "ADX_14": "ADX_14"}]

    display_graph(res.iloc[-1000:, :], data_index=res.index, open='open', close='close', high='high', low='low',
                  scatters_plot=scatters_plot, sub_plots=sub_plots)

    sys.exit(0)



