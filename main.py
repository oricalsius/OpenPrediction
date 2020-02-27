from exchangeapi.huobi.models.enumerations import TickerPeriod
from learning import MultiLayerPerceptronNN
from graphs import display_graph
from exchangeapi.huobi.rest import Rest
from architect import MLArchitect
import pandas as pd
import os

# Setting the environment
api_huobi_rest = Rest("", "", False)


def load_original_data_huobi():
    data = api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 2000, pd.DataFrame)
    data['id'] = pd.to_datetime(data['id'], infer_datetime_format=True, unit='s')
    data.set_index('id', inplace=True)

    return data


def load_original_data_cryptocompare():
    data = api_huobi_rest.get_ticker_history_cryptocompare("BTC", "USDT", to_date="2019-02-01", schema=pd.DataFrame,
                                                           api_key=api_key)
    data['id'] = pd.to_datetime(data['id'], infer_datetime_format=True, unit='s')
    data.set_index('id', inplace=True)

    return data


if __name__ == "__main__":
    window = 4; verbose = True; early_stop = True; load_model = False; save_models = True

    # Define the ML model to use
    ml_model = MultiLayerPerceptronNN(hidden_layer_sizes=(280, 75), activation='logistic',
                                      solver="adam", random_state=0, verbose=verbose,
                                      learning_rate_init=1e-3, early_stopping=early_stop, validation_fraction=0.2,
                                      tol=1e-6, alpha=0.0001, learning_rate="adaptive", max_iter=10000,
                                      batch_size=75, n_iter_no_change=10)

    # Get the exchange data
    original_data = load_original_data_huobi()

    # Initialize the Architect object
    arc = MLArchitect("saved_data/x.csv", "saved_data/y.csv", is_x_flat=False,
                      save_indicators_path="saved_data/indicators.csv", save_y_path="saved_data/y.csv",
                      save_normalize_x_model="saved_data/x_norm_model.mod", y_restoration_routine=None,
                      save_normalize_y_model="saved_data/y_norm_model.mod",
                      index_col='id', index_infer_datetime_format=True,
                      window_prediction=4, test_size=0.01, ml_model=ml_model)

    # Fit the model
    arc.ml_fit()

    score_train = arc.ml_score_train(); score_test = arc.ml_score_test()
    print(f"score train: {score_train}\nscore test: {score_test}")


    # Predict future prices
    y_predicted = arc.ml_predict()


    src_columns = ["open", "close", "high", "low"]
    columns_equivalences = []
    columns_equivalences = [(src_columns, [x + "_log_returns_" + str(i + 1) for x in src_columns], i+1)
                            for i in range(window)]

    #data_with_indicators = pd.concat([data_with_indicators, y_restored], axis=1, sort=False)

    # scatters_plot = dict(zip(y_restored.columns, y_restored.columns))
    # #sub_plots = [{"Count": "count", "QCount1": "QCount1", "QCount95": "QCount95"}, {"Vol": "vol", "QVol95": "QVol95", "QVol1": "QVol1"}]
    # sub_plots = []
    #
    # data_to_display = data_with_indicators.iloc[-600:, :]
    # display_graph(data_to_display, data_index=data_to_display.index, open='open',
    #               close='close', high='high', low='low', scatters_plot=scatters_plot, sub_plots=sub_plots)

    print("")


