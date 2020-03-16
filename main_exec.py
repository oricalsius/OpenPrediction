from exchangeapi.huobi.models.enumerations import TickerPeriod
from graphs import display_graph
from exchangeapi.huobi.rest import Rest
from architect import MLArchitect
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import explained_variance_score,  make_scorer
from sklearn import neural_network
import os


# Setting the environment
api_huobi_rest = Rest("", "", False)


def load_original_data_huobi(path: str = None):
    if path is not None and os.path.isfile(path):
        data = pd.read_csv(path)
        data['id'] = pd.to_datetime(data['id'], infer_datetime_format=True)
        save = False

    else:
        data = api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 2000, pd.DataFrame)
        data['id'] = pd.to_datetime(data['id'], infer_datetime_format=True, unit='s')

        if path is not None and not data.empty:
            save = True

    data.set_index('id', inplace=True)

    if save:
        data.to_csv(path)

    return data


def load_original_data_cryptocompare():
    data = api_huobi_rest.get_ticker_history_cryptocompare("BTC", "USDT", to_date="2019-02-01", schema=pd.DataFrame,
                                                           api_key=api_key)
    data['id'] = pd.to_datetime(data['id'], infer_datetime_format=True, unit='s')
    data.set_index('id', inplace=True)

    return data


def run_main():
    window = 4;
    verbose = True;
    early_stop = True;
    tick_to_display = 1000;

    # Get the exchange data
    exchange_data = load_original_data_huobi("saved_data/data.csv")

    # Initialize the Architect object
    arc = MLArchitect(x_data=exchange_data, y_data=None, is_x_flat=True, save_x_path="saved_data/x.csv",
                      save_indicators_path="saved_data/indicators.csv", save_y_path="saved_data/y.csv",
                      save_normalize_x_model="saved_data/x_norm_model.mod", y_restoration_routine="default",
                      save_normalize_y_model="saved_data/y_norm_model.mod",
                      index_col='id', index_infer_datetime_format=True,
                      pca_reductions=[('linear', 0.99)],
                      window_prediction=4, test_size=0.20, ml_model=None, save_ml_path="saved_data/ml_model.mod",
                      is_parallel=False, disgard_last=True, window_tuple=(5, 14, 22, 35))

    ml_model = neural_network.MLPRegressor(hidden_layer_sizes=(350, 110), activation='logistic',
                                           solver="adam", random_state=200, verbose=verbose,
                                           learning_rate_init=1e-3, early_stopping=early_stop, validation_fraction=0.20,
                                           tol=1e-6, alpha=0.001, learning_rate="adaptive", max_iter=10000,
                                           batch_size=75, n_iter_no_change=10, warm_start=False)

    # Init the ML model
    arc.ml_init_model(ml_model)

    # Define the ML model to use
    x, y = arc.get_normalized_data(arc.x, arc.y, None, arc.index_min, arc.index_max)
    x.to_csv("saved_data/x_norm.csv"); y.to_csv("saved_data/y_norm.csv")
    nb_layers = x.shape

    # Fit the model
    arc.fit()

    score_train = arc.ml_score_train(); score_test = arc.ml_score_test()
    print(f"score train: {score_train}\nscore test: {score_test}")

    # Grid Search
    param_grid = [
        {'random_state': randint(0, 10001).rvs(10, 0),
         'batch_size': np.arange(50, int(nb_layers[0] / 10), 50),
         'hidden_layer_sizes': [(x, y) for x in np.arange(nb_layers[1], nb_layers[1]*12, int(nb_layers[1]/2))
                                for y in np.arange(1, int(nb_layers[1]*4), int(nb_layers[1]*0.5))],
         }
    ]

    ts = TimeSeriesSplit(n_splits=5)
    grid = arc.sklearn_gridsearchcv(param_grid=param_grid, cv=ts, n_jobs=-1,
                            scoring=make_scorer(explained_variance_score), verbose=1, n=0.2)

    # Predict future prices
    y_predicted = arc.ml_predict(restore_y=True, original_data=exchange_data, windows=window, is_returns=True)

    # Get all data
    data_with_indicators = pd.concat([arc.data_with_indicators, y_predicted], axis=1, sort=False)

    # Display graphs
    display(data_with_indicators, y_predicted, tick_to_display, arc.x_test.index)


def display(data_with_indicators, y_predicted, tick_to_display, test_data_index):
    scatters_plot = dict(zip(y_predicted.columns, y_predicted.columns))

    src_columns = ["open", "close", "high", "low"]
    sub_plots = []

    data_to_display = data_with_indicators.iloc[-tick_to_display:, :]
    data_index_cyan = (test_data_index & data_to_display.index).sort_values()

    display_graph(data_to_display, scatters_plot=scatters_plot, sub_plots=sub_plots, data_index_cyan=data_index_cyan)


if __name__ == "__main__":
    run_main()
    print("")


