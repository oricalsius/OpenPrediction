from exchangeapi.huobi.models.enumerations import TickerPeriod
from graphs import display_graph
from exchangeapi.huobi.rest import Rest
from architect import MLArchitect
from exchangeapi.huobi.models.ticker import PandasGlobalSchema
from marshmallow import EXCLUDE

import pandas as pd
#import numpy as np
import keras
import os
import tensorflow as tf
#import datetime




# Setting the environment
api_huobi_rest = Rest("", "", False)


def set_tf_gpu(set_log_device: bool = False):
    tf.debugging.set_log_device_placement(set_log_device)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    tf.config.set_soft_device_placement(True)
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU,", "IsEager" if tf.executing_eagerly() else '')


def get_tf_gpu_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def load_data(path: str = None, exchange="huobi"):
    columns_equivalences = dict()

    if exchange == "huobi":
        columns_equivalences = {'index': 'id'}
    elif exchange == "cryptocompare":
        columns_equivalences = {'index': 'id'}

    if path is not None and os.path.isfile(path):
        data = pd.read_csv(path)
        data['id'] = pd.to_datetime(data['id'], infer_datetime_format=True)
        data.set_index('id', inplace=True)

    else:
        if exchange == "huobi":
            data = load_from_huobi()
        elif exchange == "cryptocompare":
            data = load_from_cryptocompare()
        else:
            raise Exception("No exchange has been chosen.")

        if path is not None and not data.empty:
            data.to_csv(path)

    return data, columns_equivalences


def load_from_huobi():
    data = api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 2000, PandasGlobalSchema(unknown=EXCLUDE))

    return data


def load_from_cryptocompare():
    api_key = "###########"
    data = api_huobi_rest.get_ticker_history_cryptocompare("BTC", "USDT", to_date="2016-02-01", api_key=api_key,
                                                           schema=PandasGlobalSchema(unknown=EXCLUDE))

    return data


def create_fit_model(window, columns_to_predict, columns_indicators, exchange="huobi"):
    set_tf_gpu(False)
    exchange_data, columns_equivalences = load_data(exchange)
    arc = MLArchitect(x_data=exchange_data, y_data=None, is_x_flat=True, save_x_path="saved_data/x.csv",
                      display_indicators_callable=True, data_enhancement=True,
                      columns_equivalences=columns_equivalences, save_normalize_x_model="saved_data/x_norm_model.mod",
                      save_normalize_y_model="saved_data/y_norm_model.mod", save_y_path="saved_data/y.csv",
                      y_restoration_routine="default", index_infer_datetime_format=True, pca_reductions=[('linear', 0.99)],
                      columns_to_predict=columns_to_predict, window_prediction=window,
                      columns_indicators=columns_indicators, test_size=0.20, ml_model=None,
                      save_ml_path="saved_data/ml_model.h5", is_parallel=False, disgard_last=True,
                      window_tuple=(7, 14, 21, 5))

    # arc = MLArchitect(x_data="saved_data/x.csv", y_data="saved_data/y.csv",
    #                   learning_indicators_callable=None, display_indicators_callable=None,
    #                   normalize_x_callable="saved_data/x_norm_model.mod",
    #                   normalize_y_callable="saved_data/y_norm_model.mod",
    #                   index_col='id', pca_reductions=[('linear', 0.99)],
    #                   window_prediction=4, test_size=0.20)

    x_train, y_train = arc.x_train_normalized(), arc.y_train_normalized()
    x_test, y_test = arc.x_test_normalized(), arc.y_test_normalized()

    # Init the ML model
    n_input, n_output, n_init_neurons = x_test.shape[1], y_test.shape[1], 100
    ml_model = MLArchitect.keras_build_model(0, 0, 0, n_input, n_output, n_init_neurons)
    ml_model.summary()

    arc.ml_init_model(ml_model)

    # Fit the model
    prefit_sample_data = exchange_data.loc[exchange_data.index[-5000:], ['close', 'open', 'high', 'low']]
    history = arc.fit(x=x_train.values, y=y_train.values, prefit_sample_data=prefit_sample_data,
                      prefit_simulation_size=10000, prefit_loops=2, epochs=10000, batch_size=75, verbose=1,
                      workers=4, use_multiprocessing=True, shuffle=True, validation_split=0.2,
                      callbacks=[keras.callbacks.EarlyStopping('val_accuracy', min_delta=1e-5, patience=50,
                                                               verbose=1, restore_best_weights=True)])

    loss, accuracy = ml_model.evaluate(x_test.values, y_test.values)
    y_pred = arc.ml_predict(x_test)
    mde = arc.mean_directional_accuracy(y_test.values, y_pred.values)
    mae = arc.mean_absolute_error(y_test.values, y_pred.values)
    print(f"loss: {loss} - accuracy: {accuracy} - mean_directional_accuracy: {mde} - mean_absolute_error: {mae}")

    return arc, exchange_data


def save_data_for_jupyter(arc, prefix=''):
    # data save
    x, y = arc.get_normalized_data(arc.x, arc.y, None, arc.index_min, arc.index_max)
    x.to_csv("saved_data/" + prefix + "x_norm.csv")
    y.to_csv("saved_data/" + prefix + "y_norm.csv")
    arc.x.loc[arc.index_min:arc.index_max].to_csv("saved_data/" + prefix + "x.csv")
    arc.y.loc[arc.index_min:arc.index_max].to_csv("saved_data/" + prefix + "y.csv")

    # Train data
    x_train, y_train = arc.x_train_normalized(), arc.y_train_normalized()
    x_train.to_csv("saved_data/" + prefix + "x_train_norm.csv")
    y_train.to_csv("saved_data/" + prefix + "y_train_norm.csv")

    # Test data
    x_test, y_test = arc.x_test_normalized(), arc.y_test_normalized()
    x_test.to_csv("saved_data/" + prefix + "x_test_norm.csv")
    y_test.to_csv("saved_data/" + prefix + "y_test_norm.csv")


def run_main():
    window, tick_to_display = 2, 1000
    columns_indicators = ['close', 'high', 'low']
    columns_to_predict = ["close", "high", "low"]

    # Create and fit the model
    arc, exchange_data = create_fit_model(window, columns_to_predict, columns_indicators)#, "cryptocompare")
    save_data_for_jupyter(arc, "cryp_")

    #display(arc.data_with_indicators, pd.DataFrame(), tick_to_display, arc.x_test.index)

    # Predict future prices
    y_predicted = arc.ml_predict(restore_y=True, sort_index=True, original_data=exchange_data, windows=window,
                                 is_returns=True, src_columns=columns_to_predict)

    # Get all data
    data_with_indicators = pd.concat([arc.data_with_indicators, y_predicted], axis=1, sort=False)

    # Display graphs
    display(data_with_indicators, y_predicted, tick_to_display, arc.x_test.index)


def display(data_with_indicators, y_predicted, tick_to_display, test_data_index):
    src_columns = ["open", "close", "high", "low"]
    scatters_plot = dict()
    sub_plots = [dict(zip([x + "_rsi_14" for x in src_columns], [x + "_rsi_14" for x in src_columns]))]
    sub_plots += [{"cci_14": "cci_14"}]
    data_index_cyan = None

    pivot_points = data_with_indicators.columns[data_with_indicators.columns.map(lambda x: x.startswith('pp_5_'))]
    scatters_plot.update(dict(zip(pivot_points, pivot_points)))

    if not y_predicted.empty:
        scatters_plot.update(dict(zip(y_predicted.columns, y_predicted.columns)))

    tick = -tick_to_display if tick_to_display > 0 else 0
    data_to_display = data_with_indicators.iloc[tick:, :]
    if list(test_data_index):
        data_index_cyan = (test_data_index & data_to_display.index).sort_values()

    display_graph(data_to_display, scatters_plot=scatters_plot, sub_plots=sub_plots, data_index_cyan=data_index_cyan)


if __name__ == "__main__":
    run_main()
    print("")


