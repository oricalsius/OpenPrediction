from exchangeapi.huobi.models.enumerations import TickerPeriod
from graphs import display_graph
from exchangeapi.huobi.rest import Rest
from exchangeapi.huobi.models.ticker import PandasGlobalSchema
from marshmallow import EXCLUDE
from sklearn.model_selection import train_test_split
from config import CONFIG

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pymongo

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
    elif exchange == "cosmos":
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
        elif exchange == "cosmos":
            data = load_from_cosmos_mongo()
        else:
            raise Exception("No exchange has been chosen.")

        if path is not None and not data.empty:
            data.to_csv(path)

    return data, columns_equivalences


def load_from_huobi():
    data = api_huobi_rest.get_ticker_history("btcusdt", TickerPeriod.Hour4, 2000, PandasGlobalSchema(unknown=EXCLUDE))

    return data


def load_from_cryptocompare():
    api_key = CONFIG["CRYPTOCOMPARE_KEY"]
    data = api_huobi_rest.get_ticker_history_cryptocompare("BTC", "USDT", to_date="2016-02-01", api_key=api_key,
                                                           schema=PandasGlobalSchema(unknown=EXCLUDE))

    return data


def load_from_cosmos_mongo():
    cosmos_config = CONFIG["COSMOS_DB"]
    uri = cosmos_config["CONNECTION_STRING"]
    uri = uri.format(username=cosmos_config["USERNAME"], password=cosmos_config["PASSWORD"], host=cosmos_config["HOST"],
                     port=cosmos_config["PORT"], ssl=cosmos_config["SSL"], appname=cosmos_config["APPNAME"])

    client = pymongo.MongoClient(uri)
    db = client.get_database("huobi")
    collection = db.get_collection("spot")

    nb_elt = collection.count_documents(filter={})

    return db


# Sequence creation function
def split_sequence(x_flat, y_flat, n_steps_in=None, n_steps_out=None):
    x, y = list(), list()
    i_max = np.minimum(x_flat.shape[0], y_flat.shape[0])

    if not n_steps_in:
        return np.array(x_flat[:i_max]), np.array(y_flat[:i_max])

    if n_steps_out:
        for i in range(i_max):
            nx = i + n_steps_in
            ny = nx - n_steps_out
            if nx > i_max or ny > i_max:
                break
            #seq_x, seq_y = np.flip(x_flat[i:nx], axis=0), y_flat[ny:nx]
            seq_x, seq_y = x_flat[i:nx], y_flat[ny:nx]

            x.append(seq_x)
            y.append(seq_y)
    else:
        for i in range(i_max):
            nx = i + n_steps_in
            ny = nx - 1
            if nx > i_max or ny > i_max:
                break
            seq_x, seq_y = np.flip(x_flat[i:nx], axis=0), y_flat[ny]
            #seq_x, seq_y = x_flat[i:nx], y_flat[ny]

            x.append(seq_x)
            y.append(seq_y)

    return np.array(x), np.array(y)


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



