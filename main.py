from examples import get_data_example, normalize_fit_data
from exchangeapi.huobi.models.enumerations import TickerPeriod
from learning.transforms import DataPreProcessing
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Get all necessary data. Prices, returns and processed indicators.
    X, Y = get_data_example("btcusdt", TickerPeriod.Hour4, 2000, pd.DataFrame,
                            prediction_window=4, dropna=True)

    # Splitting and shuffling train and test data
    train_x, test_x, train_y, test_y = DataPreProcessing.train_test_split(X, Y, test_size=0.2, random_state=0)

    # Get normalized train and test x
    processed_train_x, processed_test_x = normalize_fit_data(train_x, test_x, add_standard_scaling=True,
                                                             add_power_transform=True, pca_n_components=0.95,
                                                             save_model_path="processing_x_model.mod")

    # Get normalized train and test y
    processed_train_y, processed_test_y = normalize_fit_data(train_y, test_y, add_standard_scaling=True,
                                                             add_power_transform=True,
                                                             save_model_path="processing_y_model.mod")

    X.to_csv("X.csv")
    Y.to_csv("Y.csv")

    scatters_plot = {}
    #sub_plots = [{"Count": "count", "QCount1": "QCount1", "QCount95": "QCount95"}, {"Vol": "vol", "QVol95": "QVol95", "QVol1": "QVol1"}]
    sub_plots = []

    #display_graph(res.iloc[-1000:, :], data_index=res.index, open='open', close='close', high='high', low='low',
    #              scatters_plot=scatters_plot, sub_plots=sub_plots)


    print("")


