from utils import save_data_for_jupyter, display
from learninglogic import create_fit_model

import pandas as pd

columns_to_use = ['close_log_returns_1', 'high_log_returns_1', 'low_log_returns_1',
                  'close_log_returns_2', 'high_log_returns_2', 'low_log_returns_2']


def run_main():
    window, tick_to_display = 4, 1000
    columns_indicators = ['close', 'high', 'low']
    columns_to_predict = ["close", "high", "low"]
    columns_to_predict = ["close", "high", "low"]

    # Create and fit the model
    arc, exchange_data = create_fit_model(window, columns_to_predict, columns_indicators, True, "cosmos")#, "cryptocompare")
    save_data_for_jupyter(arc, "cryp_")

    #display(arc.data_with_indicators, pd.DataFrame(), tick_to_display, arc.x_test.index)

    # Predict future prices
    y_predicted = arc.ml_predict(restore_y=True, sort_index=True, original_data=exchange_data, windows=window,
                                 is_returns=True, src_columns=columns_to_predict)

    # Get all data
    data_with_indicators = pd.concat([arc.data_with_indicators, y_predicted], axis=1, sort=False)

    # Display graphs
    display(data_with_indicators, y_predicted, tick_to_display, arc.x_test.index)


if __name__ == "__main__":
    run_main()
    print("")


