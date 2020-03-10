import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from pandas import DataFrame, Series, Index


def display_graph(data_to_display: DataFrame, open: str = 'open', close: str = 'close',
                  high: str = 'high', low: str = 'low', scatters_plot: dict = {}, sub_plots: List[dict] = [],
                  data_index_cyan: Series = None):

    empty_parameters = [key for key, value in
                        {"open": open, "close": close, "high": high, "low": low}.items()
                        if (value is None) or (value == "")]

    if empty_parameters:
        raise Exception("Following columns names should not be empty: {0}".format(empty_parameters))

    # Index definition
    data_index = data_to_display.index

    if data_index_cyan is None:
        data_index_green = data_index
    else:
        data_index_green = set(data_to_display.index) - set(data_index_cyan)
        data_index_green = Index(data_index_green).sort_values()

    # Create the template for the subplots. The main figure will take 3 rows.
    specs = [[{"rowspan": 3}], [None], [None], [{}]]
    specs.extend([{}] for i in range(len(sub_plots)))
    fig = make_subplots(rows=len(sub_plots) + 4, cols=1, shared_xaxes=True, vertical_spacing=0.01, specs=specs)

    # Add Candlestick graph
    if data_index_green is not None and not data_index_green.empty:
        data_candle = data_to_display.loc[data_index_green, :]
        fig.add_trace(go.Candlestick(x=data_index_green, open=data_candle[open], high=data_candle[high],
                                     low=data_candle[low], close=data_candle[close]), row=1, col=1)

    if data_index_cyan is not None and not data_index_cyan.empty:
        data_cyan = data_to_display.loc[data_index_cyan, :]
        fig.add_trace(go.Candlestick(x=data_index_cyan, open=data_cyan[open], high=data_cyan[high], low=data_cyan[low],
                                     close=data_cyan[close], increasing_line_color='cyan',
                                     decreasing_line_color='gray'), row=1, col=1)

    # Add other scatters plots in the main graph
    for label, column_name in scatters_plot.items():
        fig.add_scatter(x=data_index, y=data_to_display[column_name], mode='lines', name=label, row=1, col=1)

    # Add subplots
    for i, plot in enumerate(sub_plots):
        for label, column_name in plot.items():
            fig.add_scatter(x=data_index, y=data_to_display[column_name], mode='lines', name=label, row=i + 5, col=1)

    #fig.update_layout(xaxis_rangeslider_visible=False)
    #fig.update_layout(autosize=True)

    fig.show()

