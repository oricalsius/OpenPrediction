import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from pandas import DataFrame, Series


def display_graph(data_to_display: DataFrame, data_index: Series, open: str, close: str,
                  high: str, low: str, scatters_plot: dict = {}, sub_plots: List[dict] = []):

    empty_parameters = [key for key, value in
                        {"open": open, "close": close, "high": high, "low": low}.items()
                        if (value is None) or (value == "")]

    if empty_parameters:
        raise Exception("Following columns names should not be empty: {0}".format(empty_parameters))

    # Create the template for the subplots. The main figure will take 3 rows.
    specs = [[{"rowspan": 3}], [None], [None], [{}]]
    specs.extend([{}] for i in range(len(sub_plots)))
    fig = make_subplots(rows=len(sub_plots) + 4, cols=1, shared_xaxes=True, vertical_spacing=0.01, specs=specs)

    # Add Candlestick graph
    fig.add_trace(go.Candlestick(x=data_index, open=data_to_display[open], high=data_to_display[high],
                                 low=data_to_display[low], close=data_to_display[close]), row=1, col=1)

    # Add other scatters plots in the main graph
    for label, column_name in scatters_plot.items():
        fig.add_scatter(x=data_index, y=data_to_display[column_name].shift(1), mode='lines', name=label, row=1, col=1)

    # Add subplots
    for i, plot in enumerate(sub_plots):
        for label, column_name in plot.items():
            fig.add_scatter(x=data_index, y=data_to_display[column_name], mode='lines', name=label, row=i + 5, col=1)

    #fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()

