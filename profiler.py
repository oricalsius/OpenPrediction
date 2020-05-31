from utils import load_data
from learning import MultiLayerPerceptronNN
from architect import MLArchitect

import cProfile
import pstats
import io


if __name__ == "__main__":
    window = 4;
    verbose = False;
    early_stop = True;
    tick_to_display = 600;

    # Define the ML model to use
    ml_model = MultiLayerPerceptronNN(hidden_layer_sizes=(280, 70), activation='logistic',
                                      solver="adam", random_state=0, verbose=verbose,
                                      learning_rate_init=1e-3, early_stopping=early_stop, validation_fraction=0.20,
                                      tol=1e-6, alpha=0.0001, learning_rate="adaptive", max_iter=10000,
                                      batch_size=75, n_iter_no_change=10)

    # Get the exchange data
    exchange_data = load_data()

    profile_path = r'.\profile'

    with cProfile.Profile() as pr:
        # Initialize the Architect object
        arc = MLArchitect(x_data=exchange_data, y_data=None, is_x_flat=True, save_x_path="saved_data/x.csv",
                          save_indicators_path="saved_data/indicators.csv", save_y_path="saved_data/y.csv",
                          save_normalize_x_model="saved_data/x_norm_model.mod", y_restoration_routine="default",
                          save_normalize_y_model="saved_data/y_norm_model.mod",
                          index_col='id', index_infer_datetime_format=True, pca_n_components=0.99,
                          window_prediction=4, test_size=0.01, ml_model=ml_model,
                          save_ml_path="saved_data/ml_model.mod")

    pr.dump_stats(profile_path + '.ori')

    with open(profile_path + '.txt', 'w') as f:
        p_stats = pstats.Stats(profile_path + '.ori', stream=f).sort_stats('cumulative')
        p_stats.print_stats()

    print("")





