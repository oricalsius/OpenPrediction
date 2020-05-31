from architect import MLArchitect
from config import CONFIG
from utils import set_tf_gpu, load_data, split_sequence

import keras


def create_fit_model(window, columns_to_predict, columns_indicators, normalize_data=True, exchange="huobi"):
    set_tf_gpu(False)
    exchange_data, columns_equivalences = load_data(exchange=exchange)
    arc = MLArchitect(x_data=exchange_data, y_data=None, is_x_flat=True, save_x_path="saved_data/x.csv",
                      display_indicators_callable=True, data_enhancement=True,
                      columns_equivalences=columns_equivalences, save_normalize_x_model="saved_data/x_norm_model.mod",
                      save_normalize_y_model="saved_data/y_norm_model.mod", save_y_path="saved_data/y.csv",
                      y_restoration_routine="default", index_infer_datetime_format=True, pca_reductions=[('linear', 0.99)],
                      columns_to_predict=columns_to_predict, window_prediction=window,
                      columns_indicators=columns_indicators, test_size=0.01, ml_model=None,
                      save_ml_path="models/ml_model.h5", is_parallel=False, disgard_last=True,
                      window_tuple=(7, 14, 21, 6))

    # arc = MLArchitect(x_data="saved_data/x.csv", y_data="saved_data/y.csv",
    #                   learning_indicators_callable=None, display_indicators_callable=None,
    #                   normalize_x_callable="saved_data/x_norm_model.mod",
    #                   normalize_y_callable="saved_data/y_norm_model.mod",
    #                   index_col='id', pca_reductions=[('linear', 0.99)],
    #                   window_prediction=4, test_size=0.20)

    if normalize_data:
        x, y = arc.get_normalized_data(arc.x, arc.y, None, arc.index_min, arc.index_max)
    else:
        x, y = arc.x.loc[arc.index_min:arc.index_max], arc.y.loc[arc.index_min:arc.index_max]

    n_steps_in, n_steps_out = 3, None
    n_shape, n = x.shape, 10

    x_train, y_train = split_sequence(x.iloc[:-n].values, y.iloc[:-n].values, n_steps_in, n_steps_out)
    x_test, y_test = split_sequence(x.iloc[-n:].values, y.iloc[-n:].values, n_steps_in, n_steps_out)

    # Init the ML model
    n_input, n_features, n_output, n_init_neurons = x.shape[1], x.shape[1], y_train.shape[1], 100

    # n_seq = 2
    # n_steps_in = int(n_steps_in / n_seq)
    # x_train = x_train.reshape((x_train.shape[0], n_seq, n_steps_in, n_features))
    # x_test = x_test.reshape((x_test.shape[0], n_seq, n_steps_in, n_features))

    ml_model = MLArchitect.keras_build_model(0, 0, 0, n_input, n_output, n_steps_in, n_steps_out, n_features,
                                             n_init_neurons)
    ml_model.summary()

    arc.ml_init_model(ml_model)

    # Fit the model
    prefit_sample_data = exchange_data.loc[exchange_data.index[-500:], ['close', 'open', 'high', 'low']]
    history = arc.fit(x=x_train, y=y_train,
                      epochs=10000, batch_size=50, verbose=1, shuffle=True, validation_split=0.2,
                      prefit_sample_data=prefit_sample_data, prefit_simulation_size=10000,
                      callbacks=[keras.callbacks.EarlyStopping('loss', min_delta=1e-5, patience=200,
                                                               verbose=1, restore_best_weights=True),
                                 keras.callbacks.ModelCheckpoint("models/best_model_2.h5", monitor='keras_r2_score',
                                                                 verbose=1, save_best_only=True, mode='max')])

    err_res = dict(zip(ml_model.metrics_names, ml_model.evaluate(x_test, y_test)))
    y_pred = arc.norm_output_inverse_transform(ml_model.predict(x_test)) if normalize_data else ml_model.predict(x_test)

    mde = arc.mean_directional_accuracy(y_test, y_pred.values)
    mae = arc.mean_absolute_error(y_test, y_pred.values)
    err_res.update({'mean_directional_accuracy': mde, 'mean_absolute_error': mae})
    print(err_res)

    return arc, exchange_data






