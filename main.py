from examples import get_data_example
from learning.transforms import DataPreProcessing
import numpy as np

if __name__ == "__main__":
    data = get_data_example(dropna=True)

    #data.iloc[:, :].to_csv("example.csv")

    Y = data[["open", "close", "high", "low"]]
    Y = np.log(Y / Y.shift(1)).shift(-1).iloc[:-1]
    X = data.iloc[:-1]
    X.drop(columns=["open", "close", "high", "low"], inplace=True)

    # Splitting and shuffling train and test data
    train_x, test_x, train_y, test_y = DataPreProcessing.train_test_split(X, Y, test_size=0.2, random_state=0)

    # Normalizing test and train x
    preprocessing_x = DataPreProcessing()
    preprocessing_x.add_standard_scaling(train_x)
    preprocessing_x.add_power_transform_scaling(train_x)
    processed_train_x, processed_test_x = preprocessing_x.transform(train_x), preprocessing_x.transform(test_x)

    # Apply PCA reduction to test and train x
    processed_train_x = preprocessing_x.pca_reduction_fit_transform(processed_train_x, 0.95, svd_solver="full")
    processed_test_x = preprocessing_x.pca_reduction_transform(processed_test_x)

    # Normalizing test and train y
    preprocessing_y = DataPreProcessing()
    preprocessing_y.add_standard_scaling(train_y)
    preprocessing_y.add_power_transform_scaling(train_y)
    processed_train_y, processed_test_y = preprocessing_y.transform(train_y), preprocessing_y.transform(test_y)

    # Save models
    preprocessing_x.save_model("processing_x_model.mod")
    preprocessing_y.save_model("processing_y_model.mod")

    print("")


