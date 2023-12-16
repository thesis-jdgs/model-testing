from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from asboostreg import SparseAdditiveBoostingRegressor


params = [
    {
        "n_estimators": 3234,
        "learning_rate": 0.34,
        "row_subsample": 0.72,
        "max_bins": 70,
        "l2_regularization": 2.05,
        "min_samples_leaf": 7,
        "max_leaves": 30,
        "redundancy_exponent": 2.00,
    },
    {
        "n_estimators": 31,
        "learning_rate": 0.27,
        "row_subsample": 0.88,
        "max_bins": 786,
        "l2_regularization": 0.9,
        "min_samples_leaf": 1,
        "max_leaves": 39,
        "redundancy_exponent": 1.00,
    },
    {
        "n_estimators": 4549,
        "learning_rate": 0.18,
        "row_subsample": 0.62,
        "max_bins": 750,
        "l2_regularization": 0.89,
        "min_samples_leaf": 15,
        "max_leaves": 60,
        "redundancy_exponent": 0.72,
    },
    {
        "n_estimators": 962,
        "learning_rate": 0.16,
        "row_subsample": 0.8,
        "max_bins": 541,
        "l2_regularization": 0.61,
        "min_samples_leaf": 18,
        "max_leaves": 19,
        "redundancy_exponent": 1.00,
        "dropout": True,
        "dropout_probability": 0.01,
        "dropout_rate": 0.03
    },
    {
        "n_estimators": 1575,
        "learning_rate": 0.23,
        "row_subsample": 0.8,
        "max_bins": 769,
        "l2_regularization": 0.27,
        "min_samples_leaf": 19,
        "max_leaves": 20,
        "redundancy_exponent": 0.40,
        "dropout": True,
        "dropout_probability": 0.01,
        "dropout_rate": 0.005,
    },
    {
        "n_estimators": 1622,
        "learning_rate": 0.26,
        "row_subsample": 0.8,
        "max_bins": 417,
        "l2_regularization": 0.08,
        "min_samples_leaf": 9,
        "max_leaves": 34,
        "redundancy_exponent": 0.16,
        "dropout": True,
        "dropout_probability": 0.03,
        "dropout_rate": 0.002,
    },
    {
        "n_estimators": 510,
        "learning_rate": 0.50,
        "row_subsample": 0.30,
        "max_bins": 57,
        "l2_regularization": 0.39,
        "min_samples_leaf": 1,
        "max_leaves": 14,
        "redundancy_exponent": 1.08,
    },
    {
        "n_estimators": 734,
        "learning_rate": 0.27,
        "row_subsample": 0.88,
        "max_bins": 786,
        "l2_regularization": 0.9,
        "min_samples_leaf": 1,
        "max_leaves": 39,
        "redundancy_exponent": 1.00,
    },
]

datos_column = [
    "215_2dplanes",
    "344_mv",
    "562_cpu_small",
    "197_cpu_act",
    "294_satellite_image",
    "227_cpu_small",
    "564_fried",
    "201_pol",
]

for datos, param in zip(datos_column, params):
    X, y = fetch_data(datos, return_X_y=True)
    X = X[:, X.std(axis=0) > 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2
    )
    n_estimators = param["n_estimators"]
    reg = SparseAdditiveBoostingRegressor(
        n_iter_no_change=n_estimators, random_state=0, **param
    )
    reg.fit(X_train, y_train, validation_set=(X_test, y_test))
    print("Dataset: ", datos)
    print("-" * 40)
    selected, total = len(reg.selection_count_), X_train.shape[1]
    print("Number of selected features: ", selected)
    print("Total number of features: ", total)
    print(f"Percentage of selected features: {selected/total*100:.2f}%", end="\n\n")
