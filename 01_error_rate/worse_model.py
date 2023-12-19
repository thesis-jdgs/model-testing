"""Test why the model degenerates on the 215_2dplanes dataset."""

import pandas as pd
from pmlb import fetch_data
from asboostreg import SparseAdditiveBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, make_scorer

from utils import median_score

median_scorer = make_scorer(median_score)


def main() -> None:
    """
    params = {
        "n_estimators": 50,
        "learning_rate": 0.99,
        "row_subsample": 0.72,
        "max_bins": 70,
        "l2_regularization": 2.05,
        "min_samples_leaf": 7,
        "max_leaves": 30,
        "redundancy_exponent": 0.00,
        "n_iter_no_change": 500,
    }
    """
    params = {
        "n_estimators": 50,
        "learning_rate": 0.27,
        "row_subsample": 0.80,
        "max_bins": 800,
        "l2_regularization": 1.0,
        "min_samples_leaf": 1,
        "max_leaves": 40,
        "redundancy_exponent": 0.0,
        "n_iter_no_change": 15,
    }
    dataset = "344_mv"
    data = fetch_data(dataset)
    X = data.drop(columns="target")
    X = X.loc[:, X.var() > 0]
    y = data["target"]
    cv = KFold()
    print(
        cross_validate(
            SparseAdditiveBoostingRegressor(**params),
            X,
            y,
            cv=cv,
            scoring="neg_mean_absolute_error",
        )
    )


if __name__ == "__main__":
    main()
