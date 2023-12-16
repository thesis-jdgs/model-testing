"""Function utilities for error rate calculation."""
from typing import Any

import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from utils import median_score


def run_experiment(
    dataset: str,
    regressor_class: Any,
    opt_regressor: Any,
    n_folds: int = 5
):
    """Run experiments for a given dataset with multiple regressors.

    Parameters:
    ----------
    dataset, str
        The name of the dataset to use.
    regressor, Any
        A regressor.
    n_folds, int, default=10
        Number of folds for cross-validation.

    """
    data = fetch_data(dataset)
    X, y = data.drop(columns=["target"]), data["target"]
    X = X.loc[:, X.var() > 0]
    kf = KFold(n_splits=n_folds)
    opt_regressor.set_params(cv=kf)
    opt_regressor.fit(X, y)
    regressor = regressor_class(
        random_state=0,
        n_iter_no_change=10_000,
        row_subsample=0.8,
        dropout=False,
        **opt_regressor.best_params_
    )
    scores = np.empty(n_folds, dtype=float)
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        scores[fold] = median_score(y_test, y_pred)
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    return scores, mean, std, regressor


def run_experiment2(
    dataset: str,
    regressor: Any,
    n_folds: int = 5
):
    """Run experiments for a given dataset with multiple regressors.

    Parameters:
    ----------
    dataset, str
        The name of the dataset to use.
    regressor, Any
        A regressor.
    n_folds, int, default=10
        Number of folds for cross-validation.

    """
    data = fetch_data(dataset)
    X, y = data.drop(columns=["target"]), data["target"]
    kf = KFold(n_splits=n_folds)
    scores = np.empty(n_folds, dtype=float)
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        scores[fold] = median_score(y_test, y_pred)
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    return scores, mean, std


def main() -> None:
    from asboostreg import SparseAdditiveBoostingRegressor
    from optuna.distributions import FloatDistribution
    from optuna.distributions import IntDistribution
    from optuna.integration import OptunaSearchCV

    params = {
        "learning_rate": FloatDistribution(0.1, 0.6, log=True),
        "max_leaves": IntDistribution(10, 70, log=True),
        "l2_regularization": FloatDistribution(0.05, 2.5, log=True),
        "max_bins": IntDistribution(50, 850, log=True),
        "min_samples_leaf": IntDistribution(1, 20),
        "redundancy_exponent": FloatDistribution(0.15, 2.2),
        #"dropout_rate": FloatDistribution(0.0, 0.1),
        #"dropout_probability": FloatDistribution(0.0, 0.2),
        #"n_estimators": IntDistribution(20, 5_100),
    }

    asreg = SparseAdditiveBoostingRegressor(
        random_state=0,
        n_iter_no_change=30,
        row_subsample=0.8,
        dropout=False,
        n_estimators=5100,
    )
    optreg = OptunaSearchCV(
        asreg,
        n_trials=100,
        n_jobs=5,
        random_state=0,
        scoring="neg_mean_absolute_error",
        param_distributions=params,
        timeout=3600,
        refit=False,
    )

    names = [
        '215_2dplanes',
        '344_mv',
        '562_cpu_small',
        '294_satellite_image',
        '573_cpu_act',
        '227_cpu_small',
        '564_fried',
        '201_pol'
    ]
    to_df = {
        'dataset': [],
        'mean': [],
        'std': [],
    }
    for i, name in enumerate(names):
        _, mean, std, regressor = run_experiment(
            name, SparseAdditiveBoostingRegressor, optreg
        )
        reg_params = {
            key: val
            for key, val in regressor.get_params().items()
            if key in params.keys()
        }
        if i == 0:
            for key in reg_params.keys():
                to_df[key] = []
        for key, value in reg_params.items():
            to_df[key].append(value)
        to_df['dataset'].append(name)
        to_df['mean'].append(mean)
        to_df['std'].append(std)
        print(regressor)
    df = pd.DataFrame(to_df)
    df["ste"] = df["std"] / np.sqrt(5)
    print(df)
    print(df.to_latex())


def main2():
    from asboostreg import SparseAdditiveBoostingRegressor
    from pmlb import regression_dataset_names

    params = {
        'n_estimators': 1876,
        'learning_rate': 0.020444049468993485,
        'max_leaves': 4,
        'l2_regularization': 0.5200040243048039,
        'max_bins': 502,
        'min_samples_leaf': 2,
        'row_subsample': 0.30343377928331733
    }
    asreg = SparseAdditiveBoostingRegressor(
        random_state=0, n_iter_no_change=15, **params
    )
    print(run_experiment2(regression_dataset_names[2], asreg))


if __name__ == "__main__":
    main()
