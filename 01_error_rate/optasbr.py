"""Function utilities for error rate calculation."""
from typing import Any

import numpy as np
from pmlb import fetch_data
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


def median_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate the median score, which is defined as 1 - (MAE / MAE_median).

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    float
        Error rate.

    """
    median = np.median(y_true)
    min_mae = np.mean(np.abs(y_true - median))
    mae = mean_absolute_error(y_true, y_pred)
    return 1.0 - mae / min_mae


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
    kf = KFold(n_splits=n_folds)
    opt_regressor.set_params(cv=kf)
    opt_regressor.fit(X, y)
    regressor = regressor_class(
        random_state=0,
        n_iter_no_change=15,
        **opt_regressor.best_params_
    )
    print(regressor)
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
    from pmlb import regression_dataset_names

    params = {
        "n_estimators": IntDistribution(500, 10_000, log=True),
        "learning_rate": FloatDistribution(0.001, 0.5, log=True),
        "max_leaves": IntDistribution(3, 64, log=True),
        "l2_regularization": FloatDistribution(0.01, 10, log=True),
        "max_bins": IntDistribution(128, 1024, log=True),
        "min_samples_leaf": IntDistribution(1, 15),
        "row_subsample": FloatDistribution(0.15, 0.9),
    }

    asreg = SparseAdditiveBoostingRegressor(
        random_state=0,
        n_iter_no_change=15,
    )
    optreg = OptunaSearchCV(
        asreg,
        n_trials=5,
        n_jobs=5,
        random_state=0,
        scoring="neg_mean_absolute_error",
        param_distributions=params,
        timeout=30,
        refit=False,
    )

    solved = [
        2,   7,   8,  16,  18,  19,  22,  23,  26,  27,  33,  35,  37,  42,
        44,  48,  49,  50,  54,  63,  66,  70,  74,  77,  79,  84,  88,  95,
        96, 101, 102, 104, 105, 107, 110, 113, 119
    ]
    datasets = list(np.array(regression_dataset_names)[solved])
    print(run_experiment(datasets[0], SparseAdditiveBoostingRegressor, optreg))


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
