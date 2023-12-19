"""Function utilities for error rate calculation."""
from typing import Any

import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import KFold

from utils import median_score


RNG = np.random.default_rng(0)


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
        n_iter_no_change=15,
        row_subsample=0.8,
        dropout=False,
        n_estimators=5100,
        redundancy_exponent=0.0,
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
    X = X.loc[:, X.var() > 0]
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


def random_relevance(
    X, y
):
    """Return a random canonical vector like X[0]."""
    return RNG.random(X.shape[1])



def main() -> None:
    from asboostreg import SparseAdditiveBoostingRegressor
    from optuna.distributions import FloatDistribution
    from optuna.distributions import IntDistribution
    from optuna.integration import OptunaSearchCV

    params = {
        "learning_rate": FloatDistribution(0.1, 0.6),
        "max_leaves": IntDistribution(10, 65, log=True),
        "l2_regularization": FloatDistribution(0.05, 2.1),
        "max_bins": IntDistribution(55, 790, log=True),
        "min_samples_leaf": IntDistribution(1, 20),
    }

    asreg = SparseAdditiveBoostingRegressor(
        random_state=0,
        n_iter_no_change=15,
        row_subsample=0.8,
        dropout=False,
        n_estimators=5100,
        redundancy_exponent=0.0,
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
        #'344_mv',
        '215_2dplanes',
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


if __name__ == "__main__":
    main()
