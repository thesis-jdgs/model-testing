"""Function utilities for error rate calculation."""
import logging
import warnings
from time import perf_counter

import mlflow
import numpy as np
from asboostreg import SparseAdditiveBoostingRegressor
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.integration import OptunaSearchCV
from pmlb import fetch_data
from sklearn.model_selection import KFold

from utils import median_score


def get_name(model):
    """Get the name of a model."""
    name = type(model).__name__
    if name == "Pipeline":
        name = model.steps[-1][0]
    return name


def run_experiment(
    dataset: str,
    regressors: list,
    n_folds: int = 5
) -> None:
    """Run experiments for a given dataset with multiple regressors.

    Parameters:
    ----------
    dataset, str
        The name of the dataset to use.
    regressors, list
        A list of regressors.
    n_folds, int, default=10
        Number of folds for cross-validation.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Running experiment for {dataset} with {len(regressors)} regressors.")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    #file_handler = logging.FileHandler(f"logs/{dataset}.log")
    #file_handler.setLevel(logging.INFO)
    #file_handler.setFormatter(formatter)
    #logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    start = perf_counter()
    data = fetch_data(dataset)
    mlflow.set_experiment(dataset)
    experiment_id = mlflow.get_experiment_by_name(dataset).experiment_id

    X, y = data.drop(columns=["target"]), data["target"]
    kf = KFold(n_splits=n_folds)

    scores = np.full(n_folds, np.nan)
    try:
        for i, regressor in enumerate(regressors, 1):
            name = get_name(regressor)
            is_asbr = name == "SparseAdditiveBoostingRegressor"
            with mlflow.start_run(run_name=f"{i}_{name}", experiment_id=experiment_id):
                mlflow.set_tags({"dataset": dataset, "model": name})
                params = regressor.get_params()
                # mlflow.log_params(params)
                logger.info(
                    f"Running model d_{dataset}/m_{i}: {name} with params {params}."
                )
                for fold, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    try:
                        if is_asbr:
                            regressor.fit(
                                X_train, y_train, validation_set=(X_test, y_test)
                            )
                        else:
                            regressor.fit(X_train, y_train)
                        y_pred = regressor.predict(X_test)
                        scores[fold] = median_score(y_test, y_pred)
                        logger.info(
                            f"Score d_{dataset}/m_{i}/f_{fold + 1}: {scores[fold]}"
                        )
                        mlflow.log_metric(f"score_fold_{fold + 1}", scores[fold])
                    except Exception as e:
                        logger.error(f"Error d_{dataset}/m_{i}/f_{fold + 1}: {e}")
                        scores[fold] = np.nan
                        mlflow.log_metric(f"score_fold_{fold + 1}", np.nan)
                    except KeyboardInterrupt:
                        logger.error(
                            f"Experiment d_{dataset}/m_{i}/f_{fold + 1} interrupted."
                        )
                with warnings.catch_warnings(record=True) as w:
                    mean = np.nanmean(scores)
                    std = np.nanstd(scores)
                if any(issubclass(warn.category, RuntimeWarning) for warn in w):
                    logger.info(f"All fits of d_{dataset}/m_{i} failed.")
                else:
                    logger.info(f"Mean score d_{dataset}/m_{i}: {mean}")
                    logger.info(f"Standard deviation score d_{dataset}/m_{i}: {std}")
                mlflow.log_metric("score_mean", mean)
                mlflow.log_metric("score_std", std)
    except Exception as e:
        logger.error(f"Error d_{dataset}: {e}")
    except KeyboardInterrupt:
        logger.error(f"Experiment d_{dataset} interrupted.")
    finally:
        #file_handler.close()
        console_handler.close()
        mlflow.end_run()
        end = perf_counter()
        logger.info(f"Experiment for {dataset} dataset took {end - start} seconds.")


def optuna_model(cv):
    base_model = SparseAdditiveBoostingRegressor(
        random_state=0,
        n_iter_no_change=15,
        max_leaves=3,
        min_samples_leaf=1,
        max_bins=3,
        row_subsample=0.8,
    )
    params = {
        "n_estimators": IntDistribution(500, 5_000, log=True),
        "learning_rate": FloatDistribution(0.18, 0.5, log=True),
        # "max_leaves": IntDistribution(3, 64),
        "l2_regularization": FloatDistribution(0.01, 5, log=True),
        # "max_bins": IntDistribution(56, 1024, log=True),
        # "min_samples_leaf": IntDistribution(1, 15),
        # "row_subsample": FloatDistribution(0.15, 0.9),
        "redundancy_exponent": FloatDistribution(0.1, 2.5),
    }
    return OptunaSearchCV(
        base_model,
        n_trials=100,
        n_jobs=5,
        random_state=1,
        scoring="neg_mean_absolute_error",
        param_distributions=params,
        timeout=3600,
        refit=False,
        cv=cv,
    )


def run_optuna_experiment(
    dataset: str,
    n_folds: int = 5
) -> None:
    """Run experiments for a given dataset with multiple regressors.

    Parameters:
    ----------
    dataset, str
        The name of the dataset to use.
    regressors, list
        A list of regressors.
    n_folds, int, default=10
        Number of folds for cross-validation.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Running experiment {dataset}.")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    start = perf_counter()
    data = fetch_data(dataset)
    mlflow.set_experiment(dataset)
    experiment_id = mlflow.get_experiment_by_name(dataset).experiment_id

    X, y = data.drop(columns=["target"]), data["target"]
    X = X.loc[:, X.var() > 0]
    kf = KFold(n_splits=n_folds)
    optreg = optuna_model(kf)
    scores = np.full(n_folds, np.nan)
    try:
        name = get_name(optreg)
        with mlflow.start_run(run_name=f"{name}", experiment_id=experiment_id):
            mlflow.set_tags({"dataset": dataset, "model": name})
            optreg.fit(X, y)
            params = optreg.best_params_
            asreg = SparseAdditiveBoostingRegressor(
                random_state=0,
                n_iter_no_change=15,
                **params
            )
            mlflow.log_params(params)
            for fold, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                try:
                    asreg.fit(X_train, y_train, validation_set=(X_test, y_test))
                    y_pred = asreg.predict(X_test)
                    scores[fold] = median_score(y_test, y_pred)
                    logger.info(
                        f"Score d_{dataset}/f_{fold + 1}: {scores[fold]}"
                    )
                    mlflow.log_metric(f"score_fold_{fold + 1}", scores[fold])
                except Exception as e:
                    logger.error(f"Error d_{dataset}/f_{fold + 1}: {e}")
                    scores[fold] = np.nan
                    mlflow.log_metric(f"score_fold_{fold + 1}", np.nan)
                except KeyboardInterrupt:
                    logger.error(
                        f"Experiment d_{dataset}/f_{fold + 1} interrupted."
                    )
            with warnings.catch_warnings(record=True) as w:
                mean = np.nanmean(scores)
                std = np.nanstd(scores)
            if any(issubclass(warn.category, RuntimeWarning) for warn in w):
                logger.info(f"All fits of d_{dataset} failed.")
            else:
                logger.info(f"Mean score d_{dataset}: {mean}")
                logger.info(f"Standard deviation score d_{dataset}: {std}")
            mlflow.log_metric("score_mean", mean)
            mlflow.log_metric("score_std", std)
    except Exception as e:
        logger.error(f"Error d_{dataset}: {e}")
    except KeyboardInterrupt:
        logger.error(f"Experiment d_{dataset} interrupted.")
    finally:
        #file_handler.close()
        console_handler.close()
        mlflow.end_run()
        end = perf_counter()
        logger.info(f"Experiment for {dataset} dataset took {end - start} seconds.")


def main() -> None:
    from sklearn.tree import DecisionTreeRegressor
    # datasets = [
    #     '1196_BNG_pharynx',
    #     '201_pol',
    #     '215_2dplanes',
    #     '227_cpu_small',
    #     '294_satellite_image',
    #     '344_mv',
    #     '562_cpu_small'
    # ]
    datasets = [
        "215_2dplanes",
        "344_mv",
        "562_cpu_small",
        "197_cpu_act",
        "294_satellite_image",
        "227_cpu_small",
        "564_fried",
        "201_pol",
    ]
    for dataset in datasets:
        run_experiment(
            dataset,
            [DecisionTreeRegressor(random_state=0, max_depth=3)],
        )


if __name__ == "__main__":
    main()
