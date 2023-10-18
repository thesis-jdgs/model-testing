"""Function utilities for error rate calculation."""
import logging
import warnings
from time import perf_counter

import mlflow
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

    file_handler = logging.FileHandler(f"logs/{dataset}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
                mlflow.log_params(params)
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
        file_handler.close()
        console_handler.close()
        mlflow.end_run()
        end = perf_counter()
        logger.info(f"Experiment for {dataset} dataset took {end - start} seconds.")


def main() -> None:
    import multiprocessing

    from asboostreg import SparseAdditiveBoostingRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from pmlb import regression_dataset_names
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeRegressor
    import xgboost as xgb

    regressors = [
        make_pipeline(
            StandardScaler(),
            RidgeCV(),
        ),
        DecisionTreeRegressor(random_state=0),
        make_pipeline(
            KBinsDiscretizer(n_bins=255, encode="ordinal", random_state=0),
            RandomForestRegressor(n_estimators=50, random_state=0),
        ),
        xgb.XGBRegressor(n_jobs=1, random_state=0),
        ExplainableBoostingRegressor(n_jobs=1, random_state=0, max_rounds=1_000),
        SparseAdditiveBoostingRegressor(
            n_estimators=1_000,
            learning_rate=0.01,
            max_leaves=32,
            l2_regularization=2.0,
            random_state=0,
            max_bins=511,
            min_samples_leaf=3,
        ),
        SparseAdditiveBoostingRegressor(
            n_estimators=7567,
            learning_rate=0.05,
            max_leaves=45,
            l2_regularization=3.7,
            random_state=0,
            max_bins=672,
            min_samples_leaf=1,
            row_subsample=0.77,
            n_iter_no_change=15
        ),
        SparseAdditiveBoostingRegressor(
            n_estimators=3281,
            learning_rate=0.08,
            max_leaves=24,
            l2_regularization=1.6,
            random_state=0,
            max_bins=896,
            min_samples_leaf=2,
            row_subsample=0.68,
            n_iter_no_change=15
        ),
        SparseAdditiveBoostingRegressor(
            n_estimators=542,
            learning_rate=0.22,
            max_leaves=58,
            l2_regularization=4.2,
            random_state=0,
            max_bins=399,
            min_samples_leaf=1,
            row_subsample=0.82,
            n_iter_no_change=15,
        ),
        SparseAdditiveBoostingRegressor(
            n_estimators=2874,
            learning_rate=0.12,
            max_leaves=16,
            l2_regularization=2.5,
            random_state=0,
            max_bins=576,
            min_samples_leaf=3,
            row_subsample=0.63,
            n_iter_no_change=15
        ),
        SparseAdditiveBoostingRegressor(
            n_estimators=734,
            learning_rate=0.27,
            max_leaves=39,
            l2_regularization=0.9,
            random_state=0,
            max_bins=786,
            min_samples_leaf=1,
            row_subsample=0.88,
            n_iter_no_change=15
        )
    ]
    datasets = regression_dataset_names[117:]
    num_processes = min(len(datasets), 5)
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(
            run_experiment,
            [
                (dataset, regressors) for dataset in datasets
            ]
        )


if __name__ == "__main__":
    main()
