from time import perf_counter_ns

import numpy as np
import pandas as pd
import pmlb
from asboostreg import SparseAdditiveBoostingRegressor
from sklearn.model_selection import train_test_split


def add_irrelevant_features(
    X: pd.DataFrame,
    n_features: int,
    eps: float = 1.0,
) -> pd.DataFrame:
    """Add irrelevant features to the dataset.

    Args:
        X (pd.DataFrame): Dataset.
        n_features (int): Number of irrelevant features to add.
        eps (float, optional): Noise level. Defaults to 1e-3.

    Returns:
        pd.DataFrame: Dataset with irrelevant features.
    """
    n_samples, n_features_orig = X.shape
    X_irrelevant = pd.DataFrame(
        data=eps * np.random.randn(n_samples, n_features),
        columns=[f"irrelevant_{i}" for i in range(n_features)],
    )
    X_transformed = pd.concat([X, X_irrelevant], axis=1)
    return X_transformed


def add_redundant_features(
    X: pd.DataFrame,
    n_features: int,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """Add redundant features to the dataset.

    Args:
        X (pd.DataFrame): Dataset.
        n_features (int): Number of redundant features to add.
        eps (float, optional): Noise level. Defaults to 1e-3.

    Returns:
        pd.DataFrame: Dataset with redundant features.
    """
    n_samples, n_features_orig = X.shape
    sample_features = np.random.choice(
        np.arange(n_features_orig), size=n_features, replace=False
    )
    X_redundant = X.iloc[:, sample_features] + eps * np.random.randn(
        n_samples, n_features
    )
    X_transformed = pd.concat([X, X_redundant], axis=1)
    return X_transformed


def print_title(title: str):
    """Print a title.

    Args:
        title (str): Title to print.
    """
    print(title)
    print("-" * len(title))


def get_score_and_selected_features(
    X: pd.DataFrame,
    y: pd.Series,
    model
) -> tuple:
    """Get the score and selected features.

    Args:
        X (pd.DataFrame): Dataset.
        y (pd.Series): Target.
        model: Model to use.

    Returns:
        score (float): Score of the model.
        selected_features (int): Percentage of selected features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0,
    )
    start = perf_counter_ns()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    end = perf_counter_ns()
    print(f"Time: {(end - start) / 1e9:.3f}s")
    select = len(model.selection_count_)
    print(
        f"Score: {score:.3f}",
        f"Selected features: {select}",
        f"Percentage: {select / X.shape[1]:.3f}",
        "",
        sep="\n",
    )

    return score, select


def main():
    df = pmlb.fetch_data("562_cpu_small")
    X = pd.DataFrame(df.drop(columns="target"))
    y = pd.Series(df["target"])
    model = SparseAdditiveBoostingRegressor(
         random_state=0,
         n_iter_no_change=15,
         max_bins=750,
         max_leaves=60,
         min_samples_leaf=15,
         n_estimators=400,
         row_subsample=0.62,
         l2_regularization=0.89,
         learning_rate=0.18,
         redundancy_exponent=0.72,
    )

    # Baseline
    print_title("Baseline")
    get_score_and_selected_features(X, y, model)

    # Add redundant features
    for i in range(1, 11):
        print_title(f"Redundant features: {i}")
        estimate_eps = np.min(np.var(X, axis=0)) / 10
        X_transformed = add_redundant_features(
            X, n_features=i, eps=estimate_eps
        )
        get_score_and_selected_features(X_transformed, y, model)

    # Add irrelevant features
    for i in range(1, 11):
        print_title(f"Irrelevant features: {i}")
        X_transformed = add_irrelevant_features(X, n_features=i)
        get_score_and_selected_features(X_transformed, y, model)


if __name__ == "__main__":
    main()
