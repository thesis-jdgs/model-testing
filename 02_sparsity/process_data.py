"""Process data and run feature selection experiments."""

from time import perf_counter_ns

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pmlb
from asboostreg import SparseAdditiveBoostingRegressor
from sklearn.model_selection import train_test_split


RNG = np.random.default_rng(0)


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
        data=eps * RNG.standard_normal(size=(n_samples, n_features)),
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
    sample_features = RNG.choice(n_features_orig, size=n_features)
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


def plot(*, redundant: list, irrelevant: list, both: list) -> None:
    n = len(redundant)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)),
        y=redundant,
        name="Redundantes",
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)),
        y=irrelevant,
        name="Irrelevantes",
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(2, n + 1))[::2],
        y=both,
        name="Irrelevantes y redundantes",
        mode="lines+markers",
    ))
    fig.update_layout(
        title="Selección de características de SABRegresor",
        xaxis_title="Número de características añadidas",
        yaxis_title="Número de características seleccionadas",
        legend_title="",
        font=dict(
            size=18,
        )
    )
    fig.update_xaxes(tick0=0, dtick=1)
    fig.update_yaxes(tick0=0, dtick=1)
    fig.show()


def main():
    dataset = "197_cpu_act"
    df = pmlb.fetch_data(dataset)
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
    red_scores = []
    red_selected = []
    estimate_eps = np.min(np.var(X, axis=0)) / 10
    for i in range(1, 21):
        print_title(f"Redundant features: {i}")
        X_transformed = add_redundant_features(
            X, n_features=i, eps=estimate_eps
        )
        score, select = get_score_and_selected_features(X_transformed, y, model)
        red_scores.append(score)
        red_selected.append(select)

    # Add irrelevant features
    irr_scores = []
    irr_selected = []
    for i in range(1, 21):
        print_title(f"Irrelevant features: {i}")
        X_transformed = add_irrelevant_features(X, n_features=i)
        irr_score, irr_select = get_score_and_selected_features(X_transformed, y, model)
        irr_scores.append(irr_score)
        irr_selected.append(irr_select)

    both_scores = []
    both_selected = []
    for i in range(1, 11):
        print_title(f"Both features: {i}")
        estimate_eps = np.min(np.var(X, axis=0)) / 10
        X_transformed = add_irrelevant_features(
            add_redundant_features(X, n_features=i, eps=estimate_eps),
            n_features=i,
        )
        score, select = get_score_and_selected_features(X_transformed, y, model)
        both_scores.append(score)
        both_selected.append(select)

    #
    plot(redundant=red_selected, irrelevant=irr_selected, both=both_selected)

    # save all the score and selected features
    np.save(f"{dataset}_red_scores.npy", red_scores)
    np.save(f"{dataset}_red_selected.npy", red_selected)
    np.save(f"{dataset}_irr_scores.npy", irr_scores)
    np.save(f"{dataset}_irr_selected.npy", irr_selected)
    np.save(f"{dataset}_both_scores.npy", both_scores)
    np.save(f"{dataset}_both_selected.npy", both_selected)


def main2():
    redundant = np.load("197_cpu_act_red_selected.npy")
    irrelevant = np.load("197_cpu_act_irr_selected.npy")
    both = np.load("197_cpu_act_both_selected.npy")
    plot(redundant=redundant, irrelevant=irrelevant, both=both)


if __name__ == "__main__":
    main2()
