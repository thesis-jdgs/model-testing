"""Plot a scatterplot of different models in the (time, score) plane."""
from time import perf_counter_ns

import numpy as np
import pmlb
import plotly.graph_objects as go
from sklearn.model_selection import KFold

from utils import median_score, no_gc


def plot_models(
    model_list: dict,
    X: np.ndarray,
    y: np.ndarray,
):
    cv = KFold()
    n = len(model_list)
    scores = np.empty((n, 5), dtype=float)
    fit_times = np.empty((n, 5), dtype=float)
    predict_times = np.empty((n, 5), dtype=float)
    for j, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for i, model in enumerate(model_list.values()):
            with no_gc():
                start = perf_counter_ns()
                model.fit(X_train, y_train)
                end = perf_counter_ns()
                start2 = perf_counter_ns()
                y_pred = model.predict(X_test)
                end2 = perf_counter_ns()
            fit_times[i, j] = end - start
            predict_times[i, j] = end2 - start2
            scores[i, j] = median_score(y_test, y_pred)
    predict_times /= 1e9
    fit_times /= 1e9
    scores_mean = np.mean(scores, axis=1)
    scores_ste = np.std(scores, axis=1) / np.sqrt(5)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_ste = np.std(fit_times, axis=1) / np.sqrt(5)
    predict_times_mean = np.mean(predict_times, axis=1)
    predict_times_ste = np.std(predict_times, axis=1) / np.sqrt(5)

    # plot fit time vs score with error bars, for each model as a different point/color
    fig = go.Figure()
    for i, (name, model) in enumerate(model_list.items()):
        fig.add_trace(
            go.Scatter(
                x=[fit_times_mean[i]],
                y=[scores_mean[i]],
                mode="markers",
                name=name,
                error_x=dict(
                    type="data",
                    array=[fit_times_ste[i]],
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=[scores_ste[i]],
                    visible=True,
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Fit time (s)",
        yaxis_title="Median score",
        legend_title="Model",
    )
    fig.show()
    # now make a barplot of the prediction time for each model, with error bars
    score_order = np.argsort(predict_times_mean)
    sorted_models = list(model_list.items())
    fig = go.Figure()
    for index in score_order:
        name, model = sorted_models[index]
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[predict_times_mean[index]],
                name=name,
                error_y=dict(
                    type="data",
                    array=[predict_times_ste[index]],
                    visible=True,
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Prediction time (s)",
    )
    fig.show()

def main():
    from asboostreg import SparseAdditiveBoostingRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeRegressor
    import xgboost as xgb

    X, y = pmlb.fetch_data("562_cpu_small", return_X_y=True)
    base_params = {
        "random_state": 0,
        "learning_rate": 0.18,
        "row_subsample": 0.62,
        "max_bins": 750,
        "l2_regularization": 0.89,
        "min_samples_leaf": 15,
        "max_leaves": 60,
        "redundancy_exponent": 0.72,
        "dropout": False,
    }
    model_list = {
        "Ridge": make_pipeline(
            StandardScaler(),
            RidgeCV(),
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=0, max_depth=3),
        "Random Forest": make_pipeline(
            KBinsDiscretizer(n_bins=255, encode="ordinal", random_state=0),
            RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=-2),
        ),
        "XGBoost":  xgb.XGBRegressor(n_jobs=-2, random_state=0),
        "EBM": ExplainableBoostingRegressor(random_state=0, max_rounds=1_000),
        "SABReg (4500)": SparseAdditiveBoostingRegressor(**base_params, n_estimators=4500),
        "SABReg (600)": SparseAdditiveBoostingRegressor(**base_params, n_estimators=600),
    }
    plot_models(model_list, X, y)


if __name__ == "__main__":
    main()
