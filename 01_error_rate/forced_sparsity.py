import warnings
from math import sqrt

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from pmlb import fetch_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from utils import median_score

median_scorer = make_scorer(median_score)


def importances(model_name):
    if model_name in {"DT", "XGB"}:
        return model.feature_importances_
    elif model_name == "EBM":
        return model.term_importances()
    elif model_name == "RF":
        return model.steps[-1][-1].feature_importances_
    return model.steps[-1][-1].coef_


models = {
    "Ridge": make_pipeline(
        StandardScaler(), RidgeCV()
    ),
    "DT": DecisionTreeRegressor(random_state=0),
    "RF": make_pipeline(
            KBinsDiscretizer(n_bins=255, encode="ordinal", random_state=0),
            RandomForestRegressor(n_estimators=50, random_state=0),
        ),
    "XGB": XGBRegressor(n_estimators=50, random_state=0),
    "EBM": ExplainableBoostingRegressor(random_state=0, max_rounds=1_000),
}

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
selected = [1, 3, 11, 15, 25, 11, 6, 25]

datos_column = [datos_column[1]]
selected = [selected[1]]

df_dict = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for datos, k in zip(datos_column, selected):
        X, y = fetch_data(datos, return_X_y=True)
        X = X[:, X.std(axis=0) > 0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.2
        )
        cv = KFold()
        for name, model in models.items():
            model.fit(X_train, y_train)
            top = importances(name).argsort()[::-1][:k]
            # now cv
            try:
                scores = cross_val_score(
                    model, X_train[:, top], y_train, cv=cv, scoring=median_scorer
                )
            except Exception:
                scores = np.full(5, np.nan)
            df_dict.append(
                {
                    "Dataset": datos,
                    "Model": name,
                    "Score": np.nanmean(scores),
                    "Ste": np.nanstd(scores) / sqrt(5),
                }
            )
            print(f"{datos} {name} {scores.mean():.3f} {scores.std():.3f}")
    print(df_dict)
    df = pd.DataFrame(df_dict)
    df.to_csv("results_215.csv", index=False)
