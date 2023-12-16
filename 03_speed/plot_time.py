"""Plot the time taken to run the simulation."""
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


npz = np.load("time_profile.npz")
df = pd.DataFrame.from_dict({item: npz[item] for item in npz.files})
df["score"] = 1.0 - df["score"] / 338.58875273168087
logging.info(f"Dataframe:\n{df}")

fit_time = df["fit_time"].values
n_estimators = df["n_estimators"].values.reshape(-1, 1)

lr_bag = BaggingRegressor(
    estimator=LinearRegression(),
    n_estimators=1000,
    random_state=0,
    oob_score=True,
    n_jobs=5,
)
lr_bag.fit(n_estimators, fit_time)
score = lr_bag.oob_score_
logging.info(f"OOB score: {score}")
mean_slope = np.mean([est.coef_[0] for est in lr_bag.estimators_])
mean_intercept = np.mean([est.intercept_ for est in lr_bag.estimators_])
logging.info(f"Function shape: {mean_slope} * x + {mean_intercept}")
df["fit_trend"] = mean_slope * df["n_estimators"] + mean_intercept


fig = make_subplots()
fig.add_trace(
    go.Scatter(
        x=df["n_estimators"],
        y=df["fit_time"],
        mode="lines",
        name="Measured fit time",
        line=dict(color="red"),
    )
)
fig.add_trace(
    go.Scatter(
        x=df["n_estimators"],
        y=df["fit_trend"],
        mode="lines",
        name=f"${mean_slope:.2f}x {mean_intercept:.2f}$",
        line=dict(color="blue", dash="dash"),
    )
)
fig.update_layout(
    title=f"Fitting time as a function of the number of estimators",
    xaxis_title="Number of estimators",
    yaxis_title="Time (s)",
    legend_title="",
    font=dict(size=18),
)
fig.show()

# predict
fig = px.scatter(
    df,
    x="n_estimators",
    y="predict_time",
    trendline="expanding",
    trendline_options=dict(function="max"),
)
fig.update_layout(
    title=f"Prediction time as a function of the number of estimators",
    xaxis_title="Number of estimators",
    yaxis_title="Time (s)",
    legend_title="",
    font=dict(size=18),
)
fig.show()
# scores vs time
fig = px.scatter(
    df,
    x="fit_time",
    y="score",
)
fig.update_layout(
    title="Model score as a function of the fitting time",
    xaxis_title="Fitting time (s)",
    yaxis_title="Score",
    legend_title="",
    font=dict(size=18),
)
fig.show()
