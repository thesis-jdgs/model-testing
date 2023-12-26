from pmlb import fetch_data


import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score


X, y = fetch_data("201_pol", return_X_y=True)
cv = KFold()
reg = make_pipeline(
    StandardScaler(),
    OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
    ),
    ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        cv=cv,
        random_state=0,
        n_jobs=-1,
    ),
)
reg.fit(X, y)
alpha, l1_ratio = reg[-1].alpha_, reg[-1].l1_ratio_
reg = make_pipeline(
    StandardScaler(),
    OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
    ),
    ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=0,
    ),
)
scores = cross_val_score(reg, X, y, cv=cv, n_jobs=-1)
print("Alpha: {:.3f}".format(alpha))
print("L1 ratio: {:.3f}".format(l1_ratio))
print(f"Mean: {np.mean(scores):.3f}")
print(f"Std: {np.std(scores):.3f}")
