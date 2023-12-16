"""Profiling the asboostreg estimator for different number of trees."""
import logging
from time import perf_counter

import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from asboostreg import SparseAdditiveBoostingRegressor

from utils import no_gc

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
X, y = fetch_data("562_cpu_small", return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2
)


repeats = 50
n_estimators = np.arange(50, 3000, 50)
fit_time_curve = np.empty_like(n_estimators, dtype=float)
predict_time_curve = np.empty_like(n_estimators, dtype=float)
scores = np.empty_like(n_estimators, dtype=float)

i = 0
success = False
times = np.empty((4, repeats), dtype=float)
difference = np.empty(repeats, dtype=float)
try:
    for i, n in enumerate(n_estimators):
        logging.info(f"n_estimators: {n}")
        with no_gc():
            for j in range(repeats):
                logging.info(f"Repeat: {j}")
                start1 = perf_counter()
                reg = SparseAdditiveBoostingRegressor(
                    n_estimators=n, n_iter_no_change=n, **base_params
                )
                reg.fit(X_train, y_train, validation_set=(X_test, y_test))
                end1 = perf_counter()
                start2 = perf_counter()
                reg.predict(X_test)
                end2 = perf_counter()
                times[:, j] = [start1, end1, start2, end2]
        np.subtract(times[1], times[0], out=difference)
        fit_time_curve[i] = np.min(difference)
        np.subtract(times[3], times[2], out=difference)
        predict_time_curve[i] = np.min(difference)
        scores[i] = reg.score_history_[-1, 1]
    success = True
except Exception as e:
    logging.error(f"Error: {e}")
finally:
    k_ = i if success else i - 1
    np.savez_compressed(
        "time_profile.npz",
        n_estimators=n_estimators[:k_],
        fit_time=fit_time_curve[:k_],
        predict_time=predict_time_curve[:k_],
        score=scores[:k_],
    )
if success:
    logging.info("Success!")
