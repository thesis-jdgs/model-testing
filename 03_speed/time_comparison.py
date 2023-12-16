from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split

from utils import median_score, no_gc


def time_model(model, X, y, repeats=50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2
    )
    times = np.empty((repeats, 4), dtype=float)
    difference = np.empty(repeats, dtype=float)
    for i in range(repeats):
        with no_gc():
            start = perf_counter()
            model.fit(X_train, y_train)
            end = perf_counter()
            start2 = perf_counter()
            y_pred = model.predict(X_test)
            end2 = perf_counter()
            times[i] = [start, end, start2, end2]
    np.subtract(times[:, 1], times[:, 0], out=difference)
    fit_time = np.min(difference)
    np.subtract(times[:, 3], times[:, 2], out=difference)
    predict_time = np.min(difference)
    score = median_score(y_test, y_pred)
    return fit_time, predict_time, score
