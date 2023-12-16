import numpy as np
from sklearn.metrics import mean_absolute_error


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
