"""Adversarial validation for error rate prediction."""
import numpy as np
from pmlb import fetch_data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb


def adversarial_validation(dataset: str):
    """Adversarial validation for error rate prediction.

    Args:
        dataset: The name of the dataset to use.

    Returns:
        The adversarial validation score.
    """
    X, y = fetch_data(dataset, return_X_y=True)
    cv = KFold()
    fold_fraction = 1 / 5
    model = xgb.XGBClassifier(
        n_jobs=-2,
        random_state=0,
        objective="binary:logistic",
        verbosity=0,
        scale_pos_weight=fold_fraction//(1-fold_fraction),
    )
    scores = []
    y_adv = np.empty_like(y)
    for train_index, test_index in cv.split(X):
        y_adv[train_index] = 0.0
        y_adv[test_index] = 1.0
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_adv, test_size=0.2, random_state=0, stratify=y_adv
        )
        model.fit(
            X_train, y_train, eval_set=[(X_test, y_test)], verbose=False
        )
        y_score = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_score)
        scores.append(score)
    return np.mean(scores), np.std(scores)/np.sqrt(len(scores))


def main():
    datasets = (
        '215_2dplanes',
        '344_mv',
        '562_cpu_small',
        '294_satellite_image',
        '573_cpu_act',
        '227_cpu_small',
        '564_fried',
        '201_pol'
    )
    for dataset in datasets:
        mean, ste = adversarial_validation(dataset)
        print(f"Score {dataset}: {mean:.3f} +/- {ste:.3f}")


if __name__ == "__main__":
    main()
