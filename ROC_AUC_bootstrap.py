from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def roc_auc_ci(
        classifier: ClassifierMixin,
        X: np.ndarray,
        y: np.ndarray,
        conf: float = 0.95,
        n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    if len(np.unique(y)) != 2:
        raise ValueError
    roc_auc_list = []
    df = np.column_stack((X, y))
    for i in range(n_bootstraps):
        y_bootstrapped = np.array([0])
        while len(np.unique(y_bootstrapped)) < 2:
            df_bootstrapped = df[np.random.choice(X.shape[0], size=X.shape[0], replace=True), :]
            X_bootstrapped = df_bootstrapped[:, :-1]
            y_bootstrapped = df_bootstrapped[:, -1]
        roc_auc = roc_auc_score(y_bootstrapped, classifier.predict_proba(X)[:, 1])
        roc_auc_list = np.append(roc_auc_list, roc_auc)
    lcb = np.quantile(roc_auc_list, (1 - conf) / 2)
    ucb = np.quantile(roc_auc_list, (1 + conf) / 2)
    return lcb, ucb