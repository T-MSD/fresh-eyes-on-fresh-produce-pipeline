"""Logistic regression on CLIP image embeddings, shared by finetune, run_test and pick_labels."""

import numpy as np
from sklearn.linear_model import LogisticRegression

ZS_WEIGHT = 8  # the 3 zero-shot columns would otherwise be drowned by 512 embedding columns


def standardise(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-6)


def features(emb: np.ndarray, probs: np.ndarray, with_zeroshot: bool) -> np.ndarray:
    """Standardised so that C means the same for every column (unit-length CLIP values are tiny)."""
    parts = [standardise(emb)]
    if with_zeroshot:
        parts.append(ZS_WEIGHT * standardise(np.log(np.clip(probs, 1e-4, 1))))
    return np.hstack(parts)


def fit_classifier(x_tr, y_tr, c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=2000).fit(x_tr, y_tr)


def proba_of(model: LogisticRegression, x, n_classes: int) -> np.ndarray:
    """Class probabilities with one column per class, even if a class is absent from training."""
    proba = np.zeros((len(x), n_classes))
    proba[:, model.classes_] = model.predict_proba(x)
    return proba


def fit_proba(x_tr, y_tr, x_te, n_classes: int, c: float) -> np.ndarray:
    return proba_of(fit_classifier(x_tr, y_tr, c), x_te, n_classes)
