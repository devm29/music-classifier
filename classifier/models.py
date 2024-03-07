"""
The candidate classifiers, the registry that makes them swappable, and the one
decision that mattered most.

**Every model is wrapped in a StandardScaler.** The 104 features are not on a
common scale — their standard deviations span 1.86 to 59.66, a factor of 32 —
and SVMs and k-NN both compute distances in that raw space, so a handful of
wide-range features dominate every comparison. Scaling is one line and it
changes the ranking of the models, which is why it is applied here rather than
left to whoever calls this.

The registry is the seam for the other half of the question. ``classifier.features``
makes *how audio becomes numbers* swappable; this makes *what scores the
numbers* swappable. A new model is a factory returning anything with
``fit``/``predict_proba``/``classes_``, registered under a name that
``classifier.train --classifier <name>`` accepts and that is written into the
model metadata, so an artifact always says what produced it.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 7

ClassifierFactory = Callable[[], BaseEstimator]

_CLASSIFIERS: dict[str, ClassifierFactory] = {}

#: What the web app serves. Calibrated because the interface shows probabilities.
DEFAULT_CLASSIFIER = "svm-rbf-calibrated"


def register_classifier(name: str, factory: ClassifierFactory) -> ClassifierFactory:
    """Add a classifier factory to the registry. Re-registering a name is an error."""
    if name in _CLASSIFIERS and _CLASSIFIERS[name] is not factory:
        raise ValueError(f"classifier {name!r} is already registered")
    _CLASSIFIERS[name] = factory
    return factory


def get_classifier(name: str | None = None) -> BaseEstimator:
    """Build a fresh, unfitted estimator by registry name."""
    name = name or DEFAULT_CLASSIFIER
    try:
        factory = _CLASSIFIERS[name]
    except KeyError:
        raise KeyError(
            f"unknown classifier {name!r}; registered: {sorted(_CLASSIFIERS)}"
        ) from None
    return factory()


def available_classifiers() -> list[str]:
    return sorted(_CLASSIFIERS)


def candidates(scaled: bool = True) -> dict[str, Pipeline]:
    """
    The classifiers the original project compared, as sklearn pipelines.

    ``scaled=False`` reproduces the original setup, which fitted these
    estimators on the raw feature matrix. It is kept so the effect of scaling
    can be measured rather than asserted.
    """
    estimators = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "SVM linear": SVC(kernel="linear", random_state=RANDOM_STATE),
        "SVM RBF": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "SVM poly": SVC(kernel="poly", random_state=RANDOM_STATE),
    }
    if not scaled:
        return {name: make_pipeline(est) for name, est in estimators.items()}
    return {name: make_pipeline(StandardScaler(), est) for name, est in estimators.items()}


def _svm_rbf_calibrated() -> Pipeline:
    """
    The configuration actually served: RBF with standardised features.

    The original README recommended a polynomial kernel; measured over the same
    data, poly is the *worst* of the five and RBF is the best.

    ``SVC(probability=True)`` is deprecated in scikit-learn 1.9, and the
    replacement is the better tool anyway: this endpoint returns a ranked list
    with probabilities, and ``CalibratedClassifierCV`` produces probabilities
    that mean something rather than Platt-scaled scores fitted by an internal
    cross-validation the caller cannot see.
    """
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False),
    )


def _logistic_regression() -> Pipeline:
    """Natively probabilistic, 0.9 points behind RBF, and far faster to fit."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    )


def _knn() -> Pipeline:
    """The weakest of the five, kept because it is the cheapest to retrain."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


register_classifier(DEFAULT_CLASSIFIER, _svm_rbf_calibrated)
register_classifier("logistic-regression", _logistic_regression)
register_classifier("knn", _knn)


def best_known() -> Pipeline:
    """The configuration that performs best on this feature set."""
    return get_classifier(DEFAULT_CLASSIFIER)
