"""
Tests for the persisted model and the prediction contract.

The repository shipped four scikit-learn 0.18 pickles that no currently
installable version can unpickle. The first test here is the one that would
have caught that: load the model that is actually shipped, with the library
that is actually pinned.
"""

from __future__ import annotations

import pytest

from classifier.data import GENRES, load_features
from classifier.predict import ModelUnavailable, load_model, predict


@pytest.fixture(scope="module")
def trained():
    try:
        return load_model()
    except ModelUnavailable:
        pytest.skip("no model trained; run `python -m classifier.train`")


class TestTheShippedModelLoads:
    def test_it_unpickles_with_the_installed_sklearn(self, trained):
        model, _ = trained
        assert hasattr(model, "predict_proba")

    def test_it_records_the_versions_it_was_built_with(self, trained):
        _, metadata = trained
        assert metadata["sklearn_version"]
        assert metadata["numpy_version"]

    def test_the_recorded_version_matches_the_installed_one(self, trained):
        # The whole reason the original models died: they were written by one
        # version and never checked against another.
        import sklearn

        _, metadata = trained
        assert metadata["sklearn_version"] == sklearn.__version__, (
            "the shipped model was trained with a different scikit-learn than "
            "the one installed; retrain with `python -m classifier.train`"
        )

    def test_the_recorded_accuracy_is_out_of_sample(self, trained):
        _, metadata = trained
        assert 0 < metadata["cv_accuracy_mean"] < 1
        assert metadata["cv_accuracy_mean"] > 1 / len(GENRES)


class TestPredictionContract:
    def test_it_returns_ranked_genres_with_probabilities(self, trained):
        X, _ = load_features()
        result = predict(X[0], top_k=3)
        assert len(result) == 3
        assert all(r["genre"] in GENRES for r in result)
        probabilities = [r["probability"] for r in result]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_probabilities_are_valid(self, trained):
        X, _ = load_features()
        for r in predict(X[10], top_k=10):
            assert 0.0 <= r["probability"] <= 1.0

    def test_top_k_is_respected(self, trained):
        X, _ = load_features()
        assert len(predict(X[0], top_k=1)) == 1

    def test_it_is_right_more_often_than_chance(self, trained):
        # Not an accuracy assertion — a smoke test that the model and the
        # label mapping agree. A wrong GENRES index would score near zero here
        # while every other test still passed.
        X, y = load_features()
        sample = range(0, len(y), 37)
        correct = sum(predict(X[i], top_k=1)[0]["genre"] == GENRES[y[i]] for i in sample)
        assert correct / len(list(sample)) > 0.4
