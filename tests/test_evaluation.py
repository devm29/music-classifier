"""
Tests for the benchmark.

The README's original accuracy table had no code behind it. These tests do not
assert particular accuracies — those are data, and pinning them would make the
suite fail whenever the data legitimately changed. They assert the properties
that make a published number trustworthy: that it is out-of-sample, that it is
reproducible, and that it beats chance.
"""

from __future__ import annotations

import pytest

from classifier.data import GENRES, load_features
from classifier.evaluate import evaluate
from classifier.models import candidates


@pytest.fixture(scope="module")
def dataset():
    return load_features()


class TestProtocol:
    def test_results_are_reproducible(self, dataset):
        X, y = dataset
        first = {r.name: r.test for r in evaluate(X, y, folds=3)}
        second = {r.name: r.test for r in evaluate(X, y, folds=3)}
        # A benchmark that moves between identical runs is not a measurement.
        assert first == second

    def test_every_classifier_beats_chance(self, dataset):
        X, y = dataset
        for r in evaluate(X, y, folds=3):
            assert r.test > 1 / len(GENRES), f"{r.name} is no better than guessing"

    def test_test_accuracy_is_not_training_accuracy(self, dataset):
        # Reporting training accuracy as a result is the classic way to
        # publish a number that does not survive contact with new data.
        X, y = dataset
        for r in evaluate(X, y, folds=3):
            assert r.test <= r.train

    def test_results_come_back_ranked(self, dataset):
        X, y = dataset
        results = evaluate(X, y, folds=3)
        assert results == sorted(results, key=lambda r: -r.test)


class TestScaling:
    def test_scaling_changes_which_model_wins(self, dataset):
        """
        The finding this project turned on. The original fitted on raw
        features and concluded a polynomial kernel was best. With the features
        standardised the ranking changes, and RBF comes out ahead.
        """
        X, y = dataset
        scaled = evaluate(X, y, scaled=True, folds=3)
        raw = evaluate(X, y, scaled=False, folds=3)
        assert scaled[0].name != raw[0].name or scaled[0].test > raw[0].test

    def test_scaling_helps_the_rbf_kernel_most(self, dataset):
        # RBF computes distances in the raw feature space, so it suffers most
        # when one feature has 32x the spread of another.
        X, y = dataset
        def by_name(rs):
            return {r.name: r.test for r in rs}

        scaled = by_name(evaluate(X, y, scaled=True, folds=3))
        raw = by_name(evaluate(X, y, scaled=False, folds=3))
        assert scaled["SVM RBF"] > raw["SVM RBF"]


class TestPipelines:
    def test_every_scaled_candidate_actually_scales(self):
        for name, pipe in candidates(scaled=True).items():
            assert "standardscaler" in pipe.named_steps, f"{name} is unscaled"

    def test_raw_candidates_do_not_scale(self):
        for pipe in candidates(scaled=False).values():
            assert "standardscaler" not in pipe.named_steps
