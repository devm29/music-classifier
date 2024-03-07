"""
Tests for the training entry point.

These run on a synthetic feature matrix rather than the real one: training is
the slowest thing in the project, and what matters here is not the accuracy it
reaches but that the metadata it records is true. A model persisted with a
wrong `class_ids` or a training-set accuracy labelled "cv" is exactly how this
repository ended up shipping four unloadable pickles and an unreproducible
table.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from classifier import train as train_module
from classifier.data import GENRES, WEB_APP_GENRES
from classifier.models import RANDOM_STATE


@pytest.fixture
def synthetic_training(monkeypatch, small_dataset):
    """Point `train()` at a small synthetic matrix instead of `Xall.npy`."""
    monkeypatch.setattr(train_module, "load_features", lambda: small_dataset)
    return small_dataset


class TestTrain:
    def test_it_returns_a_fitted_model(self, synthetic_training):
        model, _ = train_module.train()
        X, _ = synthetic_training
        assert model.predict_proba(X[:1]).shape == (1, len(GENRES))

    def test_metadata_describes_the_data_it_was_fitted_on(self, synthetic_training):
        X, _ = synthetic_training
        _, metadata = train_module.train()
        assert metadata["n_samples"] == X.shape[0]
        assert metadata["n_features"] == X.shape[1]
        assert metadata["genres"] == GENRES
        assert metadata["random_state"] == RANDOM_STATE

    def test_metadata_records_the_libraries_that_produced_it(self, synthetic_training):
        # The check that would have caught the 2017 pickles: a model is only
        # useful if you can tell what can still load it.
        import sklearn

        _, metadata = train_module.train()
        assert metadata["sklearn_version"] == sklearn.__version__
        assert metadata["numpy_version"] == np.__version__
        assert metadata["python_version"]
        assert metadata["trained_at"].endswith("+00:00"), "timestamp must be UTC"

    def test_the_recorded_accuracy_is_cross_validated_not_training_accuracy(
        self, synthetic_training
    ):
        X, y = synthetic_training
        model, metadata = train_module.train()
        in_sample = model.score(X, y)
        assert 0.0 <= metadata["cv_accuracy_mean"] <= 1.0
        assert metadata["cv_accuracy_mean"] <= in_sample + 1e-9, (
            "the persisted accuracy is not out-of-sample"
        )
        assert metadata["cv_accuracy_std"] >= 0.0

    def test_training_is_reproducible(self, synthetic_training):
        # Everything here is seeded. Two runs that disagree mean a seed went
        # missing, and the number written next to the model stops meaning
        # anything.
        first = train_module.train()[1]["cv_accuracy_mean"]
        second = train_module.train()[1]["cv_accuracy_mean"]
        assert first == second

    def test_the_metadata_is_json_serialisable(self, synthetic_training):
        # It is written out with `json.dumps`; a stray numpy scalar in there
        # would fail only at the very end of a training run.
        _, metadata = train_module.train()
        json.dumps(metadata)


class TestSubsetTraining:
    def test_it_trains_on_only_the_requested_genres(self, synthetic_training):
        model, metadata = train_module.train(WEB_APP_GENRES)
        assert metadata["genres"] == WEB_APP_GENRES
        assert len(model.classes_) == len(WEB_APP_GENRES)

    def test_a_subset_keeps_the_original_class_ids(self, synthetic_training):
        # A subset model's classes are not 0..n-1. `predict` maps them back
        # through GENRES, so renumbering them here would mislabel every
        # prediction the six-genre web app makes.
        _, metadata = train_module.train(["classical", "rock"])
        assert metadata["class_ids"] == [
            GENRES.index("classical"),
            GENRES.index("rock"),
        ]

    def test_class_ids_match_the_fitted_model(self, synthetic_training):
        model, metadata = train_module.train(WEB_APP_GENRES)
        assert sorted(int(c) for c in model.classes_) == metadata["class_ids"]

    def test_an_unknown_genre_is_refused(self, synthetic_training):
        with pytest.raises(ValueError, match="unknown genre"):
            train_module.train(["klezmer"])


class TestPersistence:
    def test_main_writes_a_model_and_its_metadata(
        self, synthetic_training, tmp_path, monkeypatch, capsys
    ):
        model_path = tmp_path / "genre_classifier.joblib"
        metadata_path = tmp_path / "genre_classifier.json"
        monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
        monkeypatch.setattr(train_module, "MODEL_PATH", model_path)
        monkeypatch.setattr(train_module, "METADATA_PATH", metadata_path)
        monkeypatch.setattr("sys.argv", ["classifier.train"])

        assert train_module.main() == 0
        assert model_path.exists()

        written = json.loads(metadata_path.read_text())
        assert written["genres"] == GENRES
        assert "accuracy" in capsys.readouterr().out

    def test_the_six_flag_trains_the_web_app_subset(
        self, synthetic_training, tmp_path, monkeypatch
    ):
        metadata_path = tmp_path / "genre_classifier.json"
        monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
        monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "m.joblib")
        monkeypatch.setattr(train_module, "METADATA_PATH", metadata_path)
        monkeypatch.setattr("sys.argv", ["classifier.train", "--six"])

        assert train_module.main() == 0
        assert json.loads(metadata_path.read_text())["genres"] == WEB_APP_GENRES
