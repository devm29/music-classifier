"""
Tests for the prediction path, end to end, without the shipped artifact.

`test_predict.py` checks the model in `artifacts/`, which is gitignored and so
absent from a fresh clone — those tests skip. These fit a model on synthetic
features, persist it, and load it back through the same code the web app uses,
so the path is covered on any checkout.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from classifier import predict as predict_module
from classifier.data import GENRES
from classifier.models import best_known
from classifier.predict import ModelUnavailable, load_model, predict


class TestLoading:
    def test_it_loads_the_persisted_model_and_metadata(self, trained_model):
        model, metadata = load_model()
        assert hasattr(model, "predict_proba")
        assert metadata["n_features"] == 104

    def test_the_model_is_loaded_once_and_cached(self, trained_model):
        # The web app classifies one upload per request; re-reading a joblib
        # file every time would be the slowest part of the response.
        assert load_model() is load_model()

    def test_a_missing_model_raises_a_message_that_says_what_to_do(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "absent.joblib")
        monkeypatch.setattr(predict_module, "METADATA_PATH", tmp_path / "absent.json")
        predict_module.load_model.cache_clear()
        try:
            with pytest.raises(ModelUnavailable, match="classifier.train"):
                load_model()
        finally:
            predict_module.load_model.cache_clear()

    def test_a_model_without_metadata_still_loads(self, tmp_path, monkeypatch, small_dataset):
        # Metadata is informational. Losing it should degrade the diagnostics,
        # not break classification.
        X, y = small_dataset
        model = best_known().fit(X, y)
        model_path = tmp_path / "m.joblib"
        joblib.dump(model, model_path)

        monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
        monkeypatch.setattr(predict_module, "METADATA_PATH", tmp_path / "missing.json")
        predict_module.load_model.cache_clear()
        try:
            loaded, metadata = load_model()
            assert metadata == {}
            assert loaded.predict(X[:1]).shape == (1,)
        finally:
            predict_module.load_model.cache_clear()


class TestPredictContract:
    def test_it_returns_exactly_top_k_entries(self, trained_model):
        X, _ = trained_model
        for k in (1, 3, len(GENRES)):
            assert len(predict(X[0], top_k=k)) == k

    def test_more_than_the_available_classes_is_not_padded(self, trained_model):
        X, _ = trained_model
        assert len(predict(X[0], top_k=99)) == len(GENRES)

    def test_results_are_ranked_by_probability(self, trained_model):
        X, _ = trained_model
        probabilities = [r["probability"] for r in predict(X[3], top_k=len(GENRES))]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_probabilities_are_a_distribution(self, trained_model):
        X, _ = trained_model
        full = predict(X[5], top_k=len(GENRES))
        assert all(0.0 <= r["probability"] <= 1.0 for r in full)
        assert sum(r["probability"] for r in full) == pytest.approx(1.0, abs=1e-3)

    def test_every_genre_name_is_a_known_genre(self, trained_model):
        X, _ = trained_model
        assert all(r["genre"] in GENRES for r in predict(X[0], top_k=len(GENRES)))

    def test_predictions_are_deterministic(self, trained_model):
        X, _ = trained_model
        assert predict(X[7], top_k=3) == predict(X[7], top_k=3)

    def test_it_accepts_a_plain_list_and_a_two_dimensional_row(self, trained_model):
        X, _ = trained_model
        from_array = predict(X[0], top_k=3)
        assert predict(list(X[0]), top_k=3) == from_array
        assert predict(X[0].reshape(1, -1), top_k=3) == from_array

    def test_a_wrong_length_vector_is_rejected(self, trained_model):
        with pytest.raises(ValueError):
            predict(np.zeros(7))

    def test_the_label_mapping_is_right(self, trained_model):
        # A wrong GENRES index would leave every other test passing while the
        # interface named the wrong genre for every upload.
        X, y = trained_model
        top = [predict(X[i], top_k=1)[0]["genre"] for i in range(0, len(y), 9)]
        truth = [GENRES[y[i]] for i in range(0, len(y), 9)]
        agree = sum(a == b for a, b in zip(top, truth, strict=False)) / len(truth)
        assert agree > 0.5


class TestSubsetModelLabelling:
    def test_a_six_genre_model_names_genres_by_their_global_ids(
        self, tmp_path, monkeypatch, small_dataset
    ):
        # The trap: a model fitted on classes [1, 9] has `classes_ == [1, 9]`,
        # not [0, 1]. Treating those as positions would report "blues" and
        # "classical" for every prediction.
        X, y = small_dataset
        wanted = [GENRES.index("classical"), GENRES.index("rock")]
        mask = np.isin(y, wanted)

        model = best_known().fit(X[mask], y[mask])
        model_path = tmp_path / "subset.joblib"
        joblib.dump(model, model_path)

        monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
        monkeypatch.setattr(predict_module, "METADATA_PATH", tmp_path / "none.json")
        predict_module.load_model.cache_clear()
        try:
            names = {r["genre"] for r in predict(X[mask][0], top_k=2)}
            assert names == {"classical", "rock"}
        finally:
            predict_module.load_model.cache_clear()


class TestAudioToPrediction:
    def test_a_wav_file_goes_all_the_way_through(self, trained_model, tmp_path):
        # The whole served path on a two-second synthetic tone: decode,
        # extract 104 features, score, rank. Nothing here needs GTZAN.
        from classifier.service import classify_file
        from tests.conftest import write_wav

        ranked = classify_file(write_wav(tmp_path / "tone.wav", seconds=2.0), top_k=3)
        assert len(ranked) == 3
        assert all(r["genre"] in GENRES for r in ranked)
        assert all(0.0 <= r["probability"] <= 1.0 for r in ranked)

    def test_an_unreadable_file_never_reaches_the_model(self, trained_model, tmp_path):
        from classifier.audio import AudioError
        from classifier.service import classify_file

        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"not audio")
        with pytest.raises(AudioError):
            classify_file(junk)
