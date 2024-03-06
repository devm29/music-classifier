"""
Shared fixtures.

Everything here is synthetic and small. No test in this suite needs the GTZAN
audio, a network download, a GPU, or a training run longer than a second or
two: the feature matrix is generated, and the models are fitted on a few dozen
rows. The tests that do touch the repository's real `Xall.npy` (it is committed,
the audio is not) are in `test_data.py` and `test_evaluation.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from classifier.data import GENRES, TRACKS_PER_GENRE

#: Number of MFCC-derived features the model consumes.
N_FEATURES = 104

#: Seed for every synthetic fixture. Fixed so a failure is reproducible.
SEED = 20250922


def make_feature_matrix(
    n_genres: int = len(GENRES),
    per_genre: int = TRACKS_PER_GENRE,
    n_features: int = N_FEATURES,
    separation: float = 6.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A learnable stand-in for the real feature matrix.

    Rows are laid out in per-genre blocks, exactly like `Xall.npy`, because the
    label reconstruction in `classifier.data` depends on that ordering. Feature
    scales deliberately span two orders of magnitude, mirroring the real
    matrix's 1.86-to-59.66 spread, so that tests about scaling are testing the
    same phenomenon the real data has.
    """
    rng = np.random.default_rng(seed)
    scales = np.geomspace(1.0, 60.0, n_features)
    centres = rng.normal(size=(n_genres, n_features)) * separation * scales

    blocks = [
        centres[g] + rng.normal(size=(per_genre, n_features)) * scales
        for g in range(n_genres)
    ]
    X = np.vstack(blocks)
    y = np.repeat(np.arange(n_genres), per_genre)
    return X, y


@pytest.fixture
def synthetic_dataset():
    """A full-size (1000 x 104) synthetic matrix with block-ordered labels."""
    return make_feature_matrix()


@pytest.fixture
def synthetic_features_file(tmp_path):
    """A synthetic matrix written to a .npy, loadable by `load_features`."""
    X, _ = make_feature_matrix()
    path = tmp_path / "Xall.npy"
    np.save(path, X)
    return path


@pytest.fixture
def small_dataset():
    """A handful of rows per genre — enough to fit, fast enough to fit often."""
    return make_feature_matrix(per_genre=12)


@pytest.fixture
def trained_model(tmp_path, monkeypatch, small_dataset):
    """
    A real, freshly fitted model persisted and wired up for `predict`.

    This covers the production prediction path — joblib round-trip, metadata,
    `model.classes_` to genre-name mapping — without depending on the model
    artifact in `artifacts/`, which is gitignored and absent from a fresh clone.
    """
    from classifier import predict as predict_module
    from classifier.models import best_known

    X, y = small_dataset
    model = best_known()
    model.fit(X, y)

    model_path = tmp_path / "genre_classifier.joblib"
    metadata_path = tmp_path / "genre_classifier.json"

    import joblib
    import sklearn

    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "genres": list(GENRES),
                "class_ids": sorted(int(v) for v in np.unique(y)),
                "cv_accuracy_mean": 0.5,
                "cv_accuracy_std": 0.01,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "sklearn_version": sklearn.__version__,
            }
        )
    )

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)
    predict_module.load_model.cache_clear()
    yield X, y
    predict_module.load_model.cache_clear()


def write_wav(path, seconds=5.0, rate=22050, channels=1, seed=1, dtype=np.int16):
    """Write a short synthetic tone to `path` and return it."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    peak = np.abs(signal).max() or 1.0
    data = (signal / peak * 32767).astype(dtype)
    if channels == 2:
        data = np.column_stack([data, data])
    wavfile.write(path, rate, data)
    return path
