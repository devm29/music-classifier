"""
Load the persisted model once, and classify feature vectors with it.

Loading is process-wide and cached: the web app classifies one upload per
request, and re-reading a joblib file every time would be the slowest thing in
the response by an order of magnitude. ``web.apps`` warms this at startup so
the first user does not pay for it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from classifier.data import GENRES
from classifier.features import get_extractor
from classifier.train import METADATA_PATH, MODEL_PATH


class ModelUnavailable(RuntimeError):
    """Raised when no usable model has been trained yet."""


class ExtractorMismatch(RuntimeError):
    """
    Raised when the model was trained on features this build cannot reproduce.

    Serving anyway is the failure mode this project already had once: numbers
    that look like predictions but are computed over a different feature
    definition. Failing loudly is the whole point of recording the extractor.
    """


@lru_cache(maxsize=1)
def load_model() -> tuple[object, dict]:
    if not MODEL_PATH.exists():
        raise ModelUnavailable(
            f"No model at {MODEL_PATH}. Run `python -m classifier.train` first."
        )
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}

    # Older artifacts predate the extractor field; absent means "the only
    # extractor that ever existed", which is the default.
    recorded = metadata.get("extractor", get_extractor().name)
    installed = get_extractor()
    if recorded != installed.name:
        raise ExtractorMismatch(
            f"the model at {MODEL_PATH.name} was trained with feature extractor "
            f"{recorded!r}, but {installed.name!r} is installed. Retrain, or "
            "register the extractor the model expects."
        )
    return joblib.load(MODEL_PATH), metadata


def model_info() -> dict:
    """Metadata about the served model, for the health and info endpoints."""
    _, metadata = load_model()
    extractor = get_extractor()
    return {
        "genres": metadata.get("genres", list(GENRES)),
        "classifier": metadata.get("classifier", "unknown"),
        "extractor": metadata.get("extractor", extractor.name),
        "extractor_version": metadata.get("extractor_version", extractor.version),
        "sample_rate": metadata.get("sample_rate", extractor.sample_rate),
        "clip_seconds": metadata.get("clip_seconds", extractor.clip_seconds),
        "cv_accuracy_mean": metadata.get("cv_accuracy_mean"),
        "cv_accuracy_std": metadata.get("cv_accuracy_std"),
        "sklearn_version": metadata.get("sklearn_version"),
        "trained_at": metadata.get("trained_at"),
        "path": str(Path(MODEL_PATH).name),
    }


def predict(features: np.ndarray, top_k: int = 3) -> list[dict]:
    """
    Classify one feature vector, returning ranked genres with probabilities.

    Returning a ranked list rather than a single label is deliberate: at 71%
    accuracy the top guess is wrong about one time in four, and a caller that
    can see the runner-up can tell a confident answer from a coin flip.
    """
    model, _ = load_model()
    features = np.asarray(features, dtype=float).reshape(1, -1)

    probabilities = model.predict_proba(features)[0]
    class_ids = list(model.classes_)

    ranked = sorted(zip(class_ids, probabilities, strict=False), key=lambda p: -p[1])[:top_k]
    return [
        {"genre": GENRES[int(cid)], "probability": round(float(p), 4)}
        for cid, p in ranked
    ]
