"""
Fit a registered classifier on the preserved feature matrix and persist it.

The repository once shipped five pickles under ``mysvm/data/``, all produced by
scikit-learn 0.18 in 2017. None could be loaded by any currently installable
scikit-learn: unpickling raised ``ModuleNotFoundError: No module named
'sklearn.svm.classes'``, because that module was removed in 0.22. The web app
therefore could not classify anything on a modern install, and nothing in the
repository said so. Those files are gone; this is what replaces them.

The artifact records what produced it — classifier name, **feature extractor
name and version**, sample rate, library versions — so ``classifier.predict``
can refuse to serve a model whose feature contract does not match the installed
extractor, instead of silently scoring the wrong numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score

from classifier.data import GENRES, WEB_APP_GENRES, load_features, subset
from classifier.features import get_extractor
from classifier.models import (
    DEFAULT_CLASSIFIER,
    RANDOM_STATE,
    available_classifiers,
    get_classifier,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = MODEL_DIR / "genre_classifier.joblib"
METADATA_PATH = MODEL_DIR / "genre_classifier.json"


def train(
    genres: list[str] | None = None,
    classifier: str | None = None,
    extractor: str | None = None,
) -> tuple[object, dict]:
    X, y = load_features()
    genres = genres or GENRES
    if genres != GENRES:
        X, y = subset(X, y, genres)

    extractor_impl = get_extractor(extractor)
    if X.shape[1] != extractor_impl.n_features:
        raise ValueError(
            f"the feature matrix has {X.shape[1]} columns but extractor "
            f"{extractor_impl.name!r} produces {extractor_impl.n_features}. "
            "Training on features the serving path cannot reproduce is exactly "
            "the skew this metadata exists to prevent."
        )

    classifier = classifier or DEFAULT_CLASSIFIER
    model = get_classifier(classifier)

    # Score before fitting on everything, so the number recorded with the
    # model is an out-of-sample estimate rather than training accuracy.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)

    model.fit(X, y)

    metadata = {
        "genres": genres,
        # The model predicts positions in the *original* GENRES list, so the
        # mapping must be stored — a subset model's class ids are not 0..n-1.
        "class_ids": sorted(int(v) for v in np.unique(y)),
        "classifier": classifier,
        "extractor": extractor_impl.name,
        "extractor_version": extractor_impl.version,
        "sample_rate": extractor_impl.sample_rate,
        "clip_seconds": extractor_impl.clip_seconds,
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(UTC).isoformat(),
        "random_state": RANDOM_STATE,
    }
    return model, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--six", action="store_true",
                        help="train on the six genres the web app originally offered")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        choices=available_classifiers(),
                        help="which registered classifier to fit")
    args = parser.parse_args()

    genres = WEB_APP_GENRES if args.six else GENRES
    model, metadata = train(genres, classifier=args.classifier)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"  classifier  {metadata['classifier']}")
    print(f"  extractor   {metadata['extractor']} v{metadata['extractor_version']} "
          f"@ {metadata['sample_rate']} Hz")
    print(f"  genres      {len(metadata['genres'])}: {', '.join(metadata['genres'])}")
    print(f"  accuracy    {metadata['cv_accuracy_mean']:.3f} "
          f"+/- {metadata['cv_accuracy_std']:.3f}  (5-fold, out of sample)")
    print(f"  sklearn     {metadata['sklearn_version']}")
    print(f"  written     {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
