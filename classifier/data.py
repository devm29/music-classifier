"""
Loading the extracted feature matrix and its labels.

The repository ships `data/Xall.npy` — a (1000, 104) matrix of features
extracted from the GTZAN collection. The audio itself is not here (and should
not be: it is ~1.2GB and not ours to redistribute), but the features are, and
they are what every result in this project rests on.

The labels are not stored anywhere. They are implied by the *ordering* of the
matrix: GTZAN is 10 genres of exactly 100 tracks each, laid out in blocks, and
the original code reconstructed the label vector with
`np.ones(n), np.ones(n)*2, ...`. That is a real fragility — the ground truth
lives in an undocumented convention about row order — so it is reconstructed
here in one place, checked, and explained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Genre order in the feature matrix. Taken from the 2017 `getlabels()`,
#: which is the only record of what row block corresponds to which genre.
#: `tests/test_features.py` keeps a transcription of that code as the
#: reference implementation the current extractor is pinned against.
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

TRACKS_PER_GENRE = 100

#: The six genres the original web app was restricted to, per the README.
WEB_APP_GENRES = ["classical", "hiphop", "jazz", "metal", "pop", "rock"]

DEFAULT_FEATURES = Path(__file__).resolve().parent.parent / "data" / "Xall.npy"


def load_features(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) with y as genre-name indices into GENRES.

    Raises if the matrix does not have the shape the block-ordering assumption
    requires — silently mislabelling every row would be far worse than failing.
    """
    path = Path(path) if path is not None else DEFAULT_FEATURES
    # `allow_pickle=True` lets a crafted .npy execute arbitrary code on load.
    # The matrix is a plain float array and needs none of it.
    X = np.load(path, allow_pickle=False)

    if X.ndim != 2:
        raise ValueError(f"{path.name} is {X.ndim}-dimensional; a 2-D matrix is required.")

    expected = len(GENRES) * TRACKS_PER_GENRE
    if X.shape[0] != expected:
        raise ValueError(
            f"{path.name} has {X.shape[0]} rows; the label reconstruction "
            f"assumes {len(GENRES)} genres x {TRACKS_PER_GENRE} tracks = {expected}. "
            "If the feature matrix was regenerated with different data, the "
            "labels must be stored alongside it rather than inferred."
        )

    y = np.repeat(np.arange(len(GENRES)), TRACKS_PER_GENRE)
    return X, y


def subset(X: np.ndarray, y: np.ndarray, genres: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the dataset to a named subset of genres."""
    unknown = [g for g in genres if g not in GENRES]
    if unknown:
        raise ValueError(f"unknown genre(s): {unknown}; known: {GENRES}")
    wanted = [GENRES.index(g) for g in genres]
    mask = np.isin(y, wanted)
    return X[mask], y[mask]


def label_names(y: np.ndarray) -> list[str]:
    return [GENRES[i] for i in y]


def track_id(row: int) -> str:
    """
    The GTZAN filename a matrix row came from, e.g. row 327 -> ``jazz.00027``.

    The rows are the sorted glob of ``wav/<genre>/*.wav`` — 10 blocks of 100 in
    genre order — and GTZAN names its files ``<genre>.000NN.au``. So the row
    index recovers the source track exactly, which is what makes the
    nearest-neighbour panel nameable. It is the same undocumented convention
    the labels rest on; if the matrix is ever regenerated from other audio,
    both this and ``load_features`` must be revisited together.
    """
    if not 0 <= row < len(GENRES) * TRACKS_PER_GENRE:
        raise IndexError(f"row {row} is outside the {len(GENRES) * TRACKS_PER_GENRE}-row matrix")
    return f"{GENRES[row // TRACKS_PER_GENRE]}.{row % TRACKS_PER_GENRE:05d}"
