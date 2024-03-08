"""
Nearest-neighbour lookup over the GTZAN feature matrix.

A probability vector says *what* the model thinks. It does not say *why*, and
for a 71%-accurate model "why" is the difference between a user trusting the
answer and ignoring it. The cheapest honest explanation available here is:
which tracks in the training set does this upload actually sit next to?

The same 104 features, standardised the same way the classifier standardises
them, with plain Euclidean distance. No new model, no new dependency, and it
is falsifiable — if the neighbours are all jazz and the prediction is metal,
the prediction is the thing that is wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import pdist

from classifier.data import load_features, track_id

#: Sample of intra-dataset distances used to turn a raw distance into a
#: percentile. Full pairwise over 1000 rows is cheap; this is the whole matrix.
_QUANTILE_POINTS = np.linspace(0, 1, 101)


@dataclass(frozen=True)
class Neighbour:
    track: str
    genre: str
    distance: float
    #: Fraction of GTZAN track pairs that are *further* apart than this one.
    closer_than: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NeighbourIndex:
    """Standardised training matrix plus the distance distribution over it."""

    z_matrix: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    distance_quantiles: np.ndarray

    def query(self, features: np.ndarray, k: int = 5) -> list[Neighbour]:
        vector = np.asarray(features, dtype=float).ravel()
        # Checked before the standardisation, not after: subtracting a
        # 104-vector from a 7-vector raises a numpy broadcast error that names
        # neither this function nor the actual problem.
        if vector.shape[0] != self.z_matrix.shape[1]:
            raise ValueError(
                f"expected {self.z_matrix.shape[1]} features, got {vector.shape[0]}"
            )
        vector = (vector - self.mean) / self.scale
        distances = np.linalg.norm(self.z_matrix - vector, axis=1)
        order = np.argsort(distances)[:k]
        neighbours = []
        for row in order:
            name = track_id(int(row))
            neighbours.append(
                Neighbour(
                    track=name,
                    genre=name.split(".")[0],
                    distance=round(float(distances[row]), 3),
                    closer_than=round(
                        1.0
                        - np.searchsorted(self.distance_quantiles, distances[row]) / 100.0,
                        2,
                    ),
                )
            )
        return neighbours


@lru_cache(maxsize=1)
def index() -> NeighbourIndex:
    """
    Build the index once per process.

    It is ~800 KB of float64 and takes a few hundred milliseconds to build, so
    it is cached exactly like the model. ``web.apps`` warms both together.
    """
    X, _ = load_features()
    mean = X.mean(axis=0)
    # Guard a zero-variance column: it would make the whole row NaN.
    scale = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    z = (X - mean) / scale

    # Condensed pairwise distances. `pdist` keeps this to the ~500k-element
    # upper triangle; materialising the full 1000x1000x104 difference tensor
    # would be 800 MB for a number we only need percentiles of.
    return NeighbourIndex(
        z_matrix=z,
        mean=mean,
        scale=scale,
        distance_quantiles=np.quantile(pdist(z), _QUANTILE_POINTS),
    )


def similar_tracks(features: np.ndarray, k: int = 5) -> list[dict]:
    """The k GTZAN tracks whose features sit closest to this vector."""
    return [n.as_dict() for n in index().query(features, k=k)]
