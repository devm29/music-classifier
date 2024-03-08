"""
Tests for the nearest-neighbour explanation.

The neighbour panel is the falsifiable half of a prediction: if the five
closest training tracks are all jazz and the label says metal, the label is the
thing that is wrong. That only holds if the lookup is actually correct, so
these tests check the standardisation, the ordering, the row-to-track-name
mapping and the degenerate cases — against a synthetic matrix, so no GTZAN
audio and no download is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from classifier import neighbours
from classifier.data import GENRES, TRACKS_PER_GENRE, track_id
from classifier.neighbours import NeighbourIndex, similar_tracks
from tests.conftest import make_feature_matrix


@pytest.fixture
def synthetic_index(monkeypatch):
    """Build the index over a synthetic matrix instead of `data/Xall.npy`."""
    X, _ = make_feature_matrix()
    monkeypatch.setattr(neighbours, "load_features", lambda *a, **k: (X, None))
    neighbours.index.cache_clear()
    yield X
    neighbours.index.cache_clear()


class TestTrackNaming:
    """The row index is the only record of which GTZAN file a row came from."""

    def test_the_first_row_of_each_block_is_track_zero_of_that_genre(self):
        for i, genre in enumerate(GENRES):
            assert track_id(i * TRACKS_PER_GENRE) == f"{genre}.00000"

    def test_the_last_row_is_the_last_rock_track(self):
        assert track_id(999) == "rock.00099"

    def test_a_row_outside_the_matrix_is_an_index_error(self):
        for bad in (-1, 1000):
            with pytest.raises(IndexError):
                track_id(bad)


class TestQuery:
    def test_a_training_row_is_its_own_nearest_neighbour(self, synthetic_index):
        X = synthetic_index
        for row in (0, 137, 999):
            top = similar_tracks(X[row], k=1)[0]
            assert top["track"] == track_id(row)
            assert top["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_neighbours_come_back_in_increasing_distance(self, synthetic_index):
        found = similar_tracks(synthetic_index[42], k=8)
        distances = [n["distance"] for n in found]
        assert distances == sorted(distances)

    def test_k_controls_how_many_come_back(self, synthetic_index):
        for k in (1, 3, 10):
            assert len(similar_tracks(synthetic_index[7], k=k)) == k

    def test_neighbours_of_a_training_row_share_its_genre(self, synthetic_index):
        # The synthetic blocks are separated, so this is a check on the index
        # rather than on the data: a broken standardisation or a transposed
        # matrix would scatter the neighbours across genres.
        row = 3 * TRACKS_PER_GENRE + 5
        found = similar_tracks(synthetic_index[row], k=5)
        assert {n["genre"] for n in found} == {GENRES[3]}

    def test_the_genre_is_derived_from_the_track_name(self, synthetic_index):
        for n in similar_tracks(synthetic_index[500], k=4):
            assert n["track"].startswith(n["genre"] + ".")

    def test_closer_than_is_a_fraction(self, synthetic_index):
        for n in similar_tracks(synthetic_index[11], k=5):
            assert 0.0 <= n["closer_than"] <= 1.0

    def test_an_identical_vector_beats_almost_every_pair_in_the_dataset(
        self, synthetic_index
    ):
        assert similar_tracks(synthetic_index[11], k=1)[0]["closer_than"] >= 0.99

    def test_a_wrong_length_vector_is_rejected(self, synthetic_index):
        with pytest.raises(ValueError, match="expected 104 features"):
            similar_tracks(np.zeros(7))


class TestIndexConstruction:
    def test_the_index_is_built_once_and_cached(self, synthetic_index):
        # It is ~800 KB of float64 plus a full pairwise distance pass; rebuilding
        # it per request would dominate the response.
        assert neighbours.index() is neighbours.index()

    def test_the_standardised_matrix_is_centred_and_scaled(self, synthetic_index):
        index = neighbours.index()
        np.testing.assert_allclose(index.z_matrix.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(index.z_matrix.std(axis=0), 1.0, atol=1e-10)

    def test_a_zero_variance_column_does_not_poison_the_whole_row(self, monkeypatch):
        # Dividing by a zero standard deviation makes the entire distance NaN,
        # and argsort over NaN returns an arbitrary order — the neighbour panel
        # would silently show five unrelated tracks.
        X, _ = make_feature_matrix()
        X[:, 3] = 1.0
        monkeypatch.setattr(neighbours, "load_features", lambda *a, **k: (X, None))
        neighbours.index.cache_clear()
        try:
            found = similar_tracks(X[0], k=3)
            assert all(np.isfinite(n["distance"]) for n in found)
            assert found[0]["track"] == track_id(0)
        finally:
            neighbours.index.cache_clear()

    def test_the_index_is_a_frozen_dataclass(self):
        index = NeighbourIndex(
            z_matrix=np.zeros((2, 2)),
            mean=np.zeros(2),
            scale=np.ones(2),
            distance_quantiles=np.zeros(101),
        )
        with pytest.raises(Exception):
            index.mean = np.ones(2)
