"""
Tests for the feature matrix and its labels.

The labels are not stored anywhere — they are implied by the row ordering of
`Xall.npy` (GTZAN is 10 genres of exactly 100 tracks, in blocks). That is the
most fragile assumption in the project: if the matrix were ever regenerated
with different data, every label would silently be wrong and every accuracy
figure meaningless. These tests make the assumption explicit and loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from classifier.data import (
    GENRES,
    TRACKS_PER_GENRE,
    WEB_APP_GENRES,
    label_names,
    load_features,
    subset,
)


class TestFeatureMatrix:
    def test_it_loads_with_the_expected_shape(self):
        X, y = load_features()
        assert X.shape == (len(GENRES) * TRACKS_PER_GENRE, 104)
        assert y.shape == (X.shape[0],)

    def test_every_genre_is_equally_represented(self):
        _, y = load_features()
        counts = np.bincount(y)
        assert set(counts) == {TRACKS_PER_GENRE}

    def test_labels_run_in_blocks_matching_the_genre_order(self):
        _, y = load_features()
        assert y[0] == 0 and y[99] == 0          # first block is blues
        assert y[100] == 1                        # second is classical
        assert y[-1] == len(GENRES) - 1           # last is rock

    def test_a_matrix_of_the_wrong_size_is_refused(self, tmp_path):
        # Silently mislabelling every row would be far worse than failing.
        bad = tmp_path / "wrong.npy"
        np.save(bad, np.zeros((37, 104)))
        with pytest.raises(ValueError, match="assumes"):
            load_features(bad)

    def test_features_are_not_on_a_common_scale(self):
        # The fact this project's original results depended on: standard
        # deviations span more than an order of magnitude, so distance-based
        # models are dominated by a few wide features unless they are scaled.
        X, _ = load_features()
        spread = X.std(axis=0)
        assert spread.max() / spread.min() > 10


class TestSubset:
    def test_it_selects_only_the_named_genres(self):
        X, y = load_features()
        Xs, ys = subset(X, y, WEB_APP_GENRES)
        assert Xs.shape[0] == len(WEB_APP_GENRES) * TRACKS_PER_GENRE
        assert {GENRES[i] for i in np.unique(ys)} == set(WEB_APP_GENRES)

    def test_it_keeps_original_class_ids(self):
        # A subset model's classes are not 0..n-1, and the prediction code
        # maps them back through GENRES. Renumbering here would mislabel
        # every prediction the web app makes.
        X, y = load_features()
        _, ys = subset(X, y, ["classical", "rock"])
        assert sorted(np.unique(ys)) == [GENRES.index("classical"), GENRES.index("rock")]

    def test_an_unknown_genre_is_refused(self):
        X, y = load_features()
        with pytest.raises(ValueError, match="unknown genre"):
            subset(X, y, ["polka"])


class TestLoadingIsSafeAndStrict:
    def test_it_loads_a_matrix_from_an_explicit_path(self, synthetic_features_file):
        # The loader must work on any correctly shaped matrix, not only the
        # one committed at the default location.
        X, y = load_features(synthetic_features_file)
        assert X.shape == (len(GENRES) * TRACKS_PER_GENRE, 104)
        assert np.array_equal(y, np.repeat(np.arange(len(GENRES)), TRACKS_PER_GENRE))

    def test_a_pickled_npy_is_refused(self, tmp_path):
        # `np.load(..., allow_pickle=True)` executes arbitrary code from the
        # file. The feature matrix is plain floats and needs none of it.
        malicious = tmp_path / "pickled.npy"
        np.save(malicious, np.array([{"payload": "anything"}], dtype=object),
                allow_pickle=True)
        with pytest.raises(ValueError):
            load_features(malicious)

    def test_a_one_dimensional_array_is_refused(self, tmp_path):
        flat = tmp_path / "flat.npy"
        np.save(flat, np.zeros(1000))
        with pytest.raises(ValueError, match="dimensional"):
            load_features(flat)

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_features(tmp_path / "absent.npy")

    def test_the_committed_matrix_is_finite(self):
        # A NaN anywhere in here would poison every model silently: sklearn
        # would reject it at fit time with a message about the array, not the
        # file.
        X, _ = load_features()
        assert np.all(np.isfinite(X))


class TestLabelNames:
    def test_it_maps_indices_to_genre_names(self):
        assert label_names(np.array([0, 9])) == ["blues", "rock"]

    def test_it_round_trips_with_the_reconstructed_labels(self):
        _, y = load_features()
        names = label_names(y)
        assert len(names) == len(y)
        assert names[0] == GENRES[0] and names[-1] == GENRES[-1]


class TestSubsetKeepsRowsAligned:
    def test_rows_and_labels_stay_paired(self, synthetic_dataset):
        # The failure that matters: a mask applied to X but not y (or applied
        # in a different order) would mislabel everything while every shape
        # assertion still passed.
        X, y = synthetic_dataset
        Xs, ys = subset(X, y, ["country", "metal"])
        for row, label in zip(Xs, ys, strict=False):
            original = np.flatnonzero((row == X).all(axis=1))
            assert label in y[original]

    def test_the_order_of_the_requested_genres_does_not_matter(self, synthetic_dataset):
        X, y = synthetic_dataset
        a = subset(X, y, ["metal", "blues"])
        b = subset(X, y, ["blues", "metal"])
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_an_empty_selection_returns_nothing_rather_than_everything(
        self, synthetic_dataset
    ):
        X, y = synthetic_dataset
        Xs, ys = subset(X, y, [])
        assert Xs.shape[0] == 0 and ys.shape[0] == 0

    def test_a_single_genre_is_allowed(self, synthetic_dataset):
        X, y = synthetic_dataset
        Xs, ys = subset(X, y, ["jazz"])
        assert set(np.unique(ys)) == {GENRES.index("jazz")}
        assert Xs.shape[0] == TRACKS_PER_GENRE
