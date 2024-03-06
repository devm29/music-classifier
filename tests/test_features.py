"""
Tests for the feature extractor and its registry.

The point of `classifier.features` is that training and serving use *one*
implementation. Two things therefore need pinning, and both are here:

1. The current extractor reproduces the 2017 maths exactly. `data/Xall.npy`
   was produced by `mysvm/feature.py`; if the extractor drifts from it, every
   number in the README becomes a claim about a matrix nobody can regenerate.
   `reference_extract_2017` below is an independent transcription of that
   code's inner loop, not an import of it — the original module is gone.
2. The preprocessing contract (rate, channels, duration) belongs to the
   extractor rather than to whichever caller happens to invoke it, because
   that split is what allowed the two copies to diverge in the first place.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc

from classifier.features import (
    DEFAULT_EXTRACTOR,
    AudioClip,
    FeatureError,
    FeatureExtractor,
    MfccStats104,
    available_extractors,
    get_extractor,
    register_extractor,
)


def reference_extract_2017(data, rate):
    """
    The 2017 `extract_all` inner loop, transcribed.

        mm = np.transpose(mfcc(data, rate))
        ff = np.mean(mm, axis=1)
        cf = np.cov(mm)
        for i in range(mm.shape[0]):
            ff = np.append(ff, np.diag(cf, i))

    This is the maths that produced every row of `data/Xall.npy`.
    """
    mm = np.transpose(mfcc(data, rate))
    ff = np.mean(mm, axis=1)
    cf = np.cov(mm)
    for i in range(mm.shape[0]):
        ff = np.append(ff, np.diag(cf, i))
    return ff


def tone(seconds=6.0, rate=22050, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    signal = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    return (signal / np.abs(signal).max() * 32000).astype(np.int16)


class TestItReproducesTheTrainingMaths:
    @pytest.mark.parametrize("seconds", [2.0, 6.0, 30.0])
    def test_it_matches_the_2017_reference_exactly(self, seconds):
        samples = tone(seconds)
        clip = AudioClip(samples.astype(np.float64), 22050)
        np.testing.assert_allclose(
            get_extractor().extract(clip),
            reference_extract_2017(samples.astype(np.float64), 22050),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_the_dimension_count_is_the_matrix_width(self):
        # 13 means + 13 + 12 + ... + 1 covariance diagonals.
        assert MfccStats104.n_features == 13 + sum(range(1, 14)) == 104

    def test_the_real_feature_matrix_has_that_width(self):
        from classifier.data import load_features

        X, _ = load_features()
        assert X.shape[1] == get_extractor().n_features


class TestThePreprocessingContract:
    """The contract travels with the extractor. That is the anti-skew rule."""

    def test_a_clip_at_the_wrong_rate_is_refused(self):
        with pytest.raises(FeatureError, match="expects 22050 Hz"):
            get_extractor().extract(AudioClip(tone(rate=44100).astype(float), 44100))

    def test_a_stereo_clip_is_refused(self):
        stereo = np.column_stack([tone(2.0), tone(2.0)]).astype(float)
        with pytest.raises(FeatureError, match="expects mono"):
            get_extractor().extract(AudioClip(stereo, 22050))

    def test_an_empty_clip_is_refused(self):
        with pytest.raises(FeatureError, match="no audio samples"):
            get_extractor().extract(AudioClip(np.array([]), 22050))

    def test_a_clip_with_one_mfcc_frame_is_refused(self):
        # np.cov over a single observation divides by N-1 = 0 and returns NaN,
        # so this used to produce a 104-vector with 91 NaNs in it.
        with pytest.raises(FeatureError, match="too short"):
            get_extractor().extract(AudioClip(np.zeros(300) + 100.0, 22050))

    def test_features_are_finite_or_it_raises(self):
        features = get_extractor().extract(AudioClip(tone().astype(float), 22050))
        assert np.all(np.isfinite(features))


class TestAudioClip:
    def test_duration_is_samples_over_rate(self):
        assert AudioClip(np.zeros(22050), 22050).duration_seconds == 1.0

    def test_slicing_takes_the_requested_window(self):
        clip = AudioClip(np.arange(22050 * 4, dtype=float), 22050)
        window = clip.slice_seconds(1.0, 3.0)
        assert window.duration_seconds == pytest.approx(2.0)
        assert window.samples[0] == 22050


class TestTheRegistrySeam:
    """The seam a future developer uses to swap in a different feature space."""

    def test_the_default_is_the_extractor_the_matrix_was_built_with(self):
        assert get_extractor().name == DEFAULT_EXTRACTOR == "mfcc-stats-104"

    def test_an_unknown_name_says_what_is_registered(self):
        with pytest.raises(KeyError, match="mfcc-stats-104"):
            get_extractor("mel-cnn-embedding")

    def test_a_new_extractor_can_be_registered_and_retrieved(self):
        class Tiny:
            name = "test-tiny-3"
            version = "0.1"
            n_features = 3
            sample_rate = 8000
            clip_seconds = 1.0

            def extract(self, clip):
                return np.array([clip.samples.mean(), clip.samples.std(), clip.samples.size])

        try:
            register_extractor(Tiny())
            assert get_extractor("test-tiny-3").n_features == 3
            assert "test-tiny-3" in available_extractors()
        finally:
            from classifier import features

            features._EXTRACTORS.pop("test-tiny-3", None)

    def test_registering_a_name_twice_is_an_error(self):
        # Silently replacing an extractor would change what every model in the
        # repository was trained against, with nothing to show for it.
        with pytest.raises(ValueError, match="already registered"):
            register_extractor(MfccStats104())

    def test_the_shipped_extractor_satisfies_the_protocol(self):
        assert isinstance(get_extractor(), FeatureExtractor)


class TestEndToEndAgreementWithDecoding:
    def test_decoding_then_extracting_matches_the_reference_on_a_wav(self, tmp_path):
        # The whole point: what `classifier.audio.decode` hands the extractor is
        # what the extractor is defined over, so the served vector equals the
        # trained one for the same audio.
        from classifier.audio import features_from_file

        path = tmp_path / "a.wav"
        samples = tone(6.0)
        wavfile.write(path, 22050, samples)

        np.testing.assert_allclose(
            features_from_file(path),
            reference_extract_2017(samples.astype(np.float64), 22050),
            rtol=1e-9,
            atol=1e-9,
        )
