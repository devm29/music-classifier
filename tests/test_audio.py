"""
Tests for audio decoding — the I/O half of feature extraction.

The original `extract()` wrapped both of its failure paths in
`except Exception: print(e)` and carried on, so an undecodable file fell
through, failed again, and returned None — which the caller then indexed into.
The request died with a TypeError that said nothing about the actual problem.
These tests pin that failures are explicit.

The maths those features are made of now lives in `classifier.features` and is
tested in `test_features.py`; what is under test here is decoding, mono
reduction, resampling, truncation, bounds and temp-file hygiene.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from classifier.audio import (
    MAX_UPLOAD_BYTES,
    AudioError,
    decode,
    features_from_file,
    resample,
    segments,
    to_mono,
)
from classifier.features import get_extractor

EXTRACTOR = get_extractor()
EXPECTED_DIMENSIONS = EXTRACTOR.n_features
CLIP_SECONDS = EXTRACTOR.clip_seconds


def write_wav(path, seconds=5, rate=22050, channels=1, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    data = (signal / np.abs(signal).max() * 32767).astype(np.int16)
    if channels == 2:
        data = np.column_stack([data, data])
    wavfile.write(path, rate, data)
    return path


class TestExtraction:
    def test_it_produces_the_dimensions_the_model_expects(self, tmp_path):
        features = features_from_file(write_wav(tmp_path / "a.wav"))
        assert features.shape == (EXPECTED_DIMENSIONS,)

    def test_extraction_is_deterministic(self, tmp_path):
        path = write_wav(tmp_path / "a.wav")
        np.testing.assert_array_equal(features_from_file(path), features_from_file(path))

    def test_stereo_is_reduced_to_mono(self, tmp_path):
        # The training features were computed from single-channel audio, so
        # passing two channels through would change the feature scale.
        mono = features_from_file(write_wav(tmp_path / "m.wav", channels=1))
        stereo = features_from_file(write_wav(tmp_path / "s.wav", channels=2))
        assert stereo.shape == mono.shape
        np.testing.assert_allclose(stereo, mono, rtol=1e-6)

    def test_only_the_first_clip_seconds_are_used(self, tmp_path):
        # The model was trained on 30-second clips. A longer file must not
        # produce features drawn from a different amount of audio.
        short = features_from_file(write_wav(tmp_path / "short.wav", seconds=CLIP_SECONDS))
        long = features_from_file(write_wav(tmp_path / "long.wav", seconds=CLIP_SECONDS + 20))
        np.testing.assert_allclose(short, long, rtol=1e-6)


class TestDecodedClipContract:
    """`decode` must hand the extractor exactly what it is defined over."""

    def test_the_clip_is_at_the_extractors_sample_rate(self, tmp_path):
        clip = decode(write_wav(tmp_path / "a.wav", rate=44100, seconds=3))
        assert clip.sample_rate == EXTRACTOR.sample_rate

    def test_the_clip_is_one_dimensional(self, tmp_path):
        assert decode(write_wav(tmp_path / "s.wav", channels=2)).samples.ndim == 1

    def test_a_long_file_is_truncated_to_the_clip_length(self, tmp_path):
        clip = decode(write_wav(tmp_path / "long.wav", seconds=CLIP_SECONDS + 30))
        assert clip.duration_seconds == pytest.approx(CLIP_SECONDS, abs=0.05)

    def test_an_oversized_file_is_refused_before_it_is_decoded(self, tmp_path):
        # An upload is untrusted. Without a byte cap, one very large "wav" is a
        # trivial memory exhaustion, and rejecting it after decoding is not
        # rejecting it.
        big = tmp_path / "big.wav"
        big.write_bytes(b"\0" * (MAX_UPLOAD_BYTES + 1))
        with pytest.raises(AudioError, match="limit is"):
            decode(big)


def write_tone(path, rate, seconds=4.0):
    """The same continuous signal sampled at whatever rate is asked for.

    Noiseless on purpose: two rates must sample the *same* waveform for a
    resampling comparison to mean anything.
    """
    t = np.arange(int(rate * seconds)) / rate
    signal = sum(0.5 ** k * np.sin(2 * np.pi * 220 * (k + 1) * t) for k in range(4))
    data = (signal / np.abs(signal).max() * 32000).astype(np.int16)
    wavfile.write(path, rate, data)
    return path


class TestResampling:
    def test_the_same_waveform_at_two_rates_agrees_after_resampling(self, tmp_path):
        # The train/serve skew this module exists to close: MFCCs are a
        # function of the sample rate, so a 44.1 kHz upload scored against a
        # 22.05 kHz-trained model is scored off-distribution.
        native = features_from_file(write_tone(tmp_path / "n.wav", 22050))
        upsampled = features_from_file(write_tone(tmp_path / "u.wav", 44100))
        # The 13 MFCC means agree to ~0.04 on a feature whose scale runs to
        # ~150; the covariance diagonals to under 1.
        np.testing.assert_allclose(native[:13], upsampled[:13], atol=0.2)
        assert np.abs(native - upsampled).max() < 1.0

    def test_skipping_the_resample_moves_the_features_materially(self, tmp_path):
        # The counterpart: without the resampling step the same waveform lands
        # somewhere else entirely. This is the measurement behind the claim in
        # `classifier.features`, not an assertion about it.
        reference = features_from_file(write_tone(tmp_path / "n.wav", 22050))

        rate, data = wavfile.read(write_tone(tmp_path / "u.wav", 44100))
        from classifier.features import AudioClip

        unresampled = EXTRACTOR.extract(
            AudioClip(to_mono(data), EXTRACTOR.sample_rate)  # lie about the rate
        )
        assert np.abs(unresampled[:13] - reference[:13]).max() > 1.0

    def test_resampling_a_matching_rate_is_a_no_op(self):
        data = np.arange(100, dtype=float)
        assert resample(data, 22050, 22050) is data

    def test_a_nonsensical_rate_is_an_audio_error(self):
        with pytest.raises(AudioError, match="nonsensical"):
            resample(np.arange(100, dtype=float), 0, 22050)

    def test_to_mono_averages_channels(self):
        stereo = np.array([[1, 3], [5, 7]], dtype=np.int16)
        np.testing.assert_allclose(to_mono(stereo), [2.0, 6.0])


class TestSegments:
    def test_windows_tile_the_clip_with_the_requested_hop(self, tmp_path):
        clip = decode(write_wav(tmp_path / "a.wav", seconds=20))
        windows = segments(clip, window=10.0, hop=5.0)
        assert [start for start, _ in windows] == [0.0, 5.0, 10.0]
        assert all(c.duration_seconds == pytest.approx(10.0, abs=0.01) for _, c in windows)

    def test_a_short_tail_is_dropped_rather_than_padded(self, tmp_path):
        # Padding a window with silence shifts its MFCC means, which would read
        # as a spurious genre change at the end of every track.
        clip = decode(write_wav(tmp_path / "a.wav", seconds=12))
        assert [start for start, _ in segments(clip, window=10.0, hop=5.0)] == [0.0]

    def test_a_clip_shorter_than_one_window_yields_nothing(self, tmp_path):
        clip = decode(write_wav(tmp_path / "a.wav", seconds=3))
        assert segments(clip, window=10.0, hop=5.0) == []

    def test_a_nonpositive_window_is_rejected(self, tmp_path):
        clip = decode(write_wav(tmp_path / "a.wav", seconds=3))
        with pytest.raises(ValueError):
            segments(clip, window=0.0, hop=1.0)


class TestFailuresAreExplicit:
    def test_a_missing_file_raises_audio_error(self):
        with pytest.raises(AudioError, match="No such file"):
            features_from_file("/nonexistent/file.wav")

    def test_a_file_that_is_not_audio_raises_audio_error(self, tmp_path):
        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"definitely not a wav file")
        with pytest.raises(AudioError, match="not readable as WAV"):
            features_from_file(junk)

    def test_an_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(AudioError, match="is empty"):
            features_from_file(empty)

    def test_an_empty_wav_raises_rather_than_returning_nothing(self, tmp_path):
        empty = tmp_path / "empty.wav"
        wavfile.write(empty, 22050, np.array([], dtype=np.int16))
        with pytest.raises(AudioError):
            features_from_file(empty)

    def test_nothing_returns_none(self, tmp_path):
        # The precise regression: the old code returned None on failure.
        for path in ["/nope.wav", str(write_wav(tmp_path / "ok.wav"))]:
            try:
                result = features_from_file(path)
            except AudioError:
                continue
            assert result is not None


class TestClipsTooShortToScore:
    def test_a_clip_shorter_than_two_mfcc_frames_raises(self, tmp_path):
        # `np.cov` over a single observation divides by N-1 = 0 and returns
        # NaN. A 300-sample file used to come back as a 104-vector with 91
        # NaNs in it, which reached the model and died inside scikit-learn's
        # input validation, nowhere near the cause.
        tiny = tmp_path / "tiny.wav"
        wavfile.write(tiny, 22050, (np.zeros(300) + 100).astype(np.int16))
        with pytest.raises(AudioError, match="too short"):
            features_from_file(tiny)

    def test_a_clip_long_enough_for_two_frames_is_accepted(self, tmp_path):
        ok = write_wav(tmp_path / "ok.wav", seconds=0.2)
        assert features_from_file(ok).shape == (EXPECTED_DIMENSIONS,)

    def test_features_are_always_finite(self, tmp_path):
        features = features_from_file(write_wav(tmp_path / "a.wav"))
        assert np.all(np.isfinite(features)), "a NaN feature scores as garbage"


class TestTemporaryFiles:
    def test_a_wav_input_is_read_in_place(self, tmp_path, monkeypatch):
        # No conversion means no temporary file and nothing to clean up.
        import tempfile as tempfile_module

        monkeypatch.setattr(
            tempfile_module, "NamedTemporaryFile",
            lambda *a, **k: pytest.fail("a .wav must not be converted"),
        )
        assert features_from_file(write_wav(tmp_path / "a.wav")).shape == (
            EXPECTED_DIMENSIONS,
        )

    def test_a_failed_conversion_does_not_leak_a_temp_file(self, tmp_path, monkeypatch):
        import classifier.audio as audio

        before = set(Path(tempfile.gettempdir()).glob("*.wav"))

        class ExplodingSegment:
            @staticmethod
            def from_file(path, fmt):
                class Song:
                    def __getitem__(self, _):
                        return self

                    def export(self, *_a, **_k):
                        raise OSError("ffmpeg died")

                return Song()

        monkeypatch.setitem(
            __import__("sys").modules, "pydub",
            type("pydub", (), {"AudioSegment": ExplodingSegment}),
        )
        source = tmp_path / "song.mp3"
        source.write_bytes(b"not really an mp3")
        with pytest.raises(AudioError):
            audio.features_from_file(source)

        after = set(Path(tempfile.gettempdir()).glob("*.wav"))
        assert after == before, "a failed conversion left a temp file behind"
