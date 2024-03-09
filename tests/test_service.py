"""
Tests for the framework-free application service.

`classifier.service` is the layer the Django views call, and it imports no
Django. These tests run without a database and without a request, which is the
property that keeps the web app a thin adapter: if the analysis logic needed
Django to be exercised, it would already have leaked into the view.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from classifier import service
from classifier.audio import AudioError, decode
from classifier.features import get_extractor
from classifier.service import (
    MAX_NEIGHBOUR_PERCENTILE,
    MIN_CONFIDENT_MARGIN,
    MIN_CONFIDENT_PROBABILITY,
    analyse_clip,
    analyse_file,
    confidence_band,
    content_hash,
    timeline,
)


def write_wav(path, seconds=25.0, rate=22050, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    signal = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.08 * rng.standard_normal(t.size)
    wavfile.write(path, rate, (signal / np.abs(signal).max() * 32000).astype(np.int16))
    return path


def ranked(*pairs):
    return [{"genre": g, "probability": p} for g, p in pairs]


def neighbour(closer_than, distance=7.0):
    return {"track": "rock.00012", "genre": "rock",
            "distance": distance, "closer_than": closer_than}


#: A nearest training track well inside the dataset's own spread, and one
#: further away than almost every pair of real tracks.
CLEAR = neighbour(0.90)
FAR = neighbour(0.02, distance=24.3)


class TestConfidenceBand:
    """The 'uncertain' band. A 71%-accurate model that reports one label is
    lying by omission; the band is what makes the interface honest."""

    def test_a_clear_winner_is_confident(self):
        band = confidence_band(ranked(("rock", 0.62), ("metal", 0.21), ("pop", 0.09)))
        assert band["band"] == "confident"
        assert band["genre"] == "rock"
        assert band["margin"] == pytest.approx(0.41)

    def test_a_near_tie_is_uncertain_however_high_the_top_is(self):
        band = confidence_band(ranked(("rock", 0.48), ("metal", 0.45)))
        assert band["band"] == "uncertain"
        assert "tie" in band["reason"]

    def test_a_low_top_probability_is_uncertain_even_with_a_wide_margin(self):
        band = confidence_band(ranked(("rock", 0.30), ("metal", 0.10), ("pop", 0.09)))
        assert band["band"] == "uncertain"
        assert "probability mass is elsewhere" in band["reason"]

    def test_the_thresholds_are_the_documented_ones(self):
        just_over = confidence_band(
            ranked(("rock", MIN_CONFIDENT_PROBABILITY + 0.01),
                   ("metal", MIN_CONFIDENT_PROBABILITY - MIN_CONFIDENT_MARGIN))
        )
        assert just_over["band"] == "confident"

    def test_a_single_entry_list_has_a_full_margin(self):
        assert confidence_band(ranked(("rock", 0.9)))["margin"] == pytest.approx(0.9)

    def test_an_empty_ranking_is_an_error_not_a_verdict(self):
        with pytest.raises(ValueError):
            confidence_band([])

    def test_the_reason_always_names_the_top_genre(self):
        for r in (ranked(("jazz", 0.7), ("blues", 0.1)), ranked(("jazz", 0.2), ("blues", 0.19))):
            assert "jazz" in confidence_band(r)["reason"]


class TestOffDistributionBand:
    """
    The detector that stops the page asserting a genre for audio the model has
    never seen anything like. Without it every synthesised clip, podcast and
    silent file comes back "confident: jazz" at 0.54.
    """


    def test_a_far_nearest_neighbour_overrides_a_confident_probability(self):
        band = confidence_band(ranked(("jazz", 0.54), ("blues", 0.22)), FAR)
        assert band["band"] == "off-distribution"
        assert "extrapolating" in band["reason"]

    def test_it_still_reports_the_genre_the_model_named(self):
        band = confidence_band(ranked(("jazz", 0.54), ("blues", 0.22)), FAR)
        assert band["genre"] == "jazz" and "jazz" in band["reason"]

    def test_a_close_neighbour_leaves_the_probability_verdict_alone(self):
        assert confidence_band(
            ranked(("rock", 0.62), ("metal", 0.21)), CLEAR
        )["band"] == "confident"

    def test_an_uncertain_result_stays_uncertain_with_a_close_neighbour(self):
        assert confidence_band(
            ranked(("rock", 0.48), ("metal", 0.45)), CLEAR
        )["band"] == "uncertain"

    def test_without_a_neighbour_lookup_the_band_is_probability_only(self):
        assert confidence_band(ranked(("rock", 0.62), ("metal", 0.21)))["band"] == "confident"

    def test_the_threshold_is_the_documented_one(self):
        just_inside = dict(CLEAR, closer_than=MAX_NEIGHBOUR_PERCENTILE)
        just_outside = dict(CLEAR, closer_than=MAX_NEIGHBOUR_PERCENTILE - 0.001)
        r = ranked(("rock", 0.62), ("metal", 0.21))
        assert confidence_band(r, just_inside)["band"] == "confident"
        assert confidence_band(r, just_outside)["band"] == "off-distribution"

    def test_an_off_distribution_clip_really_is_flagged(self, trained_model, tmp_path):
        # End to end against the real neighbour index rather than a stubbed
        # dict. An impulse grid — clicks on a quarter-second beat, no pitch and
        # no sustain — lands ~53 standardised units from its nearest GTZAN
        # track, where the furthest-apart real pair is ~21.
        rate, seconds = 22050, 25.0
        size = int(rate * seconds)
        rng = np.random.default_rng(3)
        signal = np.where(np.arange(size) % int(rate * 0.25) < 30, 1.0, 0.0)
        signal *= rng.standard_normal(size)

        path = tmp_path / "clicks.wav"
        wavfile.write(path, rate, (signal / np.abs(signal).max() * 32000).astype(np.int16))

        result = analyse_file(path)
        assert result["verdict"]["band"] == "off-distribution"
        assert result["neighbours"][0]["closer_than"] < MAX_NEIGHBOUR_PERCENTILE

    def test_an_in_distribution_clip_is_not_flagged(self, trained_model, tmp_path):
        # The detector has to discriminate, not just fire on anything
        # synthesised: a sustained tone sits ~10 units out, well inside.
        result = analyse_file(write_wav(tmp_path / "tone.wav", seconds=25))
        assert result["verdict"]["band"] != "off-distribution"


class TestContentHash:
    def test_the_same_bytes_hash_the_same(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"x" * 4096)
        b.write_bytes(b"x" * 4096)
        assert content_hash(a) == content_hash(b)

    def test_different_bytes_hash_differently(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"x")
        b.write_bytes(b"y")
        assert content_hash(a) != content_hash(b)

    def test_a_file_larger_than_one_chunk_is_streamed_whole(self, tmp_path):
        import hashlib

        big = tmp_path / "big.bin"
        payload = bytes(range(256)) * 8192  # 2 MB, two read chunks
        big.write_bytes(payload)
        assert content_hash(big) == hashlib.sha256(payload).hexdigest()


class TestTimeline:
    def test_windows_are_scored_in_order(self, trained_model, tmp_path):
        clip = decode(write_wav(tmp_path / "a.wav", seconds=25))
        rows = timeline(clip)
        assert [r["start"] for r in rows] == [0.0, 5.0, 10.0, 15.0]
        assert all(r["end"] == r["start"] + 10.0 for r in rows)

    def test_every_window_names_a_known_genre(self, trained_model, tmp_path):
        from classifier.data import GENRES

        clip = decode(write_wav(tmp_path / "a.wav", seconds=25))
        assert all(r["genre"] in GENRES for r in timeline(clip))

    def test_a_clip_shorter_than_one_window_has_an_empty_timeline(
        self, trained_model, tmp_path
    ):
        clip = decode(write_wav(tmp_path / "a.wav", seconds=4))
        assert timeline(clip) == []

    def test_an_unscorable_window_is_skipped_not_guessed(self, trained_model, tmp_path):
        # An unscorable window is not evidence of a genre. Reporting one would
        # put a label on the timeline that no audio supports.
        clip = decode(write_wav(tmp_path / "a.wav", seconds=25))
        extractor = get_extractor()

        calls = {"n": 0}
        real_extract = extractor.extract

        def flaky(window):
            calls["n"] += 1
            if calls["n"] == 2:
                from classifier.features import FeatureError

                raise FeatureError("synthetic failure")
            return real_extract(window)

        class Flaky:
            name = extractor.name
            version = extractor.version
            n_features = extractor.n_features
            sample_rate = extractor.sample_rate
            clip_seconds = extractor.clip_seconds
            extract = staticmethod(flaky)

        rows = timeline(clip, Flaky())
        assert [r["start"] for r in rows] == [0.0, 10.0, 15.0]


class TestAnalyse:
    def test_an_analysis_has_every_documented_section(self, trained_model, tmp_path):
        result = analyse_file(write_wav(tmp_path / "a.wav", seconds=25))
        assert set(result) >= {
            "ranked", "verdict", "timeline", "neighbours",
            "duration_seconds", "sample_rate", "extractor", "elapsed_ms",
        }
        assert len(result["ranked"]) == 3
        assert result["verdict"]["genre"] == result["ranked"][0]["genre"]
        assert result["sample_rate"] == get_extractor().sample_rate
        assert result["extractor"] == get_extractor().name

    def test_the_expensive_sections_can_be_switched_off(self, trained_model, tmp_path):
        clip = decode(write_wav(tmp_path / "a.wav", seconds=25))
        result = analyse_clip(clip, with_timeline=False, with_neighbours=False)
        assert result["timeline"] == [] and result["neighbours"] == []

    def test_neighbours_are_named_gtzan_tracks(self, trained_model, tmp_path):
        from classifier.data import GENRES

        result = analyse_file(write_wav(tmp_path / "a.wav", seconds=25))
        assert len(result["neighbours"]) == service.NEIGHBOUR_COUNT
        for n in result["neighbours"]:
            assert n["genre"] in GENRES
            assert n["track"].startswith(n["genre"] + ".")

    def test_it_is_deterministic(self, trained_model, tmp_path):
        path = write_wav(tmp_path / "a.wav", seconds=25)
        first, second = analyse_file(path), analyse_file(path)
        assert first["ranked"] == second["ranked"]
        assert first["timeline"] == second["timeline"]

    def test_an_undecodable_file_raises_audio_error(self, trained_model, tmp_path):
        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"nope")
        with pytest.raises(AudioError):
            analyse_file(junk)

    def test_a_clip_too_short_to_score_raises_audio_error_not_feature_error(
        self, trained_model, tmp_path
    ):
        # Callers above this layer catch AudioError. A FeatureError escaping
        # here would be a 500 on a file the user could have been told about.
        tiny = tmp_path / "tiny.wav"
        wavfile.write(tiny, 22050, (np.zeros(300) + 100).astype(np.int16))
        with pytest.raises(AudioError, match="too short"):
            analyse_file(tiny)
