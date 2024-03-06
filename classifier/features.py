"""
Feature extraction — the one implementation both training and serving use.

Why this module exists
----------------------
The 104 features this project rests on were produced in 2017 by
``mysvm/feature.py``, and the web app extracted its own features with a
*second* copy of the same maths. The two copies did not agree, which is the
classic train/serve skew: a model evaluated at 73% on the bench and scoring
noticeably off-distribution in production, with nothing in the codebase to
show why.

Two concrete divergences existed, both measurable:

* **Sample rate.** GTZAN is distributed as 22,050 Hz mono. ``python_speech_features``
  derives its mel filterbank from ``highfreq = rate / 2`` and frames at
  ``winlen * rate`` samples, so MFCCs are a function of the sample rate. Handed
  the same waveform at 44,100 Hz — which is what every MP3 a user owns actually
  is — the 104-vector lands a mean of **0.47 training standard deviations**
  away from its 22,050 Hz self, and up to 1.94 SD on individual coefficients.
  Resampling first brings that to **0.07 SD**.
* **Channel count.** The training extractor passed stereo data straight to
  ``mfcc``; the serving extractor averaged to mono. Worth 0.66 SD. GTZAN is
  mono, so the serving path was the correct one — but only by accident.

So: one extractor, it owns its own preprocessing contract (rate, channels,
duration), that contract is recorded in the model metadata, and
``classifier.predict`` refuses to serve a model whose recorded extractor does
not match the one installed. See ``tests/test_features.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from python_speech_features import mfcc


@dataclass(frozen=True)
class AudioClip:
    """A decoded, preprocessed mono waveform ready for an extractor."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.shape[0]) / self.sample_rate

    def slice_seconds(self, start: float, end: float) -> AudioClip:
        lo = int(start * self.sample_rate)
        hi = int(end * self.sample_rate)
        return AudioClip(self.samples[lo:hi], self.sample_rate)


class FeatureError(ValueError):
    """Raised when a clip cannot be turned into a usable feature vector."""


@runtime_checkable
class FeatureExtractor(Protocol):
    """
    The seam for swapping how audio becomes numbers.

    A mel-spectrogram CNN embedding would be a different implementation of this
    protocol; nothing outside this module needs to know which one is installed,
    because the preprocessing contract travels with the extractor.
    """

    name: str
    version: str
    n_features: int
    #: Rate every clip is resampled to before extraction.
    sample_rate: int
    #: Seconds of audio the extractor consumes. Longer input is truncated.
    clip_seconds: float

    def extract(self, clip: AudioClip) -> np.ndarray: ...


class MfccStats104:
    """
    Thirteen MFCCs summarised by their means and covariance diagonals.

    13 means + the 13 upper diagonals of the 13x13 covariance
    (13 + 12 + ... + 1 = 91) = 104 dimensions. This is the maths that produced
    ``mysvm/data/Xall.npy``, preserved exactly — ``tests/test_features.py``
    pins it against an independent transcription of the 2017 code.
    """

    name = "mfcc-stats-104"
    version = "1.0"
    n_features = 104
    sample_rate = 22050
    clip_seconds = 30.0

    #: ``np.cov`` over a single observation divides by N-1 = 0 and returns NaN,
    #: so a clip shorter than two MFCC frames used to yield 91 silent NaNs.
    min_frames = 2

    def extract(self, clip: AudioClip) -> np.ndarray:
        if clip.sample_rate != self.sample_rate:
            raise FeatureError(
                f"{self.name} expects {self.sample_rate} Hz; got {clip.sample_rate} Hz. "
                "Decode through classifier.audio, which resamples."
            )
        if clip.samples.ndim != 1:
            raise FeatureError(f"{self.name} expects mono; got shape {clip.samples.shape}.")
        if clip.samples.size == 0:
            raise FeatureError("clip contains no audio samples.")

        coefficients = np.transpose(mfcc(clip.samples, clip.sample_rate))
        if coefficients.shape[1] < self.min_frames:
            raise FeatureError(
                f"clip is too short: {coefficients.shape[1]} MFCC frame(s), at least "
                f"{self.min_frames} are needed to estimate a covariance."
            )

        covariance = np.cov(coefficients)
        parts = [np.mean(coefficients, axis=1)]
        parts += [np.diag(covariance, i) for i in range(coefficients.shape[0])]
        features = np.concatenate(parts)

        if not np.all(np.isfinite(features)):
            raise FeatureError("clip produced non-finite features; it cannot be scored.")
        if features.shape[0] != self.n_features:
            raise FeatureError(
                f"extracted {features.shape[0]} features, expected {self.n_features}."
            )
        return features


_EXTRACTORS: dict[str, FeatureExtractor] = {}

#: What ``Xall.npy`` was built with, and therefore what the shipped model needs.
DEFAULT_EXTRACTOR = MfccStats104.name


def register_extractor(extractor: FeatureExtractor) -> FeatureExtractor:
    """Add an extractor to the registry. Re-registering a name is an error."""
    if extractor.name in _EXTRACTORS and _EXTRACTORS[extractor.name] is not extractor:
        raise ValueError(f"extractor {extractor.name!r} is already registered")
    _EXTRACTORS[extractor.name] = extractor
    return extractor


def get_extractor(name: str | None = None) -> FeatureExtractor:
    name = name or DEFAULT_EXTRACTOR
    try:
        return _EXTRACTORS[name]
    except KeyError:
        raise KeyError(
            f"unknown extractor {name!r}; registered: {sorted(_EXTRACTORS)}"
        ) from None


def available_extractors() -> list[str]:
    return sorted(_EXTRACTORS)


register_extractor(MfccStats104())
