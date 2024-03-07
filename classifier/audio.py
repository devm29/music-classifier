"""
Decoding uploaded audio into the preprocessed clip an extractor expects.

This is the I/O half of feature extraction; the maths lives in
``classifier.features``. The split is what lets a single extractor serve both
training and inference (see that module's header for the skew it closes).

Everything here is bounded on purpose. An upload is untrusted input: it is
size-capped before it is decoded, truncated to the extractor's clip length
before it is resampled, and every failure is an ``AudioError`` naming the file
and the reason rather than a ``None`` that explodes three layers away.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

from classifier.features import AudioClip, FeatureError, FeatureExtractor, get_extractor

#: Upper bound on an accepted upload. 30 seconds is all that is ever scored, so
#: anything past a few minutes of audio is bytes we decode and throw away.
#: Without a cap, one 2 GB "wav" is a trivial memory exhaustion.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

#: Formats the upload form advertises. Only WAV is read without ffmpeg.
SUPPORTED_SUFFIXES = (".wav", ".mp3", ".au", ".flac", ".ogg", ".m4a", ".mp4", ".flv")


class AudioError(RuntimeError):
    """Raised when a file cannot be decoded into a scorable clip."""


def _to_wav(path: Path, seconds: float) -> tuple[Path, bool]:
    """Return (wav_path, is_temporary), converting via pydub/ffmpeg when needed."""
    if path.suffix.lower() == ".wav":
        return path, False

    try:
        from pydub import AudioSegment
    except ImportError as exc:  # pragma: no cover - pydub is a hard requirement
        raise AudioError(
            f"{path.suffix} files need pydub and ffmpeg; only .wav can be read without them."
        ) from exc

    try:
        song = AudioSegment.from_file(path, path.suffix.lstrip("."))
    except Exception as exc:
        raise AudioError(
            f"Could not decode {path.name}. ffmpeg must be installed to read "
            f"{path.suffix} files."
        ) from exc

    # Not a context manager on purpose: the file must outlive this block so
    # the caller can read it, and `decode` unlinks it in its own `finally`.
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
    handle.close()
    try:
        song[: int(seconds * 1000)].export(handle.name, format="wav")
    except Exception as exc:
        # Without this the temporary file is left behind on every failed
        # export, and a web app that fails on a bad upload leaks a file per
        # request into the system temp directory.
        os.unlink(handle.name)
        raise AudioError(f"Could not convert {path.name} to WAV.") from exc
    return Path(handle.name), True


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average interleaved channels down to one, as float."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Polyphase-resample a mono waveform.

    MFCCs are a function of the sample rate, so a 44.1 kHz upload scored
    against a 22.05 kHz-trained model is scored off-distribution. Resampling is
    not a nicety: it moves the feature vector from 0.47 training SDs off its
    reference to 0.07, and it roughly halves extraction cost because there is
    half as much signal to frame.
    """
    if source_rate == target_rate:
        return data
    if source_rate <= 0:
        raise AudioError(f"nonsensical sample rate: {source_rate} Hz")
    return resample_poly(data, target_rate, source_rate)


def decode(
    path: str | Path, extractor: FeatureExtractor | None = None
) -> AudioClip:
    """
    Read an audio file into the mono, correctly-rated, truncated clip the
    extractor is defined over. Raises ``AudioError`` with a reason.
    """
    extractor = extractor or get_extractor()
    path = Path(path)
    if not path.exists():
        raise AudioError(f"No such file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise AudioError(f"{path.name} is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise AudioError(
            f"{path.name} is {size / 1_048_576:.0f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB. Only the first "
            f"{extractor.clip_seconds:.0f} seconds are scored anyway."
        )

    wav_path, temporary = _to_wav(path, extractor.clip_seconds)
    try:
        try:
            rate, data = scipy.io.wavfile.read(wav_path)
        except Exception as exc:
            raise AudioError(f"{path.name} is not readable as WAV audio.") from exc

        data = to_mono(data)
        # Truncate before resampling: there is no point filtering audio that is
        # about to be discarded, and it bounds the work a long upload can cause.
        data = data[: int(extractor.clip_seconds * rate)]
        if data.size == 0:
            raise AudioError(f"{path.name} contains no audio samples.")

        return AudioClip(resample(data, int(rate), extractor.sample_rate), extractor.sample_rate)
    finally:
        if temporary:
            os.unlink(wav_path)


def features_from_file(
    path: str | Path, extractor: FeatureExtractor | None = None
) -> np.ndarray:
    """Decode a file and extract its feature vector, or raise ``AudioError``."""
    extractor = extractor or get_extractor()
    clip = decode(path, extractor)
    try:
        return extractor.extract(clip)
    except FeatureError as exc:
        raise AudioError(f"{Path(path).name}: {exc}") from exc


def segments(clip: AudioClip, window: float, hop: float) -> list[tuple[float, AudioClip]]:
    """
    Split a clip into overlapping windows, returned as (start_seconds, clip).

    Used by the per-segment timeline. Windows shorter than ``window`` at the
    tail are dropped rather than padded — padding a window with silence shifts
    its MFCC means and would show up as a spurious genre change at the end of
    every track.
    """
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    out: list[tuple[float, AudioClip]] = []
    start = 0.0
    while start + window <= clip.duration_seconds + 1e-9:
        out.append((round(start, 3), clip.slice_seconds(start, start + window)))
        start += hop
    return out
