"""
Seed the history with real analyses so the app is never empty on first boot.

The clips are synthesised here rather than shipped as binaries — no audio is
committed to this repository — but nothing about the *analysis* is faked: each
clip is written to disk, decoded, resampled and scored through exactly the code
path an uploaded file takes. Whatever genre appears is the model's real answer
to a real waveform.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from classifier.audio import AudioError
from classifier.features import get_extractor
from classifier.predict import ModelUnavailable
from web.models import Analysis
from web.services import classify_upload

SECONDS = 30.0


def _envelope(length: int, attack: int, decay: float) -> np.ndarray:
    env = np.exp(-decay * np.linspace(0.0, 1.0, length))
    env[:attack] *= np.linspace(0.0, 1.0, attack)
    return env


def _arpeggio(rate: int, rng: np.random.Generator) -> np.ndarray:
    """Clean, sparse, harmonically simple — a few sine partials per note."""
    note_length = int(rate * 0.5)
    semitones = [0, 4, 7, 12, 7, 4]
    out = np.zeros(int(rate * SECONDS))
    t = np.arange(note_length) / rate
    for index in range(int(SECONDS / 0.5)):
        freq = 261.63 * 2 ** (semitones[index % len(semitones)] / 12.0)
        note = sum(
            (0.6 ** harmonic) * np.sin(2 * np.pi * freq * (harmonic + 1) * t)
            for harmonic in range(3)
        )
        note *= _envelope(note_length, 200, 4.0)
        start = index * note_length
        out[start:start + note_length] += note
    return out + 0.005 * rng.standard_normal(out.size)


def _distorted_riff(rate: int, rng: np.random.Generator) -> np.ndarray:
    """Dense and broadband — hard-clipped low notes with a lot of harmonics."""
    note_length = int(rate * 0.25)
    semitones = [0, 0, 3, 0, 5, 3]
    out = np.zeros(int(rate * SECONDS))
    t = np.arange(note_length) / rate
    for index in range(int(SECONDS / 0.25)):
        freq = 82.41 * 2 ** (semitones[index % len(semitones)] / 12.0)
        note = np.tanh(6.0 * np.sin(2 * np.pi * freq * t))
        note *= _envelope(note_length, 80, 1.2)
        start = index * note_length
        out[start:start + note_length] += note
    return out + 0.05 * rng.standard_normal(out.size)


def _drum_loop(rate: int, rng: np.random.Generator) -> np.ndarray:
    """Transient-heavy and unpitched — filtered noise bursts on a grid."""
    out = np.zeros(int(rate * SECONDS))
    step = int(rate * 0.125)
    for index in range(int(SECONDS / 0.125)):
        start = index * step
        if index % 4 == 0:            # kick: short low sine sweep
            length = int(rate * 0.12)
            t = np.arange(length) / rate
            hit = np.sin(2 * np.pi * np.linspace(110, 45, length) * t)
            hit *= _envelope(length, 50, 6.0)
        elif index % 4 == 2:          # snare: noise plus a body tone
            length = int(rate * 0.10)
            t = np.arange(length) / rate
            hit = 0.7 * rng.standard_normal(length) + 0.4 * np.sin(2 * np.pi * 190 * t)
            hit *= _envelope(length, 20, 9.0)
        else:                         # hat: very short bright noise
            length = int(rate * 0.04)
            hit = rng.standard_normal(length) * _envelope(length, 10, 18.0)
        end = min(start + hit.size, out.size)
        out[start:end] += hit[: end - start]
    return out


CLIPS = {
    "arpeggio-in-c.wav": _arpeggio,
    "distorted-riff.wav": _distorted_riff,
    "drum-loop-120bpm.wav": _drum_loop,
}


class Command(BaseCommand):
    help = "Classify a few synthesised clips so the history is populated on first boot."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force", action="store_true",
            help="seed even if the history already has rows",
        )

    def handle(self, *args, **options):
        if Analysis.objects.exists() and not options["force"]:
            self.stdout.write("history is not empty; nothing to seed (use --force)")
            return

        rate = get_extractor().sample_rate
        rng = np.random.default_rng(20260923)

        with tempfile.TemporaryDirectory() as workdir:
            for name, build in CLIPS.items():
                signal = build(rate, rng)
                peak = np.abs(signal).max() or 1.0
                samples = (signal / peak * 0.9 * 32767).astype(np.int16)

                path = Path(workdir) / name
                scipy.io.wavfile.write(path, rate, samples)

                with path.open("rb") as handle:
                    try:
                        outcome = classify_upload(File(handle, name=name))
                    except ModelUnavailable as exc:
                        raise CommandError(
                            f"{exc}\nTrain a model before seeding: python -m classifier.train"
                        ) from exc
                    except AudioError as exc:
                        raise CommandError(f"could not seed {name}: {exc}") from exc

                row = outcome.analysis
                self.stdout.write(
                    f"  {name:<24} {row.top_genre:<10} "
                    f"{row.top_probability:.2f}  ({row.band})"
                )
        self.stdout.write(self.style.SUCCESS("seeded"))
