"""
The application service: one file in, one analysis out.

This is the layer the Django views call, and it is deliberately framework-free
— no request, no ORM, no settings import. The whole inference path can
therefore be exercised from a REPL or a test without Django present, which is
what keeps the web app a thin adapter rather than the place the logic lives.

What an analysis contains, and why each part earns its place:

``ranked``    the calibrated probabilities. The model is ~71% accurate, so a
              single label overstates what it knows.
``verdict``   a confidence band derived from the top probability *and* the
              margin over the runner-up. Two genres at 0.33/0.31 is not the
              same answer as one at 0.62, and a UI that renders both as "rock"
              is lying by omission.
``timeline``  per-window predictions across the clip. A track that reads jazz
              for ten seconds and metal for the next ten is telling you
              something a single average cannot.
``neighbours`` the closest GTZAN tracks in feature space — a falsifiable
              explanation rather than a number, and the out-of-distribution
              detector the verdict depends on.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from classifier import audio, neighbours
from classifier.features import AudioClip, FeatureError, FeatureExtractor, get_extractor
from classifier.predict import predict

#: Below this top probability nothing is worth asserting: with ten genres,
#: 0.40 still leaves 60% of the mass elsewhere.
MIN_CONFIDENT_PROBABILITY = 0.40

#: A top-two gap narrower than this is a coin flip however high the top is.
MIN_CONFIDENT_MARGIN = 0.10

#: Out-of-distribution threshold, expressed as a fraction of GTZAN track pairs
#: that are *further apart* than this clip's nearest training track.
#:
#: An SVM has no idea what it has not seen. Handed audio unlike anything in
#: GTZAN — a synthesised tone, a podcast, silence — it does not abstain; it
#: returns its off-distribution default, which for this model is jazz at about
#: 0.54 with blues second, for essentially any such input. Rendered as
#: "confident: jazz", that is the single most misleading thing this interface
#: could say.
#:
#: The neighbour index already answers the question, and it discriminates
#: rather than simply firing on anything synthetic. A real GTZAN track's
#: nearest neighbour sits at a standardised distance of ~6.9 (median; 95% are
#: under 12.8), and the furthest-apart pair of real tracks is ~21 apart at the
#: 95th percentile. Measured against the three seeded demo clips: the distorted
#: riff lands 6.6 from ``metal.00004`` — inside — while the arpeggio lands 24.2
#: and the drum loop 65.1, further from anything in GTZAN than GTZAN tracks
#: ever are from each other. So: if fewer than 5% of pairs are further apart
#: than this clip's closest training track, the model is extrapolating, and the
#: verdict says so instead of asserting a genre.
#:
#: Stated as a percentile rather than a distance so it stays meaningful if the
#: feature matrix is ever regenerated.
MAX_NEIGHBOUR_PERCENTILE = 0.05

#: Segment geometry for the timeline. Ten seconds keeps each window's
#: covariance estimate usable — the features are 13 MFCC means plus a 13x13
#: covariance, and a covariance over a handful of frames is mostly noise.
SEGMENT_WINDOW_SECONDS = 10.0
SEGMENT_HOP_SECONDS = 5.0

NEIGHBOUR_COUNT = 5
HASH_CHUNK_BYTES = 1 << 20


def content_hash(path: str | Path) -> str:
    """SHA-256 of the file, streamed. The cache key for a repeat upload."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_band(ranked: list[dict], nearest: dict | None = None) -> dict:
    """
    Turn a ranked probability list into a verdict a UI can render honestly.

    ``nearest`` is the closest training track, when one was looked up. It
    outranks the probabilities: a clip that sits nowhere near the training set
    gets an ``off-distribution`` verdict however confident the SVM sounds,
    because that confidence is about a region of feature space the model has
    never seen an example from.
    """
    if not ranked:
        raise ValueError("ranked predictions are empty")

    top = ranked[0]
    runner_up = ranked[1]["probability"] if len(ranked) > 1 else 0.0
    margin = top["probability"] - runner_up

    if nearest is not None and nearest["closer_than"] < MAX_NEIGHBOUR_PERCENTILE:
        band, reason = "off-distribution", (
            f"the nearest GTZAN track sits {nearest['distance']:.1f} away in "
            f"standardised feature space, and only {nearest['closer_than']:.0%} "
            "of the dataset's own track pairs are further apart than that. The "
            f"model still names a genre ({top['genre']}), but it is "
            "extrapolating, not recognising."
        )
    elif top["probability"] >= MIN_CONFIDENT_PROBABILITY and margin >= MIN_CONFIDENT_MARGIN:
        band, reason = "confident", (
            f"{top['probability']:.0%} on {top['genre']}, "
            f"{margin:.0%} clear of the runner-up."
        )
    elif margin < MIN_CONFIDENT_MARGIN:
        band, reason = "uncertain", (
            f"{top['genre']} and {ranked[1]['genre']} are within {margin:.0%} "
            "of each other — treat this as a tie, not an answer."
        )
    else:
        band, reason = "uncertain", (
            f"the best guess is only {top['probability']:.0%}; "
            f"{1 - top['probability']:.0%} of the probability mass is elsewhere."
        )

    return {
        "genre": top["genre"],
        "probability": top["probability"],
        "margin": round(margin, 4),
        "band": band,
        "reason": reason,
    }


def timeline(
    clip: AudioClip,
    extractor: FeatureExtractor | None = None,
    window: float = SEGMENT_WINDOW_SECONDS,
    hop: float = SEGMENT_HOP_SECONDS,
) -> list[dict]:
    """
    Predict each window of the clip in turn.

    Windows whose features cannot be extracted are skipped rather than
    reported as a genre: an unscorable window is not evidence of anything.

    Caveat worth knowing: the model was fitted on 30-second summaries, and a
    10-second window's covariance is a noisier estimate of the same quantity.
    The timeline is a texture-change indicator, not a second opinion on the
    overall label.
    """
    extractor = extractor or get_extractor()
    out: list[dict] = []
    for start, segment in audio.segments(clip, window=window, hop=hop):
        try:
            ranked = predict(extractor.extract(segment), top_k=2)
        except FeatureError:
            continue
        out.append(
            {
                "start": start,
                "end": round(start + window, 3),
                "genre": ranked[0]["genre"],
                "probability": ranked[0]["probability"],
                "runner_up": ranked[1]["genre"] if len(ranked) > 1 else None,
            }
        )
    return out


def analyse_clip(
    clip: AudioClip,
    extractor: FeatureExtractor | None = None,
    *,
    top_k: int = 3,
    with_timeline: bool = True,
    with_neighbours: bool = True,
) -> dict:
    """Score a decoded clip. The one place the inference steps are composed."""
    extractor = extractor or get_extractor()
    started = time.perf_counter()

    features = extractor.extract(clip)
    ranked = predict(features, top_k=top_k)

    # Looked up before the verdict, not after: the nearest training track is
    # what tells the verdict whether the model is recognising or extrapolating.
    similar = neighbours.similar_tracks(features, k=NEIGHBOUR_COUNT) if with_neighbours else []

    result = {
        "ranked": ranked,
        "verdict": confidence_band(ranked, similar[0] if similar else None),
        "duration_seconds": round(clip.duration_seconds, 2),
        "sample_rate": clip.sample_rate,
        "extractor": extractor.name,
        "timeline": timeline(clip, extractor) if with_timeline else [],
        "neighbours": similar,
    }
    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def analyse_file(path: str | Path, **kwargs) -> dict:
    """
    Decode and score an audio file.

    Raises ``classifier.audio.AudioError`` if the file cannot be decoded, and
    ``classifier.predict.ModelUnavailable`` / ``ExtractorMismatch`` if the
    model cannot serve it. Nothing here returns ``None`` on failure.
    """
    extractor = kwargs.pop("extractor", None) or get_extractor()
    clip = audio.decode(path, extractor)
    try:
        return analyse_clip(clip, extractor, **kwargs)
    except FeatureError as exc:
        raise audio.AudioError(f"{Path(path).name}: {exc}") from exc


def classify_file(path: str | Path, top_k: int = 3) -> list[dict]:
    """Just the ranked genres — the narrow entry point, kept for scripts."""
    features = audio.features_from_file(path)
    return predict(np.asarray(features), top_k=top_k)
