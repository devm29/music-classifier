"""
Upload orchestration: the bit between an HTTP request and `classifier.service`.

Everything Django-specific about classifying an upload lives here — saving the
file, the content-hash cache, and history pruning — so the views stay at the
altitude of "parse request, call service, serialise response".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction

from classifier import service
from classifier.audio import AudioError
from web.models import Analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifyResult:
    analysis: Analysis
    cached: bool


def classify_upload(upload: UploadedFile) -> ClassifyResult:
    """
    Persist an upload, analyse it, and return the row.

    A file whose SHA-256 is already on record short-circuits: inference is
    deterministic, so re-running ~50 ms of MFCC extraction and five window
    predictions to produce a byte-identical answer is pure waste. The duplicate
    upload is discarded rather than stored twice.

    Raises `AudioError` if the file cannot be decoded, having cleaned up the
    stored copy first — a failed upload should not leave audio on disk.
    """
    row = Analysis(
        file=upload,
        original_name=upload.name[:255],
        size_bytes=upload.size,
        sha256="",
        duration_seconds=0.0,
        top_genre="",
        top_probability=0.0,
        band="",
        result={},
    )
    row.file.save(upload.name, upload, save=False)

    try:
        digest = service.content_hash(row.file.path)
        existing = Analysis.objects.filter(sha256=digest).first()
        if existing is not None:
            row.file.delete(save=False)
            return ClassifyResult(analysis=existing, cached=True)

        result = service.analyse_file(row.file.path)
    except Exception:
        row.file.delete(save=False)
        raise

    row.sha256 = digest
    row.duration_seconds = result["duration_seconds"]
    row.top_genre = result["verdict"]["genre"]
    row.top_probability = result["verdict"]["probability"]
    row.band = result["verdict"]["band"]
    row.elapsed_ms = result["elapsed_ms"]
    row.result = result

    with transaction.atomic():
        row.save()
        prune_history()
    return ClassifyResult(analysis=row, cached=False)


def prune_history(limit: int | None = None) -> int:
    """
    Keep the history bounded, deleting the audio with each row.

    Called after every successful classification, so the table and the media
    directory cannot grow without limit on a public demo.
    """
    limit = settings.HISTORY_LIMIT if limit is None else limit
    stale = list(Analysis.objects.order_by("-created_at", "-id")[limit:])
    for row in stale:
        row.delete()
    return len(stale)


__all__ = ["AudioError", "ClassifyResult", "classify_upload", "prune_history"]
