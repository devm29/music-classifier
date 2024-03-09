"""
Persistence for the web layer: one row per analysed upload.

The row is the cache. Feature extraction plus the windowed timeline is ~50 ms of
single-threaded CPU that produces exactly the same answer for the same bytes, so the
SHA-256 of the upload is a unique key and a repeat upload is a database lookup
instead. `sha256` is unique-indexed and `created_at` is indexed because the
only two queries this app makes are "find by hash" and "the most recent N".

Uploads are pruned to a bounded history rather than kept forever: this is a
demo that accepts arbitrary audio from anyone who can reach it, and unbounded
retention of other people's files is both a disk problem and a privacy one.
"""

from __future__ import annotations

from django.db import models


class AnalysisQuerySet(models.QuerySet):
    def recent(self, limit: int):
        return self.order_by("-created_at")[:limit]


class Analysis(models.Model):
    """A classified upload and the full result document produced for it."""

    file = models.FileField(upload_to="audio")
    original_name = models.CharField(max_length=255)
    sha256 = models.CharField(max_length=64, unique=True, db_index=True)
    size_bytes = models.PositiveBigIntegerField()
    duration_seconds = models.FloatField()
    top_genre = models.CharField(max_length=32)
    top_probability = models.FloatField()
    #: "confident", "uncertain" or "off-distribution".
    band = models.CharField(max_length=24)
    #: The whole service response, so the page can be re-rendered without
    #: re-running inference and so the schema can grow without a migration.
    result = models.JSONField()
    elapsed_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    objects = AnalysisQuerySet.as_manager()

    class Meta:
        verbose_name_plural = "analyses"
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"{self.original_name} -> {self.top_genre} ({self.top_probability:.0%})"

    def delete(self, *args, **kwargs):
        """Remove the stored audio with the row; a dangling file helps nobody."""
        self.file.delete(save=False)
        return super().delete(*args, **kwargs)

    def as_dict(self) -> dict:
        payload = dict(self.result)
        payload.update(
            {
                "id": self.pk,
                "name": self.original_name,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
                "audio_url": self.file.url if self.file else None,
                "created_at": self.created_at.isoformat(),
            }
        )
        return payload
