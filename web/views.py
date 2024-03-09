"""
HTTP adapter. No inference, no feature maths, no model loading happens here —
these functions parse a request, call `web.services` or `classifier.predict`,
and serialise the answer.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from classifier.audio import MAX_UPLOAD_BYTES, SUPPORTED_SUFFIXES, AudioError
from classifier.predict import ExtractorMismatch, ModelUnavailable, model_info
from web.forms import UploadForm
from web.models import Analysis
from web.services import classify_upload

logger = logging.getLogger(__name__)


def _error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"error": message}, status=status)


@require_GET
def index(request: HttpRequest):
    """The single page. Recent analyses are rendered server-side so the
    interface is never an empty form on first paint."""
    return render(
        request,
        "web/index.html",
        {
            "recent": Analysis.objects.recent(settings.HISTORY_LIMIT),
            "history_limit": settings.HISTORY_LIMIT,
            "accepted": ", ".join(SUPPORTED_SUFFIXES),
            "max_mb": MAX_UPLOAD_BYTES // 1_048_576,
            "model": _safe_model_info(),
        },
    )


@require_GET
def about(request: HttpRequest):
    """How the thing works, and what it cannot do."""
    return render(request, "web/about.html", {"model": _safe_model_info()})


@require_POST
def classify(request: HttpRequest):
    """
    Classify one uploaded file.

    Returns the full analysis document: ranked genres, a confidence verdict,
    the per-window timeline and the nearest GTZAN tracks.
    """
    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return _error("; ".join(m for errors in form.errors.values() for m in errors))

    try:
        outcome = classify_upload(form.cleaned_data["file"])
    except AudioError as exc:
        return _error(str(exc))
    except ModelUnavailable as exc:
        return _error(str(exc), status=503)
    except ExtractorMismatch as exc:
        logger.error("refusing to serve a mismatched model: %s", exc)
        return _error(str(exc), status=503)

    payload = outcome.analysis.as_dict()
    payload["cached"] = outcome.cached
    return JsonResponse(payload, status=200)


@require_GET
def history(request: HttpRequest):
    """The most recent analyses, newest first. Bounded — never `.all()`."""
    try:
        limit = max(1, min(int(request.GET.get("limit", 10)), settings.HISTORY_LIMIT))
    except (TypeError, ValueError):
        limit = 10
    return JsonResponse(
        {"results": [row.as_dict() for row in Analysis.objects.recent(limit)]}
    )


@require_http_methods(["DELETE"])
def delete(request: HttpRequest, pk: int):
    """Drop one analysis and its audio."""
    row = Analysis.objects.filter(pk=pk).first()
    if row is None:
        # Deleting something already gone is the caller's desired end state.
        return JsonResponse({"deleted": False}, status=200)
    row.delete()
    return JsonResponse({"deleted": True}, status=200)


@require_GET
def model(request: HttpRequest):
    """What the served model is, and what it claims about itself."""
    info = _safe_model_info()
    if info is None:
        return _error("no model has been trained yet", status=503)
    return JsonResponse(info)


@require_GET
def healthz(request: HttpRequest):
    """
    Liveness plus model readiness, in one call.

    The container healthcheck hits this: a process that is up but cannot load
    its model is not healthy, and reporting 200 for it would hide exactly the
    failure that shipped here once already.
    """
    info = _safe_model_info()
    ready = info is not None
    return JsonResponse(
        {"status": "ok" if ready else "degraded", "model_loaded": ready, "model": info},
        status=200 if ready else 503,
    )


def _safe_model_info() -> dict | None:
    try:
        return model_info()
    except (ModelUnavailable, ExtractorMismatch) as exc:
        logger.warning("model unavailable: %s", exc)
        return None
