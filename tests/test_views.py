"""
Tests for the HTTP surface.

This is where the project's failures actually surfaced to a user: a malformed
body came back as a 500, a model that could not load came back as a TypeError,
and the "top three genres" panel returned the *characters* of one genre name,
so the interface displayed `r, o, c, k`. Those endpoints are gone; the
regressions they represent are pinned here against their replacements.

The audio and the model are both stubbed — what is under test is the adapter,
not scikit-learn.
"""

from __future__ import annotations

import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from classifier.audio import AudioError
from classifier.features import FeatureError
from classifier.predict import ExtractorMismatch, ModelUnavailable
from web import services as services_module
from web import views
from web.models import Analysis

ANALYSIS = {
    "ranked": [
        {"genre": "rock", "probability": 0.62},
        {"genre": "metal", "probability": 0.21},
        {"genre": "pop", "probability": 0.09},
    ],
    "verdict": {
        "genre": "rock",
        "probability": 0.62,
        "margin": 0.41,
        "band": "confident",
        "reason": "62% on rock, 41% clear of the runner-up.",
    },
    "timeline": [
        {"start": 0.0, "end": 10.0, "genre": "rock", "probability": 0.6, "runner_up": "metal"},
        {"start": 5.0, "end": 15.0, "genre": "metal", "probability": 0.4, "runner_up": "rock"},
    ],
    "neighbours": [
        {"track": "rock.00012", "genre": "rock", "distance": 8.1, "closer_than": 0.97},
    ],
    "duration_seconds": 30.0,
    "sample_rate": 22050,
    "extractor": "mfcc-stats-104",
    "elapsed_ms": 180,
}


@pytest.fixture(autouse=True)
def isolated_media(settings, tmp_path):
    """Keep uploads made by tests out of the repository's own media/ directory."""
    settings.MEDIA_ROOT = str(tmp_path / "media")
    return settings.MEDIA_ROOT


@pytest.fixture
def stub_model(monkeypatch):
    """A served model, without loading a joblib file."""
    info = {
        "genres": ["blues", "classical", "country", "disco", "hiphop",
                   "jazz", "metal", "pop", "reggae", "rock"],
        "classifier": "svm-rbf-calibrated",
        "extractor": "mfcc-stats-104",
        "extractor_version": "1.0",
        "sample_rate": 22050,
        "clip_seconds": 30.0,
        "cv_accuracy_mean": 0.712,
        "cv_accuracy_std": 0.02,
        "sklearn_version": "1.5.2",
        "trained_at": "2026-09-23T00:00:00+00:00",
        "path": "genre_classifier.joblib",
    }
    monkeypatch.setattr(views, "model_info", lambda: dict(info))
    return info


@pytest.fixture
def stub_analysis(monkeypatch):
    """Replace the whole analysis with a recorder, so views are tested alone."""
    calls = []

    def fake(path, **kwargs):
        calls.append(str(path))
        return dict(ANALYSIS)

    monkeypatch.setattr(services_module.service, "analyse_file", fake)
    return calls


def wav_upload(name="track.wav", payload=b"RIFF0000WAVEfmt "):
    return SimpleUploadedFile(name, payload, content_type="audio/wav")


def post_file(client, upload):
    return client.post(reverse("classify"), {"file": upload})


class TestClassifyEndpoint:
    def test_it_returns_the_full_ranked_analysis(self, client, db, stub_analysis):
        response = post_file(client, wav_upload())
        assert response.status_code == 200
        body = response.json()
        assert [r["genre"] for r in body["ranked"]] == ["rock", "metal", "pop"]
        assert body["verdict"]["band"] == "confident"

    def test_genres_are_returned_as_names_not_as_characters(
        self, client, db, stub_analysis
    ):
        # The regression: `', '.join(genre)` joined the characters of the top
        # genre name, so the interface showed "r, o, c, k".
        body = post_file(client, wav_upload()).json()
        assert body["verdict"]["genre"] == "rock"
        assert "r, o, c, k" not in json.dumps(body)

    def test_it_records_the_analysis(self, client, db, stub_analysis):
        post_file(client, wav_upload())
        row = Analysis.objects.get()
        assert row.top_genre == "rock"
        assert row.band == "confident"
        assert row.result["timeline"]

    def test_a_request_with_no_file_is_a_client_error_not_a_crash(self, client, db):
        response = client.post(reverse("classify"), {})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_an_unsupported_extension_is_refused_with_the_reason(self, client, db):
        response = post_file(client, wav_upload(name="notes.txt"))
        assert response.status_code == 400
        assert ".txt" in response.json()["error"]

    def test_an_empty_file_is_refused(self, client, db):
        response = post_file(client, wav_upload(payload=b""))
        assert response.status_code == 400
        assert "empty" in response.json()["error"]

    def test_an_oversized_file_is_refused_before_decoding(self, client, db, monkeypatch):
        monkeypatch.setattr("web.forms.MAX_UPLOAD_BYTES", 16)
        response = post_file(client, wav_upload(payload=b"x" * 64))
        assert response.status_code == 400
        assert "limit" in response.json()["error"]

    def test_an_undecodable_upload_returns_the_reason(self, client, db, monkeypatch):
        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(AudioError("ffmpeg is not installed")),
        )
        response = post_file(client, wav_upload())
        assert response.status_code == 400
        assert "ffmpeg" in response.json()["error"]

    def test_a_missing_model_is_a_server_state_not_a_bad_request(
        self, client, db, monkeypatch
    ):
        # 503, not 400: the upload was fine, the server is not ready.
        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("no model yet")),
        )
        response = post_file(client, wav_upload())
        assert response.status_code == 503
        assert "no model" in response.json()["error"]

    def test_a_mismatched_extractor_refuses_to_serve(self, client, db, monkeypatch):
        # Serving anyway means numbers that look like predictions but were
        # computed over a different feature definition.
        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(ExtractorMismatch("retrain")),
        )
        assert post_file(client, wav_upload()).status_code == 503

    def test_a_failed_analysis_leaves_no_stored_audio(self, client, db, monkeypatch, settings):
        from pathlib import Path

        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(AudioError("broken")),
        )
        post_file(client, wav_upload())
        assert not Analysis.objects.exists()
        stored = list(Path(settings.MEDIA_ROOT).rglob("*")) if Path(settings.MEDIA_ROOT).exists() else []
        assert [p for p in stored if p.is_file()] == []

    def test_an_unexpected_feature_error_is_not_swallowed(self, client, db, monkeypatch):
        # A bare `except` that turned every failure into a 200 is how the
        # original returned None-shaped answers. Unknown failures must surface.
        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(FeatureError("unexpected")),
        )
        with pytest.raises(FeatureError):
            post_file(client, wav_upload())

    def test_get_is_not_allowed(self, client, db):
        assert client.get(reverse("classify")).status_code == 405


class TestContentHashCache:
    def test_the_same_file_twice_is_analysed_once(self, client, db, stub_analysis):
        first = post_file(client, wav_upload()).json()
        second = post_file(client, wav_upload()).json()
        assert len(stub_analysis) == 1, "identical bytes were re-analysed"
        assert second["cached"] is True
        assert first["cached"] is False
        assert second["id"] == first["id"]
        assert Analysis.objects.count() == 1

    def test_different_bytes_are_analysed_separately(self, client, db, stub_analysis):
        post_file(client, wav_upload(payload=b"RIFFaaaaWAVE"))
        post_file(client, wav_upload(payload=b"RIFFbbbbWAVE"))
        assert len(stub_analysis) == 2
        assert Analysis.objects.count() == 2

    def test_a_duplicate_upload_is_not_stored_twice(
        self, client, db, stub_analysis, settings
    ):
        from pathlib import Path

        post_file(client, wav_upload())
        post_file(client, wav_upload())
        files = [p for p in Path(settings.MEDIA_ROOT).rglob("*") if p.is_file()]
        assert len(files) == 1


class TestHistoryIsBounded:
    def test_history_is_pruned_to_the_limit(self, client, db, stub_analysis, settings):
        settings.HISTORY_LIMIT = 3
        for i in range(6):
            post_file(client, wav_upload(payload=f"RIFF{i:04d}WAVE".encode()))
        assert Analysis.objects.count() == 3

    def test_pruning_removes_the_oldest(self, client, db, stub_analysis, settings):
        settings.HISTORY_LIMIT = 2
        for i in range(4):
            post_file(client, wav_upload(name=f"t{i}.wav",
                                         payload=f"RIFF{i:04d}WAVE".encode()))
        names = set(Analysis.objects.values_list("original_name", flat=True))
        assert names == {"t2.wav", "t3.wav"}

    def test_the_history_endpoint_never_returns_more_than_the_limit(
        self, client, db, stub_analysis, settings
    ):
        settings.HISTORY_LIMIT = 3
        for i in range(5):
            post_file(client, wav_upload(payload=f"RIFF{i:04d}WAVE".encode()))
        # Was the classic unbounded `.all()`: a caller asking for 10000 rows got them.
        body = client.get(reverse("history"), {"limit": 10000}).json()
        assert len(body["results"]) <= 3

    def test_a_nonsense_limit_falls_back_to_the_default(self, client, db, stub_analysis):
        post_file(client, wav_upload())
        assert client.get(reverse("history"), {"limit": "lots"}).status_code == 200

    def test_history_is_newest_first(self, client, db, stub_analysis):
        for i in range(3):
            post_file(client, wav_upload(name=f"t{i}.wav",
                                         payload=f"RIFF{i:04d}WAVE".encode()))
        names = [r["name"] for r in client.get(reverse("history")).json()["results"]]
        assert names == ["t2.wav", "t1.wav", "t0.wav"]


class TestDelete:
    def test_it_removes_the_row_and_its_audio(self, client, db, stub_analysis, settings):
        from pathlib import Path

        pk = post_file(client, wav_upload()).json()["id"]
        response = client.delete(reverse("delete", args=[pk]))
        assert response.status_code == 200 and response.json()["deleted"] is True
        assert not Analysis.objects.filter(pk=pk).exists()
        assert [p for p in Path(settings.MEDIA_ROOT).rglob("*") if p.is_file()] == []

    def test_deleting_something_already_gone_is_not_a_crash(self, client, db):
        # Was `Music.objects.get(id=...)` with no guard: a retried request hit
        # DoesNotExist and came back as a 500.
        response = client.delete(reverse("delete", args=[9999]))
        assert response.status_code == 200 and response.json()["deleted"] is False

    def test_get_is_not_allowed(self, client, db):
        assert client.get(reverse("delete", args=[1])).status_code == 405


class TestPages:
    def test_the_index_renders_with_no_model_present(self, client, db, monkeypatch):
        # A fresh clone has no artifact. The page must still say what to do
        # rather than 500.
        monkeypatch.setattr(
            views, "model_info",
            lambda: (_ for _ in ()).throw(ModelUnavailable("none")),
        )
        response = client.get(reverse("index"))
        assert response.status_code == 200
        assert b"classifier.train" in response.content

    def test_the_index_shows_the_served_model(self, client, db, stub_model):
        assert b"svm-rbf-calibrated" in client.get(reverse("index")).content

    def test_the_index_lists_recent_analyses(self, client, db, stub_analysis, stub_model):
        post_file(client, wav_upload(name="dominoes.wav"))
        assert b"dominoes.wav" in client.get(reverse("index")).content

    def test_the_about_page_renders(self, client, db, stub_model):
        assert client.get(reverse("about")).status_code == 200


class TestModelAndHealth:
    def test_the_model_endpoint_reports_the_served_artifact(self, client, db, stub_model):
        body = client.get(reverse("model")).json()
        assert body["classifier"] == "svm-rbf-calibrated"
        assert body["extractor"] == "mfcc-stats-104"

    def test_the_model_endpoint_is_503_with_no_model(self, client, db, monkeypatch):
        monkeypatch.setattr(
            views, "model_info",
            lambda: (_ for _ in ()).throw(ModelUnavailable("none")),
        )
        assert client.get(reverse("model")).status_code == 503

    def test_healthz_is_ok_when_the_model_loads(self, client, db, stub_model):
        response = client.get(reverse("healthz"))
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok", "model_loaded": True, "model": stub_model
        }

    def test_healthz_is_degraded_when_the_model_does_not_load(
        self, client, db, monkeypatch
    ):
        # A process that is up but cannot load its model is not healthy, and
        # a 200 here would hide exactly the failure that shipped once already.
        monkeypatch.setattr(
            views, "model_info",
            lambda: (_ for _ in ()).throw(ModelUnavailable("none")),
        )
        response = client.get(reverse("healthz"))
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False
