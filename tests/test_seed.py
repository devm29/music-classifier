"""
Tests for the persistence model and the demo seed.

`seed_demo` is what makes "one command, no manual steps" true: the compose
stack comes up with a populated history instead of an empty form. It must
therefore be idempotent, must fail loudly rather than half-seed when no model
is trained, and must never fabricate a result — the clips are synthesised, the
analysis is real.

Inference is stubbed here so the suite stays fast; `test_service.py` covers the
real analysis path.
"""

from __future__ import annotations

from io import StringIO

import pytest
from django.core.management import CommandError, call_command

from classifier.predict import ModelUnavailable
from web import services as services_module
from web.models import Analysis

pytestmark = pytest.mark.django_db

RESULT = {
    "ranked": [{"genre": "metal", "probability": 0.55}, {"genre": "rock", "probability": 0.2}],
    "verdict": {"genre": "metal", "probability": 0.55, "margin": 0.35,
                "band": "confident", "reason": "55% on metal."},
    "timeline": [],
    "neighbours": [],
    "duration_seconds": 30.0,
    "sample_rate": 22050,
    "extractor": "mfcc-stats-104",
    "elapsed_ms": 120,
}


@pytest.fixture(autouse=True)
def isolated_media(settings, tmp_path):
    settings.MEDIA_ROOT = str(tmp_path / "media")


@pytest.fixture
def stub_analysis(monkeypatch):
    monkeypatch.setattr(services_module.service, "analyse_file", lambda *a, **k: dict(RESULT))


def seed(*args):
    out = StringIO()
    call_command("seed_demo", *args, stdout=out)
    return out.getvalue()


class TestSeedDemo:
    def test_it_populates_the_history(self, stub_analysis):
        output = seed()
        assert Analysis.objects.count() == 3
        assert "seeded" in output

    def test_each_seeded_row_is_a_real_stored_file(self, stub_analysis, settings):
        from pathlib import Path

        seed()
        files = [p for p in Path(settings.MEDIA_ROOT).rglob("*") if p.is_file()]
        assert len(files) == 3
        assert all(row.file.name for row in Analysis.objects.all())

    def test_the_clips_are_distinct_so_the_cache_does_not_collapse_them(
        self, stub_analysis
    ):
        # Three identical waveforms would hash the same and the history would
        # come up with one row, which is the empty-ish state seeding exists to
        # avoid.
        seed()
        assert len({row.sha256 for row in Analysis.objects.all()}) == 3

    def test_seeding_twice_does_not_duplicate(self, stub_analysis):
        seed()
        output = seed()
        assert Analysis.objects.count() == 3
        assert "nothing to seed" in output

    def test_force_reseeds_an_already_populated_history(self, stub_analysis):
        seed()
        assert "seeded" in seed("--force")

    def test_no_model_is_a_command_error_that_says_what_to_run(self, monkeypatch):
        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("no model")),
        )
        with pytest.raises(CommandError, match="classifier.train"):
            seed()

    def test_a_failed_seed_leaves_nothing_behind(self, monkeypatch, settings):
        from pathlib import Path

        monkeypatch.setattr(
            services_module.service, "analyse_file",
            lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("no model")),
        )
        with pytest.raises(CommandError):
            seed()
        assert not Analysis.objects.exists()
        root = Path(settings.MEDIA_ROOT)
        assert not root.exists() or [p for p in root.rglob("*") if p.is_file()] == []


class TestAnalysisModel:
    def test_the_string_form_names_the_genre_and_confidence(self, stub_analysis):
        seed()
        assert "->" in str(Analysis.objects.first())

    def test_as_dict_merges_the_row_onto_the_stored_result(self, stub_analysis):
        seed()
        payload = Analysis.objects.first().as_dict()
        assert payload["verdict"]["genre"] == "metal"
        assert payload["ranked"] == RESULT["ranked"]
        assert payload["sha256"] and payload["audio_url"] and payload["created_at"]

    def test_deleting_a_row_deletes_its_audio(self, stub_analysis, settings):
        from pathlib import Path

        seed()
        row = Analysis.objects.first()
        path = Path(row.file.path)
        assert path.exists()
        row.delete()
        assert not path.exists()

    def test_recent_is_bounded_and_newest_first(self, stub_analysis):
        seed()
        rows = list(Analysis.objects.recent(2))
        assert len(rows) == 2
        assert rows[0].created_at >= rows[1].created_at

    def test_the_hash_is_unique(self, stub_analysis):
        from django.db import IntegrityError

        seed()
        existing = Analysis.objects.first()
        with pytest.raises(IntegrityError):
            Analysis.objects.create(
                original_name="dup.wav", sha256=existing.sha256, size_bytes=1,
                duration_seconds=1.0, top_genre="rock", top_probability=0.5,
                band="uncertain", result={},
            )


class TestStartupWarmUp:
    """
    `WebConfig.ready` runs one full analysis at boot so the first real upload
    is not the one that pays for the joblib load, the neighbour index and the
    first FFT. It must never be able to stop the process from starting.
    """

    def test_it_is_skipped_when_switched_off(self, monkeypatch):
        from django.apps import apps

        monkeypatch.setenv("WARM_MODEL", "0")
        monkeypatch.setattr(
            "classifier.service.analyse_clip",
            lambda *a, **k: pytest.fail("warm-up ran with WARM_MODEL=0"),
        )
        apps.get_app_config("web").ready()

    def test_it_runs_a_full_analysis_when_switched_on(self, monkeypatch):
        from django.apps import apps

        seen = []
        monkeypatch.setenv("WARM_MODEL", "1")
        monkeypatch.setattr(
            "classifier.service.analyse_clip",
            lambda clip, extractor=None, **k: seen.append(clip.duration_seconds),
        )
        apps.get_app_config("web").ready()
        # The whole clip length, so the ten-second timeline windows warm too.
        assert seen == [30.0]

    def test_a_missing_model_does_not_stop_the_process_starting(self, monkeypatch):
        from django.apps import apps

        monkeypatch.setenv("WARM_MODEL", "1")
        monkeypatch.setattr(
            "classifier.service.analyse_clip",
            lambda *a, **k: (_ for _ in ()).throw(ModelUnavailable("none")),
        )
        apps.get_app_config("web").ready()  # must not raise
