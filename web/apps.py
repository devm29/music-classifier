"""App configuration, including the startup warm-up."""

from __future__ import annotations

import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class WebConfig(AppConfig):
    name = "web"
    verbose_name = "Genre classification"

    def ready(self) -> None:
        """
        Run one complete analysis of a synthetic clip at process start.

        Not just `load_model()`: everything on the request path caches
        something the first time it runs, and only exercising the whole path
        warms all of it.

        * the joblib model and the 1000-row neighbour index are behind
          `lru_cache`, so the first user would otherwise pay for both while
          every other request on that worker blocks behind the GIL;
        * `python_speech_features` and the FFT underneath it cost ~0.9 s on
          their first call against ~0.05 s afterwards — and they warm *per
          input length*, so warming a one-second buffer does nothing for the
          ten-second timeline windows.

        Measured in the container: a cold upload took 4.9 s, then 1.7 s, then
        settled at ~0.35 s. Warming with a full 30-second analysis — the same
        shapes a real upload produces — costs ~1.7 s of worker boot, inside
        the healthcheck's start period, and the first real request is already
        at the steady state.

        Opt out with `WARM_MODEL=0` — `manage.py migrate`, `collectstatic` and
        the test suite have no use for it, and it makes them slower.
        """
        if os.environ.get("WARM_MODEL", "1") in ("0", "false", "False"):
            return
        try:
            import numpy as np

            from classifier.features import AudioClip, get_extractor
            from classifier.service import analyse_clip

            extractor = get_extractor()
            length = int(extractor.sample_rate * extractor.clip_seconds)
            samples = np.sin(2 * np.pi * 220 * np.arange(length) / extractor.sample_rate)
            analyse_clip(AudioClip(samples, extractor.sample_rate), extractor)
        except Exception as exc:  # pragma: no cover - startup must not hard-fail
            # A missing model is a legitimate state: the image trains at build
            # time, but a local checkout may not have run it yet. /healthz
            # reports it; the app still serves its pages and says what to do.
            logger.warning("model warm-up skipped: %s", exc)
