# ---------------------------------------------------------------------------
# Stage 1 — build the virtualenv and train the model.
#
# Training happens at build time, not at boot: the feature matrix is in the
# repository (the audio is not, and does not need to be), fitting takes a few
# seconds, and it means the image ships an artifact that the installed
# scikit-learn can actually load. The repository used to ship five pickles
# produced by scikit-learn 0.18 that no currently installable version can
# unpickle, so the app could not classify anything on a fresh install.
#
# Build tooling, the compiler toolchain and the dev requirements all stay in
# this stage; the runtime image gets the virtualenv, the code and the artifact.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Only what training and collectstatic need, so a template edit does not
# invalidate the dependency layer above.
COPY classifier ./classifier
COPY config ./config
COPY web ./web
COPY data ./data
COPY manage.py ./

# The artifact records the extractor it was trained with; `classifier.predict`
# refuses to serve a model whose extractor does not match the installed one.
RUN python -m classifier.train

# DEBUG=0 so this uses the hashed, pre-compressed WhiteNoise storage that the
# runtime image serves.
ENV DEBUG=0 SECRET_KEY=build-time-only PYTHONPATH=/build
RUN python manage.py collectstatic --noinput


# ---------------------------------------------------------------------------
# Stage 2 — runtime.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=config.settings \
    MEDIA_ROOT=/app/var/media \
    DATABASE_PATH=/app/var/db.sqlite3

# ffmpeg only: pydub shells out to it for anything that is not a plain WAV.
# No compiler, no pip cache, no dev requirements.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app classifier ./classifier
COPY --chown=app:app config ./config
COPY --chown=app:app web ./web
COPY --chown=app:app data ./data
COPY --chown=app:app manage.py docker-entrypoint.sh ./
COPY --from=builder --chown=app:app /build/artifacts ./artifacts
COPY --from=builder --chown=app:app /build/staticfiles ./staticfiles

RUN chmod +x docker-entrypoint.sh && mkdir -p /app/var/media && chown -R app:app /app/var

USER app
EXPOSE 8000

# A process that is up but cannot load its model is not healthy. /healthz
# reports 503 in that case, and urlopen raises on it, so the check fails.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz/', timeout=4)"]

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["gunicorn", "config.wsgi", "--bind", "0.0.0.0:8000", \
     "--workers", "2", "--threads", "2", "--timeout", "90", "--log-file", "-"]
