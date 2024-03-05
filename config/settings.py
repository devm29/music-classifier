"""
Django settings.

Everything environment-dependent is read from the environment with a
development default, so the same image runs locally and behind a real host
without a second settings file.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "no")


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


# SECURITY WARNING: keep the secret key used in production secret.
# It was committed in plain text once; it comes from the environment now, with
# a development fallback that is obviously not a secret.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-key-change-me")

DEBUG = _flag("DEBUG", "1")

ALLOWED_HOSTS = _csv("ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0,[::1]")
CSRF_TRUSTED_ORIGINS = _csv("CSRF_TRUSTED_ORIGINS", "http://localhost:8310")

INSTALLED_APPS = [
    "web.apps.WebConfig",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise serves the collected, hashed static files in the runtime
    # image, so the container needs gunicorn and nothing else in front of it.
    # Under DEBUG the staticfiles app serves them straight from the app
    # directories instead, and adding WhiteNoise there only produces a warning
    # about a `staticfiles/` directory a developer has no reason to create.
    *([] if DEBUG else ["whitenoise.middleware.WhiteNoiseMiddleware"]),
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # SQLite is the right call here: the only queries are a unique-key
        # lookup and "the most recent N". An empty `db` file used to be
        # committed at the repository root; this one is gitignored.
        "NAME": os.environ.get("DATABASE_PATH", str(BASE_DIR / "db.sqlite3")),
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    # Hashed, pre-compressed filenames in the runtime image, so the container
    # needs gunicorn and nothing else in front of it. The manifest backend is
    # deliberately not used under DEBUG: it requires `collectstatic` to have
    # run, which is true of the image and not of a fresh checkout or a test
    # run, and failing there gives a stack trace about a manifest rather than
    # about whatever the developer was actually doing.
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        if DEBUG
        else "whitenoise.storage.CompressedManifestStaticFilesStorage"
    },
}

MEDIA_URL = "/media/"
# Uploads go to <repo>/media/: the path docker-compose mounts as a volume, the
# path the Dockerfile creates and chowns, and the path .gitignore excludes.
# Pointing this anywhere else means uploads land inside the container image and
# vanish on the next `docker compose up --build`.
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", str(BASE_DIR / "media"))

# A demo that serves uploaded audio back for playback. A real deployment puts
# media behind the web server or object storage and turns this off.
SERVE_MEDIA = _flag("SERVE_MEDIA", "1")

#: How many analyses to keep. Every successful classification prunes past this,
#: so neither the table nor media/ grows without bound.
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "20"))

# Bound what a request can push into memory before it is rejected. The upload
# form enforces the same ceiling with a readable message; this is the backstop.
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 41 * 1024 * 1024

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")},
}

if not DEBUG:
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SESSION_COOKIE_SECURE = _flag("SECURE_COOKIES", "1")
    CSRF_COOKIE_SECURE = _flag("SECURE_COOKIES", "1")
    X_FRAME_OPTIONS = "DENY"
