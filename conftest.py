"""
Root conftest, imported before pytest-django calls `django.setup()`.

Its only job is to switch off the startup warm-up: `web.apps.WebConfig.ready`
loads the joblib model and builds the 1000x1000 neighbour index, which is the
right thing for a server process and pure cost for a suite that stubs both.
"""

import os

os.environ.setdefault("WARM_MODEL", "0")
os.environ.setdefault("SECRET_KEY", "test-only-key")
