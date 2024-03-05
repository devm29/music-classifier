#!/bin/sh
# Bring the container to a usable state before serving: migrate, then seed the
# history so the first page a reviewer sees is populated rather than an empty
# upload form. `seed_demo` is a no-op when the history already has rows, so a
# restart does not duplicate anything.
set -e

python manage.py migrate --noinput
python manage.py seed_demo || echo "seed skipped"

exec "$@"
