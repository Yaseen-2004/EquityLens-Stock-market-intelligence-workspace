"""Vercel entry point.

Vercel's Python runtime imports this module and serves the WSGI callable it
finds, so all this needs to do is expose the Flask app. Everything else lives
in app.py, which still runs standalone with `python app.py`.
"""
import os
import sys

# The project root is one level up; Vercel runs with the repo root as cwd but
# does not put it on sys.path for a module inside api/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402

# Vercel looks for `app` (or `application`). Both are provided.
application = app
