"""Shared pytest setup.

Unit tests never download or load the HuggingFace models: model-backed calls
are replaced with small fakes, and the action server's import-time preload is
disabled. The dashboard runs against a throwaway SQLite database.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before actions.actions / dashboard/app.py are imported.
os.environ.setdefault("PRELOAD_MODELS", "false")
os.environ.setdefault("ORDER_PROVIDER", "mock")
os.environ.setdefault("JWT_SECRET_KEY", "t" * 64)
os.environ.setdefault("DASHBOARD_USERNAME", "admin")
os.environ.setdefault("DASHBOARD_PASSWORD", "test-password-123")
os.environ.setdefault("DASHBOARD_INGEST_TOKEN", "i" * 64)
os.environ.setdefault("RATELIMIT_STORAGE_URI", "memory://")
