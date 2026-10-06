"""Test-suite settings applied before any backend module is imported."""
import os

# Tests drive research jobs explicitly (tests/test_research_jobs.py); an app
# started by TestClient must not spawn a real worker process.
os.environ.setdefault("RESEARCH_WORKER_MODE", "off")
