"""QS_PROCESS_ROLE (backend/config.py): in production each process requires
only the secrets it uses. The API requires all of them, a research worker its
database URL and the server signing key, a research job process none. Each
check imports the code in a fresh interpreter with a controlled environment;
none of them connects to a database (port 1 refuses anything)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend import worker

ROOT = Path(__file__).resolve().parent.parent
# What the research worker service is given in docker-compose.production.yml.
WORKER = {"ENVIRONMENT": "production", "QS_PROCESS_ROLE": "research-worker",
          "DATABASE_URL": "postgresql+psycopg://qs_research_worker:x@127.0.0.1:1/quantumsentinel",
          "SERVER_DSA_PRIVATE_KEY": "eA==", "SERVER_DSA_PUBLIC_KEY": "eA==",  # base64 placeholders
          "PQC_PROVIDER": "external", "PQC_PROVIDER_URL": "https://pqc.invalid"}


def run(code: str, settings: dict) -> subprocess.CompletedProcess:
    # The platform's settings only (as a job process gets them), then these.
    env = worker.job_environment(os.environ)
    del env["QS_PROCESS_ROLE"]
    env.update(settings)
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=120)


def test_a_research_worker_starts_without_the_apis_secrets():
    result = run("import backend.config as c; print(c.PROCESS_ROLE)", WORKER)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "research-worker"


@pytest.mark.parametrize("missing,message", [
    ("SERVER_DSA_PRIVATE_KEY", "SERVER_DSA_PRIVATE_KEY and SERVER_DSA_PUBLIC_KEY are required"),
    ("DATABASE_URL", "DATABASE_URL is required"),
    ("PQC_PROVIDER_URL", "external liboqs/HSM PQC provider"),
])
def test_a_research_worker_still_requires_what_it_uses(missing, message):
    settings = {name: value for name, value in WORKER.items() if name != missing}
    result = run("import backend.config", settings)
    assert result.returncode != 0 and message in result.stderr


@pytest.mark.parametrize("role", [{"QS_PROCESS_ROLE": "api"}, {}], ids=["api", "unset"])
def test_the_api_still_requires_every_secret(role):
    settings = {name: value for name, value in WORKER.items() if name != "QS_PROCESS_ROLE"}
    result = run("import backend.config", {**settings, **role})
    assert result.returncode != 0
    assert "JWT_PRIVATE_KEY and JWT_PUBLIC_KEY are required in production" in result.stderr


def test_a_job_process_requires_nothing():
    result = run("import backend.config as c; print(c.PROCESS_ROLE)",
                 {"ENVIRONMENT": "production", "QS_PROCESS_ROLE": "research-job"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "research-job"


def test_an_unknown_role_is_refused():
    result = run("import backend.config", {"QS_PROCESS_ROLE": "admin"})
    assert result.returncode != 0 and "QS_PROCESS_ROLE must be one of" in result.stderr


def test_the_api_refuses_to_run_as_a_research_process():
    result = run("import backend.main", {"QS_PROCESS_ROLE": "research-worker",
                                         "DATABASE_URL": WORKER["DATABASE_URL"]})
    assert result.returncode != 0
    assert "the API cannot run as QS_PROCESS_ROLE=research-worker" in result.stderr


def test_a_worker_refuses_to_run_as_a_job_process():
    """That role passes production's checks with no signing key at all."""
    result = run("import sys; from backend import worker; sys.exit(worker.main([]))",
                 {**WORKER, "QS_PROCESS_ROLE": "research-job"})
    assert result.returncode == 2
    assert "cannot run as QS_PROCESS_ROLE=research-job" in result.stderr


def test_a_research_worker_cannot_reach_users_private_keys():
    """It is not given the key that protects them; using them is a bug and fails."""
    code = ("from backend.services import security_service as s\n"
            "try:\n    s.protect_private_key('k')\nexcept RuntimeError as e:\n    print(e)\n")
    result = run(code, WORKER)
    assert result.returncode == 0, result.stderr
    assert "not available to a research-worker process" in result.stdout
    assert "PRIVATE_KEY_ENCRYPTION_KEY is not set" not in result.stderr
