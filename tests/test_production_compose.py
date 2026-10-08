"""docker-compose.production.yml: which service holds which database credential.

The API connects as qs_app, migrations run as qs_migrator, backups as
qs_backup; only postgres and the one-off provision service see the bootstrap
superuser's password. Services that load the production env file blank every
database password variable, so the file cannot leak one into them.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.production.yml"
PASSWORDS = ("POSTGRES_PASSWORD", "DB_APP_PASSWORD", "DB_MIGRATOR_PASSWORD", "DB_BACKUP_PASSWORD")
# The services allowed to receive each password.
HOLDERS = {
    "POSTGRES_PASSWORD": {"postgres", "provision"},
    "DB_APP_PASSWORD": {"quantumsentinel", "provision"},
    "DB_MIGRATOR_PASSWORD": {"migrate", "provision"},
    "DB_BACKUP_PASSWORD": {"backup", "provision"},
}


@pytest.fixture(scope="module")
def services():
    return yaml.safe_load(COMPOSE.read_text())["services"]


def _text(service) -> str:
    return json.dumps({key: service.get(key) for key in ("environment", "command", "entrypoint")})


def test_each_service_connects_as_its_own_role(services):
    def url(name):
        return services[name]["environment"]["DATABASE_URL"]
    assert url("quantumsentinel").startswith("postgresql+psycopg://qs_app:${DB_APP_PASSWORD")
    assert url("migrate").startswith("postgresql+psycopg://qs_migrator:${DB_MIGRATOR_PASSWORD")
    assert url("provision").startswith("postgresql+psycopg://quantumsentinel:${POSTGRES_PASSWORD")
    assert "-U qs_backup" in _text(services["backup"])
    assert services["backup"]["environment"]["PGPASSWORD"].startswith("${DB_BACKUP_PASSWORD")


def test_each_password_reaches_only_its_holders(services):
    for password, holders in HOLDERS.items():
        referencing = {name for name, service in services.items() if "${" + password in _text(service)}
        assert referencing == holders, password


def test_services_that_load_the_env_file_blank_every_password_variable(services):
    """Their own password reaches them inside DATABASE_URL; the variables
    themselves would otherwise come straight from the env file."""
    loading = [name for name, service in services.items() if "env_file" in service and name != "provision"]
    assert loading == ["migrate", "quantumsentinel"]
    for name in loading:
        for password in PASSWORDS:
            assert services[name]["environment"].get(password) == "", (name, password)


def test_the_api_starts_only_after_migrations_succeed(services):
    assert services["quantumsentinel"]["depends_on"]["migrate"] == {"condition": "service_completed_successfully"}
    assert services["migrate"]["restart"] == "no"
    assert services["migrate"]["command"][-1] == "migrate"
    assert services["provision"]["profiles"] == ["provision"]
    assert services["provision"]["command"][-1] == "provision-roles"


def test_every_service_can_start_with_every_capability_dropped(services):
    """Started as root, the postgres and redis entrypoints chown their data and
    switch user, which needs CHOWN, SETUID and SETGID; without them neither
    starts. They run as their images' own users instead. nginx's root master
    hands its temp directories and workers to the nginx user."""
    for name, service in services.items():
        assert service.get("cap_drop") == ["ALL"], name
    assert services["postgres"]["user"] == "postgres"
    assert services["redis"]["user"] == "redis"
    assert set(services["nginx"]["cap_add"]) == {"NET_BIND_SERVICE", "CHOWN", "SETUID", "SETGID"}


def test_external_research_workers_need_a_worker_service(services):
    """With RESEARCH_WORKER_MODE=external the API starts no worker, so research
    jobs are accepted and never run unless some service runs backend.worker."""
    external = [name for name, service in services.items()
                if (service.get("environment") or {}).get("RESEARCH_WORKER_MODE") == "external"]
    if external:
        assert any("backend.worker" in _text(service) for service in services.values()), external


def _compose_cli() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _compose_cli(), reason="needs the Docker Compose CLI (not a running daemon)")
def test_resolved_configuration_confines_each_password(tmp_path):
    """The same check on what Compose itself resolves (anchors, env_file
    merging, overrides), with an env file that holds every password."""
    marks = {password: f"mark{index}{password.lower()}" for index, password in enumerate(PASSWORDS)}
    env = tmp_path / "env"
    env.write_text("".join(f"{key}={value}\n" for key, value in marks.items())
                   + "REDIS_PASSWORD=markredis\n"
                   + f"DATABASE_URL=postgresql+psycopg://quantumsentinel:{marks['POSTGRES_PASSWORD']}"
                     "@postgres:5432/quantumsentinel\n")
    resolved = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "--project-directory", str(ROOT), "--env-file", str(env),
         "--profile", "provision", "--profile", "backup", "config", "--format", "json"],
        capture_output=True, text=True, check=True,
        env={**{key: value for key, value in os.environ.items() if key not in PASSWORDS},
             "PRODUCTION_ENV_FILE": str(env)})
    seen = set()
    for name, service in json.loads(resolved.stdout)["services"].items():
        text = _text(service)
        held = {password for password, mark in marks.items() if mark in text}
        allowed = {password for password, holders in HOLDERS.items() if name in holders}
        assert held <= allowed, (name, held - allowed)
        seen.add(name)
    assert {"quantumsentinel", "migrate", "provision", "backup"} <= seen
