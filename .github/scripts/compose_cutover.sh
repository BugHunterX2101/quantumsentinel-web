#!/usr/bin/env bash
# Deploys docker-compose.production.yml for real and moves it off the bootstrap
# superuser onto the least-privilege roles, the way an operator would:
#   A. the superuser-era stack (the API wired as in 205de7b's compose file), with data in it
#   B. `up` with the current file before provisioning: migrate cannot log in
#   C. provision (while the API runs), then `up`: migrate as qs_migrator, API as qs_app
#   D. who is connected, what the API's environment holds, data from era A, research jobs
#   E. the backup service dumps as qs_backup and copies the dataset blobs;
#      the blobs are restored onto an empty datasets volume
#   F. the API restarts as qs_app
#   G. a fresh install: provision, the backup service first, then up
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OLD=(docker compose -p qsci -f "$RUNNER_TEMP/compose.superuser.yml" --project-directory . --env-file .env.production)
NEW=(docker compose -p qsci -f docker-compose.production.yml --env-file .env.production)

ready() {
  for _ in $(seq 1 60); do
    curl -skf https://localhost/health/ready > /dev/null && return 0
    sleep 5
  done
  return 1
}
sessions() {
  "${NEW[@]}" exec -T postgres psql -U quantumsentinel -d quantumsentinel -Atc \
    "SELECT string_agg(DISTINCT usename, ',' ORDER BY usename) FROM pg_stat_activity WHERE application_name = 'quantumsentinel'"
}
# The API and the research worker each connect as their own role.
both_roles() {
  for _ in $(seq 1 60); do
    [ "$(sessions)" = "qs_app,qs_research_worker" ] && return 0
    sleep 2
  done
  echo "sessions: $(sessions)"
  return 1
}
# The research worker stores a dataset blob on the `datasets` volume and
# prints "<dataset id> <sha256 of the blob>".
store_blob() {
  "${NEW[@]}" exec -T research-worker python - <<'EOF' | tr -d '\r'
from backend.config import DATASET_ROOT
from backend.services.dataset_format import BARS_V1, Dataset, DatasetIdentity, Table
from backend.services.dataset_store import DatasetStore

assert str(DATASET_ROOT) == "/data/datasets", DATASET_ROOT
identity = DatasetIdentity("CI_EOD", "ci", "yahoo-personal-use", "1d", "exchange-local", "split",
                           "current_listings")
bars = Table.build(BARS_V1, {"symbol": ["CI"], "session_date": ["2024-01-02"], "open": [1.0], "high": [1.0],
                             "low": [1.0], "close": [1.0], "volume": [1.0]})
stored = DatasetStore(DATASET_ROOT).put(Dataset(identity, (bars,)))
assert DatasetStore(DATASET_ROOT).get(stored.dataset_id, stored.sha256).dataset_id == stored.dataset_id
print(stored.dataset_id, stored.sha256)
EOF
}
# Waits for the backup service's first pass and prints its summary line.
backup_pass() {
  for _ in $(seq 1 60); do
    line=$("${NEW[@]}" logs --no-log-prefix backup 2>&1 | grep -E 'backup (ok|failed)' | tail -1 || true)
    case "$line" in
      *"backup ok"*) echo "$line"; return 0 ;;
      *"backup failed"*) "${NEW[@]}" logs --no-log-prefix backup 2>&1 | tail -20; return 1 ;;
    esac
    sleep 2
  done
  "${NEW[@]}" logs --no-log-prefix backup 2>&1 | tail -20
  return 1
}

echo "::group::Image, TLS certificate and a generated production environment"
docker build -q -t quantumsentinel-web:production . > /dev/null
mkdir -p deploy/tls
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=localhost \
  -keyout deploy/tls/privkey.pem -out deploy/tls/fullchain.pem 2> /dev/null
chmod 644 deploy/tls/privkey.pem  # nginx runs with every capability dropped
docker run --rm -i quantumsentinel-web:production python - > .env.production <<'EOF'
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
def pem(data): return data.decode().replace("\n", "\\n")
def token(): return secrets.token_urlsafe(32)
settings = {
    "ENVIRONMENT": "production",
    "POSTGRES_PASSWORD": token(), "REDIS_PASSWORD": token(),
    "DB_APP_PASSWORD": token(), "DB_RESEARCH_WORKER_PASSWORD": token(),
    "DB_MIGRATOR_PASSWORD": token(), "DB_BACKUP_PASSWORD": token(),
    "REDIS_URL": "redis://:REPLACED_BY_COMPOSE@redis:6379/0",
    "REFRESH_TOKEN_SECRET": token(), "CSRF_SECRET": token(),
    "JWT_PRIVATE_KEY": pem(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                             serialization.NoEncryption())),
    "JWT_PUBLIC_KEY": pem(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                        serialization.PublicFormat.SubjectPublicKeyInfo)),
    "WEBHOOK_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    "PRIVATE_KEY_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    # Production refuses to start without an external PQC provider configured. This job
    # checks the database roles, not PQC, so it meets that startup check with a placeholder.
    "PQC_PROVIDER": "ci-placeholder", "PQC_PROVIDER_URL": "https://pqc.invalid",
    "CORS_ORIGINS": "https://localhost", "ALLOWED_HOSTS": "localhost",
    "PAPER_INITIAL_CASH": "100000", "PAPER_MAX_POSITION_FRACTION": "0.05",
}
print("\n".join(f"{name}={value}" for name, value in settings.items()))
EOF
docker run --rm quantumsentinel-web:production python -m backend.manage generate-server-key \
  | grep -E '^(SERVER_DSA_|TRUSTED_SERVER_DSA_)' >> .env.production
# The superuser-era deployment: the API wired as in 205de7b, and postgres,
# redis and nginx as they are now. 205de7b's own versions of those three
# cannot start: with every capability dropped, postgres and redis fail to
# switch to their unprivileged users.
git show 205de7b:docker-compose.production.yml > "$RUNNER_TEMP/compose.205de7b.yml"
docker run --rm -i --user "$(id -u)" -v "$RUNNER_TEMP:/w" -v "$PWD:/repo:ro" \
  quantumsentinel-web:production python - <<'EOF'
import yaml
old = yaml.safe_load(open("/w/compose.205de7b.yml"))
new = yaml.safe_load(open("/repo/docker-compose.production.yml"))
for name in ("postgres", "redis", "nginx"):
    old["services"][name] = new["services"][name]
old["services"]["quantumsentinel"]["networks"] = new["services"]["quantumsentinel"]["networks"]
old["networks"] = new["networks"]
with open("/w/compose.superuser.yml", "w") as out:
    yaml.safe_dump(old, out, sort_keys=False)
EOF
echo "::endgroup::"

echo "::group::A. the superuser-era stack"
"${OLD[@]}" up -d --build
ready
python3 .github/scripts/compose_smoke.py seed era-a@example.com
echo "sessions: $(sessions)"
test "$(sessions)" = "quantumsentinel"
echo "::endgroup::"

echo "::group::B. up with the current file before provisioning"
set +e
"${NEW[@]}" up -d > "$RUNNER_TEMP/phase_b.log" 2>&1
rc=$?
set -e
tail -5 "$RUNNER_TEMP/phase_b.log"
echo "up exited $rc"
test "$rc" -ne 0
"${NEW[@]}" logs migrate 2>&1 | grep -E 'qs_migrator' | tail -2
if curl -skf https://localhost/health/ready > /dev/null; then
  echo "PHASE_B_API=still-serving"
else
  echo "PHASE_B_API=down-until-provisioned"
fi
echo "::endgroup::"

echo "::group::C. provision while the API runs, then up"
"${NEW[@]}" --profile provision run --rm provision
"${NEW[@]}" up -d
ready
echo "::endgroup::"

echo "::group::D. checks"
both_roles
echo "sessions: $(sessions)"
test "$("${NEW[@]}" ps -a --format '{{.ExitCode}}' migrate)" = "0"
"${NEW[@]}" logs migrate 2>&1 | tail -1
superuser_password=$(grep '^POSTGRES_PASSWORD=' .env.production | cut -d= -f2-)
for service in quantumsentinel research-worker migrate; do
  if "${NEW[@]}" run --rm --no-deps --entrypoint env "$service" | grep -qF "$superuser_password"; then
    echo "$service holds the superuser password"; exit 1
  fi
done
if "${NEW[@]}" exec -T quantumsentinel env | grep -qF "$superuser_password"; then
  echo "the running API holds the superuser password"; exit 1
fi
echo "the API, research-worker and migrate containers do not hold the superuser password"
python3 .github/scripts/compose_smoke.py verify era-a@example.com
# The jobs ran in the research-worker container: worker ids start with its hostname.
worker_host=$("${NEW[@]}" exec -T research-worker python -c "import socket; print(socket.gethostname())" | tr -d '\r')
ran_on=$("${NEW[@]}" exec -T postgres psql -U quantumsentinel -d quantumsentinel -Atc \
  "SELECT string_agg(DISTINCT split_part(worker_id, ':', 1), ',') FROM research_jobs WHERE status = 'succeeded'")
echo "research jobs ran on: $ran_on (research-worker container: $worker_host)"
test "$ran_on" = "$worker_host"
# The research worker holds none of the API's secrets, and its job process
# (alive after those jobs, kept for the next one) holds no secret at all.
for secret in JWT_PRIVATE_KEY WEBHOOK_ENCRYPTION_KEY PRIVATE_KEY_ENCRYPTION_KEY REFRESH_TOKEN_SECRET \
              CSRF_SECRET REDIS_PASSWORD; do
  value=$(grep "^$secret=" .env.production | cut -d= -f2-)
  test -n "$value"
  if "${NEW[@]}" exec -T research-worker env | grep -qF -- "$value"; then
    echo "research-worker holds $secret"; exit 1
  fi
done
echo "research-worker holds none of the API's secrets"
"${NEW[@]}" exec -T research-worker python - <<'EOF'
from pathlib import Path

for proc in Path("/proc").glob("[0-9]*"):
    try:
        if b"spawn_main" not in (proc / "cmdline").read_bytes():
            continue
        names = {entry.split(b"=", 1)[0].decode() for entry in (proc / "environ").read_bytes().split(b"\0") if entry}
    except OSError:
        continue
    print(f"job process {proc.name} environment: {' '.join(sorted(names))}")
    leaked = names & {"DATABASE_URL", "SERVER_DSA_PRIVATE_KEY", "JWT_PRIVATE_KEY", "CSRF_SECRET", "REDIS_URL",
                      "PRIVATE_KEY_ENCRYPTION_KEY", "WEBHOOK_ENCRYPTION_KEY", "REFRESH_TOKEN_SECRET"}
    assert not leaked, leaked
    assert "DATASET_ROOT" in names, "a job process must know where to store dataset snapshots"
    break
else:
    raise SystemExit("no job process found")
EOF
# The research worker stores dataset blobs on the `datasets` volume.
read -r dataset_id dataset_sha < <(store_blob)
test -n "$dataset_sha"
echo "research-worker stored $dataset_id ($dataset_sha) on the datasets volume"
# A blob the worker is still writing: a 0600 .tmp-* file the backup service
# cannot read. Its backup pass must skip it, not fail on it.
"${NEW[@]}" exec -T -e DATASET_ID="$dataset_id" research-worker python - <<'EOF'
import os
from backend.config import DATASET_ROOT
from backend.services.dataset_store import DatasetStore

shard = DatasetStore(DATASET_ROOT).path_for(os.environ["DATASET_ID"]).parent
fd = os.open(shard / ".tmp-ci-planted", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(fd, b"unfinished")
os.close(fd)
EOF
echo "planted an unfinished 0600 blob write next to it"
# Networks: nginx cannot resolve the database; the database has no route out;
# the API and the research worker reach the internet (market data). Each
# negative check has a positive control with the same busybox wget, so a
# missing tool or a TLS problem cannot pass for isolation.
fetch() {  # service url [wget options]: prints wget's own words, returns its exit code
  "${NEW[@]}" exec -T "$1" wget -q -T 5 -O /dev/null "${@:3}" "$2" 2>&1 | tr -d '\r'
  return "${PIPESTATUS[0]}"
}
"${NEW[@]}" exec -T nginx nginx -v 2>&1 | tr -d '\r'  # deploy/nginx.conf's `resolve` needs 1.27.3+
# The API answers only the hosts in ALLOWED_HOSTS (400 otherwise).
fetch nginx http://quantumsentinel:8000/health/ready --header "Host: localhost"
echo "nginx reaches the API"
if out=$(fetch nginx http://postgres:5432); then
  echo "nginx reached postgres"; exit 1
fi
echo "nginx -> postgres: $out"
case "$out" in *"bad address"*) echo "nginx cannot resolve postgres" ;; *) exit 1 ;; esac
fetch nginx http://example.com
echo "nginx reaches the internet"
if out=$(fetch postgres http://example.com); then
  echo "postgres reached the internet"; exit 1
fi
echo "postgres -> internet: $out"
echo "postgres has no route to the internet"
for service in quantumsentinel research-worker; do
  "${NEW[@]}" exec -T "$service" python -c \
    "import urllib.request; urllib.request.urlopen('https://example.com', timeout=15)"
  echo "$service reaches the internet"
done
# The API keys the /api/auth/* rate limit, login lockout and audit records on
# the client's address, which it takes from X-Forwarded-For. Through nginx a
# client must not be able to choose it: 11 requests to one /api/auth/ path,
# each claiming a different address, and the 11th is over the limit of 10 per
# address. The path is new, so no earlier request counts against it.
probe="/api/auth/ci-xff-$RANDOM$RANDOM"
codes=()
for i in $(seq 1 11); do
  codes+=("$(curl -sk -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
    -H "X-Forwarded-For: 203.0.113.$i" -d '{}' "https://localhost$probe")")
done
echo "11 requests to $probe, each with its own X-Forwarded-For: ${codes[*]}"
for code in "${codes[@]:0:10}"; do
  case "$code" in 429|5??|000) echo "unexpected $code before the limit"; exit 1 ;; esac
done
test "${codes[10]}" = 429
echo "a client-supplied X-Forwarded-For does not change the address the API limits"
echo "::endgroup::"

echo "::group::E. backup as qs_backup, with the dataset blobs, and a dataset restore"
"${NEW[@]}" --profile backup up -d backup
backup_pass
"${NEW[@]}" exec -T backup sh -c 'pg_restore -l /backups/quantumsentinel-*.dump > /tmp/toc'
tables=$("${NEW[@]}" exec -T backup sh -c 'grep -c " TABLE DATA " /tmp/toc')
owners=$("${NEW[@]}" exec -T backup sh -c 'grep " TABLE public " /tmp/toc | awk "{print \$NF}" | sort -u | tr "\n" " "')
echo "dump: $tables tables, owned by: $owners"
test "$tables" -ge 21
test "$owners" = "qs_owner "
# The backed-up blob verifies with the application's own store; the
# unfinished write was left out.
"${NEW[@]}" --profile restore run --rm -T --entrypoint python \
  -e DATASET_ID="$dataset_id" -e DATASET_SHA="$dataset_sha" restore-datasets - <<'EOF'
import os
from backend.services.dataset_store import DatasetStore

store = DatasetStore("/backups/datasets")
store.get(os.environ["DATASET_ID"], os.environ["DATASET_SHA"])
shard = store.path_for(os.environ["DATASET_ID"]).parent
assert not (shard / ".tmp-ci-planted").exists(), "the unfinished write was backed up"
EOF
echo "the backup holds $dataset_id, verified against its hash, and not the unfinished write"
# Restore drill: lose the datasets volume, restore it from the backup, and
# the worker reads the blob and can still write next to it.
"${NEW[@]}" --profile backup rm -sf research-worker backup
docker volume rm qsci_datasets
"${NEW[@]}" --profile restore run --rm restore-datasets 2>&1 | tr -d '\r' | tee "$RUNNER_TEMP/restore.log"
grep -qE 'restore ok: [1-9][0-9]* dataset blobs copied' "$RUNNER_TEMP/restore.log"
"${NEW[@]}" up -d --no-deps research-worker
"${NEW[@]}" exec -T -e DATASET_ID="$dataset_id" -e DATASET_SHA="$dataset_sha" research-worker python - <<'EOF'
import os
import tempfile
from backend.config import DATASET_ROOT
from backend.services.dataset_store import DatasetStore

store = DatasetStore(DATASET_ROOT)
store.get(os.environ["DATASET_ID"], os.environ["DATASET_SHA"])
shard = store.path_for(os.environ["DATASET_ID"]).parent
for directory in (DATASET_ROOT, shard.parent, shard):
    assert directory.stat().st_uid == os.getuid(), (str(directory), directory.stat().st_uid, os.getuid())
fd, name = tempfile.mkstemp(prefix=".tmp-", dir=shard)
os.close(fd)
os.unlink(name)
EOF
echo "restored $dataset_id onto an empty datasets volume; the worker owns every directory and can write in them"
"${NEW[@]}" --profile backup up -d backup
echo "::endgroup::"

echo "::group::F. the API and the research worker restart as their own roles"
"${NEW[@]}" restart quantumsentinel research-worker
ready
both_roles
"${NEW[@]}" exec -T research-worker python -c \
  "from backend.config import DATASET_ROOT; from backend.services.dataset_store import DatasetStore; DatasetStore(DATASET_ROOT).get('$dataset_id')"
echo "the dataset blob survived the restart and still verifies"
# The rate-limit counter expires in the real Redis even when an earlier
# request left it without an expiry (PERSIST stands in for a failure between
# INCR and EXPIRE): the next request gives it one back, so it cannot count
# forever and limit its principal and path for good.
rcli() { "${NEW[@]}" exec -T redis sh -c 'redis-cli -a "$REDIS_PASSWORD" --no-auth-warning "$@"' rcli "$@" | tr -d '\r'; }
rpath="/api/ci-rate-$RANDOM$RANDOM"
curl -sk -o /dev/null "https://localhost$rpath"
rkey=$(rcli --scan --pattern "qs:rate:*:$rpath")
test -n "$rkey"
echo "$rkey: count $(rcli get "$rkey"), ttl $(rcli ttl "$rkey")"
test "$(rcli ttl "$rkey")" -gt 0
rcli persist "$rkey" > /dev/null
test "$(rcli ttl "$rkey")" = -1
curl -sk -o /dev/null "https://localhost$rpath"
count=$(rcli get "$rkey")
ttl=$(rcli ttl "$rkey")
echo "after PERSIST and one more request: count $count, ttl $ttl"
test "$count" = 2
test "$ttl" -gt 0
# WebSocket slots, through the API's own code against the real Redis: three
# per user, the fourth refused, the counter always with an expiry.
"${NEW[@]}" exec -T quantumsentinel python - <<'EOF'
import asyncio
import os

from backend import main


async def run():
    user = "ci-ws-" + os.urandom(4).hex()
    key = f"qs:ws:conn:{user}"
    got = [await main._ws_acquire(user) for _ in range(4)]
    count, ttl = int(await main._redis_client.get(key)), await main._redis_client.ttl(key)
    for _ in range(3):
        await main._ws_release(user, True)
    left = int(await main._redis_client.get(key))
    print(f"websocket slots: {got}, counter {count}, ttl {ttl}, after release {left}")
    assert got == [(True, True)] * 3 + [(False, False)], got
    assert count == 3 and 0 < ttl <= main._WS_COUNTER_TTL, (count, ttl)
    assert left == 0, left

asyncio.run(run())
EOF
echo "::endgroup::"

echo "::group::G. a fresh install, as the README deploys it"
"${NEW[@]}" --profile provision --profile backup down -v
"${NEW[@]}" --profile provision run --rm provision
# The backup service mounts the new datasets volume (read-only) before the
# research worker does; the worker must still own it and store blobs in it.
"${NEW[@]}" --profile backup up -d backup
backup_pass
"${NEW[@]}" up -d
ready
both_roles
read -r fresh_id fresh_sha < <(store_blob)
test -n "$fresh_sha"
echo "research-worker stored $fresh_id in a datasets volume the backup service mounted first"
python3 .github/scripts/compose_smoke.py seed fresh@example.com
echo "::endgroup::"
echo "cutover verified"
