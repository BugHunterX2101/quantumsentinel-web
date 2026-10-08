#!/usr/bin/env bash
# Deploys docker-compose.production.yml for real and moves it off the bootstrap
# superuser onto the least-privilege roles, the way an operator would:
#   A. the superuser-era stack (the API wired as in 205de7b's compose file), with data in it
#   B. `up` with the current file before provisioning: migrate cannot log in
#   C. provision (while the API runs), then `up`: migrate as qs_migrator, API as qs_app
#   D. who is connected, what the API's environment holds, data from era A, research jobs
#   E. the backup service dumps as qs_backup
#   F. the API restarts as qs_app
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
    "DB_APP_PASSWORD": token(), "DB_MIGRATOR_PASSWORD": token(), "DB_BACKUP_PASSWORD": token(),
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
echo "sessions: $(sessions)"
test "$(sessions)" = "qs_app"
test "$("${NEW[@]}" ps -a --format '{{.ExitCode}}' migrate)" = "0"
"${NEW[@]}" logs migrate 2>&1 | tail -1
superuser_password=$(grep '^POSTGRES_PASSWORD=' .env.production | cut -d= -f2-)
for service in quantumsentinel migrate; do
  if "${NEW[@]}" run --rm --no-deps --entrypoint env "$service" | grep -qF "$superuser_password"; then
    echo "$service holds the superuser password"; exit 1
  fi
done
if "${NEW[@]}" exec -T quantumsentinel env | grep -qF "$superuser_password"; then
  echo "the running API holds the superuser password"; exit 1
fi
echo "the API and migrate containers do not hold the superuser password"
python3 .github/scripts/compose_smoke.py verify era-a@example.com
echo "::endgroup::"

echo "::group::E. backup as qs_backup"
"${NEW[@]}" --profile backup up -d backup
for _ in $(seq 1 60); do
  "${NEW[@]}" exec -T backup sh -c 'pg_restore -l /backups/*.dump > /tmp/toc' 2> /dev/null && break
  sleep 2
done
tables=$("${NEW[@]}" exec -T backup sh -c 'grep -c " TABLE DATA " /tmp/toc')
owners=$("${NEW[@]}" exec -T backup sh -c 'grep " TABLE public " /tmp/toc | awk "{print \$NF}" | sort -u | tr "\n" " "')
echo "dump: $tables tables, owned by: $owners"
test "$tables" -ge 21
test "$owners" = "qs_owner "
echo "::endgroup::"

echo "::group::F. the API restarts as qs_app"
"${NEW[@]}" restart quantumsentinel
ready
test "$(sessions)" = "qs_app"
echo "::endgroup::"
echo "cutover verified"
