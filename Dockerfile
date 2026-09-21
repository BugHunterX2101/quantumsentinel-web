FROM python:3.12-slim AS cpp-builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
COPY cpp ./cpp
RUN pip install --no-cache-dir "pybind11>=2.12" setuptools wheel \
    && pip wheel --no-build-isolation --no-cache-dir --no-deps --wheel-dir /wheels ./cpp

FROM python:3.12-slim
# Security: run as non-root, no new privileges
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /sbin/nologin app
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=cpp-builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/qs_fast-*.whl && rm -rf /wheels
COPY backend ./backend
COPY frontend ./frontend
# Ensure app user owns only what it needs; root owns the rest (read-only at runtime)
RUN chown -R app:app /app
EXPOSE 8000
USER app
# FORWARDED_ALLOW_IPS controls which immediate peers' X-Forwarded-For/X-Real-IP
# headers gunicorn/uvicorn will trust to derive request.client.host (used for
# login-lockout keys and IP-based rate limiting). Defaults to loopback-only so
# a bare `docker run` of this image never trusts a spoofable header; deployments
# that sit behind a reverse proxy (see docker-compose.production.yml) must set
# this explicitly to the proxy's address/CIDR.
CMD ["sh", "-c", "exec gunicorn backend.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers ${WEB_CONCURRENCY:-2} --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\" --access-logfile - --error-logfile -"]
