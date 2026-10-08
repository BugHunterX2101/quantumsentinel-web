"""HTTP checks for the production stack behind nginx (stdlib only; used by compose_cutover.sh).

    compose_smoke.py seed <email>      register a user and write data as them
    compose_smoke.py verify <email>    that user still works; a new user runs research jobs
"""
import http.cookiejar
import json
import math
import ssl
import sys
import time
import urllib.error
import urllib.request

BASE = "https://localhost"
PASSWORD = "Str0ng!Passw0rd#2026"
TLS = ssl.create_default_context()
TLS.check_hostname = False
TLS.verify_mode = ssl.CERT_NONE  # the job's own self-signed certificate


class Client:
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
                                                  urllib.request.HTTPSHandler(context=TLS))
        self.headers = {"Origin": BASE, "Content-Type": "application/json"}

    def call(self, method, path, body=None):
        request = urllib.request.Request(BASE + path, method=method, headers=self.headers,
                                         data=None if body is None else json.dumps(body).encode())
        try:
            with self.opener.open(request, timeout=120) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode(errors="replace")[:300]

    def login(self, email, register):
        if register:
            status, body = self.call("POST", "/api/auth/register", {"email": email, "password": PASSWORD})
            check(status in (200, 201), f"register {email}", status, body)
        status, body = self.call("POST", "/api/auth/login", {"email": email, "password": PASSWORD})
        check(status == 200, f"login {email}", status, body)
        self.headers["X-CSRF-Token"] = body["csrf_token"]

    def settle(self, path, body):
        status, job = self.call("POST", path, body)
        check(status == 202, f"queue {path}", status, job)
        deadline = time.monotonic() + 300
        while job["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(1)
            status, job = self.call("GET", job["poll_url"])
        check(job["status"] == "succeeded", f"job {path}", job["status"], job.get("error"))
        return job["result"]


def check(ok, what, *detail):
    print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  {detail}"), flush=True)
    if not ok:
        sys.exit(1)


def seed(email):
    client = Client()
    client.login(email, register=True)
    status, body = client.call("GET", "/api/portfolio/account")
    check(status == 200, "account", status, body)
    status, body = client.call("POST", "/api/integrations/webhooks",
                               {"url": "https://example.com/hook", "event_types": ["order.filled"]})
    check(status == 201, "create a webhook", status, body)


def verify(email):
    old = Client()
    old.login(email, register=False)
    status, hooks = old.call("GET", "/api/integrations/webhooks")
    check(status == 200 and len(hooks) >= 1, "data written before the cutover is still there", status, hooks)
    client = Client()
    client.login(f"after-{int(time.time())}@example.com", register=True)
    bars, price = [], 100.0
    for i in range(300):
        opened, price = price, price * (1 + 0.01 * math.sin(i / 7))
        bars.append({"timestamp": 1700000000 + i * 60, "open": opened, "high": max(opened, price) * 1.002,
                     "low": min(opened, price) * 0.998, "close": price, "volume": 1000})
    status, body = client.call("POST", "/api/experiments/create", {
        "strategy_id": "obi_momentum", "dataset_id": "ci", "dataset": bars, "parameters": {"events_per_bar": 50}})
    check(status == 201, "create an experiment", status, body)
    client.settle(f"/api/experiments/{body['experiment_id']}/run", None)
    returns = [0.0004 + 0.01 * math.sin(i * 1.3) for i in range(1000)]
    result = client.settle("/api/research/stat-test", {"returns": returns, "n_bootstrap": 200, "n_permutations": 200})
    check(result["bootstrap_sharpe"]["n_bootstrap"] == 200, "stat-test result", result)
    status, events = client.call("GET", "/api/security/audit-log?limit=20")
    check(status == 200 and len(events) >= 3 and all(e["verified"] for e in events),
          "audit events written as qs_app are signed and verify", status, events)


if __name__ == "__main__":
    {"seed": seed, "verify": verify}[sys.argv[1]](sys.argv[2])
