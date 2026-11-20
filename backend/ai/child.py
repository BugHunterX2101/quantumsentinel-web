"""An AI job's own process: no database, no secret, bounded resources.

The AI worker runs each job's target (backend/ai/kinds.py) in a fresh
interpreter that it starts with subprocess.Popen. The process gets:

* an environment built from an allow-list (CHILD_ENVIRONMENT), so it holds
  no database URL or password. QS_PROCESS_ROLE is ai-job, under which
  backend.config refuses to load;
* no file descriptor of the worker's (close_fds), so no database socket;
* ``python -E -s``, so no PYTHON* variable or user site directory shapes it;
* on Linux, limits on memory, CPU time, open files and core dumps. macOS
  refuses to lower RLIMIT_AS below the address space an arm64 interpreter
  already reserves, so elsewhere the job runs without them; production runs
  on Linux.

The worker writes ``{"target": ..., "payload": ...}`` to its stdin. It
answers with the result as JSON on stdout and exits 0, or exits non-zero with
a one-line reason on stderr. Anything the target prints goes to stderr, so it
cannot corrupt the result.
"""
from __future__ import annotations

import importlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB_ROLE = "ai-job"

# What a job process may inherit from its worker. Proxy settings carry the
# egress proxy (R4-2); the Windows names are what an interpreter needs to
# start there at all.
CHILD_ENVIRONMENT = frozenset({
    "PATH", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TEMP", "TMP",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR",
    "ENVIRONMENT",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
})


def child_environment(environ) -> dict[str, str]:
    """The environment a job process starts with, from its worker's."""
    kept = {name: value for name, value in environ.items()
            if name.upper() in CHILD_ENVIRONMENT or name.upper().startswith("LC_")}
    kept["QS_PROCESS_ROLE"] = JOB_ROLE
    return kept


def command() -> list[str]:
    return [sys.executable, "-E", "-s", "-m", "backend.ai.child"]


def _linux_limits(memory_bytes: int, cpu_seconds: int):
    def apply():  # runs in the child between fork and exec
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    return apply


def start(memory_bytes: int, timeout_seconds: float, environ=None) -> subprocess.Popen:
    """Start a job process; the caller sends its job with ``communicate``."""
    posix = os.name == "posix"
    linux = sys.platform.startswith("linux")
    return subprocess.Popen(
        command(), cwd=ROOT, env=child_environment(os.environ if environ is None else environ),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
        preexec_fn=_linux_limits(memory_bytes, math.ceil(timeout_seconds) + 5) if linux else None,
        start_new_session=posix)


def request(target: str, payload: dict) -> bytes:
    return json.dumps({"target": target, "payload": payload}).encode()


class JobFailed(Exception):
    pass


def outcome(returncode: int, stdout: bytes, stderr: bytes, max_bytes: int) -> dict:
    """The job's result from a finished process, or JobFailed."""
    if returncode != 0:
        reason = stderr.decode("utf-8", "replace").strip().splitlines()
        raise JobFailed(reason[-1][:500] if reason else f"exited with status {returncode}")
    if len(stdout) > max_bytes:
        raise JobFailed(f"result larger than {max_bytes} bytes")
    try:
        result = json.loads(stdout)
    except ValueError:
        raise JobFailed("result is not JSON") from None
    if not isinstance(result, dict):
        raise JobFailed("result is not a JSON object")
    return result


def run_target(target: str, payload: dict) -> dict:
    from .kinds import TARGETS

    if target not in TARGETS:
        raise PermissionError(f"not an AI job target: {target!r}")
    module_name, _, function = target.partition(":")
    return getattr(importlib.import_module(module_name), function)(payload)


def main() -> int:
    out = sys.stdout
    sys.stdout = sys.stderr  # whatever the target prints stays out of the result
    try:
        if os.environ.get("QS_PROCESS_ROLE") != JOB_ROLE:
            raise PermissionError(f"an AI job process runs as QS_PROCESS_ROLE={JOB_ROLE}")
        job = json.load(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
        result = run_target(job["target"], job.get("payload") or {})
        text = json.dumps(result, default=str)
    except Exception as exc:  # noqa: BLE001 - reported to the worker as one line
        print(f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500], file=sys.stderr)
        return 1
    out.write(text)
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
