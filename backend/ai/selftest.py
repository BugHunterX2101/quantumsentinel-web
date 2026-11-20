"""The ``ai_selftest`` job: what an AI job process runs with.

It reports the names (never the values) of the environment variables the
process has, its process role, any module of the trusted tier that has been
imported (there must be none), and its resource limits. An operator queues
it with ``manage ai-selftest``. ``hold_seconds`` keeps the job running for a
while, so that switching its stage off mid-run can be seen to stop it.
"""
from __future__ import annotations

import os
import sys
import time

MAX_HOLD_SECONDS = 30


def trusted_modules() -> list[str]:
    return sorted(name for name in sys.modules
                  if name.startswith("backend.") and not (name == "backend.ai" or name.startswith("backend.ai.")))


def _limits() -> dict:
    if os.name != "posix":
        return {}
    import resource

    return {name: resource.getrlimit(getattr(resource, name))[1]
            for name in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_CORE")}


def run(payload: dict) -> dict:
    hold = payload.get("hold_seconds", 0)
    if isinstance(hold, bool) or not isinstance(hold, (int, float)) or not 0 <= hold <= MAX_HOLD_SECONDS:
        raise ValueError(f"hold_seconds must be a number from 0 to {MAX_HOLD_SECONDS}")
    time.sleep(hold)
    return {
        "process_role": os.environ.get("QS_PROCESS_ROLE"),
        "environment": sorted(os.environ),
        "trusted_modules": trusted_modules(),
        "limits": _limits(),
        "pid": os.getpid(),
        "python": sys.version.split()[0],
    }
