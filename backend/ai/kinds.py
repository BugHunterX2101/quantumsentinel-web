"""The kinds of AI job and the code each runs.

A kind's target is ``module:function`` under backend.ai. The function runs
in a job process (backend/ai/child.py) with no database connection and no
secret; it takes the job's payload and returns the job's JSON result. The
job process refuses any target not listed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import flags


@dataclass(frozen=True)
class AiKind:
    name: str
    target: str
    stage: str
    timeout_seconds: float
    max_attempts: int = 1


KINDS: dict[str, AiKind] = {
    # Reports the job process's environment and loaded modules, so an
    # operator can see what an AI job runs with (manage ai-selftest).
    "ai_selftest": AiKind("ai_selftest", "backend.ai.selftest:run", "selftest", timeout_seconds=60),
}
TARGETS = frozenset(kind.target for kind in KINDS.values())

if not all(kind.target.startswith("backend.ai.") and kind.stage in flags.STAGES for kind in KINDS.values()):
    raise RuntimeError("every AI kind runs code under backend.ai and belongs to a known stage")
