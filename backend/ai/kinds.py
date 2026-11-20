"""The kinds of AI job and the code each runs.

A kind's target is ``module:function`` under backend.ai. The function runs
in a job process (backend/ai/child.py) with no database connection and no
secret; it takes a payload and returns JSON. The job process refuses any
target not listed here.

A kind without a driver runs its target once, with the job's payload, and
the target's result becomes the job's result. A kind with a driver
(``module:function`` under backend.ai) is run by that function inside the
AI worker instead: it may run the target many times, one job process each,
and checks and records what they return (backend/ai/harvest.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import flags


@dataclass(frozen=True)
class AiKind:
    name: str
    target: str
    stage: str
    # The whole job's deadline.
    timeout_seconds: float
    max_attempts: int = 1
    driver: str | None = None
    # Each job process's deadline (and CPU limit); the job's deadline if None.
    child_timeout_seconds: float | None = None
    # The most a job process may print; AI_JOB_RESULT_MAX_BYTES if None.
    child_output_max_bytes: int | None = None


KINDS: dict[str, AiKind] = {
    # Reports the job process's environment and loaded modules, so an
    # operator can see what an AI job runs with (manage ai-selftest).
    "ai_selftest": AiKind("ai_selftest", "backend.ai.selftest:run", "selftest", timeout_seconds=60),
    # Harvests one arXiv set's metadata into the document registry
    # (manage ai-arxiv-sync). Each job process fetches and parses one
    # OAI-PMH page; a full page parses to about 2.5 MB of JSON.
    "arxiv_sync": AiKind("arxiv_sync", "backend.ai.oai_fetch:fetch_page", "arxiv_sync",
                         timeout_seconds=6 * 3600, max_attempts=3, driver="backend.ai.harvest:run",
                         child_timeout_seconds=180, child_output_max_bytes=32 * 1024 * 1024),
}
TARGETS = frozenset(kind.target for kind in KINDS.values())


def check(kinds: dict[str, AiKind]) -> None:
    """Raise RuntimeError unless every kind runs code under backend.ai and
    belongs to a known stage."""
    if not all(kind.target.startswith("backend.ai.") and kind.stage in flags.STAGES
               and (kind.driver is None or kind.driver.startswith("backend.ai."))
               for kind in kinds.values()):
        raise RuntimeError("every AI kind runs code under backend.ai and belongs to a known stage")


check(KINDS)
