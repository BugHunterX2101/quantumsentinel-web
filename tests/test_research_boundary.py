"""Research code cannot reach trading or identity code.

A research job runs code on users' inputs (datasets, parameters, strategies)
in a job process; its worker records the outcome as qs_research_worker. The
code either of them can import is the closure of every import statement,
module-level or inside a function, starting from the worker and from the
job targets. None of it may be the API, the operator CLI, authentication,
trading, the paper broker, portfolios, integrations or Redis, so research
code can never call them, whatever input it is given.
"""
import ast
import importlib
from pathlib import Path

import pytest

from backend.services import research_jobs

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {
    "backend.main", "backend.manage", "backend.crypto.message_protocol",
    "backend.services.auth_service", "backend.services.trading_service", "backend.services.paper_broker",
    "backend.services.portfolio_service", "backend.services.order_security",
    "backend.services.integration_service", "backend.services.redis_store",
}
RESEARCH_TASKS = "backend.services.research_tasks"


def _path(module: str) -> Path | None:
    base = ROOT / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _imports(module: str) -> set[str]:
    """Backend modules ``module`` imports anywhere in its source."""
    path = _path(module)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module
            found.add(target)
            # `from package import name` imports the submodule when there is one.
            found |= {f"{target}.{alias.name}" for alias in node.names if _path(f"{target}.{alias.name}")}
    return {name for name in found if name.split(".")[0] == "backend" and _path(name)}


def closure(*roots: str) -> dict[str, str | None]:
    """Every module reachable from ``roots``, each with the module that first imports it."""
    reached, todo = {root: None for root in roots}, list(roots)
    while todo:
        module = todo.pop()
        for imported in _imports(module):
            if imported not in reached:
                reached[imported] = module
                todo.append(imported)
    return reached


def _chain(reached: dict, module: str) -> str:
    chain = [module]
    while reached[chain[-1]] is not None:
        chain.append(reached[chain[-1]])
    return " <- ".join(chain)


def test_every_job_kind_runs_research_code():
    for name, kind in research_jobs.KINDS.items():
        module, _, attr = kind.target.partition(":")
        assert module == RESEARCH_TASKS, name
        assert getattr(importlib.import_module(module), attr).__module__ == RESEARCH_TASKS, name
        if kind.finalize is not None:
            assert kind.finalize.startswith("backend.services.experiment_registry:"), name


@pytest.mark.parametrize("root", ["backend.worker", RESEARCH_TASKS])
def test_research_code_cannot_reach_trading_or_identity(root):
    reached = closure(root)
    assert {_chain(reached, module) for module in FORBIDDEN & set(reached)} == set()


def test_the_check_sees_every_kind_of_import():
    """Without these, the test above could pass by seeing nothing."""
    assert all(_path(module) for module in FORBIDDEN)
    assert "backend.services.trading_service" in _imports("backend.main")       # from .services import x
    assert "backend.services.walk_forward" in _imports(RESEARCH_TASKS)          # inside a function
    assert "backend.services.research_trials" in _imports(RESEARCH_TASKS)       # from . import x
    assert "backend.services.security_service" in closure("backend.worker")     # transitively
