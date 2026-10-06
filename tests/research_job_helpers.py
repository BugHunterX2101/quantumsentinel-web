"""Tasks for exercising the real job process (imported in the child process)."""
import os
import time

from backend.services.research_tasks import TaskOutput


def echo_task(params: dict) -> TaskOutput:
    return TaskOutput({"echo": params, "pid": os.getpid()}, {"echoed": True})


def sleep_task(params: dict) -> TaskOutput:
    time.sleep(params.get("seconds", 60))
    return TaskOutput({"slept": True}, {})


def crash_task(params: dict) -> TaskOutput:
    os._exit(3)
