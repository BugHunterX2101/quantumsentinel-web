"""Tasks for exercising the real job process (imported in the child process),
and helpers that run a queued research job to completion in a test."""
import json
import os
import time

from backend.services.research_tasks import TaskOutput


class InlineJobProcess:
    """Runs tasks in this process (so monkeypatches apply); no time limit."""

    def run(self, target, params, timeout, on_tick, tick_seconds=1.0):
        from backend import worker
        return worker._run_target(target, params)

    def stop(self):
        pass


def run_queued(db, response, job_process=None) -> dict:
    """Run the job a research endpoint queued on a worker; return its final view."""
    from sqlalchemy.orm import sessionmaker

    from backend import models, worker
    from backend.services import research_jobs
    assert response.status_code == 202
    job_id = json.loads(response.body)["job_id"]
    runner = worker.Worker(job_process=job_process or InlineJobProcess(),
                           session_factory=sessionmaker(bind=db.get_bind()),
                           lease_seconds=60, timeout_seconds=300, poll_seconds=0.1)
    assert runner.run_once() is True
    db.expire_all()
    return research_jobs.view(db, db.get(models.ResearchJob, job_id))


def result_of(db, response, job_process=None) -> dict:
    """The result of the job a research endpoint queued; fails if it did not succeed."""
    view = run_queued(db, response, job_process)
    assert view["status"] == "succeeded", view
    return view["result"]


def echo_task(params: dict) -> TaskOutput:
    return TaskOutput({"echo": params, "pid": os.getpid()}, {"echoed": True})


def sleep_task(params: dict) -> TaskOutput:
    time.sleep(params.get("seconds", 60))
    return TaskOutput({"slept": True}, {})


def crash_task(params: dict) -> TaskOutput:
    os._exit(3)
