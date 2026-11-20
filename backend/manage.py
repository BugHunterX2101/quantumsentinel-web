"""Operator CLI.

    python -m backend.manage set-role <email> <user|risk_admin|admin>
    python -m backend.manage migrate
    python -m backend.manage import-sqlite <path/to/quantumsentinel.db>
    python -m backend.manage generate-server-key
    python -m backend.manage repair-positions
    python -m backend.manage verify-audit-chain
    python -m backend.manage provision-roles
    python -m backend.manage set-holdout <FAMILY> <YYYY-MM-DD> --by <who> --reason <why>
    python -m backend.manage holdouts
    python -m backend.manage final-evaluate <VALIDATION_JOB_ID> --period <P> --capital <C> --by <who> --reason <why>
    python -m backend.manage set-flag <FLAG> <on|off> --by <who> --reason <why>
    python -m backend.manage flags
    python -m backend.manage ai-selftest --by <who> [--hold <seconds>]
    python -m backend.manage ai-arxiv-sync <SET> --by <who> [--from <YYYY-MM-DD>] [--max-pages <N>]
    python -m backend.manage ai-metadata-sync <openalex|crossref> --by <who> [--max-docs <N>]
    python -m backend.manage ai-job <AI_JOB_ID>

Roles are granted only here — out of band, by someone with shell access to
the deployment — never through the web API, so no self-registered account
can acquire operator privileges.

``migrate`` applies pending schema migrations (the API also does this at
startup unless DB_MIGRATE_ON_STARTUP=false). ``import-sqlite`` moves the data
of a deployment that ran on SQLite (releases before 1.3) into the configured
PostgreSQL database. ``generate-server-key`` prints a new server ML-DSA
signing key as the SERVER_DSA_* settings (run it before those are set: a
production configuration refuses to load without them). ``repair-positions``
rebuilds the positions of users left with duplicate position rows by the
concurrent-fill race that earlier releases had. ``verify-audit-chain`` checks
every link of the audit chain and exits non-zero if any fails; schedule it,
because the API's default check covers only links added since the worker's
previous check.

``provision-roles`` creates the least-privilege database roles
(backend/db_roles.py) and hands the schema to them. Run it as the database
superuser, with DATABASE_URL naming that superuser. A role that does not
exist yet takes its password from DB_MIGRATOR_PASSWORD, DB_APP_PASSWORD,
DB_RESEARCH_WORKER_PASSWORD, DB_EVALUATOR_PASSWORD, DB_AI_WORKER_PASSWORD or
DB_BACKUP_PASSWORD (or the matching *_FILE setting); given for a role that
exists, the password is changed. It is safe to run again: it repairs any
drift from the intended privileges. It is safe while the application runs:
it takes the table locks it needs all at once or not at all, so it cannot
deadlock with a request, and if tables stay in use for 10 s it exits 1
without changing anything.

``set-holdout`` holds back a dataset family's data from a date on: research
jobs see only what came before it (backend/services/dataset_holdouts.py). The
date can only move later; moving it releases the old holdout to research.
``holdouts`` prints each family's current start.

``final-evaluate`` queues the one-shot holdout evaluation of a passing
strategy validation, named by the job that recorded it, with the period and
capital its evaluation ran with (backend/services/holdout_evaluation.py). The
research evaluator runs it and spends the attempt as it starts. The command
prints the job id, and the owner sees only whether the grade passed.

``set-flag`` switches a feature flag of the AI tier (backend/ai/flags.py)
on or off. A stage of the AI tier runs only while ``research_intelligence``
and its own flag are both on, and every flag starts off. The change is
audited and takes effect without a restart: the AI worker checks the flags
when it claims a job and while one runs. ``flags`` prints them.
``ai-selftest`` queues a job that reports what an AI job process runs with:
the names of its environment variables, its process role, and any module of
the trusted tier it imported (there must be none). ``ai-arxiv-sync`` queues
a harvest of one arXiv set's metadata into the document registry
(backend/ai/harvest.py), from where the last sync of that set stopped;
``--from`` gives the first day for a set never harvested before. At most one
sync is queued or running at a time; schedule it daily.
``ai-metadata-sync`` queues lookups of the registry's documents at OpenAlex
or Crossref (backend/ai/enrich.py): every document never looked up there,
changed since, or due for a recheck. At most one is queued or running per
service; schedule it daily, after the arXiv sync. ``ai-job`` prints an AI
job's state and result.
"""
import argparse
import sys
from pathlib import Path

from . import models
from .database import SessionLocal, init_db

ROLES = ("user", "risk_admin", "admin")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.manage")
    commands = parser.add_subparsers(dest="command", required=True)
    set_role = commands.add_parser("set-role", help="grant or revoke an operator role")
    set_role.add_argument("email")
    set_role.add_argument("role", choices=ROLES)
    commands.add_parser("migrate", help="apply pending database schema migrations")
    import_sqlite = commands.add_parser(
        "import-sqlite", help="copy every row of a SQLite database into the (empty) PostgreSQL database")
    import_sqlite.add_argument("path", type=Path)
    commands.add_parser("generate-server-key",
                        help="print a new ML-DSA-65 server signing key as SERVER_DSA_* settings")
    commands.add_parser("repair-positions",
                        help="rebuild the positions of users left with duplicate position rows")
    commands.add_parser("verify-audit-chain",
                        help="verify every link of the audit chain; exit 1 if any fails")
    commands.add_parser("provision-roles",
                        help="create the least-privilege database roles and hand them the schema "
                             "(run as the database superuser)")
    set_holdout = commands.add_parser(
        "set-holdout", help="hold back a dataset family's data from a date on (the date can only move later)")
    set_holdout.add_argument("family")
    set_holdout.add_argument("start", type=_date, help="first held-back session, YYYY-MM-DD")
    set_holdout.add_argument("--by", required=True, help="who sets it")
    set_holdout.add_argument("--reason", required=True, help="why")
    commands.add_parser("holdouts", help="print each dataset family's holdout start")
    final_evaluate = commands.add_parser(
        "final-evaluate", help="queue the one-shot holdout evaluation of a passing strategy validation")
    final_evaluate.add_argument("validation", help="the job id that recorded the passing verdict")
    final_evaluate.add_argument("--period", required=True, help="the period its evaluation ran with, e.g. 2y")
    final_evaluate.add_argument("--capital", required=True, type=float,
                                help="the initial capital its evaluation ran with")
    final_evaluate.add_argument("--by", required=True, help="who queues it")
    final_evaluate.add_argument("--reason", required=True, help="why")
    set_flag = commands.add_parser("set-flag", help="switch a feature flag of the AI tier on or off")
    set_flag.add_argument("flag")
    set_flag.add_argument("state", choices=("on", "off"))
    set_flag.add_argument("--by", required=True, help="who sets it")
    set_flag.add_argument("--reason", required=True, help="why")
    commands.add_parser("flags", help="print the AI tier's feature flags")
    ai_selftest = commands.add_parser("ai-selftest", help="queue an AI job that reports what AI jobs run with")
    ai_selftest.add_argument("--by", required=True, help="who queues it")
    ai_selftest.add_argument("--hold", type=float, default=0, help="seconds the job keeps running (at most 30)")
    arxiv_sync = commands.add_parser("ai-arxiv-sync", help="queue a harvest of an arXiv set's metadata")
    arxiv_sync.add_argument("set", help="an arXiv set spec, such as q-fin")
    arxiv_sync.add_argument("--by", required=True, help="who queues it")
    arxiv_sync.add_argument("--from", dest="start", help="first day, for a set never harvested (YYYY-MM-DD)")
    arxiv_sync.add_argument("--max-pages", type=int, help="stop after this many pages")
    metadata_sync = commands.add_parser("ai-metadata-sync",
                                        help="queue lookups of the registry's documents at OpenAlex or Crossref")
    metadata_sync.add_argument("source", help="openalex or crossref")
    metadata_sync.add_argument("--by", required=True, help="who queues it")
    metadata_sync.add_argument("--max-docs", type=int, help="stop after this many documents")
    ai_job = commands.add_parser("ai-job", help="print an AI job's state and result")
    ai_job.add_argument("job")
    args = parser.parse_args(argv)

    if args.command == "generate-server-key":
        return _generate_server_key()
    if args.command == "provision-roles":
        return _provision_roles()
    if args.command == "repair-positions":
        init_db()
        return _repair_positions(SessionLocal)
    if args.command == "verify-audit-chain":
        init_db()
        return _verify_audit_chain(SessionLocal)
    if args.command == "migrate":
        from .database import migrate
        print(f"database schema at revision {migrate()}")
        return 0
    if args.command == "import-sqlite":
        return _import_sqlite(args.path)
    if args.command == "set-holdout":
        init_db()
        return _set_holdout(SessionLocal, args.family, args.start, args.by, args.reason)
    if args.command == "holdouts":
        init_db()
        return _print_holdouts(SessionLocal)
    if args.command == "final-evaluate":
        init_db()
        return _final_evaluate(SessionLocal, args.validation, args.period, args.capital, args.by, args.reason)
    if args.command == "set-flag":
        init_db()
        return _set_flag(SessionLocal, args.flag, args.state == "on", args.by, args.reason)
    if args.command == "flags":
        init_db()
        return _print_flags(SessionLocal)
    if args.command == "ai-selftest":
        init_db()
        return _ai_selftest(SessionLocal, args.by, args.hold)
    if args.command == "ai-arxiv-sync":
        init_db()
        return _ai_arxiv_sync(SessionLocal, args.set, args.start, args.max_pages, args.by)
    if args.command == "ai-metadata-sync":
        init_db()
        return _ai_metadata_sync(SessionLocal, args.source, args.max_docs, args.by)
    if args.command == "ai-job":
        init_db()
        return _print_ai_job(SessionLocal, args.job)

    init_db()
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == args.email.strip().lower()).first()
        if user is None:
            print(f"no user with email {args.email}", file=sys.stderr)
            return 1
        old_role = user.role or "user"
        user.role = args.role
        db.commit()
        from .services import security_service
        security_service.write_audit_log(db, None, "ROLE_CHANGED", "user", user.id,
                                         {"email": user.email, "old_role": old_role, "new_role": args.role})
        print(f"{user.email}: {old_role} -> {args.role}")
        return 0
    finally:
        db.close()


def _date(text: str):
    import datetime as dt
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date (YYYY-MM-DD): {text!r}") from None


def _set_holdout(session_factory, family: str, start, set_by: str, reason: str) -> int:
    from .services import dataset_holdouts

    with session_factory() as db:
        try:
            row = dataset_holdouts.set_start(db, family, start, set_by=set_by, reason=reason)
        except dataset_holdouts.HoldoutError as exc:
            print(f"set-holdout: {exc}", file=sys.stderr)
            return 1
        print(f"{row.family}: research sees data before {row.holdout_start.isoformat()}")
    return 0


def _final_evaluate(session_factory, validation: str, period: str, capital: float, queued_by: str,
                    reason: str) -> int:
    from .services import holdout_evaluation, research_jobs, security_service

    queued_by, reason = (queued_by or "").strip(), (reason or "").strip()
    if not queued_by or not reason:
        print("final-evaluate: say who queues it and why", file=sys.stderr)
        return 1
    with session_factory() as db:
        try:
            job = holdout_evaluation.queue(db, validation, period, capital)
        except holdout_evaluation.HoldoutEvaluationError as exc:
            print(f"final-evaluate: {exc}", file=sys.stderr)
            return 1
        except research_jobs.QueueFull as exc:
            print(f"final-evaluate: the owner has too many research jobs queued or running: {exc}",
                  file=sys.stderr)
            return 1
        security_service.write_audit_log(db, job.user_id, "HOLDOUT_EVALUATION_QUEUED", "research_job", job.id,
                                         {"validation": validation, "queued_by": queued_by[:320],
                                          "reason": reason})
        print(f"holdout evaluation queued: job {job.id}")
    return 0


def _ai_connection(db):
    """The session's psycopg connection, inside its transaction, and the AI schema's name."""
    from . import db_roles

    conn = db.connection().connection.driver_connection
    return conn, db_roles.ai_schema_for(conn.execute("SELECT current_schema()").fetchone()[0])


_ON_OFF = {True: "on", False: "off"}


def _set_flag(session_factory, name: str, on: bool, set_by: str, reason: str) -> int:
    from .ai import flags
    from .services import security_service

    with session_factory() as db:
        conn, schema = _ai_connection(db)
        try:
            previous = flags.set_flag(conn, schema, name, on, set_by, reason)
        except flags.FlagError as exc:
            print(f"set-flag: {exc}", file=sys.stderr)
            return 1
        db.commit()
        security_service.write_audit_log(db, None, "AI_FLAG_SET", "ai_flag", name,
                                         {"enabled": on, "previous": previous, "set_by": set_by.strip()[:320],
                                          "reason": reason.strip()})
        print(f"{name}: {_ON_OFF[on]} (was {_ON_OFF[previous]})")
    return 0


def _print_flags(session_factory) -> int:
    from .ai import flags

    with session_factory() as db:
        conn, schema = _ai_connection(db)
        rows = flags.listing(conn, schema)
    for row in rows:
        effective = "" if row["name"] == flags.MASTER else f" (effective: {_ON_OFF[row['effective']]})"
        if row["set_at"]:
            detail = f"  set {row['set_at']:%Y-%m-%d %H:%M} by {row['set_by']}: {row['reason']}"
        else:
            detail = "  never set"
        print(f"{row['name']:24s} {_ON_OFF[row['enabled']]}{effective}{detail}")
    return 0


def _ai_selftest(session_factory, queued_by: str, hold: float) -> int:
    from .ai import queue
    from .services import security_service

    with session_factory() as db:
        conn, schema = _ai_connection(db)
        try:
            job_id = queue.enqueue(conn, schema, "ai_selftest", {"hold_seconds": hold}, queued_by)
        except queue.EnqueueError as exc:
            print(f"ai-selftest: {exc}", file=sys.stderr)
            return 1
        db.commit()
        security_service.write_audit_log(db, None, "AI_JOB_QUEUED", "ai_job", job_id,
                                         {"kind": "ai_selftest", "queued_by": queued_by.strip()[:320]})
        print(f"AI job queued: {job_id}")
    return 0


def _ai_arxiv_sync(session_factory, set_spec: str, start: str | None, max_pages: int | None,
                   queued_by: str) -> int:
    from .ai import harvest, queue
    from .services import security_service

    payload = {"set": set_spec}
    if start is not None:
        payload["from"] = start
    if max_pages is not None:
        payload["max_pages"] = max_pages
    try:
        harvest.check_payload(payload)
    except ValueError as exc:
        print(f"ai-arxiv-sync: {exc}", file=sys.stderr)
        return 1
    with session_factory() as db:
        conn, schema = _ai_connection(db)
        try:
            job_id = queue.enqueue(conn, schema, "arxiv_sync", payload, queued_by)
        except queue.EnqueueError as exc:
            print(f"ai-arxiv-sync: {exc}", file=sys.stderr)
            return 1
        db.commit()
        security_service.write_audit_log(db, None, "AI_JOB_QUEUED", "ai_job", job_id,
                                         {"kind": "arxiv_sync", "queued_by": queued_by.strip()[:320],
                                          **payload})
        print(f"AI job queued: {job_id}")
    return 0


def _ai_metadata_sync(session_factory, source: str, max_docs: int | None, queued_by: str) -> int:
    from .ai import enrich, queue
    from .services import security_service

    payload = {"source": source}
    if max_docs is not None:
        payload["max_docs"] = max_docs
    try:
        enrich.check_payload(payload)
    except ValueError as exc:
        print(f"ai-metadata-sync: {exc}", file=sys.stderr)
        return 1
    with session_factory() as db:
        conn, schema = _ai_connection(db)
        try:
            job_id = queue.enqueue(conn, schema, "metadata_sync", payload, queued_by)
        except queue.EnqueueError as exc:
            print(f"ai-metadata-sync: {exc}", file=sys.stderr)
            return 1
        db.commit()
        security_service.write_audit_log(db, None, "AI_JOB_QUEUED", "ai_job", job_id,
                                         {"kind": "metadata_sync", "queued_by": queued_by.strip()[:320],
                                          **payload})
        print(f"AI job queued: {job_id}")
    return 0


def _print_ai_job(session_factory, job_id: str) -> int:
    import json

    from .ai import queue

    with session_factory() as db:
        conn, schema = _ai_connection(db)
        job = queue.get(conn, schema, job_id)
    if job is None:
        print(f"ai-job: no AI job {job_id}", file=sys.stderr)
        return 1
    print(json.dumps(job, default=str, indent=2))
    return 0


def _print_holdouts(session_factory) -> int:
    from .services import dataset_holdouts

    with session_factory() as db:
        starts = dataset_holdouts.current(db)
    for family, start in sorted(starts.items()):
        print(f"{family:28s} {start.isoformat()}")
    if not starts:
        print("no dataset family has a holdout")
    return 0


def _generate_server_key() -> int:
    """Print a new server signing key, ready to paste into the environment.

    The private key is a 32-byte FIPS 204 seed, which signs through native
    ML-DSA. Expanded keys made by earlier releases keep working but sign
    through the much slower dilithium-py. Replacing a key never orphans what
    it signed: every key a process has signed with is kept in
    server_signing_keys, which is what audit verification reads.
    """
    import datetime as dt
    import hashlib

    from .crypto import pqc

    pk, sk, _ = pqc.dsa_keygen()
    print(f"SERVER_DSA_PRIVATE_KEY={pqc.b64(sk)}")
    print(f"SERVER_DSA_PUBLIC_KEY={pqc.b64(pk)}")
    print(f"SERVER_DSA_CREATED_AT={dt.datetime.now(dt.timezone.utc).isoformat()}")
    print(f"TRUSTED_SERVER_DSA_FINGERPRINT={hashlib.sha256(pk).hexdigest()}")
    return 0


def _provision_roles(target_engine=None, roles=None) -> int:
    """Bring the schema to the latest revision, then provision the roles.

    Prints what changed, never a password. Exits 1 if the schema has tables
    the allow-list does not name: the application role cannot use them.
    """
    from . import db_roles
    from .config import _setting
    from .database import engine, migrate

    roles = roles or db_roles.ROLES
    target = target_engine or engine
    migrate(target, roles)
    passwords = {role: _setting(name) for role, name in (
        (roles.migrator, "DB_MIGRATOR_PASSWORD"), (roles.app, "DB_APP_PASSWORD"),
        (roles.research_worker, "DB_RESEARCH_WORKER_PASSWORD"), (roles.evaluator, "DB_EVALUATOR_PASSWORD"),
        (roles.ai_worker, "DB_AI_WORKER_PASSWORD"), (roles.backup, "DB_BACKUP_PASSWORD"))}
    try:
        with target.begin() as connection:
            summary = db_roles.provision(connection.connection.driver_connection, roles,
                                         {role: pw for role, pw in passwords.items() if pw})
    except (PermissionError, ValueError, RuntimeError) as exc:
        print(f"provision-roles: {exc}", file=sys.stderr)
        return 1
    print(f"schema {summary['schema']}: roles created: {', '.join(summary['created']) or 'none'}; "
          f"objects given to {roles.owner}: {summary['ownership_moved']}; "
          f"passwords set: {', '.join(role for role, pw in passwords.items() if pw) or 'none'}")
    print(f"connect the application as {roles.app}, a separate research worker as "
          f"{roles.research_worker}, the research evaluator as {roles.evaluator}, the AI worker as "
          f"{roles.ai_worker}, migrations as {roles.migrator}, backups as {roles.backup}")
    if summary["unlisted"]:
        print("tables missing from db_roles.APP_TABLE_PRIVILEGES, which the application cannot use: "
              + ", ".join(summary["unlisted"]), file=sys.stderr)
        return 1
    return 0


def _verify_audit_chain(session_factory) -> int:
    """Check every audit chain link; print the result as JSON."""
    import json

    from .services import security_service

    with session_factory() as db:
        status = security_service.audit_chain_status(db, full=True)
    print(json.dumps(status))
    return 0 if status["valid"] else 1


def _repair_positions(session_factory) -> int:
    """Rebuild the positions of every user with more than one row for an asset.

    Earlier releases rebuilt positions after a fill had committed, without a
    lock, so two fills for one user at the same moment could leave duplicate
    rows. Each user's next fill rebuilds them correctly; this repairs users who
    have not traded since. Each rebuild holds the user's paper account row, as
    a fill does, so it is safe to run while the API is serving orders.
    """
    from sqlalchemy import func, select

    from .services import portfolio_service

    with session_factory() as db:
        user_ids = sorted(set(db.execute(
            select(models.Position.user_id)
            .group_by(models.Position.user_id, models.Position.asset)
            .having(func.count() > 1)
        ).scalars()))
        db.rollback()
        for user_id in user_ids:
            db.execute(select(models.PaperAccount.user_id)
                       .where(models.PaperAccount.user_id == user_id).with_for_update())
            portfolio_service.rebuild_positions(db, user_id)
            db.commit()
    print(f"rebuilt the positions of {len(user_ids)} user(s) with duplicate position rows")
    return 0


def _import_sqlite(path: Path, target_engine=None) -> int:
    """Copy all rows from a SQLite database file, all-or-nothing.

    The target is migrated first and must hold no rows. Every table is copied
    in foreign-key order inside one transaction; the row counts and the
    audit hash chain are verified before it commits, so a failure of any kind
    leaves PostgreSQL exactly as it was.
    """
    from sqlalchemy import create_engine, func, inspect, select
    from sqlalchemy.orm import Session

    from .database import Base, engine, migrate
    from .services import security_service

    if not path.is_file():
        print(f"no such SQLite database: {path}", file=sys.stderr)
        return 1
    target = target_engine or engine
    migrate(target)
    source = create_engine(f"sqlite:///file:{path.resolve().as_posix()}?mode=ro&uri=true")
    try:
        source_tables = set(inspect(source).get_table_names())
        copied: dict[str, int] = {}
        with target.begin() as dst:
            occupied = [t.name for t in Base.metadata.sorted_tables
                        if dst.execute(select(func.count()).select_from(t)).scalar()]
            if occupied:
                print("refusing to import: the PostgreSQL database already has rows in "
                      + ", ".join(occupied), file=sys.stderr)
                return 1
            with source.connect() as src:
                for table in Base.metadata.sorted_tables:
                    if table.name not in source_tables:
                        continue
                    present = {c["name"] for c in inspect(source).get_columns(table.name)}
                    columns = [c for c in table.columns if c.name in present]
                    rows = [dict(r._mapping) for r in src.execute(select(*columns))]
                    for start in range(0, len(rows), 1000):
                        dst.execute(table.insert(), rows[start:start + 1000])
                    stored = dst.execute(select(func.count()).select_from(table)).scalar()
                    if stored != len(rows):
                        raise RuntimeError(f"{table.name}: copied {len(rows)} rows but {stored} are stored")
                    copied[table.name] = stored
            chain = security_service.audit_chain_status(Session(bind=dst))
            if not chain["valid"]:
                raise RuntimeError(f"audit chain does not verify after import: {chain}")
    finally:
        source.dispose()
    for name, count in copied.items():
        print(f"{name:28s} {count}")
    print(f"imported {sum(copied.values())} rows from {path}; audit chain verified "
          f"({chain['links']} links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
