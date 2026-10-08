"""Operator CLI.

    python -m backend.manage set-role <email> <user|risk_admin|admin>
    python -m backend.manage migrate
    python -m backend.manage import-sqlite <path/to/quantumsentinel.db>
    python -m backend.manage generate-server-key
    python -m backend.manage repair-positions
    python -m backend.manage verify-audit-chain
    python -m backend.manage provision-roles

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
exist yet takes its password from DB_MIGRATOR_PASSWORD, DB_APP_PASSWORD or
DB_BACKUP_PASSWORD (or the matching *_FILE setting); given for a role that
exists, the password is changed. It is safe to run again: it repairs any
drift from the intended privileges. It is safe while the application runs:
it takes the table locks it needs all at once or not at all, so it cannot
deadlock with a request, and if tables stay in use for 10 s it exits 1
without changing anything.
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
        (roles.backup, "DB_BACKUP_PASSWORD"))}
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
    print(f"connect the application as {roles.app}, migrations as {roles.migrator}, "
          f"backups as {roles.backup}")
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
