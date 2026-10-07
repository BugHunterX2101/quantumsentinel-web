"""Operator CLI.

    python -m backend.manage set-role <email> <user|risk_admin|admin>
    python -m backend.manage migrate
    python -m backend.manage import-sqlite <path/to/quantumsentinel.db>

Roles are granted only here — out of band, by someone with shell access to
the deployment — never through the web API, so no self-registered account
can acquire operator privileges.

``migrate`` applies pending schema migrations (the API also does this at
startup unless DB_MIGRATE_ON_STARTUP=false). ``import-sqlite`` moves the data
of a deployment that ran on SQLite (releases before 1.3) into the configured
PostgreSQL database.
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
    args = parser.parse_args(argv)

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
