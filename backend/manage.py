"""Operator provisioning CLI.

    python -m backend.manage set-role <email> <user|risk_admin|admin>

Roles are granted only here — out of band, by someone with shell access to
the deployment — never through the web API, so no self-registered account
can acquire operator privileges.
"""
import argparse
import sys

from . import models
from .database import SessionLocal, init_db

ROLES = ("user", "risk_admin", "admin")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.manage")
    commands = parser.add_subparsers(dest="command", required=True)
    set_role = commands.add_parser("set-role", help="grant or revoke an operator role")
    set_role.add_argument("email")
    set_role.add_argument("role", choices=ROLES)
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
