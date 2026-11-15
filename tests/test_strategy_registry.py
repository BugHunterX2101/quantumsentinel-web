"""Strategy versions on PostgreSQL (services/strategy_registry.py, migration
0006): one row per spec content, tied to its id by CHECKs, never changed,
held and tagged per user."""
import hashlib
import json

import pytest
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from backend import models
from backend.services import strategy_registry as sr
from backend.services import strategy_spec as ss
from backend.services.strategy_registry import StrategyRegistryError
from test_strategy_spec import spec


@pytest.fixture
def engine(make_engine):
    engine = make_engine(migrated=True)
    with engine.begin() as conn:
        for uid in ("u1", "u2"):
            conn.execute(text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, 'h')"),
                         {"id": uid, "email": f"{uid}@example.com"})
    return engine


@pytest.fixture
def db(engine):
    with Session(engine) as session:
        yield session


def register(db, value=None, user_id="u1"):
    version = sr.register(db, value if value is not None else spec(), user_id=user_id)
    db.commit()
    return version


def count(db, table):
    return db.execute(text(f"SELECT count(*) FROM {table}")).scalar()


# ── registering ───────────────────────────────────────────────────────────────

def test_a_spec_is_stored_as_its_canonical_bytes_under_its_id(db):
    parsed = ss.parse(spec())
    version = register(db)
    assert version.version_id == ss.version_id(parsed)
    assert bytes(version.spec_canonical) == ss.canonical(parsed)
    assert (version.dsl, version.template, version.family) == ("qs-strategy-v1", "ma_crossover", "EQUITY_EOD")
    assert sr.spec_of(version) == parsed
    assert db.get(models.UserStrategyVersion, ("u1", version.version_id)) is not None


def test_one_spec_written_twice_is_one_version_held_by_each_writer(db):
    first = register(db, spec(**{"universe.symbols": ["MSFT", "aapl", "GOOGL"]}), user_id="u1")
    created = first.created_at
    again = register(db, spec(), user_id="u2")  # the same spec once canonical
    register(db, spec(), user_id="u2")
    db.expire_all()
    assert again.version_id == first.version_id
    assert count(db, "strategy_versions") == 1
    assert db.get(models.StrategyVersion, first.version_id).created_at == created
    holders = db.execute(select(models.UserStrategyVersion.user_id).order_by("user_id")).scalars().all()
    assert holders == ["u1", "u2"]


def test_a_spec_that_is_not_dsl_v1_is_refused_and_nothing_is_written(db):
    with pytest.raises(ValidationError):
        sr.register(db, spec(**{"signal.slow": 10}), user_id="u1")
    db.rollback()
    assert count(db, "strategy_versions") == count(db, "user_strategy_versions") == 0


def test_versions_are_listed_and_found_only_for_their_holders(db):
    a = register(db, spec(), user_id="u1")
    b = register(db, spec(**{"signal.fast": 10}), user_id="u1")
    register(db, spec(**{"signal.fast": 10}), user_id="u2")
    assert [(v.version_id, tags) for v, tags in sr.list_for_user(db, "u1")] == \
        [(a.version_id, []), (b.version_id, [])]
    assert [v.version_id for v, _ in sr.list_for_user(db, "u2")] == [b.version_id]
    assert sr.get_for_user(db, "u2", b.version_id).version_id == b.version_id
    assert sr.get_for_user(db, "u2", a.version_id) is None
    assert sr.resolve(db, "u2", a.version_id) is None


# ── tags ──────────────────────────────────────────────────────────────────────

def test_a_tag_names_one_of_its_users_versions_forever(db):
    a = register(db)
    b = register(db, spec(**{"signal.fast": 10}))
    sr.tag(db, "u1", a.version_id, "1.0.0")
    sr.tag(db, "u1", a.version_id, "baseline")
    sr.tag(db, "u1", b.version_id, "1.1.0")
    db.commit()
    assert sr.tag(db, "u1", a.version_id, "1.0.0").version_id == a.version_id  # again: a no-op
    with pytest.raises(StrategyRegistryError, match="'1.0.0' already names another version"):
        sr.tag(db, "u1", b.version_id, "1.0.0")
    db.rollback()
    assert [(v.version_id, tags) for v, tags in sr.list_for_user(db, "u1")] == \
        [(a.version_id, ["1.0.0", "baseline"]), (b.version_id, ["1.1.0"])]
    assert sr.resolve(db, "u1", "1.0.0").version_id == a.version_id
    assert sr.resolve(db, "u1", "1.1.0").version_id == b.version_id
    assert sr.resolve(db, "u1", "2.0.0") is None


def test_tags_belong_to_their_user(db):
    a = register(db, user_id="u1")
    register(db, user_id="u2")
    sr.tag(db, "u1", a.version_id, "1.0.0")
    sr.tag(db, "u2", a.version_id, "mine")
    db.commit()
    # Each sees only their own tags on the shared version, so a listing
    # never shows that someone else holds it.
    assert [(v.version_id, tags) for v, tags in sr.list_for_user(db, "u1")] == [(a.version_id, ["1.0.0"])]
    assert [(v.version_id, tags) for v, tags in sr.list_for_user(db, "u2")] == [(a.version_id, ["mine"])]
    assert sr.resolve(db, "u2", "1.0.0") is None
    assert sr.resolve(db, "u1", "mine") is None
    # The same label is free for another user, even for another version.
    b = register(db, spec(**{"signal.fast": 10}), user_id="u2")
    sr.tag(db, "u2", b.version_id, "1.0.0")
    db.commit()
    assert sr.resolve(db, "u2", "1.0.0").version_id == b.version_id
    assert sr.resolve(db, "u1", "1.0.0").version_id == a.version_id


@pytest.mark.parametrize("label", ["", ".hidden", "-x", "has space", "a" * 65, "é", None, 1])
def test_what_a_tag_cannot_be(db, label):
    version = register(db)
    with pytest.raises(StrategyRegistryError, match="a tag is 1-64"):
        sr.tag(db, "u1", version.version_id, label)
    assert count(db, "user_strategy_tags") == 0


def test_a_version_the_user_does_not_hold_cannot_be_tagged(db):
    version = register(db, user_id="u1")
    with pytest.raises(StrategyRegistryError, match="no such strategy version"):
        sr.tag(db, "u2", version.version_id, "stolen")
    with pytest.raises(StrategyRegistryError, match="no such strategy version"):
        sr.tag(db, "u1", "qs-sv-sha256-v1:" + "0" * 64, "nothing")


def test_a_label_another_request_gave_first_is_refused_or_returned(engine, db):
    """tag() read no row for the label, then a concurrent request committed
    one: the insert's conflict rolls back only its savepoint, and tag() says
    what the label now names."""
    a = register(db)
    b = register(db, spec(**{"signal.fast": 10}))
    with Session(engine) as other:
        sr.tag(other, "u1", a.version_id, "1.0.0")
        other.commit()
    real_get = db.get
    stale = {"left": 1}

    def get(entity, key, **kwargs):
        if entity is models.UserStrategyTag and stale["left"]:
            stale["left"] -= 1
            return None
        return real_get(entity, key, **kwargs)

    db.get = get
    with pytest.raises(StrategyRegistryError, match="already names another version"):
        sr.tag(db, "u1", b.version_id, "1.0.0")
    # A real stale read means the row is not in this session yet.
    db.expunge(db.get(models.UserStrategyTag, ("u1", "1.0.0")))
    stale["left"] = 1
    assert sr.tag(db, "u1", a.version_id, "1.0.0").version_id == a.version_id
    # The caller's transaction survived both conflicts.
    sr.tag(db, "u1", b.version_id, "1.1.0")
    db.commit()
    assert sorted(db.execute(text("SELECT label FROM user_strategy_tags")).scalars()) == ["1.0.0", "1.1.0"]


# ── what the database refuses ─────────────────────────────────────────────────

def _insert_version(db, raw: bytes, **changes):
    parsed = json.loads(raw)
    row = {"version_id": "qs-sv-sha256-v1:" + hashlib.sha256(raw).hexdigest(), "dsl": parsed["dsl"],
           "template": parsed["signal"]["template"], "family": parsed["universe"]["family"], "raw": raw}
    row.update(changes)
    db.execute(text("INSERT INTO strategy_versions (version_id, dsl, template, family, spec_canonical, created_at)"
                    " VALUES (:version_id, :dsl, :template, :family, :raw, now())"), row)


def _refused(db, raw, **changes):
    with pytest.raises(IntegrityError) as caught:
        _insert_version(db, raw, **changes)
    db.rollback()
    return caught.value.orig.diag.constraint_name


CANONICAL = ss.canonical(ss.parse(spec()))


def test_an_id_that_is_not_the_spec_hash_is_refused(db):
    assert _refused(db, CANONICAL, version_id="qs-sv-sha256-v1:" + "0" * 64) == "ck_strategy_versions_id_is_spec_hash"
    assert _refused(db, CANONICAL, version_id="qs-sv-sha256-v2:" + hashlib.sha256(CANONICAL).hexdigest()) \
        == "ck_strategy_versions_id_is_spec_hash"


@pytest.mark.parametrize("column, forged", [("dsl", "qs-strategy-v2"), ("template", "momentum"),
                                            ("family", "CRYPTO_SPOT_EOD")])
def test_columns_that_disagree_with_the_spec_are_refused(db, column, forged):
    assert _refused(db, CANONICAL, **{column: forged}) == "ck_strategy_versions_columns_are_spec"


def test_a_tag_of_a_version_its_user_does_not_hold_is_refused(db):
    version = register(db, user_id="u1")
    with pytest.raises(IntegrityError) as caught:
        db.execute(text("INSERT INTO user_strategy_tags (user_id, label, version_id, created_at)"
                        " VALUES ('u2', 'x', :v, now())"), {"v": version.version_id})
    assert caught.value.orig.diag.constraint_name == "fk_user_strategy_tags_held_version"
    db.rollback()


def test_a_label_must_look_like_one(db):
    version = register(db)
    with pytest.raises(IntegrityError) as caught:
        db.execute(text("INSERT INTO user_strategy_tags (user_id, label, version_id, created_at)"
                        " VALUES ('u1', ' x', :v, now())"), {"v": version.version_id})
    assert caught.value.orig.diag.constraint_name == "ck_user_strategy_tags_label"
    db.rollback()


@pytest.mark.parametrize("statement", [
    "UPDATE strategy_versions SET created_at = now()",
    "UPDATE strategy_versions SET family = family",
    "DELETE FROM strategy_versions",
    "TRUNCATE strategy_versions CASCADE",
    "UPDATE user_strategy_versions SET created_at = now()",
    "DELETE FROM user_strategy_versions",
    "TRUNCATE user_strategy_versions CASCADE",
    "UPDATE user_strategy_tags SET label = 'moved'",
    "UPDATE user_strategy_tags SET version_id = version_id",
    "DELETE FROM user_strategy_tags",
    "TRUNCATE user_strategy_tags",
])
def test_recorded_rows_never_change_even_for_the_owner(db, statement):
    version = register(db)
    sr.tag(db, "u1", version.version_id, "1.0.0")
    db.commit()
    with pytest.raises(DBAPIError, match="immutable") as caught:
        db.execute(text(statement))
    assert caught.value.orig.sqlstate == "23001"  # restrict_violation
    db.rollback()
    assert [count(db, t) for t in ("strategy_versions", "user_strategy_versions", "user_strategy_tags")] == [1, 1, 1]


# ── reading a version back ────────────────────────────────────────────────────

def test_a_stored_spec_todays_dsl_refuses_is_not_returned(db):
    """The database checks only the id and the columns: a spec an earlier
    release accepted and today's validators refuse is refused on reading."""
    value = spec(**{"signal.slow": 10})  # slow must exceed fast
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    _insert_version(db, raw)
    db.commit()
    version = db.execute(select(models.StrategyVersion)).scalar_one()
    with pytest.raises(StrategyRegistryError, match="no longer validates"):
        sr.spec_of(version)


def test_stored_bytes_that_are_not_canonical_are_not_returned(db):
    raw = json.dumps(spec(), indent=2).encode()  # the same spec, not canonically encoded
    _insert_version(db, raw)
    db.commit()
    version = db.execute(select(models.StrategyVersion)).scalar_one()
    with pytest.raises(StrategyRegistryError, match="no longer encodes to its id"):
        sr.spec_of(version)
