"""The strategy API (backend/main.py, /api/research/strategies): register a
spec as a version, list and read the caller's versions, tag them."""
import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from backend import main, models, schemas
from backend.services import strategy_spec as ss
from test_strategy_spec import spec


@pytest.fixture
def db(make_engine):
    session = sessionmaker(bind=make_engine(migrated=True))()
    yield session
    session.close()


def make_user(db, email="researcher@example.com"):
    user = models.User(email=email, password_hash="x", role="user")
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def owner(db):
    return make_user(db)


def register(db, user, value=None, tag=None):
    return main.register_strategy(schemas.StrategyRegisterRequest(spec=value or spec(), tag=tag), user=user, db=db)


def tag(db, user, version_id, label):
    return main.tag_strategy(version_id, schemas.StrategyTagRequest(label=label), user=user, db=db)


def status_of(call) -> int:
    with pytest.raises(HTTPException) as caught:
        call()
    return caught.value.status_code


def count(db, table):
    return db.execute(text(f"SELECT count(*) FROM {table}")).scalar()


def actions(db, user):
    return db.execute(select(models.AuditLog.action, models.AuditLog.resource_id)
                      .where(models.AuditLog.user_id == user.id).order_by(models.AuditLog.created_at)).all()


def test_registering_returns_the_version_and_writes_an_audit_event(db, owner):
    parsed = ss.parse(spec())
    view = register(db, owner)
    assert view["version_id"] == ss.version_id(parsed)
    assert view["spec"] == parsed.model_dump(mode="json")
    assert (view["dsl"], view["template"], view["family"], view["tags"]) == \
        ("qs-strategy-v1", "ma_crossover", "EQUITY_EOD", [])
    assert actions(db, owner) == [("STRATEGY_REGISTERED", view["version_id"])]


def test_a_version_is_read_back_by_its_id_or_a_tag(db, owner):
    view = register(db, owner, tag="1.0.0")
    assert view["tags"] == ["1.0.0"]
    assert main.get_strategy(view["version_id"], user=owner, db=db) == view
    assert main.get_strategy("1.0.0", user=owner, db=db) == view
    assert status_of(lambda: main.get_strategy("2.0.0", user=owner, db=db)) == 404


def test_held_since_is_when_the_caller_registered_never_the_first_registrant(db, owner):
    """The version row keeps the first registrant's time; showing it would
    tell a later registrant that someone else wrote the same spec."""
    other = make_user(db, "other@example.com")
    first = register(db, owner)
    later = register(db, other)
    assert later["version_id"] == first["version_id"]
    assert later["held_since"] > first["held_since"]
    first_registered = db.get(models.StrategyVersion, first["version_id"]).created_at
    assert later["held_since"] != first_registered.isoformat()
    assert "created_at" not in later
    assert main.get_strategy(first["version_id"], user=other, db=db)["held_since"] == later["held_since"]


def test_versions_and_tags_are_only_the_callers(db, owner):
    other = make_user(db, "other@example.com")
    mine = register(db, owner, tag="1.0.0")
    theirs = register(db, other, spec(**{"signal.fast": 10}), tag="1.0.0")
    assert [v["version_id"] for v in main.list_strategies(50, user=owner, db=db)] == [mine["version_id"]]
    assert main.get_strategy("1.0.0", user=other, db=db)["version_id"] == theirs["version_id"]
    assert status_of(lambda: main.get_strategy(mine["version_id"], user=other, db=db)) == 404
    assert status_of(lambda: tag(db, other, mine["version_id"], "stolen")) == 404


def test_the_list_is_last_registered_first_and_limited(db, owner):
    ids = [register(db, owner, spec(**{"signal.fast": fast}))["version_id"] for fast in (5, 10, 15)]
    assert [v["version_id"] for v in main.list_strategies(50, user=owner, db=db)] == ids[::-1]
    assert [v["version_id"] for v in main.list_strategies(2, user=owner, db=db)] == ids[:0:-1]


def test_tagging_is_audited_repeatable_and_never_moves_a_tag(db, owner):
    a = register(db, owner)["version_id"]
    b = register(db, owner, spec(**{"signal.fast": 10}))["version_id"]
    assert tag(db, owner, a, "baseline")["tags"] == ["baseline"]
    assert tag(db, owner, a, "baseline")["tags"] == ["baseline"]
    assert status_of(lambda: tag(db, owner, b, "baseline")) == 409
    assert status_of(lambda: tag(db, owner, b, " bad")) == 422
    assert status_of(lambda: tag(db, owner, "qs-sv-sha256-v1:" + "0" * 64, "x")) == 404
    assert main.get_strategy("baseline", user=owner, db=db)["version_id"] == a
    assert [action for action, _ in actions(db, owner)].count("STRATEGY_TAGGED") == 2


def test_a_registration_whose_tag_is_refused_registers_nothing(db, owner):
    assert status_of(lambda: register(db, owner, tag="-bad")) == 422
    other = register(db, owner, spec(**{"signal.fast": 10}), tag="taken")
    assert status_of(lambda: register(db, owner, tag="taken")) == 409
    assert count(db, "strategy_versions") == count(db, "user_strategy_versions") == 1
    assert [action for action, _ in actions(db, owner)] == ["STRATEGY_REGISTERED"]
    assert main.get_strategy("taken", user=owner, db=db)["version_id"] == other["version_id"]


def test_an_invalid_spec_is_the_usual_422_and_writes_nothing(db, owner):
    with pytest.raises(RequestValidationError):
        register(db, owner, spec(**{"signal.slow": 10}))

    def get_db():
        yield db

    main.app.dependency_overrides[main.get_current_user] = lambda: owner
    main.app.dependency_overrides[main.get_db] = get_db
    try:
        response = TestClient(main.app).post("/api/research/strategies", json={"spec": spec(**{"signal.slow": 10})})
    finally:
        main.app.dependency_overrides.pop(main.get_current_user, None)
        main.app.dependency_overrides.pop(main.get_db, None)
    assert response.status_code == 422, response.text
    [error] = response.json()["detail"]
    assert error["loc"][:3] == ["body", "spec", "signal"]
    assert count(db, "strategy_versions") == count(db, "user_strategy_versions") == 0
    assert actions(db, owner) == []
