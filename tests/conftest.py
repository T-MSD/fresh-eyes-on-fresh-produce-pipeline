import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, text
from sqlalchemy.engine import URL, make_url

from config.db.connection import get_connection, make_engine
from src.api import dependencies
from src.main import app
from tests.db_builders import add_camera, add_dataset, add_product

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def alembic_config(connection: Connection) -> Config:
    cfg = Config(toml_file=PYPROJECT)
    cfg.attributes["connection"] = connection
    return cfg


def database_url_for_tests() -> URL:
    raw = os.environ.get("TEST_DATABASE_URL")
    if not raw:
        pytest.skip("TEST_DATABASE_URL is not set")
    url = make_url(raw)
    if not (url.database or "").endswith("_test"):
        pytest.exit(
            f"Refusing to use {url.database!r}: TEST_DATABASE_URL must name a *_test database",
            returncode=1,
        )
    return url


def recreate_database(url: URL) -> None:
    admin = make_engine(
        url.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin.dispose()


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    url = database_url_for_tests()
    recreate_database(url)
    engine = make_engine(url.render_as_string(hide_password=False))
    with engine.begin() as conn:
        command.upgrade(alembic_config(conn), "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as conn:
        transaction = conn.begin()
        yield conn
        transaction.rollback()


@pytest.fixture
def dataset(db: Connection) -> int:
    return add_dataset(db)


@pytest.fixture
def camera(db: Connection, dataset: int) -> int:
    add_product(db, dataset)
    return add_camera(db, dataset, "6215")


@pytest.fixture
def client(db: Connection) -> Iterator[TestClient]:
    app.dependency_overrides[get_connection] = lambda: db
    app.dependency_overrides[dependencies.transaction] = lambda: db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
