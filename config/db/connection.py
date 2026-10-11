import os
from collections.abc import Iterator
from functools import cache
from typing import Any

from sqlalchemy import Connection, Engine, create_engine


def make_engine(url: str | None = None, **kwargs: Any) -> Engine:
    """Build an engine for `url`, or for the `DATABASE_URL` environment variable."""
    return create_engine(url or os.environ["DATABASE_URL"], **kwargs)


@cache
def get_engine() -> Engine:
    return make_engine()


def get_connection() -> Iterator[Connection]:
    with get_engine().connect() as connection:
        yield connection
