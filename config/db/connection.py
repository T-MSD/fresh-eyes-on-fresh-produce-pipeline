import os
from typing import Any

from sqlalchemy import Engine, create_engine


def make_engine(url: str | None = None, **kwargs: Any) -> Engine:
    """Build an engine for `url`, or for the `DATABASE_URL` environment variable."""
    return create_engine(url or os.environ["DATABASE_URL"], **kwargs)
