from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from config.db.connection import make_engine
from config.db.queries import latest_run_id


@lru_cache
def get_engine() -> Engine:
    return make_engine()


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session


DatabaseSession = Annotated[Session, Depends(get_session)]


def get_latest_run_id(session: DatabaseSession) -> int | None:
    return latest_run_id(session)


LatestRunId = Annotated[int | None, Depends(get_latest_run_id)]
