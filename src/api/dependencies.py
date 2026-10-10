from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config.db.queries import latest_run_id


def database_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.db_bind) as session:
        yield session


DatabaseSession = Annotated[Session, Depends(database_session)]


def most_recent_run(session: DatabaseSession) -> int:
    run_id = latest_run_id(session)
    if run_id is None:
        raise HTTPException(status_code=404, detail="No run available")
    return run_id


MostRecentRun = Annotated[int, Depends(most_recent_run)]
