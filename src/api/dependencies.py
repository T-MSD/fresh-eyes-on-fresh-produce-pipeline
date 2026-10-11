from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy import Connection

from config.db.connection import get_connection
from config.db.queries import SelectedRun, StoredFrame, get_latest_run, get_stored_frame

DatabaseConnection = Annotated[Connection, Depends(get_connection)]


def completed_run(connection: DatabaseConnection) -> SelectedRun:
    run = get_latest_run(connection)
    if run is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    return run


CompletedRun = Annotated[SelectedRun, Depends(completed_run)]


def stored_frame(
    camera: str, filename: str, connection: DatabaseConnection, run: CompletedRun
) -> StoredFrame:
    frame = get_stored_frame(connection, run, camera, filename)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found in the latest completed run")
    return frame


RequestedFrame = Annotated[StoredFrame, Depends(stored_frame)]
