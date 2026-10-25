from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Connection

from config.db.connection import get_connection

DatabaseConnection = Annotated[Connection, Depends(get_connection)]


def transaction(connection: DatabaseConnection) -> Iterator[Connection]:
    """A connection whose writes commit before the response is sent, or roll back on error."""
    with connection.begin():
        yield connection


Transaction = Annotated[Connection, Depends(transaction, scope="function")]
