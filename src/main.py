from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Connection, Engine

from config.db.connection import make_engine
from src.api.routes import router


def create_app(bind: Engine | Connection | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.db_bind = bind if bind is not None else make_engine()
        try:
            yield
        finally:
            if bind is None:
                app.state.db_bind.dispose()

    app = FastAPI(title="Fresh Eyes API", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
