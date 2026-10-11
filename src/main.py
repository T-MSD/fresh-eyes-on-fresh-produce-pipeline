from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Connection, Engine

from config.db.connection import make_engine
from src.api.hour_range import router as hour_range_router
from src.api.routes import router
from src.api.tasks import router as tasks_router


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
    app.include_router(hour_range_router)
    app.include_router(tasks_router)
    return app


app = create_app()
