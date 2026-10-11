import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.hour_range import router as hour_range_router
from src.api.routes import router
from src.api.tasks import router as tasks_router


def create_app() -> FastAPI:
    app = FastAPI(title="Fresh Eyes API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", "").split(",")
            if origin.strip()
        ],
        allow_methods=["GET", "POST"],
    )
    app.include_router(router)
    app.include_router(hour_range_router)
    app.include_router(tasks_router)
    return app


app = create_app()
