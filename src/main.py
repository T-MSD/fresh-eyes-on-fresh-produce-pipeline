import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.runs.router import router as runs_router
from src.store.router import router as store_router
from src.tasks.router import router as tasks_router


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
    app.include_router(store_router)
    app.include_router(runs_router)
    app.include_router(tasks_router)
    return app


app = create_app()
