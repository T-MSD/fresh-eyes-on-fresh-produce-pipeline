from fastapi import FastAPI

from src.api.tasks import router

app = FastAPI(title="Fresh Eyes API")
app.include_router(router)
