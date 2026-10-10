from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="Fresh Eyes API")
app.include_router(router)
