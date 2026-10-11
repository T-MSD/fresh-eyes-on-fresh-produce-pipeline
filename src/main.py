from fastapi import FastAPI

from src.api.frame_areas import router as frames_router
from src.api.hour_range import router as hour_range_router
from src.api.tasks import router as tasks_router

app = FastAPI(title="Fresh Eyes API")
app.include_router(frames_router)
app.include_router(hour_range_router)
app.include_router(tasks_router)
