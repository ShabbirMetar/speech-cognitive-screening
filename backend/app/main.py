from fastapi import FastAPI

from app.api.health import router as health_router


app = FastAPI(title="Speech-Based Cognitive Decline Screening")
app.include_router(health_router, prefix="/api")
