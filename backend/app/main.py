"""FastAPI application factory for the frozen screening prototype."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.health import router as health_router
from .api.screening import router as screening_router
from .services.screening_service import ScreeningService


LOGGER = logging.getLogger(__name__)


def create_app(
    service_factory: Callable[[], ScreeningService] | None = None,
) -> FastAPI:
    """Build the API and load heavy frozen assets once during lifespan startup."""

    runtime_factory = service_factory or ScreeningService.load_default

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.screening_service = None
        try:
            application.state.screening_service = runtime_factory()
            LOGGER.info("frozen_screening_runtime_ready")
        except Exception:
            # The API stays alive for a transparent readiness response. Detailed errors stay server-side.
            LOGGER.exception("frozen_screening_runtime_unavailable")
        yield
        application.state.screening_service = None

    app = FastAPI(
        title="Speech-Based Cognitive Screening Research Prototype",
        description=(
            "A research-only preliminary speech-pattern screening API. "
            "It is not a medical diagnostic service."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(screening_router, prefix="/api/v1")
    return app


app = create_app()
