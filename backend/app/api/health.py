from fastapi import APIRouter, Request


router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(request: Request) -> dict[str, str | bool]:
    """Distinguish an alive API process from a ready frozen inference runtime."""

    service = getattr(request.app.state, "screening_service", None)
    screening_ready = bool(service is not None and service.is_ready)
    return {
        "status": "ok" if screening_ready else "degraded",
        "application": "Speech-Based Cognitive Decline Screening",
        "screening_model_ready": screening_ready,
    }
