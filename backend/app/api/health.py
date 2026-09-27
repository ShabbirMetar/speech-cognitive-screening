from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a lightweight service status without accessing research data."""
    return {
        "status": "ok",
        "application": "Speech-Based Cognitive Decline Screening",
    }
