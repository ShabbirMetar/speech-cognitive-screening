from pathlib import Path
import sys

from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import create_app


class _ReadyService:
    is_ready = True

    @staticmethod
    def model_info() -> dict[str, object]:
        return {
            "model_type": "Logistic Regression",
            "model_version": "1.0.0",
            "asr": "faster-whisper base.en",
            "feature_representation": "ASR_RATE_PLUS_CTD",
            "feature_count": 51,
            "research_threshold": 0.53,
            "disclaimer": "Research screening prototype; this result is not a medical diagnosis.",
        }


def test_health_check() -> None:
    with TestClient(create_app(lambda: _ReadyService())) as client:
        response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "Speech-Based Cognitive Decline Screening",
        "screening_model_ready": True,
    }
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_is_limited_to_local_vite_origins() -> None:
    app = create_app(lambda: _ReadyService())
    middleware = next(item for item in app.user_middleware if item.cls is CORSMiddleware)
    assert middleware.kwargs["allow_origins"] == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    assert middleware.kwargs["allow_credentials"] is False
