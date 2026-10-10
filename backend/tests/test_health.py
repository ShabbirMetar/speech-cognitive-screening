from pathlib import Path
import sys

from fastapi.testclient import TestClient


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
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "Speech-Based Cognitive Decline Screening",
        "screening_model_ready": True,
    }
