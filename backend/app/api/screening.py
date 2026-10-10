"""Endpoints for frozen, research-only speech screening inference."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from .schemas import ModelInfo, ScreeningAnalysisResponse
from ..services.screening_service import (
    AudioInputError,
    FeatureExtractionError,
    ModelRuntimeUnavailable,
    ScreeningRuntimeError,
    ScreeningService,
    TranscriptionError,
)


LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/screening", tags=["Research screening"])

def _service_or_503(request: Request) -> ScreeningService:
    service = getattr(request.app.state, "screening_service", None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The frozen screening runtime is not ready.",
        )
    return service


async def _read_upload(upload: UploadFile, *, max_bytes: int) -> bytes:
    """Read an upload with a hard byte limit without trusting its filename."""

    try:
        content = await upload.read(max_bytes + 1)
    finally:
        await upload.close()

    if not content:
        raise AudioInputError("An uploaded audio file was empty.")
    if len(content) > max_bytes:
        raise AudioInputError("An uploaded audio file exceeded the configured size limit.")
    return content


def _safe_api_error(error: Exception) -> HTTPException:
    if isinstance(error, AudioInputError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error))
    if isinstance(error, TranscriptionError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech transcription could not be completed for one of the recordings.",
        )
    if isinstance(error, FeatureExtractionError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Speech features could not be extracted from one of the recordings.",
        )
    if isinstance(error, ModelRuntimeUnavailable):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The frozen screening runtime is not available.",
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="The screening request could not be completed.",
    )


@router.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Read non-sensitive frozen screening-model information",
)
async def model_info(request: Request) -> ModelInfo:
    """Return public configuration only; no participant or development data."""

    return ModelInfo.model_validate(_service_or_503(request).model_info())


@router.post(
    "/analyze",
    response_model=ScreeningAnalysisResponse,
    summary="Analyze SFT, PFT, and CTD recordings with the frozen research pipeline",
    description=(
        "Upload one recording for each task. Audio is processed transiently and is not persisted by "
        "default. This research prototype returns a preliminary speech-based screening signal, not a diagnosis."
    ),
)
async def analyze_screening(
    request: Request,
    sft_audio: UploadFile | None = File(
        default=None,
        description="Semantic Fluency Task recording (WAV is supported robustly).",
    ),
    pft_audio: UploadFile | None = File(
        default=None,
        description="Phonemic Fluency Task recording (WAV is supported robustly).",
    ),
    ctd_audio: UploadFile | None = File(
        default=None,
        description="Cookie Theft Description recording (WAV is supported robustly).",
    ),
) -> ScreeningAnalysisResponse:
    """Run inference only; the frozen pipeline is never fitted in this endpoint."""

    service = _service_or_503(request)
    requested_uploads = {"SFT": sft_audio, "PFT": pft_audio, "CTD": ctd_audio}
    missing = [task for task, upload in requested_uploads.items() if upload is None]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required recording(s): {', '.join(missing)}.",
        )

    request_id = str(uuid4())
    try:
        audio_bytes = {
            task: await _read_upload(upload, max_bytes=service.max_upload_bytes)
            for task, upload in requested_uploads.items()
            if upload is not None
        }
        payload = await run_in_threadpool(service.analyze_uploads, request_id, audio_bytes)
        return ScreeningAnalysisResponse.model_validate(payload)
    except ScreeningRuntimeError as error:
        LOGGER.info("screening_request_failed request_id=%s category=%s", request_id, type(error).__name__)
        raise _safe_api_error(error) from None
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.warning("screening_request_failed request_id=%s category=%s", request_id, type(error).__name__)
        raise _safe_api_error(Exception()) from None
