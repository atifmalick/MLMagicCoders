"""
FastAPI Gateway — Inference Job Orchestrator
=============================================
Runs on: Render Free Tier (or any host)
Role:    Ingest client inference requests, store job state in Supabase,
         and expose endpoints for status polling.

Endpoints:
    POST   /submit-job          — Create a new inference job
    GET    /job-status/{job_id} — Poll for job result
    GET    /health              — Health check
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import httpx
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("gateway")

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY: str = os.environ.get("SUPABASE_SERVICE_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.warning(
        "SUPABASE_URL and/or SUPABASE_SERVICE_KEY not set. "
        "The gateway will start but all requests will fail."
    )

JOBS_TABLE: str = os.environ.get("JOBS_TABLE", "jobs")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ML Inference Gateway",
    version="1.0.0",
    description="Production-grade inference job orchestrator",
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SubmitJobRequest(BaseModel):
    """Schema for incoming inference requests."""

    input_payload: str = Field(
        ...,
        min_length=1,
        max_length=10_000_000,  # 10 MB as base64
        description="Base64-encoded input data for the model",
    )
    input_format: str = Field(
        default="text/plain",
        max_length=64,
        description='MIME type / format hint e.g. "text/plain", "image/png"',
    )

    @field_validator("input_payload")
    @classmethod
    def payload_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("input_payload must not be empty or whitespace-only")
        return stripped


class SubmitJobResponse(BaseModel):
    """Returned after successfully creating a job."""

    job_id: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Full job record returned on status poll."""

    job_id: str
    status: str
    input_payload: str
    input_format: str
    output_payload: Optional[str] = None
    output_format: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    worker_id: Optional[str] = None
    model_version: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Supabase REST helpers
# ---------------------------------------------------------------------------


async def _supabase_request(
    method: str,
    path: str,
    json_body: Optional[dict] = None,
) -> httpx.Response:
    """Make an authenticated request to the Supabase REST API."""
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{path.lstrip('/')}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method.upper() == "GET":
                resp = await client.get(url, headers=headers, params=json_body)
            elif method.upper() == "POST":
                resp = await client.post(url, headers=headers, json=json_body or {})
            elif method.upper() == "PATCH":
                resp = await client.patch(url, headers=headers, json=json_body or {})
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Supabase HTTP error [%s]: %s — %s",
                exc.response.status_code,
                exc.response.text[:500],
                url,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("Supabase connection error: %s — %s", exc, url)
            raise
    return resp


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple liveness probe."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post(
    "/submit-job",
    response_model=SubmitJobResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "Database unavailable"},
    },
)
async def submit_job(request: SubmitJobRequest):
    """
    Accept a client inference request, persist it to Supabase with
    status PENDING, and return the generated job_id.
    """
    logger.info("Received /submit-job — payload length=%d", len(request.input_payload))

    body = {
        "input_payload": request.input_payload,
        "input_format": request.input_format,
        "status": "PENDING",
    }

    try:
        resp = await _supabase_request("POST", JOBS_TABLE, json_body=body)
        records = resp.json()
    except Exception as exc:
        logger.exception("Failed to insert job into Supabase")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not persist job: {exc}",
        )

    if not records:
        logger.error("Supabase returned empty record set after insert")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Database returned no record after insert",
        )

    record = records[0]
    job_id = record.get("id", str(uuid.uuid4()))
    created_at = record.get("created_at", datetime.now(timezone.utc).isoformat())

    logger.info("Job created — id=%s", job_id)
    return SubmitJobResponse(
        job_id=job_id,
        status="PENDING",
        created_at=created_at,
    )


@app.get(
    "/job-status/{job_id}",
    response_model=JobStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        502: {"model": ErrorResponse, "description": "Database unavailable"},
    },
)
async def get_job_status(job_id: str):
    """
    Return the full job record for a given job_id.
    The client can poll this endpoint until status == 'COMPLETED'
    or 'FAILED'.
    """
    logger.info("Received /job-status/%s", job_id)

    # Validate UUID format gracefully (avoid SQL errors)
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid job_id format: {job_id}",
        )

    query = {
        "id": f"eq.{job_id}",
        "select": "*",
    }

    try:
        resp = await _supabase_request("GET", JOBS_TABLE, json_body=query)
        records = resp.json()
    except Exception as exc:
        logger.exception("Failed to query Supabase for job %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not query job status: {exc}",
        )

    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    record = records[0]
    logger.info("Job %s status is %s", job_id, record.get("status"))

    return JobStatusResponse(
        job_id=record["id"],
        status=record["status"],
        input_payload=record.get("input_payload", ""),
        input_format=record.get("input_format", "text/plain"),
        output_payload=record.get("output_payload"),
        output_format=record.get("output_format"),
        error_message=record.get("error_message"),
        created_at=record.get("created_at", ""),
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        worker_id=record.get("worker_id"),
        model_version=record.get("model_version"),
    )


# ---------------------------------------------------------------------------
# Entry point (for debugging locally)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        log_level="info",
    )