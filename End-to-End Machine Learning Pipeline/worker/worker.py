"""
Inference Worker — Polls Supabase for PENDING jobs and runs model inference
===========================================================================
Runs on: Hugging Face Spaces (Free CPU basic instance)
Model:   distilbert-base-uncased (lightweight ~67M params text classifier)
         Can be swapped for any Hugging Face model by changing MODEL_NAME.
"""

import base64
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | worker | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    stream=sys.stdout,
)
logger = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Configuration — all via environment variables
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY: str = os.environ.get("SUPABASE_SERVICE_KEY", "")
JOBS_TABLE: str = os.environ.get("JOBS_TABLE", "jobs")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
POLL_INTERVAL_SECONDS: int = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))

# Worker identity — useful when multiple workers are deployed
WORKER_ID: str = os.environ.get("WORKER_ID", platform.node())

# ---------------------------------------------------------------------------
# Global model pipeline (loaded once at startup)
# ---------------------------------------------------------------------------
_classifier_pipeline: Optional[Pipeline] = None


def _load_model_pipeline() -> Pipeline:
    """
    Download and cache the model + tokenizer, then return a Hugging Face
    `Pipeline` object ready for inference.
    """
    global _classifier_pipeline

    if _classifier_pipeline is not None:
        return _classifier_pipeline

    logger.info("Loading model: %s", MODEL_NAME)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _classifier_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Force CPU
        )
        logger.info("Model loaded successfully: %s", MODEL_NAME)
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", MODEL_NAME, exc)
        raise

    return _classifier_pipeline


# ---------------------------------------------------------------------------
# Supabase REST helpers (synchronous — no async needed in this worker loop)
# ---------------------------------------------------------------------------


def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _fetch_pending_jobs(limit: int = 1) -> List[Dict[str, Any]]:
    """
    Query Supabase for the oldest PENDING job(s).
    Returns a list of job dicts (empty list if none found).
    """
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{JOBS_TABLE}"
    params = {
        "status": "eq.PENDING",
        "order": "created_at.asc",
        "limit": str(limit),
        "select": "*",
    }
    try:
        resp = requests.get(url, headers=_supabase_headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching pending jobs — will retry")
        return []
    except requests.exceptions.RequestException as exc:
        logger.error("Error fetching pending jobs: %s", exc)
        return []


def _claim_job(job_id: str) -> bool:
    """
    Atomically transition a job from PENDING -> IN_PROGRESS.
    Returns True if the claim succeeded (i.e., no other worker grabbed it).
    """
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{JOBS_TABLE}"
    params = {"id": f"eq.{job_id}"}
    body = {
        "status": "IN_PROGRESS",
        "worker_id": WORKER_ID,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    headers = _supabase_headers()
    headers["Prefer"] = "return=minimal"  # We only care about rows-affected
    try:
        resp = requests.patch(url, headers=headers, params=params, json=body, timeout=15)
        resp.raise_for_status()
        # Supabase returns 204 with no body on success; we check rows affected
        # via the content-range header or simply verify the request succeeded.
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to claim job %s: %s", job_id, exc)
        return False


def _complete_job(
    job_id: str,
    output_payload: str,
    output_format: str,
    model_version: str,
) -> bool:
    """Mark a job as COMPLETED with its output artifact."""
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{JOBS_TABLE}"
    params = {"id": f"eq.{job_id}"}
    body = {
        "status": "COMPLETED",
        "output_payload": output_payload,
        "output_format": output_format,
        "model_version": model_version,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        resp = requests.patch(url, headers=_supabase_headers(), params=params, json=body, timeout=15)
        resp.raise_for_status()
        logger.info("Job %s completed successfully", job_id)
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to mark job %s as COMPLETED: %s", job_id, exc)
        return False


def _fail_job(job_id: str, error_message: str) -> bool:
    """Mark a job as FAILED with an error description."""
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{JOBS_TABLE}"
    params = {"id": f"eq.{job_id}"}
    body = {
        "status": "FAILED",
        "error_message": error_message[:1000],  # Truncate to DB column limit
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        resp = requests.patch(url, headers=_supabase_headers(), params=params, json=body, timeout=15)
        resp.raise_for_status()
        logger.info("Job %s marked as FAILED", job_id)
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to mark job %s as FAILED: %s", job_id, exc)
        return False


# ---------------------------------------------------------------------------
# Model inference logic
# ---------------------------------------------------------------------------


def _run_text_classification(text: str) -> Dict[str, Any]:
    """
    Run distilbert text classification and return the result.
    The pipeline returns a list like: [{'label': 'POSITIVE', 'score': 0.998}]
    """
    pipe = _load_model_pipeline()
    results = pipe(text)
    if not results:
        raise ValueError("Model returned empty result")
    return results[0]


def _inference_handler(input_payload: str, input_format: str) -> Dict[str, Any]:
    """
    Dispatch an inference request based on input_format.
    Returns a dict with:
        - output_payload (base64-encoded string)
        - output_format  (MIME type)
        - model_version  (str)
    """
    # Decode the base64 input
    try:
        decoded_bytes = base64.b64decode(input_payload)
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 input_payload: {exc}") from exc

    if input_format.startswith("text/"):
        # Text classification inference
        text = decoded_bytes.decode("utf-8", errors="replace")
        logger.info("Running text classification on %d characters", len(text))

        result = _run_text_classification(text)

        output = json.dumps(result, ensure_ascii=False)
        output_b64 = base64.b64encode(output.encode("utf-8")).decode("ascii")

        return {
            "output_payload": output_b64,
            "output_format": "application/json",
            "model_version": MODEL_NAME,
        }
    elif input_format.startswith("image/"):
        # Image inference placeholder — swap in a vision model (e.g., ViT)
        # For now we do a simple pass-through / echo to demonstrate the pipeline.
        logger.info("Received image input — running passthrough (vision model TBD)")
        output_b64 = base64.b64encode(
            json.dumps({"note": "Image model not yet configured; echoing input"}).encode("utf-8")
        ).decode("ascii")
        return {
            "output_payload": output_b64,
            "output_format": "application/json",
            "model_version": f"{MODEL_NAME}-vision-placeholder",
        }
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------


def run_worker_loop() -> None:
    """
    Infinite loop that:
        1. Fetches the oldest PENDING job
        2. Claims it (PENDING -> IN_PROGRESS)
        3. Runs inference
        4. Marks COMPLETED or FAILED
        5. Sleeps POLL_INTERVAL_SECONDS if no job found
    """
    logger.info(
        "Worker starting — id=%s | model=%s | poll_interval=%ds | db=%s",
        WORKER_ID,
        MODEL_NAME,
        POLL_INTERVAL_SECONDS,
        SUPABASE_URL,
    )

    consecutive_empty_polls = 0

    while True:
        try:
            # --- 1. Fetch pending job ---
            jobs = _fetch_pending_jobs(limit=1)

            if not jobs:
                consecutive_empty_polls += 1
                backoff = min(
                    POLL_INTERVAL_SECONDS * (2 ** (consecutive_empty_polls - 1)),
                    120,  # Cap at 2 minutes
                )
                logger.debug(
                    "No pending jobs (empty poll #%d) — sleeping %ds",
                    consecutive_empty_polls,
                    backoff,
                )
                time.sleep(backoff)
                continue

            # Reset backoff on successful fetch
            consecutive_empty_polls = 0
            job = jobs[0]
            job_id = job["id"]
            logger.info("Discovered pending job %s", job_id)

            # --- 2. Claim the job ---
            if not _claim_job(job_id):
                logger.warning("Failed to claim job %s — another worker may have it", job_id)
                continue

            # --- 3. Run inference with retry logic ---
            last_error: Optional[str] = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(
                        "Processing job %s (attempt %d/%d)",
                        job_id,
                        attempt,
                        MAX_RETRIES,
                    )
                    result = _inference_handler(
                        input_payload=job["input_payload"],
                        input_format=job.get("input_format", "text/plain"),
                    )

                    # --- 4. Mark COMPLETED ---
                    _complete_job(
                        job_id=job_id,
                        output_payload=result["output_payload"],
                        output_format=result["output_format"],
                        model_version=result["model_version"],
                    )
                    last_error = None
                    break  # Success — exit retry loop

                except Exception as exc:
                    last_error = f"Attempt {attempt}/{MAX_RETRIES}: {exc}"
                    logger.exception("Inference attempt %d failed for job %s", attempt, job_id)
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)  # Exponential backoff 2s, 4s, 8s

            # --- 5. Mark FAILED if all retries exhausted ---
            if last_error:
                error_msg = f"All {MAX_RETRIES} attempts failed for job {job_id}. Last: {last_error}"
                logger.error(error_msg)
                _fail_job(job_id, error_msg)

        except KeyboardInterrupt:
            logger.info("Worker received shutdown signal — exiting")
            break
        except Exception as exc:
            logger.exception("Unhandled error in main loop: %s", exc)
            time.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Validate required env vars early
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_KEY")

    if missing:
        logger.critical(
            "Missing required environment variables: %s. Cannot start worker.",
            ", ".join(missing),
        )
        sys.exit(1)

    # Preload the model at startup so the first job doesn't pay cold-start cost
    logger.info("Preloading model %s ...", MODEL_NAME)
    try:
        _load_model_pipeline()
    except Exception as exc:
        logger.critical("Model preloading failed: %s. Worker cannot start.", exc)
        sys.exit(1)

    run_worker_loop()