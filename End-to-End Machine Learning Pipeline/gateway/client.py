"""
Pipeline Client — Example script to interact with the ML Inference Pipeline
============================================================================
This is a ready-to-use Python client for the pipeline. After deploying
the Gateway and Worker, set GATEWAY_URL below and run this script.

Usage:
    python client.py

What it does:
    1. Submits a text sentiment-analysis job to the gateway
    2. Polls for the result until COMPLETED or FAILED
    3. Decodes and prints the model prediction
"""

import argparse
import base64
import json
import sys
import time
from typing import Any, Dict, Optional

import requests


class PipelineClient:
    """Client for the ML Inference Pipeline Gateway."""

    def __init__(self, gateway_url: str, timeout: int = 60):
        """
        Args:
            gateway_url: Base URL of the deployed Render gateway.
                         Example: "https://ml-pipeline-gateway.onrender.com"
            timeout:     Max seconds to wait for a job to complete when polling.
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """Verify the gateway is alive."""
        resp = requests.get(f"{self.gateway_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Submit a Job
    # ------------------------------------------------------------------
    def submit_job(self, text: str, input_format: str = "text/plain") -> str:
        """
        Submit a text inference job.

        Args:
            text:         The raw text string to classify.
            input_format: MIME type hint (default: 'text/plain').

        Returns:
            job_id: The UUID of the created job.

        Raises:
            requests.HTTPError: If the gateway returns an error.
        """
        # Base64-encode the input text
        payload_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")

        body = {
            "input_payload": payload_b64,
            "input_format": input_format,
        }

        resp = requests.post(
            f"{self.gateway_url}/submit-job",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id: str = data["job_id"]
        print(f"  ✓ Job submitted — ID: {job_id}")
        print(f"  ✓ Status: {data['status']}")
        print(f"  ✓ Created at: {data['created_at']}")
        return job_id

    # ------------------------------------------------------------------
    # Poll for Job Status
    # ------------------------------------------------------------------
    def poll_until_complete(self, job_id: str, poll_interval: int = 3) -> Dict[str, Any]:
        """
        Poll the gateway until the job completes or fails.

        Args:
            job_id:        The UUID of the job to poll.
            poll_interval: Seconds between poll attempts.

        Returns:
            The full job record dict.

        Raises:
            TimeoutError: If the job doesn't finish within self.timeout.
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {self.timeout}s"
                )

            resp = requests.get(
                f"{self.gateway_url}/job-status/{job_id}",
                timeout=10,
            )
            resp.raise_for_status()
            record = resp.json()

            status = record["status"]
            print(f"  ⌛ Polling... status={status}  (elapsed: {elapsed:.1f}s)")

            if status in ("COMPLETED", "FAILED"):
                return record

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Decode Output
    # ------------------------------------------------------------------
    @staticmethod
    def decode_output(output_b64: str) -> Any:
        """
        Decode a base64-encoded output payload.
        Attempts to parse JSON first, otherwise returns raw string.
        """
        decoded_bytes = base64.b64decode(output_b64)
        decoded_str = decoded_bytes.decode("utf-8", errors="replace")

        # Try to pretty-print as JSON
        try:
            return json.loads(decoded_str)
        except json.JSONDecodeError:
            return decoded_str

    # ------------------------------------------------------------------
    # Full Pipeline Run
    # ------------------------------------------------------------------
    def run_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Convenience method: submit a text sentiment job and wait for result.

        Args:
            text: The text to analyze.

        Returns:
            The decoded inference result dict (e.g. {"label": "POSITIVE", "score": 0.998}).
        """
        print(f"\n{'='*60}")
        print(f"Submitting sentiment analysis for: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        print(f"{'='*60}\n")

        # 1. Health check
        health = self.health_check()
        print(f"  ✓ Gateway healthy: {health['status']}")

        # 2. Submit
        job_id = self.submit_job(text)

        # 3. Poll
        record = self.poll_until_complete(job_id)

        print(f"\n{'='*60}")
        print(f"Job Result (Status: {record['status']})")
        print(f"{'='*60}")

        if record["status"] == "COMPLETED":
            # 4. Decode
            output = self.decode_output(record["output_payload"])
            print(f"\n  Model: {record.get('model_version', 'unknown')}")
            print(f"  Worker: {record.get('worker_id', 'unknown')}")
            print(f"  Completed in: {record.get('completed_at', 'N/A')}")
            print(f"\n  ╔══ PREDICTION ══╗")
            print(f"  ║  {output}")
            print(f"  ╚════════════════╝\n")
            return output
        else:
            print(f"\n  ✗ Job FAILED: {record.get('error_message', 'Unknown error')}")
            return {"error": record.get("error_message")}


# ------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ML Inference Pipeline Client — submit text for sentiment analysis"
    )
    parser.add_argument(
        "--gateway",
        default="http://localhost:8000",
        help="Gateway URL (default: http://localhost:8000, change to https://your-app.onrender.com)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to analyze. If omitted, uses a default example.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Max seconds to wait for completion (default: 120)",
    )

    args = parser.parse_args()

    # Default example text if none provided
    sample_text = (
        args.text
        or "This movie is absolutely fantastic and wonderful! I loved every minute of it."
    )

    client = PipelineClient(gateway_url=args.gateway, timeout=args.timeout)
    try:
        result = client.run_sentiment_analysis(sample_text)
        return 0
    except requests.exceptions.ConnectionError:
        print(
            f"\n✗ Could not connect to {args.gateway}.\n"
            "  Make sure the gateway is running.\n"
            "  For local dev:  cd gateway && uvicorn app:app --reload\n"
            "  For production: deploy to Render first, then pass the URL with --gateway",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"\n✗ Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())