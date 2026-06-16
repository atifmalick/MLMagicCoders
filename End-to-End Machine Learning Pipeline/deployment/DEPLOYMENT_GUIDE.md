# End-to-End ML Pipeline — 100% Free-Tier Deployment Guide

This guide walks you through deploying the **End-to-End Machine Learning Pipeline** using strictly free-tier services. **No credit card is required** for any of the services below.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Supabase Database Setup](#3-step-1--supabase-database-setup)
4. [Step 2 — Deploy the FastAPI Gateway on Render](#4-step-2--deploy-the-fastapi-gateway-on-render)
5. [Step 3 — Deploy the Inference Worker on Hugging Face Spaces](#5-step-3--deploy-the-inference-worker-on-hugging-face-spaces)
6. [Step 4 — Configure GitHub Actions CI/CD](#6-step-4--configure-github-actions-cicd)
7. [Step 5 — Test the Full Pipeline End-to-End](#7-step-5--test-the-full-pipeline-end-to-end)
8. [Environment Variables Reference](#8-environment-variables-reference)
9. [Monitoring and Debugging](#9-monitoring-and-debugging)
10. [Cost Confirmation — Why This Stays Free](#10-cost-confirmation--why-this-stays-free)

---

## 1. Architecture Overview

```
┌──────────────┐     POST /submit-job     ┌──────────────────┐
│              │ ──────────────────────►   │                  │
│   Client     │                           │  FastAPI Gateway │
│  (curl/App)  │                           │  (Render Free)   │
│              │ ◄──────────────────────   │                  │
└──────────────┘     GET /job-status/id    └────────┬─────────┘
                                                    │
                                                    │ Write/Read via
                                                    │ Supabase REST API
                                                    ▼
                                          ┌──────────────────┐
                                          │                  │
                                          │    Supabase DB   │
                                          │  (PostgreSQL)    │
                                          │                  │
                                          └────────┬─────────┘
                                                    │
                                          Polls for PENDING
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │                  │
                                          │  HF Spaces       │
                                          │  Inference       │
                                          │  Worker (CPU)    │
                                          │                  │
                                          └──────────────────┘
```

### Data Flow

1. **Client** sends a base64-encoded payload to `POST /submit-job` on the Gateway.
2. **Gateway** validates the payload, inserts a row into Supabase with status `PENDING`, and returns the `job_id`.
3. **Worker** (on Hugging Face Spaces) polls Supabase for `PENDING` jobs, claims one (status → `IN_PROGRESS`), runs inference, then updates to `COMPLETED` with the output payload.
4. **Client** polls `GET /job-status/{job_id}` until status is `COMPLETED` or `FAILED`, then reads the `output_payload`.

---

## 2. Prerequisites

Before you begin, create free accounts on these services:

| Service | Sign-up Link | Notes |
|---------|-------------|-------|
| **GitHub** | https://github.com/signup | You likely already have one |
| **Supabase** | https://supabase.com/dashboard/sign-up | No credit card required |
| **Render** | https://render.com/register | Free tier — no credit card needed |
| **Hugging Face** | https://huggingface.co/join | Free account with unlimited public Spaces |

### Local Tooling

- **Git** — `git --version` (install from https://git-scm.com/)
- **Docker** (optional, for local testing) — `docker --version`
- **Python 3.11+** (optional, for local testing)
- A terminal (PowerShell, bash, or similar)

---

## 3. Step 1 — Supabase Database Setup

### 3.1 Create a Supabase Project

1. Log in to https://supabase.com
2. Click **New project**
3. Fill in:
   - **Name**: `ml-pipeline`
   - **Database Password**: Generate a strong one and save it securely
   - **Region**: Choose the closest to you (or `us-east-1` for lowest latency)
   - **Pricing Plan**: **Free** (verified: no credit card needed)
4. Click **Create new project** and wait ~2 minutes for provisioning.

### 3.2 Get Your API Credentials

Once the project is ready:

1. Go to **Project Settings** → **API** (in the left sidebar)
2. Under **Project URL**, copy the **URL** — it looks like: `https://xxxxxxxxxxxx.supabase.co`
3. Under **Project API keys**, copy the **`service_role` key** (NOT the `anon` key). This has full admin access to the database.

> **⚠️ Security**: Never expose the `service_role` key in client-side code. The Gateway and Worker store it as server-side environment variables.

### 3.3 Run the Database Migration

You have two options:

#### Option A: Supabase SQL Editor (Easiest)

1. In the Supabase Dashboard, go to **SQL Editor** (left sidebar)
2. Click **New query**
3. Open the file `database/001_jobs_table.sql` from this repository
4. Copy-paste the entire contents into the editor
5. Click **Run** (▶)
6. Verify: you should see `CREATE TABLE`, indexes, and triggers created without errors

#### Option B: Local `psql` Client

```bash
# Install PostgreSQL client if needed
# Then run:
PGPASSWORD="your-db-password" psql -h db.xxxxxxxxxxxx.supabase.co \
  -p 5432 -d postgres -U postgres -f database/001_jobs_table.sql
```

### 3.4 Verify the Table

1. Go to **Table Editor** in Supabase
2. You should see the `jobs` table with columns: `id`, `input_payload`, `input_format`, `status`, `output_payload`, `output_format`, `error_message`, `retry_count`, `worker_id`, `model_version`, `created_at`, `started_at`, `completed_at`

---

## 4. Step 2 — Deploy the FastAPI Gateway on Render

### 4.1 Push the Code to GitHub

If you haven't already:

```bash
# Initialize repository
cd /path/to/End-to-End Machine Learning Pipeline
git init
git add .
git commit -m "Initial commit: complete ML pipeline"

# Create a repository on GitHub (do NOT initialize with README)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/ml-pipeline.git
git branch -M main
git push -u origin main
```

### 4.2 Create a Render Web Service

1. Log in to https://dashboard.render.com
2. Click **New +** → **Web Service**
3. **Connect your GitHub repository**: Authorize Render to access your repos, then select `ml-pipeline`
4. Configure the service:

| Field | Value |
|-------|-------|
| **Name** | `ml-pipeline-gateway` |
| **Region** | `Frankfurt (EU)` or `Oregon (US)` — closest to your users |
| **Branch** | `main` |
| **Root Directory** | `gateway` (this is critical — Render looks here for the Dockerfile) |
| **Runtime** | `Docker` (Render auto-detects the Dockerfile) |
| **Plan** | **Free** ($0/month — spins down after 15 min of inactivity) |

5. Under **Environment Variables**, add the following **two secrets**:

| Key | Value |
|-----|-------|
| `SUPABASE_URL` | Your Supabase project URL (e.g., `https://xxxxxxxxxxxx.supabase.co`) |
| `SUPABASE_SERVICE_KEY` | Your Supabase `service_role` key |

6. Click **Create Web Service**

### 4.3 Wait for Deployment

- Render will build the Docker image and deploy it (~2–3 minutes)
- Once deployed, your service will be available at: `https://ml-pipeline-gateway.onrender.com`
- The free tier spins down after 15 minutes of inactivity. The first request after idle will take ~30–60 seconds to cold-start.

### 4.4 Verify the Gateway

```bash
# Health check
curl https://ml-pipeline-gateway.onrender.com/health

# Expected response:
# {"status":"healthy","timestamp":"2026-06-15T14:30:00+00:00"}
```

---

## 5. Step 3 — Deploy the Inference Worker on Hugging Face Spaces

### 5.1 Create a Hugging Face Space

1. Log in to https://huggingface.co
2. Click your profile avatar → **New Space**
3. Configure:

| Field | Value |
|-------|-------|
| **Space Name** | `ml-pipeline-worker` |
| **License** | `MIT` |
| **Space SDK** | **Docker** (this uses our custom Dockerfile) |
| **Space Hardware** | **CPU basic** (Free — 2 vCPU, 16 GB RAM) — verified no credit card needed |
| **Visibility** | `Public` (or `Private` if you prefer — still free) |

4. Click **Create Space**

### 5.2 Connect Your Repository

In your new Space:

1. Go to the **Settings** tab
2. Scroll to **Repository secrets** (formerly "Variables and secrets")
3. Add the following **secrets** (these become environment variables in the container):

| Secret Key | Secret Value |
|------------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Your Supabase `service_role` key |
| `MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english` (or any model) |
| `POLL_INTERVAL_SECONDS` | `15` |
| `MAX_RETRIES` | `3` |

4. Still in **Settings**, scroll to **Space Docker** and confirm:
   - **Dockerfile path**: `worker/Dockerfile`
   - **Build context**: `worker/`

### 5.3 Push Code to the Space

The Space needs the code. You have two options:

#### Option A: GitHub Sync (Recommended)

In the Space **Settings** → **GitHub Sync**:
- Enable "GitHub Sync"
- Select your `ml-pipeline` repository
- Set branch to `main`
- The Space will auto-deploy on every push to `main`

#### Option B: Manual Git Push

```bash
# Add the Space as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ml-pipeline-worker

# Push the worker directory
git subtree push --prefix worker hf main
```

### 5.4 Wait for Deployment

- The Space will build the Docker image (~5–10 minutes the first time due to PyTorch/transformers download)
- Once built, the worker starts automatically and begins polling Supabase

### 5.5 Verify the Worker

Check the **logs** in the Hugging Face Space:

1. Go to your Space → **Runtime** tab
2. Click **Factory** → **Reboot** if needed
3. Look at the logs. You should see:
   ```
   2026-06-15T14:32:00 | worker | INFO | Model loaded successfully: distilbert-base-uncased-finetuned-sst-2-english
   2026-06-15T14:32:01 | worker | INFO | Worker starting — id=...
   2026-06-15T14:32:01 | worker | DEBUG | No pending jobs (empty poll #1) — sleeping 15s
   ```

---

## 6. Step 4 — Configure GitHub Actions CI/CD

The CI/CD workflow is already defined in `.github/workflows/docker-build.yml`. It will:

- **Build** both Docker images on every push/PR
- **Run a smoke test** on the Gateway container
- **Verify imports** on the Worker container
- **Lint** Python code with flake8

### 6.1 Enable GitHub Actions

No additional configuration is needed. GitHub Actions is enabled by default on all repositories.

Just push to `main` to trigger the workflow:

```bash
git add .
git commit -m "ci: trigger initial pipeline build"
git push origin main
```

Then go to your repository on GitHub → **Actions** tab to see the workflow run.

---

## 7. Step 5 — Test the Full Pipeline End-to-End

### 7.1 Submit a Job

```bash
# Replace with your Render gateway URL
GATEWAY_URL="https://ml-pipeline-gateway.onrender.com"

# Base64-encode some text
INPUT_TEXT=$(echo -n "This movie is absolutely fantastic and wonderful!" | base64 -w0)

# Submit the job
curl -s -X POST "$GATEWAY_URL/submit-job" \
  -H "Content-Type: application/json" \
  -d "{
    \"input_payload\": \"$INPUT_TEXT\",
    \"input_format\": \"text/plain\"
  }"
```

Expected response (201 Created):
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "PENDING",
  "created_at": "2026-06-15T14:35:00+00:00"
}
```

### 7.2 Poll for the Result

```bash
# Replace with your actual job_id from above
JOB_ID="a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Poll until COMPLETED or FAILED
while true; do
  RESPONSE=$(curl -s "$GATEWAY_URL/job-status/$JOB_ID")
  STATUS=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATUS"

  if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "FAILED" ]; then
    echo "$RESPONSE" | python -m json.tool
    break
  fi
  sleep 5
done
```

Expected result (COMPLETED):
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "COMPLETED",
  "input_payload": "VGhpcyBtb3ZpZSBpcyBhYnNvbHV0ZWx5IGZhbnRhc3RpYyBhbmQgd29uZGVyZnVsIQ==",
  "input_format": "text/plain",
  "output_payload": "eyJsYWJlbCI6ICJQT1NJVElWRSIsICJzY29yZSI6IDAuOTk4fQ==",
  "output_format": "application/json",
  "error_message": null,
  "created_at": "2026-06-15T14:35:00+00:00",
  "started_at": "2026-06-15T14:35:10+00:00",
  "completed_at": "2026-06-15T14:35:12+00:00",
  "worker_id": "worker-abc123",
  "model_version": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

### 7.3 Decode the Output

```bash
# Decode the base64 output_payload to see the actual prediction
OUTPUT_B64=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['output_payload'])")
echo "$OUTPUT_B64" | base64 -d 2>/dev/null || python -c "import base64; print(base64.b64decode('$OUTPUT_B64').decode())"

# Expected:
# {"label": "POSITIVE", "score": 0.998}
```

---

## 8. Environment Variables Reference

### Gateway (Render)

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | ✅ | Supabase project URL (e.g., `https://xxxxxxxxxxxx.supabase.co`) |
| `SUPABASE_SERVICE_KEY` | ✅ | Supabase `service_role` key (full API access) |
| `JOBS_TABLE` | ❌ | Database table name (default: `jobs`) |

### Worker (Hugging Face Spaces)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SUPABASE_URL` | ✅ | — | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ✅ | — | Supabase `service_role` key |
| `JOBS_TABLE` | ❌ | `jobs` | Database table name |
| `MODEL_NAME` | ❌ | `distilbert-base-uncased-finetuned-sst-2-english` | Hugging Face model ID |
| `POLL_INTERVAL_SECONDS` | ❌ | `10` | Seconds between polling cycles (when no jobs) |
| `MAX_RETRIES` | ❌ | `3` | Max inference retries before marking job FAILED |
| `WORKER_ID` | ❌ | Hostname | Unique worker identifier (auto-set to machine hostname) |

---

## 9. Monitoring and Debugging

### Gateway Logs (Render)

1. Go to https://dashboard.render.com
2. Click on your `ml-pipeline-gateway` service
3. Click **Logs** tab
4. You can filter by severity (`info`, `error`, etc.)

### Worker Logs (Hugging Face Spaces)

1. Go to https://huggingface.co/spaces/YOUR_USERNAME/ml-pipeline-worker
2. Click the **Runtime** tab → **Factory** button
3. Scroll through the logs. Key lines to watch for:
   - `Model loaded successfully` — confirms the model downloaded and cached
   - `Discovered pending job` — worker found work
   - `Job X completed successfully` — inference finished
   - `Failed to claim job` — another worker may have grabbed it (harmless)

### Database (Supabase)

1. Go to your Supabase project dashboard
2. **Table Editor** → `jobs` — view all records directly
3. **SQL Editor** — run queries to inspect state:
   ```sql
   SELECT status, COUNT(*) FROM jobs GROUP BY status;
   SELECT * FROM jobs ORDER BY created_at DESC LIMIT 10;
   ```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Gateway returns 502 on `/submit-job` | Missing or incorrect `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` | Verify env vars in Render dashboard |
| Worker logs show `Failed to claim job` | Race condition with multiple workers | This is normal — the first claimer wins |
| First request after idle is slow | Render free tier cold-start | Wait up to 60 seconds; subsequent requests are fast |
| Worker shows `Model loaded successfully` but never processes jobs | Worker can't reach Supabase (network/firewall) | Check `SUPABASE_URL` is publicly accessible |
| `curl /submit-job` hangs | Render service is spinning up | Wait and retry; consider a keep-alive ping |

---

## 10. Cost Confirmation — Why This Stays Free

| Service | Plan | Monthly Free Allowance | Our Usage | Cost |
|---------|------|----------------------|-----------|------|
| **GitHub** | Free | Unlimited public repos, 2000 CI minutes/month | ~50 CI minutes/month | **$0** |
| **Supabase** | Free | 500 MB database, 2 GB bandwidth, 50,000 rows | Negligible (text payloads) | **$0** |
| **Render** | Free | 1 web service (spins down after inactivity), 512 MB RAM, 100 GB bandwidth | 1 web service | **$0** |
| **Hugging Face Spaces** | Free (CPU basic) | Unlimited public Spaces, 2 vCPU, 16 GB RAM, persistent storage | 1 Space | **$0** |
| **Total** | | | | **$0.00/month** |

### What happens at the free tier limits?

- **Supabase Free**: Projects are paused after 1 week of inactivity (can be unpaused from dashboard). 500 MB database is more than enough for text inference payloads.
- **Render Free**: Services spin down after 15 minutes of no traffic. They auto-restart on the next request (with a ~30–60s cold start). No usage cap beyond fair use.
- **Hugging Face Spaces Free**: Unlimited runtime for public Spaces. No spinning down. The CPU basic instance is more than sufficient for a transformer model like DistilBERT.
- **No credit card** is required at sign-up for any of these services.

**Verification**: All pricing pages were checked at the time of writing:
- https://supabase.com/pricing (Free tier: No credit card)
- https://render.com/pricing (Free tier: No credit card)
- https://huggingface.co/pricing (Free Spaces: No credit card)

---

## Quick-Start Summary

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/ml-pipeline.git
cd ml-pipeline

# 2. Deploy to Supabase (via SQL Editor):
#    - Run database/001_jobs_table.sql in Supabase SQL Editor

# 3. Deploy Gateway to Render:
#    - Create Web Service → Docker → root dir = gateway
#    - Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars

# 4. Deploy Worker to Hugging Face Spaces:
#    - Create Space → Docker → root dir = worker
#    - Set SUPABASE_URL, SUPABASE_SERVICE_KEY, MODEL_NAME env vars

# 5. Test it!
curl -X POST https://YOUR-RENDER-URL.onrender.com/submit-job \
  -H "Content-Type: application/json" \
  -d '{"input_payload": "VGhpcyBpcyBhbiBhbWF6aW5nIG1vdmllIQ==", "input_format": "text/plain"}'
```

---

*End of Deployment Guide. Happy building! 🚀*