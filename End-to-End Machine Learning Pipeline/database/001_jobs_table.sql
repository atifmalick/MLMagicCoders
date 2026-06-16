-- =============================================================
-- Database Migration: 001_jobs_table
-- Description: Creates the core `jobs` table for tracking
-- inference job state through the pipeline lifecycle.
-- Target: Supabase PostgreSQL (Free Tier)
-- =============================================================

-- Enable pgcrypto for gen_random_uuid() if not already enabled
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- | Status Enum: Ensures only valid states are stored
DO $$ BEGIN
    CREATE TYPE job_status AS ENUM (
        'PENDING',
        'IN_PROGRESS',
        'COMPLETED',
        'FAILED'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- | Main Jobs Table
CREATE TABLE IF NOT EXISTS jobs (
    -- Primary Key: UUID v4 (secure, non-sequential)
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Input payload: base64-encoded input data (e.g., image bytes, text)
    input_payload   TEXT NOT NULL,

    -- Input format hint for the worker (e.g. "image/png", "text/plain")
    input_format    VARCHAR(64) NOT NULL DEFAULT 'text/plain',

    -- Current lifecycle status
    status          job_status NOT NULL DEFAULT 'PENDING',

    -- Output payload: base64-encoded result artifact (NULL until COMPLETED)
    output_payload  TEXT,

    -- Output format hint for the client (e.g. "image/png", "application/json")
    output_format   VARCHAR(64),

    -- Error message preserved on FAILURE
    error_message   TEXT,

    -- Processing metadata
    retry_count     SMALLINT NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    worker_id       VARCHAR(128),           -- Identifies which worker instance processed the job
    model_version   VARCHAR(64),            -- Model/version used for inference

    -- Timestamps (Supabase fills created_at automatically if configured)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,            -- When a worker picked up the job
    completed_at    TIMESTAMPTZ             -- When processing finished (success or failure)
);

-- | Indexes – critical for polling performance

-- 1. Status + created_at: The worker polls for PENDING jobs ordered by age
CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs (status, created_at ASC)
    WHERE status = 'PENDING';

-- 2. Primary lookup for /job-status/{job_id}
CREATE INDEX IF NOT EXISTS idx_jobs_id
    ON jobs (id);

-- 3. Allows quick status-based aggregations
CREATE INDEX IF NOT EXISTS idx_jobs_status_only
    ON jobs (status);

-- | Row-Level Security (RLS) — disabled by default on Supabase
-- If you enable RLS later, these policies protect the table.
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

-- Allow only authenticated servers with the service_role key to read/write.
-- The service_role key bypasses RLS by default; these policies act as a
-- safety net if you accidentally lower the security level.
CREATE POLICY "service_role_all_access"
    ON jobs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- An anonymous read-only policy if you ever need public status checks
-- (uncomment if desired)
-- CREATE POLICY "anon_read_completed"
--     ON jobs
--     FOR SELECT
--     TO anon
--     USING (status = 'COMPLETED');

-- | Auto-update completed_at trigger
CREATE OR REPLACE FUNCTION update_completed_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('COMPLETED', 'FAILED') AND OLD.status NOT IN ('COMPLETED', 'FAILED') THEN
        NEW.completed_at = now();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_completed_at ON jobs;
CREATE TRIGGER trg_jobs_completed_at
    BEFORE UPDATE OF status
    ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_completed_at();

-- | Auto-update started_at trigger
CREATE OR REPLACE FUNCTION update_started_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'IN_PROGRESS' AND OLD.status = 'PENDING' THEN
        NEW.started_at = now();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_started_at ON jobs;
CREATE TRIGGER trg_jobs_started_at
    BEFORE UPDATE OF status
    ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_started_at();