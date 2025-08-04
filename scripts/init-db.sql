-- Database initialization script for Call Analysis System
-- This script is run when the PostgreSQL container starts for the first time

-- Create the main database (if not exists)
SELECT 'CREATE DATABASE call_analysis'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'call_analysis')\gexec

-- Connect to the call_analysis database
\c call_analysis;

-- Create extensions that might be useful
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches
CREATE EXTENSION IF NOT EXISTS "unaccent"; -- For text normalization

-- Create a read-only user for reporting/analytics
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analytics_reader') THEN
        CREATE ROLE analytics_reader WITH LOGIN PASSWORD 'analytics_readonly_password';
    END IF;
END
$$;

-- Create indexes that will be useful for common queries
-- Note: The actual tables will be created by Alembic migrations

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analytics_reader;

-- Log initialization completion
\echo 'Database initialization completed successfully'