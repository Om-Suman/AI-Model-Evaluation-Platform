"""
database.py
-----------
Handles all MySQL connectivity and schema management.

Responsibilities:
  - Load credentials from environment
  - Provide a reusable connection helper
  - Create all required tables on first run (idempotent)

Tables created:
  models      – catalogue of supported LLMs
  datasets    – metadata for every uploaded CSV
  evaluations – one row per evaluation run
  metrics     – per-run aggregate metric scores
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# ─── Connection config from environment ───────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "ai_eval_platform"),
}


def get_connection():
    """Return a live mysql.connector connection.  Raises on failure."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        raise RuntimeError(f"[DB] Connection failed: {e}") from e


# ─── DDL statements ───────────────────────────────────────────────────────────

_CREATE_DB = f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']};"

_CREATE_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    id          INT          AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(120) NOT NULL UNIQUE,
    provider    VARCHAR(60)  NOT NULL,           -- 'together' | 'huggingface'
    model_id    VARCHAR(255) NOT NULL,           -- API identifier string
    description TEXT,
    created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    id          INT          AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    filename    VARCHAR(255) NOT NULL,
    row_count   INT          NOT NULL DEFAULT 0,
    uploaded_at DATETIME     DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS evaluations (
    id               INT     AUTO_INCREMENT PRIMARY KEY,
    dataset_id       INT     NOT NULL,
    model_id         INT     NOT NULL,
    status           VARCHAR(30) DEFAULT 'pending',  -- pending|running|done|error
    total_questions  INT     DEFAULT 0,
    started_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    finished_at      DATETIME,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (model_id)   REFERENCES models(id)
);
"""

_CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id              INT     AUTO_INCREMENT PRIMARY KEY,
    evaluation_id   INT     NOT NULL,
    accuracy        FLOAT,
    bleu_score      FLOAT,
    rouge1          FLOAT,
    rouge2          FLOAT,
    rougeL          FLOAT,
    avg_latency_ms  FLOAT,
    recorded_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
);
"""

# Pre-seeded models (INSERT IGNORE keeps it idempotent)
_SEED_MODELS = """
INSERT IGNORE INTO models (name, provider, model_id, description) VALUES
('Llama-3 8B',    'together',     'meta-llama/Llama-3-8b-chat-hf',
 'Meta Llama 3 8B instruction-tuned chat model via Together.ai'),
('Llama-3 70B',   'together',     'meta-llama/Llama-3-70b-chat-hf',
 'Meta Llama 3 70B instruction-tuned chat model via Together.ai'),
('Mistral 7B',    'together',     'mistralai/Mistral-7B-Instruct-v0.2',
 'Mistral 7B instruct model via Together.ai'),
('Mixtral 8x7B',  'together',     'mistralai/Mixtral-8x7B-Instruct-v0.1',
 'Mixtral 8x7B MoE model via Together.ai'),
('Gemma 7B',      'together',     'google/gemma-7b-it',
 'Google Gemma 7B instruct model via Together.ai'),
('Gemma 2B',      'huggingface',  'google/gemma-2b-it',
 'Google Gemma 2B instruct model via HuggingFace Inference API');
"""


def init_db():
    """
    Create the database (if missing) and all tables, then seed model catalogue.
    Call once at application startup.
    """
    # First connect without specifying a database to create it
    cfg_no_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    try:
        conn = mysql.connector.connect(**cfg_no_db)
        cur  = conn.cursor()
        cur.execute(_CREATE_DB)
        conn.commit()
        cur.close()
        conn.close()
    except Error as e:
        raise RuntimeError(f"[DB] Could not create database: {e}") from e

    # Now connect with the target database and create tables
    conn = get_connection()
    cur  = conn.cursor()
    for ddl in [_CREATE_MODELS, _CREATE_DATASETS, _CREATE_EVALUATIONS, _CREATE_METRICS]:
        cur.execute(ddl)
    cur.execute(_SEED_MODELS)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Schema initialised successfully.")
