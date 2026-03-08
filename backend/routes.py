"""
routes.py
---------
All FastAPI route handlers.  Imported and registered in main.py.

Endpoints:
  POST /upload-dataset      – accept a CSV file, store metadata in MySQL
  POST /evaluate-model      – trigger an evaluation run (synchronous)
  GET  /evaluation-results  – list all past evaluations with their metrics
  GET  /evaluation-results/{id} – detail for one evaluation
  GET  /models              – list all available LLMs
  GET  /datasets            – list all uploaded datasets
  GET  /health              – liveness check
"""

from __future__ import annotations

import os
import shutil
from typing import List

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
import pandas as pd

from database import get_connection
from evaluation_engine import run_evaluation
from models import (
    APIResponse,
    DatasetMeta,
    EvaluationRequest,
    EvaluationResult,
    MetricSummary,
    ModelInfo,
)

router = APIRouter()

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}


# ─── Models ───────────────────────────────────────────────────────────────────

@router.get("/models", response_model=List[ModelInfo], tags=["Models"])
def list_models():
    """Return all LLMs stored in the models table."""
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM models ORDER BY name")
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows


# ─── Datasets ─────────────────────────────────────────────────────────────────

@router.post("/upload-dataset", response_model=DatasetMeta, tags=["Datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    name: str        = Form(...),
):
    """
    Accept a CSV with columns [question, ground_truth_answer].
    Save the file to disk and record metadata in MySQL.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Save to datasets directory
    dest_path = os.path.join(DATASETS_DIR, file.filename)
    with open(dest_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Validate CSV structure
    try:
        df = pd.read_csv(dest_path)
    except Exception as e:
        os.remove(dest_path)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = {"question", "ground_truth_answer"} - set(df.columns)
    if missing:
        os.remove(dest_path)
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: question, ground_truth_answer. Missing: {missing}",
        )

    row_count = len(df.dropna(subset=["question", "ground_truth_answer"]))

    # Persist metadata
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (name, filename, row_count) VALUES (%s, %s, %s)",
        (name, file.filename, row_count),
    )
    conn.commit()
    dataset_id = cur.lastrowid
    cur.close(); conn.close()

    return DatasetMeta(
        id=dataset_id,
        name=name,
        filename=file.filename,
        row_count=row_count,
    )


@router.get("/datasets", response_model=List[DatasetMeta], tags=["Datasets"])
def list_datasets():
    """Return all uploaded datasets."""
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows


# ─── Evaluations ──────────────────────────────────────────────────────────────

@router.post("/evaluate-model", response_model=EvaluationResult, tags=["Evaluation"])
def evaluate_model(request: EvaluationRequest):
    """
    Run a full evaluation of the specified model on the specified dataset.
    This call is synchronous – it returns once the run is complete.

    For large datasets consider wrapping this with BackgroundTasks or Celery.
    """
    try:
        result = run_evaluation(
            dataset_id=request.dataset_id,
            model_db_id=request.model_id,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


@router.get("/evaluation-results", tags=["Evaluation"])
def list_evaluation_results():
    """
    Return all past evaluation runs with their metrics.
    Joins evaluations + metrics + models + datasets tables.
    """
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT
            e.id            AS evaluation_id,
            d.name          AS dataset_name,
            m.name          AS model_name,
            e.status,
            e.total_questions,
            e.started_at,
            e.finished_at,
            mt.accuracy,
            mt.bleu_score,
            mt.rouge1,
            mt.rouge2,
            mt.rougeL,
            mt.avg_latency_ms
        FROM evaluations e
        JOIN datasets    d  ON d.id  = e.dataset_id
        JOIN models      m  ON m.id  = e.model_id
        LEFT JOIN metrics mt ON mt.evaluation_id = e.id
        ORDER BY e.started_at DESC
        """
    )
    rows = cur.fetchall()
    cur.close(); conn.close()

    # Serialise datetime objects for JSON
    for row in rows:
        for key in ("started_at", "finished_at"):
            if row[key]:
                row[key] = row[key].isoformat()

    return rows


@router.get("/evaluation-results/{evaluation_id}", tags=["Evaluation"])
def get_evaluation_detail(evaluation_id: int):
    """Return metrics for a single evaluation run."""
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT
            e.id            AS evaluation_id,
            d.name          AS dataset_name,
            m.name          AS model_name,
            e.status,
            e.total_questions,
            e.started_at,
            e.finished_at,
            mt.accuracy,
            mt.bleu_score,
            mt.rouge1,
            mt.rouge2,
            mt.rougeL,
            mt.avg_latency_ms
        FROM evaluations e
        JOIN datasets    d  ON d.id  = e.dataset_id
        JOIN models      m  ON m.id  = e.model_id
        LEFT JOIN metrics mt ON mt.evaluation_id = e.id
        WHERE e.id = %s
        """,
        (evaluation_id,),
    )
    row = cur.fetchone()
    cur.close(); conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_id} not found.")

    for key in ("started_at", "finished_at"):
        if row[key]:
            row[key] = row[key].isoformat()

    return row
