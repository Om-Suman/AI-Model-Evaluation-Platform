"""
evaluation_engine.py
--------------------
Orchestrates an end-to-end evaluation run:

  1. Load the CSV dataset from disk
  2. Look up the selected model's API provider / model-id from MySQL
  3. For each question, call the appropriate LLM (Together.ai or HuggingFace)
  4. Collect model responses + latencies
  5. Compute aggregate metrics via metrics.py
  6. Persist all results to MySQL (evaluations + metrics tables)
  7. Return a structured EvaluationResult

Provider routing:
  provider == 'together'     → Together.ai REST API  (via together SDK)
  provider == 'huggingface'  → HuggingFace Inference API (via huggingface_hub)
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv

from database import get_connection
from metrics import compute_all_metrics
from models import QuestionResult, MetricSummary, EvaluationResult

load_dotenv()

TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Maximum tokens the model should generate per answer
MAX_NEW_TOKENS = 256


# ─── LLM call helpers ─────────────────────────────────────────────────────────

def _call_together(model_id: str, question: str) -> tuple[str, float]:
    """
    Send one question to Together.ai and return (response_text, latency_ms).
    Uses the REST chat-completions endpoint via the `together` SDK.
    """
    from together import Together  # lazy import to avoid hard dep at module load

    client = Together(api_key=TOGETHER_API_KEY)

    start = time.monotonic()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": question}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,   # deterministic for reproducibility
    )
    latency_ms = (time.monotonic() - start) * 1000

    answer = response.choices[0].message.content.strip()
    return answer, latency_ms


def _call_huggingface(model_id: str, question: str) -> tuple[str, float]:
    """
    Send one question to the HuggingFace Inference API and return
    (response_text, latency_ms).
    """
    from huggingface_hub import InferenceClient  # lazy import

    client = InferenceClient(token=HUGGINGFACE_API_KEY)

    start = time.monotonic()
    response = client.text_generation(
        prompt=f"Question: {question}\nAnswer:",
        model=model_id,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.01,
    )
    latency_ms = (time.monotonic() - start) * 1000

    return response.strip(), latency_ms


def _call_llm(provider: str, model_id: str, question: str) -> tuple[str, float]:
    """Route to the correct LLM provider."""
    if provider == "together":
        return _call_together(model_id, question)
    elif provider == "huggingface":
        return _call_huggingface(model_id, question)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _get_model_info(model_db_id: int) -> dict:
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, name, provider, model_id FROM models WHERE id = %s",
        (model_db_id,),
    )
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        raise ValueError(f"Model with id={model_db_id} not found in database.")
    return row


def _get_dataset_info(dataset_id: int) -> dict:
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, name, filename FROM datasets WHERE id = %s",
        (dataset_id,),
    )
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        raise ValueError(f"Dataset with id={dataset_id} not found in database.")
    return row


def _create_evaluation_record(dataset_id: int, model_id: int) -> int:
    """Insert a new evaluation row and return its auto-generated id."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO evaluations (dataset_id, model_id, status, started_at)
           VALUES (%s, %s, 'running', %s)""",
        (dataset_id, model_id, datetime.utcnow()),
    )
    conn.commit()
    eval_id = cur.lastrowid
    cur.close(); conn.close()
    return eval_id


def _update_evaluation_done(eval_id: int, total_questions: int):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """UPDATE evaluations
           SET status='done', finished_at=%s, total_questions=%s
           WHERE id=%s""",
        (datetime.utcnow(), total_questions, eval_id),
    )
    conn.commit()
    cur.close(); conn.close()


def _update_evaluation_error(eval_id: int, error_msg: str):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "UPDATE evaluations SET status='error', finished_at=%s WHERE id=%s",
        (datetime.utcnow(), eval_id),
    )
    conn.commit()
    cur.close(); conn.close()


def _save_metrics(eval_id: int, metrics: dict):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO metrics
               (evaluation_id, accuracy, bleu_score, rouge1, rouge2, rougeL, avg_latency_ms)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (
            eval_id,
            metrics["accuracy"],
            metrics["bleu_score"],
            metrics["rouge1"],
            metrics["rouge2"],
            metrics["rougeL"],
            metrics["avg_latency_ms"],
        ),
    )
    conn.commit()
    cur.close(); conn.close()


# ─── Public API ───────────────────────────────────────────────────────────────

def run_evaluation(dataset_id: int, model_db_id: int) -> EvaluationResult:
    """
    Full evaluation pipeline.  Returns an EvaluationResult Pydantic model.

    Steps:
      1. Resolve model & dataset metadata from DB
      2. Read CSV from disk
      3. Create an evaluation record (status=running)
      4. Loop over rows → call LLM → collect results
      5. Compute aggregate metrics
      6. Persist metrics, mark evaluation as done
      7. Return structured result
    """
    # ── 1. Fetch metadata ────────────────────────────────────────────────────
    model_info   = _get_model_info(model_db_id)
    dataset_info = _get_dataset_info(dataset_id)

    # ── 2. Read CSV ──────────────────────────────────────────────────────────
    datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    csv_path     = os.path.join(datasets_dir, dataset_info["filename"])

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {"question", "ground_truth_answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["question", "ground_truth_answer"])

    # ── 3. Create evaluation record ──────────────────────────────────────────
    eval_id = _create_evaluation_record(dataset_id, model_db_id)

    question_results: List[QuestionResult] = []
    ground_truths:    List[str]            = []
    predictions:      List[str]            = []
    latencies_ms:     List[float]          = []

    try:
        # ── 4. Evaluate row by row ───────────────────────────────────────────
        for _, row in df.iterrows():
            question    = str(row["question"]).strip()
            ground_truth = str(row["ground_truth_answer"]).strip()

            try:
                response, latency_ms = _call_llm(
                    provider=model_info["provider"],
                    model_id=model_info["model_id"],
                    question=question,
                )
            except Exception as llm_err:
                # Soft fail: log and continue so one bad row doesn't abort run
                print(f"[EvalEngine] LLM call failed for question '{question[:60]}': {llm_err}")
                response   = ""
                latency_ms = 0.0

            is_correct = ground_truth.lower() == response.lower()

            question_results.append(QuestionResult(
                question=question,
                ground_truth=ground_truth,
                model_response=response,
                is_correct=is_correct,
                latency_ms=latency_ms,
            ))
            ground_truths.append(ground_truth)
            predictions.append(response)
            latencies_ms.append(latency_ms)

        # ── 5. Compute metrics ───────────────────────────────────────────────
        metrics_dict = compute_all_metrics(ground_truths, predictions, latencies_ms)

        # ── 6. Persist ───────────────────────────────────────────────────────
        _save_metrics(eval_id, metrics_dict)
        _update_evaluation_done(eval_id, len(question_results))

    except Exception as e:
        _update_evaluation_error(eval_id, str(e))
        raise

    # ── 7. Build and return result ───────────────────────────────────────────
    return EvaluationResult(
        evaluation_id=eval_id,
        dataset_name=dataset_info["name"],
        model_name=model_info["name"],
        status="done",
        total_questions=len(question_results),
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        metrics=MetricSummary(
            evaluation_id=eval_id,
            **metrics_dict,
        ),
        question_results=question_results,
    )
