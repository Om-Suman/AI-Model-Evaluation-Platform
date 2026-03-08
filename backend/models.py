"""
models.py
---------
Pydantic data-transfer objects (DTOs) used by FastAPI for:
  - Request body validation
  - Response serialisation
  - Internal data exchange between modules
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── Model catalogue ──────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Represents one LLM in the models table."""
    id:          int
    name:        str
    provider:    str
    model_id:    str
    description: Optional[str] = None
    created_at:  Optional[datetime] = None

    class Config:
        from_attributes = True


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DatasetMeta(BaseModel):
    """Metadata returned after a successful dataset upload."""
    id:          int
    name:        str
    filename:    str
    row_count:   int
    uploaded_at: Optional[datetime] = None


# ─── Evaluation request ───────────────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    """Body expected by POST /evaluate-model."""
    dataset_id: int  = Field(..., description="ID of the uploaded dataset")
    model_id:   int  = Field(..., description="ID of the model to evaluate")


# ─── Per-question result ──────────────────────────────────────────────────────

class QuestionResult(BaseModel):
    """Stores the outcome for a single question in the evaluation run."""
    question:       str
    ground_truth:   str
    model_response: str
    is_correct:     bool
    latency_ms:     float


# ─── Aggregate metrics ────────────────────────────────────────────────────────

class MetricSummary(BaseModel):
    """Aggregate scores for one evaluation run."""
    evaluation_id:  int
    accuracy:       float
    bleu_score:     float
    rouge1:         float
    rouge2:         float
    rougeL:         float
    avg_latency_ms: float


# ─── Full evaluation result (API response) ────────────────────────────────────

class EvaluationResult(BaseModel):
    """Complete result payload returned by GET /evaluation-results."""
    evaluation_id:   int
    dataset_name:    str
    model_name:      str
    status:          str
    total_questions: int
    started_at:      Optional[datetime]
    finished_at:     Optional[datetime]
    metrics:         Optional[MetricSummary] = None
    question_results: Optional[List[QuestionResult]] = None


# ─── Generic API response wrapper ────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool
    message: str
    data:    Optional[dict] = None
