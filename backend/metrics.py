"""
metrics.py
----------
Pure computation layer – no I/O, no DB calls.

Functions:
  compute_accuracy   – exact-match accuracy across all questions
  compute_bleu       – corpus-level BLEU using the HuggingFace evaluate library
  compute_rouge      – ROUGE-1, ROUGE-2, ROUGE-L (F1) using evaluate
  compute_all_metrics – convenience wrapper that returns a single dict
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

# HuggingFace evaluate (wraps sacrebleu + rouge_score under the hood)
import evaluate as hf_evaluate


# Pre-load metrics once at module import (saves repeated disk hits)
_bleu_metric  = hf_evaluate.load("sacrebleu")
_rouge_metric = hf_evaluate.load("rouge")


# ─── Accuracy ─────────────────────────────────────────────────────────────────

def compute_accuracy(
    ground_truths: List[str],
    predictions:   List[str],
) -> float:
    """
    Case-insensitive exact-match accuracy.

    Args:
        ground_truths: List of reference answers.
        predictions:   List of model-generated answers.

    Returns:
        Float in [0.0, 1.0].
    """
    if not ground_truths:
        return 0.0

    correct = sum(
        gt.strip().lower() == pred.strip().lower()
        for gt, pred in zip(ground_truths, predictions)
    )
    return round(correct / len(ground_truths), 4)


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def compute_bleu(
    ground_truths: List[str],
    predictions:   List[str],
) -> float:
    """
    Corpus-level BLEU score (sacrebleu, score in [0, 100] normalised to [0, 1]).

    sacrebleu expects:
      predictions  – list of strings
      references   – list of lists of strings (multiple refs per prediction)
    """
    if not predictions:
        return 0.0

    references = [[gt] for gt in ground_truths]   # one ref per prediction
    result = _bleu_metric.compute(
        predictions=predictions,
        references=references,
    )
    # sacrebleu returns score in 0–100 range; normalise to 0–1
    return round(result["score"] / 100.0, 4)


# ─── ROUGE ────────────────────────────────────────────────────────────────────

def compute_rouge(
    ground_truths: List[str],
    predictions:   List[str],
) -> Tuple[float, float, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns:
        Tuple (rouge1, rouge2, rougeL) each in [0.0, 1.0].
    """
    if not predictions:
        return 0.0, 0.0, 0.0

    result = _rouge_metric.compute(
        predictions=predictions,
        references=ground_truths,
        use_stemmer=True,
    )
    return (
        round(result["rouge1"], 4),
        round(result["rouge2"], 4),
        round(result["rougeL"], 4),
    )


# ─── Latency helpers ──────────────────────────────────────────────────────────

def compute_avg_latency(latencies_ms: List[float]) -> float:
    """Return mean latency in milliseconds, rounded to 2 dp."""
    if not latencies_ms:
        return 0.0
    return round(float(np.mean(latencies_ms)), 2)


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def compute_all_metrics(
    ground_truths: List[str],
    predictions:   List[str],
    latencies_ms:  List[float],
) -> dict:
    """
    Run all metric calculations and return a single result dict.

    Keys:
        accuracy, bleu_score, rouge1, rouge2, rougeL, avg_latency_ms
    """
    rouge1, rouge2, rougeL = compute_rouge(ground_truths, predictions)

    return {
        "accuracy":       compute_accuracy(ground_truths, predictions),
        "bleu_score":     compute_bleu(ground_truths, predictions),
        "rouge1":         rouge1,
        "rouge2":         rouge2,
        "rougeL":         rougeL,
        "avg_latency_ms": compute_avg_latency(latencies_ms),
    }
