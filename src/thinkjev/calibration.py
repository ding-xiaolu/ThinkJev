"""Scalar temperature scaling and calibration metrics.

Vendored from SemIf `benchmarks/calibrate.py` at commit
1f2dea3e25379f9dfc98cb83c324f00ab5deda37 (https://github.com/TheoLeeCJ/SemIf),
Copyright (c) 2026 TheoLeeCJ and contributors, MIT License; see THIRD_PARTY.md. Only the
functions ThinkJev uses are kept, unchanged. The S1 readout is vendored alongside it in
`_semif.py`.
"""
from __future__ import annotations

import math

import numpy as np


def check_temperature(temperature: float) -> float:
    """A temperature must be finite and positive; otherwise dividing logits can flip the argmax."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Temperature must be finite and > 0, got {temperature!r}")
    return temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def mean_nll(pairs: list[dict], temperature: float) -> float:
    total = 0.0
    for pair in pairs:
        probs = softmax(pair["logits"] / temperature)
        total += -math.log(max(probs[pair["true_index"]], 1e-12))
    return total / len(pairs)


def fit_temperature(pairs: list[dict], bounds=(0.05, 20.0), iterations=60) -> float:
    """Minimize mean NLL over T by golden-section search. NLL is convex in 1/T."""
    ratio = (math.sqrt(5) - 1) / 2
    low, high = bounds
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = mean_nll(pairs, left), mean_nll(pairs, right)
    for _ in range(iterations):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = mean_nll(pairs, left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = mean_nll(pairs, right)
    return (low + high) / 2


def scored(pairs: list[dict], temperature: float) -> list[dict]:
    """Per-row calibrated top-label confidence and correctness (correctness is T-invariant)."""
    check_temperature(temperature)
    out = []
    for pair in pairs:
        probs = softmax(pair["logits"] / temperature)
        row = {
            "id": pair["id"],
            "group": pair["group"],
            "family": pair["family"],
            "confidence": float(probs.max()),
            "correct": int(probs.argmax() == pair["true_index"]),
        }
        if "workload" in pair:
            row["workload"] = pair["workload"]
        out.append(row)
    return out


def reliability_bins(rows: list[dict], bins=10) -> list[dict]:
    result = []
    for index in range(bins):
        part = [r for r in rows if min(bins - 1, int(r["confidence"] * bins)) == index]
        if part:
            result.append(
                {
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "n": len(part),
                    "mean_confidence": sum(r["confidence"] for r in part) / len(part),
                    "accuracy": sum(r["correct"] for r in part) / len(part),
                }
            )
    return result


def ece(rows: list[dict], bins=10) -> float:
    """Expected calibration error, top-label, equal-width bins."""
    total = 0.0
    for entry in reliability_bins(rows, bins):
        total += abs(entry["accuracy"] - entry["mean_confidence"]) * entry["n"] / len(rows)
    return total


def fold_map(pairs: list[dict], folds=5, seed=217) -> dict:
    """Deterministic ``{group_id: fold}`` assignment (group-disjoint; variants share a group)."""
    groups = sorted({pair["group"] for pair in pairs})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return {group: index % folds for index, group in enumerate(groups)}
