"""Jev-style typed questions over the S1/S2 cascade: `ThinkJev.system_one(state, questions)`.

The request and response shapes follow TypeSafe's public HTTP API for `POST /v1/systemone`
(https://docs.typesafe.ai/api): a `state` plus named Choice, Noul, and Score questions in, one
typed answer per question out, with Jev's documented confidence formulas. ThinkJev is not
affiliated with TypeSafe; only the public request format is shared.

Each question becomes one SemIf row (state, criterion, options) and runs through
`CascadeScorer`: S1 reads the option logits; when the calibrated S1 confidence is below the
threshold, the same model thinks on the cached prefix and S2 reads the same slots. Reported
probabilities are calibrated with the temperature of whichever system answered.

Limits that differ from Jev: at most 16 options or levels per question (SemIf's answer slots),
questions run one after another, and `usage.output_tokens` counts thought tokens.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

from ._semif import load_causal_model, softmax, validate_row
from .cascade import CascadeScorer

THINK_MODES = ("auto", "never", "always")
CALIBRATIONS = Path(__file__).parent / "calibrations"
_POLICY = {"auto": "gated", "never": "never", "always": "always"}


class RequestError(ValueError):
    """A request that fails validation (HTTP 422)."""


def _text(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def question_row(question_id: str, state, question: dict) -> tuple[str, dict]:
    """One typed question as (type, SemIf row); option IDs are the keys answers are reported under."""
    if not isinstance(question, dict):
        raise RequestError(f"{question_id}: a question must be an object")
    kind, instructions = question.get("type"), question.get("instructions")
    if instructions in (None, "", [], {}):
        raise RequestError(f"{question_id}: instructions are required")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise RequestError(f"{question_id}: a choice needs a criteria map with at least two options")
        options = [{"id": str(key), "description": _text(value, str(key))} for key, value in criteria.items()]
    elif kind == "noul":
        criteria = criteria or {}
        if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
            raise RequestError(f"{question_id}: noul criteria may only have true and false")
        options = [{"id": "true", "description": _text(criteria.get("true"), "Yes")},
                   {"id": "false", "description": _text(criteria.get("false"), "No")}]
    elif kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise RequestError(f"{question_id}: a score needs an ordered criteria list with at least two levels")
        options = [{"id": str(level), "description": _text(value)} for level, value in enumerate(criteria)]
    else:
        raise RequestError(f"{question_id}: type must be choice, noul, or score")
    row = {"id": question_id, "state": state, "question": _text(instructions), "options": options}
    try:
        validate_row(row)
    except ValueError as error:
        raise RequestError(f"{question_id}: {error}") from error
    return kind, row


def choice_confidence(probabilities: list[float]) -> float:
    """(p_max - 1/n) / (1 - 1/n): 1 when all mass is on one option, 0 when spread evenly."""
    n = len(probabilities)
    return max(0.0, (max(probabilities) - 1 / n) / (1 - 1 / n))


def score_confidence(probabilities: list[float]) -> float:
    """1 - (mean level distance from the peak) / (the same for an even spread from the middle), floored at 0."""
    n = len(probabilities)
    peak = max(range(n), key=probabilities.__getitem__)
    spread = sum(p * abs(level - peak) for level, p in enumerate(probabilities))
    even = sum(abs(level - (n - 1) / 2) for level in range(n)) / n
    return max(0.0, 1 - spread / even)


def answer(kind: str, row: dict, probabilities: list[float]) -> dict:
    ids = [option["id"] for option in row["options"]]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities[ids.index("true")]}
    if kind == "choice":
        return {"type": "choice", "choice": ids[max(range(len(ids)), key=probabilities.__getitem__)],
                "probabilities": dict(zip(ids, probabilities)), "confidence": choice_confidence(probabilities)}
    return {"type": "score", "score": sum(level * p for level, p in enumerate(probabilities)),
            "legend": {option["id"]: option["description"] for option in row["options"]},
            "probabilities": dict(zip(ids, probabilities)), "confidence": score_confidence(probabilities)}


def presets() -> list[str]:
    """Names of the bundled calibrations (one per release model)."""
    return sorted(path.stem for path in CALIBRATIONS.glob("*.json"))


def preset(name: str) -> dict:
    """A bundled calibration: model, revision, calibration path, and the settings it was fit with."""
    path = CALIBRATIONS / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown preset {name!r}; bundled presets: {', '.join(presets())}")
    data = json.loads(path.read_text())
    return {"model": data["model"], "revision": data["revision"], "calibration": path, **data["settings"]}


def calibration_from_report(path: Path, rotations: bool, tolerance: float = 0.01) -> dict:
    """S1/S2 temperatures and the dev-selected gate threshold from an evaluate_cascade report.

    When the report holds an anytime readout, also its checkpoints, the dev temperature of each
    checkpoint, the dev-selected anytime tau, and the test accuracy at each budget (shown by the
    demo page as the reference curve).
    """
    report = json.loads(path.read_text())
    gate = "confidence_rot" if rotations else "confidence"
    s1 = report["temperature_rot"] if rotations else report["temperature"]
    if s1 is None or report.get("s2_temperature") is None:
        raise ValueError(f"{path}: report lacks the temperatures for this configuration")
    points = [point for point in report["dev_selected_operating_points"]
              if point["policy"] == gate and point["tolerance"] == tolerance]
    if not points or not math.isfinite(points[0]["threshold"]):
        raise ValueError(f"{path}: no finite dev-selected {gate} threshold at tolerance {tolerance}")
    fitted = {"s1_temperature": s1, "s2_temperature": report["s2_temperature"], "threshold": points[0]["threshold"],
              "source": str(path), "gate": gate, "tolerance": tolerance}
    anytime = report.get("anytime_test")
    if anytime and anytime.get("gate") == gate:
        taus = [p for p in anytime["dev_selected_operating_points"] if p["policy"] == "anytime" and p["tolerance"] == tolerance]
        fitted.update({
            "checkpoints": [c["t"] for c in anytime["per_checkpoint"]],
            "checkpoint_temperatures": {c["t"]: c["temperature"] for c in anytime["per_checkpoint"]},
            "anytime_tau": taus[0]["tau"] if taus else None,
            "reference_curve": [{"t": c["t"], "accuracy": c["accuracy"], "ece": c["ece"]} for c in anytime["per_checkpoint"]]
            + [{"t": "full", "accuracy": anytime["full_thought"]["accuracy"], "ece": anytime["full_thought"]["ece"]}],
        })
    return fitted


class ThinkJev:
    """One loaded model answering Jev-style questions; S2 runs only when S1 is unsure (think='auto')."""

    def __init__(self, model, tokenizer, metadata: dict, name: str, s1_temperature: float = 1.0,
                 s2_temperature: float = 1.0, threshold: float = 0.9, think_budget: int = 1024,
                 rotations: bool = False, max_tokens: int = 4096, checkpoints: tuple[int, ...] = (),
                 checkpoint_temperatures: dict | None = None, anytime_tau: float | None = None,
                 reference_curve: list | None = None):
        """`checkpoints`: thought lengths at which the answer slots are also read (the budget-t answer),
        each calibrated with `checkpoint_temperatures[t]` (default: the S2 temperature). `anytime_tau`
        is where the anytime policy would stop (default: the gate threshold); the thought always runs
        to its end here, so the demo shows every budget."""
        if not 0 < threshold <= 1:
            raise ValueError("Threshold must be in (0, 1]")
        if anytime_tau is not None and not 0 < anytime_tau <= 1:
            raise ValueError("anytime_tau must be in (0, 1]")
        temps = {int(t): float(value) for t, value in (checkpoint_temperatures or {}).items()}
        for value in (s1_temperature, s2_temperature, *temps.values()):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Temperatures must be finite and positive")
        self.name, self.metadata = name, metadata
        self.s1_temperature, self.s2_temperature, self.threshold = s1_temperature, s2_temperature, threshold
        self.checkpoints = tuple(sorted(int(t) for t in checkpoints))
        self.checkpoint_temperatures = {t: temps.get(t, s2_temperature) for t in self.checkpoints}
        self.anytime_tau = threshold if anytime_tau is None else anytime_tau
        self.reference_curve = reference_curve or []
        self.scorer = CascadeScorer(model, tokenizer, metadata, max_tokens, think_budget, "gated",
                                    s1_temperature, threshold, rotations, checkpoints=self.checkpoints)
        self.lock = threading.Lock()

    def info(self) -> dict:
        """What the demo page needs to draw: thresholds, checkpoints, budget, and the reference curve."""
        return {"model": self.name, "threshold": self.threshold, "anytime_tau": self.anytime_tau,
                "checkpoints": list(self.checkpoints), "think_budget": self.scorer.think_budget,
                "rotations": self.scorer.rotations, "reference_curve": self.reference_curve,
                "calibration": (self.metadata.get("calibration") or {}).get("source")}

    @classmethod
    def load(cls, model: str, revision: str, name: str | None = None, device: str = "auto",
             dtype: str = "bfloat16", calibration: str | Path | None = None, **settings) -> "ThinkJev":
        loaded, tokenizer, metadata = load_causal_model(model, revision, device, dtype)
        if calibration is not None:
            fitted = calibration_from_report(Path(calibration), settings.get("rotations", False))
            keys = ("s1_temperature", "s2_temperature", "threshold", "checkpoints", "checkpoint_temperatures",
                    "anytime_tau", "reference_curve")
            settings = {**{key: fitted[key] for key in keys if key in fitted}, **settings}
            metadata = {**metadata, "calibration": {k: v for k, v in fitted.items() if k != "reference_curve"}}
        return cls(loaded, tokenizer, metadata, name or f"thinkjev-{Path(model).name.lower()}", **settings)

    @classmethod
    def from_preset(cls, name: str, **settings) -> "ThinkJev":
        """A release model with its bundled calibration, e.g. `ThinkJev.from_preset("qwen3.5-4b")`.

        Keyword arguments override the preset's settings (rotations, think_budget) and the fitted values.
        """
        fixed = preset(name)
        model, revision = fixed.pop("model"), fixed.pop("revision")
        return cls.load(model, revision, **{**fixed, **settings})

    def _calibrated(self, ids: list[str], logits: list[float], temperature: float) -> tuple[dict, float]:
        probabilities = softmax([value / temperature for value in logits])
        return dict(zip(ids, probabilities)), max(probabilities)

    def decide(self, kind: str, row: dict, think: str = "auto", on_event=None) -> tuple[dict, dict]:
        """One question: the typed answer and what ThinkJev did to produce it.

        `on_event` receives calibrated events as they happen: `s1` (probabilities, confidence,
        threshold, whether it thinks), `thinking` (tokens so far), and `checkpoint` (t, the
        budget-t probabilities and confidence, and whether the anytime policy would stop here).
        """
        if think not in THINK_MODES:
            raise RequestError(f"think must be one of {', '.join(THINK_MODES)}")
        ids = [option["id"] for option in row["options"]]
        stopped = []

        def relay(event: dict) -> None:
            if event["type"] == "s1":
                probabilities, _ = self._calibrated(ids, event["debiased_logits"] or event["option_logits"],
                                                    self.s1_temperature)
                on_event({"type": "s1", "probabilities": probabilities, "confidence": event["s1_confidence"],
                          "threshold": self.threshold, "escalate": event["escalated"], "seconds": event["seconds"]})
            elif event["type"] == "checkpoint":
                probabilities, confidence = self._calibrated(ids, event["option_logits"],
                                                             self.checkpoint_temperatures[event["t"]])
                stop = confidence >= self.anytime_tau and not stopped
                if stop:
                    stopped.append(event["t"])
                on_event({"type": "checkpoint", "t": event["t"], "probabilities": probabilities,
                          "confidence": confidence, "anytime_stop": stop})
            else:
                on_event(event)

        with self.lock:
            self.scorer.policy = _POLICY[think]
            result = self.scorer.score(row, relay if on_event else None)
        s1_logits = result.get("debiased_logits") or result["option_logits"]
        s1 = softmax([value / self.s1_temperature for value in s1_logits])
        trace = {"system": "s1", "s1_confidence": result["s1_confidence"], "threshold": self.threshold,
                 "s1_probabilities": dict(zip(ids, s1)),
                 "input_tokens": result["prefix_tokens"] + result["s1_suffix_tokens"],
                 "think_tokens": 0, "seconds": result["total_seconds"]}
        probabilities = s1
        if result["escalated"]:
            probabilities = softmax([value / self.s2_temperature for value in result["s2_option_logits"]])
            trace.update({"system": "s2", "think_tokens": result["s2_think_tokens"],
                          "thought_closed": result["s2_closed_naturally"], "thought": result["s2_thought"]})
            if self.checkpoints:
                reads = []
                for read in result.get("s2_anytime", []):
                    read_probabilities, confidence = self._calibrated(ids, read["option_logits"],
                                                                      self.checkpoint_temperatures[read["t"]])
                    reads.append({"t": read["t"], "probabilities": read_probabilities, "confidence": confidence})
                stop = next((read["t"] for read in reads if read["confidence"] >= self.anytime_tau), "full")
                trace.update({"anytime": reads, "anytime_tau": self.anytime_tau, "anytime_stop": stop,
                              "s2_confidence": max(probabilities)})
        return answer(kind, row, probabilities), trace

    def system_one(self, state, questions: dict, think: str = "auto", include_thought: bool = False,
                   on_event=None) -> dict:
        """A Jev-format response for `questions` over `state`, plus a `thinkjev` map of per-question traces.

        `on_event`, if given, is called with each question's events (tagged with `question`) as they
        happen, and with an `answer` event when a question is done.
        """
        if not isinstance(questions, dict) or not questions:
            raise RequestError("questions must be a nonempty map")
        typed = {question_id: question_row(question_id, state, question) for question_id, question in questions.items()}
        answers, traces, prompt_tokens, thought_tokens = {}, {}, 0, 0
        for question_id, (kind, row) in typed.items():
            tagged = (lambda event, q=question_id: on_event({"question": q, **event})) if on_event else None
            answers[question_id], trace = self.decide(kind, row, think, tagged)
            if tagged:
                shown = {k: v for k, v in trace.items() if include_thought or k != "thought"}
                tagged({"type": "answer", "answer": answers[question_id], "trace": shown})
            if not include_thought:
                trace.pop("thought", None)
            traces[question_id] = trace
            prompt_tokens += trace["input_tokens"]
            thought_tokens += trace["think_tokens"]
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": thought_tokens},
                "thinkjev": traces}
