"""Exploratory JevBench check: ThinkJev's policies on JevBench's public items, paired with
JevBench's published per-task outcomes of other systems (Jev 1.13.0 among them).

    python benchmarks/evaluate_jevbench.py --rows build/jevbench/rows.jsonl \
        --cascade results/jevbench-qwen3.5-4b-b4096-cascade.jsonl \
        --report results/qwen3.5-4b-b4096-report.json --jevbench build/jevbench-src \
        --output results/jevbench-qwen3.5-4b-report-v2.json

Nothing is fit here. The S1 (rotation) and S2 temperatures, the anytime checkpoint temperatures,
and the gate and anytime thresholds come from the model's release report (dev-selected at
tolerance 0.01, `anytime_test.dev_selected_operating_points`), so on JevBench they are a
cross-domain transfer. Policies: `s1` (rotation-debiased S1), `s2` (always think), `gate`, and
`anytime`, as in `evaluate_cascade.py`.

Scoring follows JevBench (`jevbench/scoring.py`): an item is correct when the argmax label equals
the expected label, ties going to the lexicographically smallest label. Other systems enter only
through `results/v1.2/jevbench-v1.2-per-task.json` in the pinned checkout ('c' is correct); this
script calls no external API. Each policy is paired with each system by an exact two-sided
McNemar test, on all public items and on the hard tier. Outputs are create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate_cascade  # noqa: E402
from export_jevbench import JEVBENCH_COMMIT  # noqa: E402
from thinkjev import calibration  # noqa: E402

PER_TASK = Path("results/v1.2/jevbench-v1.2-per-task.json")
# The two thinking APIs (most accurate listed systems) were added after the first reports were read
# and are exploratory; those reports carry only the first five.
SYSTEMS = ("jev-1.13.0", "openjev-thinking", "djev-thinking", "semif-qwen3.5-4b", "reflex-27b",
           "deepseek-flash", "gpt-5.6-luna")
POLICIES = ("s1", "s2", "gate", "anytime")
GATE = "confidence_rot"
TOLERANCE = 0.01


def jevbench_correct(probabilities: np.ndarray, labels: list[str], expected: str) -> bool:
    """JevBench's argmax rule: the highest label wins, ties to the lexicographically smallest."""
    best, best_p = None, -1.0
    for label, p in sorted(zip(labels, probabilities.tolist())):
        if p > best_p:
            best, best_p = label, p
    return best == expected


def mcnemar(ours: list[bool], theirs: list[bool]) -> dict:
    """Exact two-sided McNemar test: b = only ours right, c = only theirs right."""
    b = sum(1 for x, y in zip(ours, theirs) if x and not y)
    c = sum(1 for x, y in zip(ours, theirs) if y and not x)
    p = 1.0 if b + c == 0 else min(1.0, 2 * evaluate_cascade.binomial_cdf(min(b, c), b + c, 0.5))
    return {"n": len(ours), "ours_only": b, "theirs_only": c, "p_value": p}


def calibration_points(report: dict) -> dict:
    anytime = report["anytime_test"]
    if anytime["gate"] != GATE:
        raise ValueError(f"report's anytime gate is {anytime['gate']}, expected {GATE}")
    taus = {point["policy"]: point["tau"] for point in anytime["dev_selected_operating_points"]
            if point["tolerance"] == TOLERANCE}
    return {"s1_temperature": report["temperature_rot"], "s2_temperature": anytime["full_thought"]["temperature"],
            "stages": [(point["t"], point["temperature"]) for point in anytime["per_checkpoint"]],
            "gate_tau": taus["gate"], "anytime_tau": taus["anytime"]}


def policy_reads(item: dict, cal: dict) -> dict[str, tuple[np.ndarray, int, bool]]:
    """Per policy: the calibrated probabilities it answers with, the thought tokens it spends, and
    whether it thought at all (an anytime stop at t = 0 thinks but spends no tokens)."""
    s1 = (calibration.softmax(item["logits_rot"] / cal["s1_temperature"]), 0, False)
    s2 = (calibration.softmax(item["s2_logits"] / cal["s2_temperature"]), item["think_tokens"], True)
    accept = lambda tau: s1[0].max() >= tau  # noqa: E731
    anytime = s1 if accept(cal["anytime_tau"]) else s2
    if not accept(cal["anytime_tau"]):
        for t, temperature in cal["stages"]:
            if t >= item["think_tokens"]:
                break
            probs = calibration.softmax(item["anytime"][t] / temperature)
            if probs.max() >= cal["anytime_tau"]:
                anytime = (probs, t, True)
                break
    return {"s1": s1, "s2": s2, "gate": s1 if accept(cal["gate_tau"]) else s2, "anytime": anytime}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(rows_path: Path, cascade_path: Path, report_path: Path, checkout: Path,
             systems: tuple[str, ...] = SYSTEMS) -> dict:
    commit = (checkout / "COMMIT").read_text().strip()
    if commit != JEVBENCH_COMMIT:
        raise ValueError(f"{checkout}: JevBench commit {commit}, expected {JEVBENCH_COMMIT}")
    rows = {row["id"]: row for row in evaluate_cascade._jsonl(rows_path)}
    items = evaluate_cascade.load(rows_path, cascade_path, None)
    report = json.loads(report_path.read_text())
    cal = calibration_points(report)
    per_task = json.loads((checkout / PER_TASK).read_text())
    tiers = {task["id"]: task["tier"] for task in per_task["tasks"] if task["public"]}
    if set(tiers) != set(rows) or len(items) != len(rows):
        raise ValueError(f"{len(items)} scored rows, {len(rows)} rows, {len(tiers)} JevBench public items")

    outcomes = {policy: [] for policy in POLICIES}
    per_row = []
    for item in items:
        row = rows[item["id"]]
        expected = row["jevbench_labels"][row["label"]]
        entry = {"id": item["id"], "tier": tiers[item["id"]]}
        for policy, (probs, tokens, thought) in policy_reads(item, cal).items():
            correct = jevbench_correct(probs, row["jevbench_labels"], expected)
            outcomes[policy].append({"correct": correct, "confidence": float(probs.max()), "tokens": tokens,
                                     "thought": thought})
            entry[policy] = {"correct": correct, "think_tokens": tokens,
                             "probabilities": dict(zip(row["jevbench_labels"], np.round(probs, 6).tolist()))}
        per_row.append(entry)

    # The same policies as evaluate_cascade scores them (index argmax): they must agree but for ties.
    for item in items:
        item["confidence_rot"] = float(calibration.softmax(item["logits_rot"] / cal["s1_temperature"]).max())
    reference = {"gate": evaluate_cascade.gate_outcomes(items, GATE, cal["gate_tau"]),
                 "anytime": evaluate_cascade.anytime_outcomes(items, GATE, cal["stages"], cal["anytime_tau"])}
    for policy, (correct, tokens) in reference.items():
        if [o["tokens"] for o in outcomes[policy]] != tokens.astype(int).tolist():
            raise ValueError(f"{policy}: thought tokens differ from evaluate_cascade")
        if sum(o["correct"] != bool(c) for o, c in zip(outcomes[policy], correct)) > 0:
            print(f"note: {policy} differs from index argmax on tied rows")

    order = [item["id"] for item in items]
    hard = [tiers[i] == "hard" for i in order]

    def accuracy(flags: list[bool]) -> dict:
        by_tier = {}
        for tier in sorted(set(tiers.values())):
            part = [f for f, i in zip(flags, order) if tiers[i] == tier]
            by_tier[tier] = {"correct": int(sum(part)), "n": len(part), "accuracy": float(np.mean(part))}
        return {"correct": int(sum(flags)), "n": len(flags), "accuracy": float(np.mean(flags)), "by_tier": by_tier}

    policies = {}
    for policy, rows_out in outcomes.items():
        flags = [o["correct"] for o in rows_out]
        policies[policy] = {**accuracy(flags),
                            "ece": calibration.ece([{"confidence": o["confidence"], "correct": o["correct"]}
                                                    for o in rows_out]),
                            "mean_think_tokens": float(np.mean([o["tokens"] for o in rows_out])),
                            "escalation_rate": float(np.mean([o["thought"] for o in rows_out]))}

    others, paired = {}, {policy: {} for policy in POLICIES}
    for name in systems:
        system = per_task["systems"][name]
        theirs = [system["public_tasks"][i][0] == "c" for i in order]
        others[name] = {"display": system.get("display", name), **accuracy(theirs)}
        for policy in POLICIES:
            ours = [o["correct"] for o in outcomes[policy]]
            paired[policy][name] = {
                "all": mcnemar(ours, theirs),
                "hard": mcnemar([x for x, h in zip(ours, hard) if h], [y for y, h in zip(theirs, hard) if h])}

    return {"schema": "thinkjev-jevbench-v1",
            "inputs": {"rows": str(rows_path), "cascade": str(cascade_path), "report": str(report_path),
                       "jevbench_commit": commit, "per_task_file": str(PER_TASK),
                       "per_task_revision": per_task.get("revision"),
                       "sha256": {"cascade": _sha256(cascade_path), "report": _sha256(report_path),
                                  "per_task": _sha256(checkout / PER_TASK)}},
            "calibration": {**{k: v for k, v in cal.items() if k != "stages"},
                            "stages": [{"t": t, "temperature": temp} for t, temp in cal["stages"]],
                            "source": "release report, dev-selected at tolerance 0.01; nothing fit on JevBench"},
            "policies": policies, "systems": others, "paired": paired, "rows": per_row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--cascade", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="The model's release evaluate_cascade report")
    parser.add_argument("--jevbench", type=Path, required=True, help="JevBench checkout with a COMMIT file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; outputs are create-only")
    result = evaluate(args.rows, args.cascade, args.report, args.jevbench)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for policy, entry in result["policies"].items():
        jev = result["paired"][policy]["jev-1.13.0"]
        print(f"{policy:8s} {entry['correct']}/{entry['n']}  hard {entry['by_tier']['hard']['correct']}/"
              f"{entry['by_tier']['hard']['n']}  vs Jev: +{jev['all']['ours_only']}/-{jev['all']['theirs_only']} "
              f"p={jev['all']['p_value']:.3f}")


if __name__ == "__main__":
    main()
