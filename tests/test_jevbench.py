"""JevBench export and evaluation on a synthetic checkout; no weights and no network."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import evaluate_jevbench  # noqa: E402
import export_jevbench  # noqa: E402

TASKS = {
    "easy": [{"id": "easy-intent-00", "family": "intent", "group": "g-easy", "state": "Refund my order.",
              "question": {"type": "choice", "instructions": "Which intent?",
                           "criteria": {"refund": "Wants money back", "track": None}},
              "labels": ["refund", "track"], "expected": "refund"}],
    "original": [{"id": "original-policy-01-0", "family": "policy", "group": "g-orig",
                  "state": {"amount": 30}, "question": {"type": "noul", "instructions": "Is it permitted?",
                                                        "criteria": {"true": "Allowed", "false": "Denied"}},
                  "labels": ["no", "yes"], "expected": "no"}],
    "hard": [{"id": "hard-x-ordinal-01", "family": "ordinal", "group": "g-hard", "state": "Two users blocked.",
              "question": {"type": "score", "instructions": "Rate the impact.",
                           "criteria": ["None", "Minor", "Major", "Severe"]},
              "labels": ["0", "1", "2", "3"], "expected": 2}],
}
TIER = {"easy-intent-00": "easy", "original-policy-01-0": "standard", "hard-x-ordinal-01": "hard"}


def checkout(directory: Path, commit: str = export_jevbench.JEVBENCH_COMMIT) -> Path:
    (directory / "datasets" / "public").mkdir(parents=True)
    (directory / "COMMIT").write_text(commit + "\n")
    for name, tasks in TASKS.items():
        (directory / "datasets" / "public" / f"{name}.jsonl").write_text(
            "".join(json.dumps(task) + "\n" for task in tasks))
    per_task = {"revision": "v1.3.0", "tasks": [{"id": i, "tier": t, "public": True} for i, t in TIER.items()],
                "systems": {"jev-1.13.0": {"display": "Jev 1.13.0", "public_tasks": {
                    "easy-intent-00": ["c", 0.2], "original-policy-01-0": ["c", 0.2], "hard-x-ordinal-01": ["w", 0.3]}}}}
    (directory / "results" / "v1.2").mkdir(parents=True)
    (directory / evaluate_jevbench.PER_TASK).write_text(json.dumps(per_task))
    return directory


def test_rows_map_back_to_jevbench_labels(tmp_path):
    rows = {row["id"]: row for row in export_jevbench.export(checkout(tmp_path))}
    noul = rows["original-policy-01-0"]
    assert [o["id"] for o in noul["options"]] == ["true", "false"] and noul["jevbench_labels"] == ["yes", "no"]
    assert noul["label"] == 1 and noul["state"] == {"amount": 30}
    score = rows["hard-x-ordinal-01"]
    assert score["jevbench_labels"] == ["0", "1", "2", "3"] and score["label"] == 2
    choice = rows["easy-intent-00"]
    assert choice["options"][1] == {"id": "track", "description": "track"} and choice["label"] == 0
    assert {row["tier"] for row in rows.values()} == {"easy", "standard", "hard"}


def test_export_refuses_another_commit(tmp_path):
    with pytest.raises(ValueError, match="commit"):
        export_jevbench.export(checkout(tmp_path, commit="0" * 40))


def test_scoring_rule_and_mcnemar():
    import numpy as np

    assert evaluate_jevbench.jevbench_correct(np.array([0.5, 0.5]), ["yes", "no"], "no")  # tie: smallest label
    assert not evaluate_jevbench.jevbench_correct(np.array([0.5, 0.5]), ["yes", "no"], "yes")
    test = evaluate_jevbench.mcnemar([True] * 5 + [False], [False] * 5 + [False])
    assert (test["ours_only"], test["theirs_only"]) == (5, 0) and test["p_value"] == pytest.approx(0.0625)


def test_evaluate_applies_release_calibration(tmp_path):
    jb = checkout(tmp_path / "jb")
    rows = export_jevbench.export(jb)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # S1 sure and right on the easy row; unsure elsewhere. The noul row's S2 is right only at t=128.
    s1 = {"easy-intent-00": [4.0, 0.0], "original-policy-01-0": [0.2, 0.0], "hard-x-ordinal-01": [0.0, 0.1, 0.0, 0.0]}
    s2 = {"easy-intent-00": [3.0, 0.0], "original-policy-01-0": [2.0, 0.0], "hard-x-ordinal-01": [0.0, 0.0, 3.0, 0.0]}
    t128 = {"easy-intent-00": [3.0, 0.0], "original-policy-01-0": [0.0, 5.0], "hard-x-ordinal-01": [0.0, 0.0, 1.0, 0.0]}
    cascade = []
    for row in rows:
        k = len(row["options"])
        cascade.append({"id": row["id"], "option_ids": [o["id"] for o in row["options"]],
                        "option_logits": s1[row["id"]], "debiased_logits": s1[row["id"]],
                        "rotation_option_logits": [s1[row["id"]]] * k, "rotation_seconds": 0.01,
                        "s2_option_logits": s2[row["id"]], "s2_think_tokens": 300, "s2_closed_naturally": True,
                        "prefill_seconds": 0.1, "copy_seconds": 0.0, "s1_seconds": 0.01,
                        "s2_anytime": [{"t": 0, "option_logits": [0.0] * k}, {"t": 128, "option_logits": t128[row["id"]]}]})
    cascade_path = tmp_path / "cascade.jsonl"
    cascade_path.write_text("".join(json.dumps(pred) + "\n" for pred in cascade))
    report = {"temperature_rot": 1.0, "anytime_test": {
        "gate": "confidence_rot", "full_thought": {"temperature": 1.0},
        "per_checkpoint": [{"t": 0, "temperature": 1.0}, {"t": 128, "temperature": 1.0}],
        "dev_selected_operating_points": [{"tolerance": 0.01, "policy": "gate", "tau": 0.9},
                                          {"tolerance": 0.01, "policy": "anytime", "tau": 0.9},
                                          {"tolerance": 0.0, "policy": "gate", "tau": 0.1}]}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    result = evaluate_jevbench.evaluate(rows_path, cascade_path, report_path, jb, systems=("jev-1.13.0",))
    by_id = {row["id"]: row for row in result["rows"]}
    assert by_id["easy-intent-00"]["gate"]["think_tokens"] == 0 and by_id["easy-intent-00"]["gate"]["correct"]
    assert not by_id["original-policy-01-0"]["s2"]["correct"]  # S2 says yes, gold is no
    assert by_id["original-policy-01-0"]["anytime"] == {**by_id["original-policy-01-0"]["anytime"],
                                                       "correct": True, "think_tokens": 128}
    assert result["policies"]["s2"]["correct"] == 2 and result["policies"]["anytime"]["correct"] == 3
    assert result["policies"]["anytime"]["by_tier"]["hard"] == {"correct": 1, "n": 1, "accuracy": 1.0}
    pair = result["paired"]["anytime"]["jev-1.13.0"]
    assert (pair["all"]["ours_only"], pair["all"]["theirs_only"]) == (1, 0) and pair["hard"]["n"] == 1
    assert result["systems"]["jev-1.13.0"]["correct"] == 2
    pytest.importorskip("matplotlib")
    import make_figures

    out = tmp_path / "jevbench-report.json"
    out.write_text(json.dumps(result))
    keys = {key: make_figures._tex(value, kind) for key, value, kind, _ in make_figures.jevbench_numbers(out)}
    assert keys["jb_anytime_correct"] == "3" and keys["jb_sys_jev1130_correct"] == "2"
    assert keys["jb_anytime_vs_jev1130_all_ours"] == "1" and keys["jb_rows"] == "3"
