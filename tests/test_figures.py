"""Figures and headline table from a synthetic evaluate_cascade report; no weights needed."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import evaluate_cascade  # noqa: E402
import make_figures  # noqa: E402

FAMILIES = ("dates", "counting", "rules", "negation", "arithmetic", "logic_grid")
OTHERS = (("banking20", "intent", 0.92), ("wanli", "nli", 0.70), ("bbh", "date_understanding", 0.55),
          ("mmlu_pro", "physics", 0.45))
CHECKPOINTS = (0, 128, 256, 512, 1024, 2048)


def synthetic_rows(directory: Path, seed: int = 5) -> dict:
    """Rows where S1 weakens with difficulty on owned families and S2 mostly recovers."""
    rng = np.random.default_rng(seed)
    specs = [("owned", family, 0.95 - 0.17 * (level - 1), level) for family in FAMILIES for level in (1, 2, 3, 4)]
    specs += [(source, family, accuracy, None) for source, family, accuracy in OTHERS]
    gold, cascade, router = [], [], []
    for source, family, s1_accuracy, level in specs:
        for index in range(40):
            row_id = f"{source}/{family}/{level}/{index}"
            truth = int(rng.integers(3))
            s1_right = rng.random() < s1_accuracy
            glance = float(np.clip(rng.normal(0.65 if s1_right else 0.45, 0.15), 0.01, 0.99))  # a weak pre-router
            router.append({"id": f"router/{row_id}", "option_ids": ["glance", "work"],
                           "probabilities": [glance, 1 - glance], "total_seconds": 0.05})
            s1 = rng.normal(0, 0.4, 3)
            s1[truth if s1_right else (truth + 1) % 3] += rng.uniform(0.5, 3.0) if s1_right else rng.uniform(0, 1.5)
            s2 = rng.normal(0, 0.3, 3)
            s2[truth if rng.random() < 0.9 else (truth + 2) % 3] += 4
            think = int(rng.integers(200, 1800))
            anytime = [{"t": t, "option_logits": ((1 - t / think) * s1 + t / think * s2).tolist()}
                       for t in CHECKPOINTS if t < think]
            gold.append({"id": row_id, "group_id": row_id, "family": family, "lang": "en", "difficulty": level,
                         "cascade_split": "dev" if index % 5 < 2 else "test", "cascade_source": source,
                         "state": "s", "question": f"{family} question", "label": truth,
                         "options": [{"id": x, "description": x} for x in "abc"]})
            cascade.append({"id": row_id, "option_ids": list("abc"), "option_logits": s1.tolist(),
                            "rotation_option_logits": [s1.tolist()] * 3, "debiased_logits": s1.tolist(),
                            "rotation_seconds": 0.01, "s2_option_logits": s2.tolist(), "s2_think_tokens": think,
                            "s2_closed_naturally": True, "s2_seconds": None, "prefill_seconds": 0.02,
                            "copy_seconds": 0.0, "s1_seconds": 0.01, "s2_anytime": anytime})
    paths = {"rows": directory / "rows.jsonl", "cascade": directory / "cascade.jsonl", "router": directory / "router.jsonl"}
    for name, data in (("rows", gold), ("cascade", cascade), ("router", router)):
        paths[name].write_text("".join(json.dumps(item) + "\n" for item in data))
    return paths


def synthetic_report(directory: Path, monkeypatch, router: bool = True) -> Path:
    paths = synthetic_rows(directory)
    report = directory / "report.json"
    argv = ["evaluate_cascade.py", "--rows", str(paths["rows"]), "--cascade", str(paths["cascade"]),
            "--report", str(report), "--bootstrap", "10"]
    monkeypatch.setattr(sys, "argv", argv + (["--router", str(paths["router"])] if router else []))
    evaluate_cascade.main()
    return report


def test_combined_policy_is_fit_on_dev_and_needs_the_router(tmp_path, monkeypatch):
    report = synthetic_report(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    assert data["policies"][-3:] == ["router_supervised", "combined", "oracle"]
    fit = data["combined_fit"]
    assert fit["features"] == ["confidence_rot", "router_zero_shot"] and fit["label"] == "s1_rot_correct"
    assert set(fit["standardized_weights"]) == {"confidence_rot", "router_zero_shot"}
    assert all(w > 0 for w in fit["standardized_weights"].values())  # both signals point toward S1 being right
    assert 0.5 < fit["dev_cv_auroc"] < 1.0
    assert "combined" in data["normalized_area_test"]["difference_vs_primary"]
    assert {"combined"} <= {p["policy"] for p in data["dev_selected_operating_points"]}
    assert {"combined"} <= data["curves_test"].keys() & data["cost_curves_test"].keys()
    assert all(0.0 <= r["combined"] <= 1.0 for r in data["row_scores"])
    # The accepted answer is the debiased S1 read, as under confidence_rot.
    assert evaluate_cascade._s1_key("combined") == evaluate_cascade._s1_key("confidence_rot")
    (tmp_path / "no_router").mkdir()
    without = synthetic_report(tmp_path / "no_router", monkeypatch, router=False)
    plain = json.loads(without.read_text())
    assert "combined" not in plain["policies"] and plain["combined_fit"] is None
    keys = {key for key, *_ in make_figures.combined_numbers(plain)}
    assert keys == set()  # old reports: nothing to write, nothing breaks


def test_cost_curve_runs_from_always_s2_to_always_s1():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    s1, s2, tokens = np.array([1, 1, 0, 0.0]), np.array([1, 0, 1, 1.0]), np.array([10, 20, 30, 40.0])
    accuracy, spent = evaluate_cascade.cost_curve(scores, s1, s2, tokens)
    assert accuracy.tolist() == pytest.approx([0.75, 0.75, 0.75, 0.5])
    assert spent.tolist() == pytest.approx([25.0, 22.5, 10.0, 0.0])


def test_figures_and_summary_from_a_report(tmp_path, monkeypatch):
    report = synthetic_report(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    assert {"cost_curves_test", "anytime_test"} <= data.keys()
    assert all("system_accuracy" in entry for entry in data["per_family_test"])
    out = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--report", str(report), "--label", "Synthetic",
                                      "--out", str(out), "--formats", "png"])
    make_figures.main()
    assert {path.name for path in out.iterdir()} == {"pareto.png", "tasks.png", "difficulty.png", "anytime.png",
                                                    "risk.png", "summary.md", "numbers.tex", "per_source.tex"}
    summary = (out / "summary.md").read_text()
    assert "Always S2 accuracy" in summary and "Gate share of always-S2 thinking" in summary
    headline = make_figures.headline(data)
    assert headline["always_s2_accuracy"] == data["baselines_test"]["always_s2_accuracy"]
    groups = make_figures._groups(data)
    assert {g["block"] for g in groups} == {make_figures.MULTI_STEP, make_figures.ROUTINE, "Public reasoning"}
    numbers = (out / "numbers.tex").read_text().splitlines()[2:]
    assert all(line.startswith("\\tjset{") and " % `" in line for line in numbers)
    keys = {line.split("}")[0].removeprefix("\\tjset{") for line in numbers}
    assert {"always_s2_accuracy", "gate_thinking_share", "block_all_system", "block_weak_escalation"} <= keys
    blocks = make_figures.blocks(data)
    assert blocks["all"]["rows"] == data["rows"]["test"]
    assert blocks["all"]["s2"] == pytest.approx(data["baselines_test"]["always_s2_accuracy"])
    assert {"h2_escalation", "s1_ltt_interval_at_10pct", "anytime_acc_t0", "disagree_weak_fix",
            "weak_d4_escalation", "difficulty_rising"} <= keys
    assert any(key.startswith("area_diff_") for key in keys) and any(key.startswith("area_ci_") for key in keys)
    assert {"area_combined", "area_diff_combined", "area_ci_combined", "combined_eps1_accuracy",
            "combined_eps1_escalation", "combined_eps1_share", "combined_eps0_share", "combined_eps2_share",
            "auroc_combined", "combined_weight_confidence_rot", "combined_weight_router_zero_shot"} <= keys
    assert {"s1_ece_uncalibrated", "s1_order_flip_rate", "temp_s1", "temp_t512", "temp_full", "gate_eps0_escalation",
            "gate_eps2_share", "anytime_eps1_accuracy", "lang_en_escalation", "stop_s1", "stop_checkpoint",
            "stop_full", "stop_checkpoint_of_escalated"} <= keys
    assert len(keys) == len(numbers)  # no key is written twice
    values = dict(re.findall(r"^\\tjset\{(\w+)\}\{(.*?)\} % ", "\n".join(numbers), re.M))
    stops = [float(values[k].rstrip("\\%")) for k in ("stop_s1", "stop_checkpoint", "stop_full")]
    assert sum(stops) == pytest.approx(100, abs=0.2)  # the replay accounts for every test row
    assert values["lang_en_rows"] == str(sum(e["rows"] for e in data["per_family_test"] if e["source"] == "owned"))
    per_source = (out / "per_source.tex").read_text().splitlines()[1:]
    assert sum(int(line.split(" & ")[4]) for line in per_source if " & " in line) == data["rows"]["test"]
    with pytest.raises(SystemExit):  # create-only without --regenerate
        make_figures.main()


def test_regenerate_adds_tables_and_latency(tmp_path, monkeypatch):
    report = synthetic_report(tmp_path, monkeypatch)
    data = json.loads(report.read_text())
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"id": r["id"], "source": r["cascade_source"],
                                            "split": r["cascade_split"]}) + "\n" for r in rows))
    latency = tmp_path / "latency.jsonl"
    latency.write_text("".join(json.dumps({"prefix_tokens": 100 + i, "prefill_seconds": 0.1, "copy_seconds": 0.004,
                                           "s1_seconds": 0.05, "rotation_seconds": 0.2, "s2_seconds": 30.0 + i,
                                           "s2_think_tokens": 1000}) + "\n" for i in range(3)))
    out = tmp_path / "figures"
    argv = ["make_figures.py", "--report", str(report), "--label", "Synthetic", "--out", str(out), "--formats", "png"]
    monkeypatch.setattr(sys, "argv", argv)
    make_figures.main()
    before = (out / "numbers.tex").read_text()
    monkeypatch.setattr(sys, "argv", argv + ["--regenerate", "--manifest", str(manifest), "--latency", str(latency)])
    make_figures.main()
    after = (out / "numbers.tex").read_text()
    assert set(before.splitlines()) <= set(after.splitlines())  # same report, same values
    assert "\\tjset{lat_s1_marginal_ms}{54}" in after and "\\tjset{lat_s2_seconds}{31.0}" in after
    row_id = next(r["id"] for r in data["row_scores"] if r["split"] == "test")
    monkeypatch.setattr(sys, "argv", argv + ["--regenerate", "--example", row_id,
                                             "--example-rows", str(tmp_path / "rows.jsonl")])
    make_figures.main()
    example = dict(re.findall(r"^\\tjset\{(\w+)\}\{(.*)\} % ", (out / "example.tex").read_text(), re.M))
    assert example["ex_id"].replace("\\_", "_") == row_id
    assert sum(float(example[f"ex_s1_p{x}"]) for x in "abc".upper()) == pytest.approx(1, abs=0.01)
    assert example["ex_stop"] in {"s1", "full"} | {f"t{t}" for t in CHECKPOINTS}
    table = (out / "data.tex").read_text().splitlines()
    assert table[-1] == f"All & & {data['rows']['dev']} & {data['rows']['test']} & & \\\\"


def test_risk_interval_brackets_the_rate():
    low, high = make_figures.risk_interval(0.05, 0.5, 2000)
    assert low < 0.05 < high
    assert make_figures.risk_interval(None, 0.0, 2000) is None
    assert make_figures.risk_interval(0.0, 0.1, 2000)[0] == 0.0
