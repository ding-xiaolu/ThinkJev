"""Release figures and a headline table from one `evaluate_cascade.py` report.

    python benchmarks/make_figures.py --report results/qwen3.5-4b-b4096-report.json \
        --label "Qwen3.5-4B" --out results/figures/qwen3.5-4b

Writes into a new directory; `--regenerate` rewrites an existing one from the same report (the
directory is derived, the report is the result):

- `pareto`: test accuracy against mean thought tokens per decision for the gate, the anytime
  policy, the pre-fork router, and the oracle, with always-S1 and always-S2 marked and the
  gate's dev-selected operating point;
- `tasks`: per task group, S1, S2, and ThinkJev accuracy beside the share of rows that thought;
- `difficulty`: escalation rate and S1 accuracy by difficulty level, one panel per owned family;
- `anytime`: S2 accuracy and calibration error at each think budget;
- `risk`: target versus realized test risk of S1 answers accepted under the Learn-then-Test threshold;
- `summary.md`: the headline numbers, each traceable to a report field;
- `numbers.tex`: the same numbers plus everything the technical report quotes (per-block accuracy and
  escalation, H2, every risk target with its interval, every anytime checkpoint, H1 areas with
  their bootstrap intervals, disagreement per block, per-group and per-level accuracy, the
  rotation ablation and uncalibrated ECE, the dev temperature of every readout, the gate and
  anytime policy at each dev tolerance, block C by language, and where the anytime policy stops)
  as `\tjset` lines, each commented with its report field;
- `per_source.tex`: table rows per source, language, and difficulty level from `per_family_test`;
- with `--manifest`, `data.tex`: table rows of dev and test counts per source;
- with `--latency`, latency medians from a one-process scorer run, added to `numbers.tex`;
- with `--jevbench`, the `evaluate_jevbench.py` report's accuracies and McNemar counts as `jb_*` keys;
- with `--example ID --example-rows FILE`, `example.tex`: one row's text, its calibrated S1 and
  per-checkpoint probabilities, and where the anytime policy stops, for the paper's overview figure.

Every number comes from the report (or the named manifest, latency, or JevBench file); nothing is recomputed
from rows. Two quantities are derived: the exact binomial interval of a realized risk, computed
from the report's accepted share and risk, and where the anytime policy stops (S1, a checkpoint,
or the full thought), replayed from the report's temperatures and tau on the cascade file the
report names; the replay must reproduce the report's mean thought tokens or it is refused. Needs matplotlib (the
`figures` extra). Colors are fixed per entity across figures: ThinkJev's gate blue, the anytime
policy orange, the pre-fork router aqua, references gray.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from evaluate_cascade import clopper_pearson_upper

SURFACE, INK, MUTED, FAINT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a1", "#e8e7e3"
GATE, ANYTIME, ROUTER = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (light)

# Rows from these sources are labeled by family: multi-step decisions whose families follow the
# failure modes TypeSafe lists for Jev 1.13 (https://docs.typesafe.ai/model-jaggedness/jev-1.13),
# plus chess. Display names only; the block keys below ("jev", "weak") stay as they are.
OWNED = {"owned", "hard", "chess"}
ROUTINE, MULTI_STEP = "Routine decisions", "Multi-step decisions"
BLOCKS = {
    "authored": ROUTINE, "wanli": ROUTINE, "every": ROUTINE, "banking20": ROUTINE,
    "clinc_intent": ROUTINE, "clinc_escalate": ROUTINE, "typed_decisions": ROUTINE,
    "bbh": "Public reasoning", "mmlu_pro": "Public reasoning", "gpqa_diamond": "Public reasoning",
    "super_gpqa": "Public reasoning", "musr": "Public reasoning",
}
BLOCK_ORDER = (MULTI_STEP, ROUTINE, "Public reasoning", "Other")
FAMILY_NAMES = {
    "dates": "Date comparison", "date_deadline": "Date deadlines", "counting": "Counting",
    "arithmetic": "Arithmetic", "rules": "Chained rules", "chained_rules": "Chained rules",
    "negation": "Stacked negation", "stacked_negation": "Stacked negation", "logic_grid": "Logic grid",
    "graph": "Graph paths", "scheduling": "Scheduling", "chess": "Chess",
    "double_negation": "Double negation (hard200)", "multihop_rules": "Multi-hop rules (hard200)",
}
SOURCE_NAMES = {
    "authored": "SemIf authored", "wanli": "NLI (WANLI)", "every": "Every judgments",
    "banking20": "Intent routing (BANKING77)", "clinc_intent": "Intent routing (CLINC)",
    "clinc_escalate": "Escalation (CLINC)", "typed_decisions": "Typed decisions", "bbh": "BIG-Bench Hard",
    "mmlu_pro": "MMLU-Pro", "gpqa_diamond": "GPQA Diamond", "super_gpqa": "SuperGPQA", "musr": "MuSR",
}
OWNED_NAMES = {"owned": "Generated families", "hard": "hard200 fixture", "chess": "Chess (Lichess)"}
# As stated in benchmarks/DATA_SOURCES.md; the generated sources are this repository's (MIT).
LICENSES = {
    "authored": "MIT", "wanli": "CC-BY-4.0", "every": "unstated", "banking20": "CC-BY-4.0",
    "clinc_intent": "CC-BY-3.0", "clinc_escalate": "CC-BY-3.0", "typed_decisions": "Apache-2.0",
    "bbh": "MIT", "mmlu_pro": "MIT", "gpqa_diamond": "CC-BY-4.0", "super_gpqa": "ODC-BY-1.0",
    "musr": "CC-BY-4.0", "owned": "MIT", "hard": "MIT", "chess": "CC0-1.0",
}
RISKS = (0.01, 0.02, 0.05, 0.1)


def _rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman correlation without scipy (average ranks for ties)."""
    import numpy as np

    def ranks(values):
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values))
        out[order] = np.arange(len(values))
        for value in set(values):
            tied = [i for i, v in enumerate(values) if v == value]
            out[tied] = out[tied].mean()
        return out

    if len(xs) < 3:
        return float("nan")
    a, b = ranks(list(xs)), ranks(list(ys))
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": FAINT, "axes.labelcolor": MUTED, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 1.8, "lines.markersize": 6, "svg.fonttype": "none",
    })


def _percent(axis, which: str = "y") -> None:
    from matplotlib.ticker import PercentFormatter

    getattr(axis, f"{which}axis").set_major_formatter(PercentFormatter(1.0, decimals=0))


def _point(report: dict, policy: str, tolerance: float = 0.01) -> dict | None:
    anytime = report.get("anytime_test")
    if not anytime:
        return None
    return next((p for p in anytime["dev_selected_operating_points"]
                 if p["policy"] == policy and p["tolerance"] == tolerance), None)


def gate_policy(report: dict) -> str:
    return report["per_family_test_gate"]["policy"]


def headline(report: dict) -> dict:
    """The numbers the README and paper quote, with the report field each comes from."""
    base = report["baselines_test"]
    gate = gate_policy(report)
    out = {
        "rows_test": report["rows"]["test"],
        "always_s1_accuracy": base["always_s1_accuracy"],
        "always_s2_accuracy": base["always_s2_accuracy"],
        "always_s2_mean_think_tokens": base["s2_mean_think_tokens"],
        "s1_wrong_s2_right": base["s1_wrong_s2_right"],
        "s1_right_s2_wrong": base["s1_right_s2_wrong"],
        "gate": gate,
    }
    for policy in ("gate", "anytime"):
        point = _point(report, policy)
        if point:
            out[f"{policy}_test_accuracy"] = point["test_accuracy"]
            out[f"{policy}_mean_think_tokens"] = point["test_mean_think_tokens"]
            out[f"{policy}_thinking_share"] = point["test_mean_think_tokens"] / base["s2_mean_think_tokens"]
    view = "rotation" if gate == "confidence_rot" else "raw"
    readout = report["s1_readouts_test"][view]
    out["s1_ece"] = readout["ece"]
    risk = next((r for r in readout["coverage_at_risk"] if r["risk"] == 0.05), None)
    if risk:
        out["s1_ltt_coverage_at_5pct"] = risk["ltt_test_coverage"]
        out["s1_ltt_test_risk_at_5pct"] = risk["ltt_test_risk"]
    return out


def write_summary(report: dict, label: str, path: Path) -> None:
    h = headline(report)
    lines = [f"# {label}: headline numbers", "",
             f"Test rows: {h['rows_test']}. Every number below is read from the evaluate_cascade report.", "",
             "| Quantity | Value | Report field |", "|---|---:|---|",
             f"| Always S1 accuracy | {h['always_s1_accuracy']:.1%} | `baselines_test.always_s1_accuracy` |",
             f"| Always S2 accuracy | {h['always_s2_accuracy']:.1%} | `baselines_test.always_s2_accuracy` |",
             f"| Always S2 mean thought tokens | {h['always_s2_mean_think_tokens']:.0f} | `baselines_test.s2_mean_think_tokens` |",
             f"| S1 wrong, S2 right / S1 right, S2 wrong | {h['s1_wrong_s2_right']} / {h['s1_right_s2_wrong']} | `baselines_test` |"]
    for policy, name in (("gate", "Gate"), ("anytime", "Anytime")):
        if f"{policy}_test_accuracy" in h:
            field = f"`anytime_test.dev_selected_operating_points` ({policy}, tolerance 0.01)"
            lines += [f"| {name} accuracy (dev-selected, always-S2 − 1 pt) | {h[f'{policy}_test_accuracy']:.1%} | {field} |",
                      f"| {name} share of always-S2 thinking | {h[f'{policy}_thinking_share']:.0%} | same, ÷ always-S2 tokens |"]
    lines.append(f"| S1 ECE (calibrated) | {h['s1_ece']:.3f} | `s1_readouts_test.*.ece` |")
    if "s1_ltt_coverage_at_5pct" in h:
        risk = h["s1_ltt_test_risk_at_5pct"]
        lines.append(f"| S1 coverage at 5% target risk (LTT) / realized risk | {h['s1_ltt_coverage_at_5pct']:.1%} / "
                     f"{'—' if risk is None else f'{risk:.1%}'} | `s1_readouts_test.*.coverage_at_risk` |")
    path.write_text("\n".join(lines) + "\n")


BLOCK_KEYS = {MULTI_STEP: "weak", ROUTINE: "jev", "Public reasoning": "public", "Other": "other"}


def blocks(report: dict) -> dict:
    """Row-weighted S1, S2, gate-system accuracy and escalation per block and overall, from `per_family_test`."""
    totals = defaultdict(lambda: {"rows": 0, "s1": 0.0, "s2": 0.0, "system": 0.0, "escalation": 0.0})
    for g in _groups(report):
        for key in (BLOCK_KEYS.get(g["block"], "other"), "all"):
            totals[key]["rows"] += g["rows"]
            for field in ("s1", "s2", "system", "escalation"):
                totals[key][field] += g[field] * g["rows"]
    return {key: {"rows": t["rows"], **{f: t[f] / t["rows"] for f in ("s1", "s2", "system", "escalation")}}
            for key, t in totals.items()}


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "$-$")


def _tex(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "" if kind in ("pctint", "decint") else "--"  # an absent interval follows a "--" value
    if kind == "pctint":
        return f"[{100 * value[0]:.1f}, {100 * value[1]:.1f}]\\%"
    if kind == "decint":
        return f"[{_signed(value[0])}, {_signed(value[1])}]"
    return {"pct": f"{100 * value:.1f}\\%", "share": f"{100 * value:.0f}\\%", "int": f"{value:.0f}",
            "ece": f"{value:.3f}", "dec": f"{value:.3f}", "dec2": f"{value:.2f}", "signed": _signed(value), "pp": f"{100 * value:.1f}",
            "ms": f"{1000 * value:.0f}", "sec": f"{value:.1f}",
            "p": "$p < 0.001$" if value < 0.001 else f"$p = {value:.3f}$"}[kind]


def block_key(source: str) -> str:
    return "weak" if source in OWNED else BLOCK_KEYS.get(BLOCKS.get(source, "Other"), "other")


def risk_interval(risk: float | None, coverage: float, rows: int, level: float = 0.95):
    """Exact two-sided binomial interval of a realized risk over the accepted test rows.

    The accepted count and errors are recovered from the report's share and rate; rows are treated
    as independent, so the interval ignores grouping and is narrower than a group-level one.
    """
    accepted = round(coverage * rows)
    if risk is None or accepted == 0:
        return None
    errors = round(risk * accepted)
    alpha = (1 - level) / 2
    lower = 0.0 if errors == 0 else 1 - clopper_pearson_upper(accepted - errors, accepted, alpha)
    return lower, clopper_pearson_upper(errors, accepted, alpha)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _difficulty_cells(report: dict) -> dict:
    """{family name: {level: [rows, escalated rows, S1-correct rows, S2-correct rows]}} for owned rows."""
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0.0]))
    for entry in report["per_family_test"]:
        if entry["source"] in OWNED and entry.get("difficulty") is not None:
            cell = cells[FAMILY_NAMES.get(entry["family"], entry["family"])][entry["difficulty"]]
            cell[0] += entry["rows"]
            cell[1] += entry["escalation_rate"] * entry["rows"]
            cell[2] += entry["s1_accuracy"] * entry["rows"]
            cell[3] += entry["s2_accuracy"] * entry["rows"]
    return cells


def _rising(cells: dict) -> list[float]:
    """Rank correlation of escalation with difficulty level, per family."""
    return [_rank_correlation(sorted(cells[name]), [cells[name][level][1] / cells[name][level][0]
                                                    for level in sorted(cells[name])]) for name in sorted(cells)]


def analysis_numbers(report: dict) -> list[tuple]:
    """(key, value, kind, report field) rows beyond the headline: H2, H3, H5, H1, power, groups."""
    base, rows_test, gate = report["baselines_test"], report["rows"]["test"], gate_policy(report)
    out = [("oracle_accuracy", base["oracle_best_accuracy"], "pct", "`baselines_test.oracle_best_accuracy`"),
           ("s2_closed_naturally", base["s2_closed_naturally_rate"], "pct",
            "`baselines_test.s2_closed_naturally_rate`"),
           ("disagree_total", base["s1_wrong_s2_right"] + base["s1_right_s2_wrong"], "int",
            "`baselines_test.s1_wrong_s2_right` + `s1_right_s2_wrong`")]
    point = next((p for p in report.get("dev_selected_operating_points", [])
                  if p["policy"] == gate and p["tolerance"] == 0.01), None)
    if point:
        field = f"`dev_selected_operating_points` ({gate}, tolerance 0.01, {{}})"
        out += [("h2_accuracy", point["test_accuracy"], "pct", field.format("test_accuracy")),
                ("h2_gap", base["always_s2_accuracy"] - point["test_accuracy"], "pp",
                 field.format("always-S2 accuracy minus test_accuracy, in points")),
                ("h2_escalation", 1 - point["test_coverage"], "pct", field.format("1 - test_coverage"))]
    for view in ("raw", "rotation"):
        readout = report["s1_readouts_test"][view]
        out += [(f"s1_{view}_accuracy", readout["accuracy"], "pct", f"`s1_readouts_test.{view}.accuracy`"),
                (f"s1_{view}_ece", readout["ece"], "ece", f"`s1_readouts_test.{view}.ece`")]
    view = "rotation" if gate == "confidence_rot" else "raw"
    readouts = report["s1_readouts_test"]
    out.append(("s1_ece_uncalibrated", readouts[view]["ece_t1"], "ece", f"`s1_readouts_test.{view}.ece_t1`"))
    if "rotation_batch_prior" in readouts:
        out += [("s1_prior_accuracy", readouts["rotation_batch_prior"]["accuracy"], "pct",
                 "`s1_readouts_test.rotation_batch_prior.accuracy`"),
                ("s1_prior_ece", readouts["rotation_batch_prior"]["ece"], "ece",
                 "`s1_readouts_test.rotation_batch_prior.ece`")]
    flip = (report.get("order_flip_rate_test") or {}).get("all")
    if flip is not None:
        out.append(("s1_order_flip_rate", flip, "pct", "`order_flip_rate_test.all`"))
    out.append(("temp_s1", readouts[view]["temperature"], "dec2", f"`s1_readouts_test.{view}.temperature`"))
    entries = {e["risk"]: e for e in readouts[view]["coverage_at_risk"]}
    certified = 0
    for risk in RISKS:
        e = entries.get(risk)
        if not e:
            continue
        name, field = f"{round(100 * risk)}pct", f"`s1_readouts_test.{view}.coverage_at_risk` (risk {risk}, {{}})"
        interval = risk_interval(e["ltt_test_risk"], e["ltt_test_coverage"], rows_test)
        certified += e["ltt_test_coverage"] > 0
        out += [(f"s1_ltt_coverage_at_{name}", e["ltt_test_coverage"], "pct", field.format("ltt_test_coverage")),
                (f"s1_ltt_test_risk_at_{name}", e["ltt_test_risk"], "pct", field.format("ltt_test_risk")),
                (f"s1_ltt_interval_at_{name}", interval, "pctint",
                 field.format("exact binomial 95% interval of ltt_test_risk over ltt_test_coverage x rows.test"))]
    out.append(("s1_ltt_targets_certified", certified, "int",
                f"`s1_readouts_test.{view}.coverage_at_risk` (targets with ltt_test_coverage > 0)"))
    anytime = report.get("anytime_test")
    if anytime:
        for c in anytime["per_checkpoint"]:
            field = f"`anytime_test.per_checkpoint` (t = {c['t']}, {{}})"
            out += [(f"anytime_acc_t{c['t']}", c["accuracy"], "pct", field.format("accuracy")),
                    (f"anytime_ece_t{c['t']}", c["ece"], "ece", field.format("ece")),
                    (f"anytime_tokens_t{c['t']}", c["mean_think_tokens"], "int", field.format("mean_think_tokens"))]
        full = anytime["full_thought"]
        out += [("anytime_acc_full", full["accuracy"], "pct", "`anytime_test.full_thought.accuracy`"),
                ("anytime_ece_full", full["ece"], "ece", "`anytime_test.full_thought.ece`")]
        out += [(f"temp_t{c['t']}", c["temperature"], "dec2", f"`anytime_test.per_checkpoint` (t = {c['t']}, temperature)")
                for c in anytime["per_checkpoint"]]
        out.append(("temp_full", full["temperature"], "dec2", "`anytime_test.full_thought.temperature`"))
        out += tolerance_numbers(report)
        out += stop_numbers(report)
    out += language_numbers(report)
    out += combined_numbers(report)
    for policy, value in (report.get("auroc_s1_correct_test") or {}).items():
        out.append((f"auroc_{policy}", value, "dec", f"`auroc_s1_correct_test.{policy}`"))
    area = report.get("normalized_area_test")
    if area:
        for policy, value in area["area"].items():
            out.append((f"area_{policy}", value, "dec", f"`normalized_area_test.area.{policy}`"))
        for policy, diff in area["difference_vs_primary"].items():
            field = f"`normalized_area_test.difference_vs_primary.{policy}` ({area['primary']} minus policy, {{}})"
            out += [(f"area_diff_{policy}", diff["primary_minus_policy"], "signed", field.format("primary_minus_policy")),
                    (f"area_ci_{policy}", diff["ci95"], "decint", field.format("ci95"))]
    by_source = (report.get("disagreement_test") or {}).get("by_source", {})
    totals = defaultdict(lambda: [0, 0])
    for source, counts in by_source.items():
        totals[block_key(source)][0] += counts["s1_wrong_s2_right"]
        totals[block_key(source)][1] += counts["s1_right_s2_wrong"]
    for key, (fix, broke) in sorted(totals.items()):
        field = f"`disagreement_test.by_source` (summed over block '{key}', {{}})"
        out += [(f"disagree_{key}_fix", fix, "int", field.format("s1_wrong_s2_right")),
                (f"disagree_{key}_break", broke, "int", field.format("s1_right_s2_wrong"))]
    for g in _groups(report):
        field = f"`per_family_test` ({{}}, row-weighted over group '{g['name']}')"
        slug = _slug(g["name"])
        out.append((f"grp_{slug}_rows", g["rows"], "int", field.format("rows")))
        for name, source in (("s1", "s1_accuracy"), ("s2", "s2_accuracy"), ("system", "system_accuracy"),
                             ("escalation", "escalation_rate")):
            out.append((f"grp_{slug}_{name}", g[name], "pct", field.format(source)))
    cells = _difficulty_cells(report)
    if cells:
        rising = _rising(cells)
        field = "`per_family_test` (owned families, rank correlation of escalation_rate with difficulty {})"
        out += [("difficulty_families", len(cells), "int", field.format("computed per family")),
                ("difficulty_rising", sum(r > 0 for r in rising), "int", field.format("> 0"))]
        levels = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for family in cells.values():
            for level, cell in family.items():
                levels[level] = [a + b for a, b in zip(levels[level], cell)]
        for level, (n, escalated, s1, s2) in sorted(levels.items()):
            field = f"`per_family_test` ({{}}, row-weighted over owned families at difficulty {level})"
            out += [(f"weak_d{level}_rows", n, "int", field.format("rows")),
                    (f"weak_d{level}_escalation", escalated / n, "pct", field.format("escalation_rate")),
                    (f"weak_d{level}_s1", s1 / n, "pct", field.format("s1_accuracy")),
                    (f"weak_d{level}_s2", s2 / n, "pct", field.format("s2_accuracy"))]
    return out


def jevbench_numbers(path: Path) -> list[tuple]:
    """`jb_*` keys from an evaluate_jevbench report: each policy on JevBench's public items, the
    systems it is paired with (from JevBench's published per-task outcomes), and McNemar counts."""
    report = json.loads(path.read_text())
    name = f"`{path.name}`"
    rows = []
    for policy, entry in report["policies"].items():
        field = f"{name} `policies.{policy}`"
        hard = entry["by_tier"]["hard"]
        rows += [(f"jb_{policy}_correct", entry["correct"], "int", field + " (correct)"),
                 (f"jb_{policy}_acc", entry["accuracy"], "pct", field + " (accuracy)"),
                 (f"jb_{policy}_hard_correct", hard["correct"], "int", field + " (by_tier.hard.correct)"),
                 (f"jb_{policy}_hard_acc", hard["accuracy"], "pct", field + " (by_tier.hard.accuracy)"),
                 (f"jb_{policy}_ece", entry["ece"], "ece", field + " (ece)"),
                 (f"jb_{policy}_think", entry["mean_think_tokens"], "int", field + " (mean_think_tokens)"),
                 (f"jb_{policy}_escalation", entry["escalation_rate"], "pct", field + " (escalation_rate)")]
        for system, pair in report["paired"][policy].items():
            key = "".join(ch for ch in system if ch.isalnum())
            for part in ("all", "hard"):
                test = pair[part]
                field = f"{name} `paired.{policy}.{system}.{part}`"
                rows += [(f"jb_{policy}_vs_{key}_{part}_ours", test["ours_only"], "int", field + " (ours_only)"),
                         (f"jb_{policy}_vs_{key}_{part}_theirs", test["theirs_only"], "int", field + " (theirs_only)"),
                         (f"jb_{policy}_vs_{key}_{part}_p", test["p_value"], "p", field + " (p_value)")]
    for system, entry in report["systems"].items():
        key = "".join(ch for ch in system if ch.isalnum())
        field = f"{name} `systems.{system}`, from JevBench's per-task file"
        rows += [(f"jb_sys_{key}_correct", entry["correct"], "int", field + " (correct)"),
                 (f"jb_sys_{key}_acc", entry["accuracy"], "pct", field + " (accuracy)"),
                 (f"jb_sys_{key}_hard_correct", entry["by_tier"]["hard"]["correct"], "int",
                  field + " (by_tier.hard.correct)"),
                 (f"jb_sys_{key}_hard_acc", entry["by_tier"]["hard"]["accuracy"], "pct",
                  field + " (by_tier.hard.accuracy)")]
    first = next(iter(report["policies"].values()))
    rows += [("jb_rows", first["n"], "int", f"{name} `policies.*.n`"),
             ("jb_hard_rows", first["by_tier"]["hard"]["n"], "int", f"{name} `policies.*.by_tier.hard.n`")]
    return rows


def latency_numbers(path: Path) -> list[tuple]:
    """Medians of a one-process scorer run (one row per decision, batch 1)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    name = f"`{path.name}`"

    def median(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    out = [("lat_rows", len(rows), "int", f"{name} (rows)"),
           ("lat_prefix_tokens", median(r["prefix_tokens"] for r in rows), "int", f"{name} (median prefix_tokens)")]
    for key, field in (("prefill", "prefill_seconds"), ("copy", "copy_seconds"), ("s1", "s1_seconds"),
                       ("rotation", "rotation_seconds")):
        out.append((f"lat_{key}_ms", median(r.get(field) for r in rows), "ms", f"{name} (median {field}, ms)"))
    out += [("lat_s1_marginal_ms", median(r["copy_seconds"] + r["s1_seconds"] for r in rows), "ms",
             f"{name} (median copy_seconds + s1_seconds, ms)"),
            ("lat_s2_seconds", median(r.get("s2_seconds") for r in rows), "sec", f"{name} (median s2_seconds)"),
            ("lat_s2_think_tokens", median(r.get("s2_think_tokens") for r in rows), "int",
             f"{name} (median s2_think_tokens)")]
    return out


def tolerance_numbers(report: dict) -> list[tuple]:
    """The gate and the anytime policy at each dev tolerance (always-S2 accuracy minus 0, 1, 2 points).

    Gate values come from `dev_selected_operating_points` (the H2 source, which carries coverage);
    anytime values from `anytime_test.dev_selected_operating_points`. Shares divide by always-S2's
    mean thought tokens.
    """
    gate, s2_tokens = gate_policy(report), report["baselines_test"]["s2_mean_think_tokens"]
    out = []
    for tolerance in (0.0, 0.01, 0.02):
        name = f"eps{round(100 * tolerance)}"
        point = next((p for p in report.get("dev_selected_operating_points", [])
                      if p["policy"] == gate and p["tolerance"] == tolerance), None)
        if point:
            field = f"`dev_selected_operating_points` ({gate}, tolerance {tolerance}, {{}})"
            out += [(f"gate_{name}_accuracy", point["test_accuracy"], "pct", field.format("test_accuracy")),
                    (f"gate_{name}_escalation", 1 - point["test_coverage"], "pct", field.format("1 - test_coverage")),
                    (f"gate_{name}_share", point["test_mean_think_tokens"] / s2_tokens, "share",
                     field.format("test_mean_think_tokens / `baselines_test.s2_mean_think_tokens`"))]
        point = _point(report, "anytime", tolerance)
        if point:
            field = f"`anytime_test.dev_selected_operating_points` (anytime, tolerance {tolerance}, {{}})"
            out += [(f"anytime_{name}_accuracy", point["test_accuracy"], "pct", field.format("test_accuracy")),
                    (f"anytime_{name}_share", point["test_mean_think_tokens"] / s2_tokens, "share",
                     field.format("test_mean_think_tokens / `baselines_test.s2_mean_think_tokens`"))]
    return out


def combined_numbers(report: dict) -> list[tuple]:
    """The exploratory `combined` policy (added after the release runs were read): its dev-selected operating
    points at each tolerance, its AUROC for S1 correctness, and the fitted standardized weights.

    Empty for reports written before the policy existed; its areas come with every other policy's.
    """
    if "combined" not in report.get("policies", []):
        return []
    s2_tokens = report["baselines_test"]["s2_mean_think_tokens"]
    out = []
    for tolerance in (0.0, 0.01, 0.02):
        name = f"eps{round(100 * tolerance)}"
        point = next((p for p in report.get("dev_selected_operating_points", [])
                      if p["policy"] == "combined" and p["tolerance"] == tolerance), None)
        if point:
            field = f"`dev_selected_operating_points` (combined, tolerance {tolerance}, {{}})"
            out += [(f"combined_{name}_accuracy", point["test_accuracy"], "pct", field.format("test_accuracy")),
                    (f"combined_{name}_escalation", 1 - point["test_coverage"], "pct", field.format("1 - test_coverage")),
                    (f"combined_{name}_share", point["test_mean_think_tokens"] / s2_tokens, "share",
                     field.format("test_mean_think_tokens / `baselines_test.s2_mean_think_tokens`"))]
    fit = report.get("combined_fit") or {}
    for feature, weight in (fit.get("standardized_weights") or {}).items():
        out.append((f"combined_weight_{feature}", weight, "dec", f"`combined_fit.standardized_weights.{feature}`"))
    return out


def language_numbers(report: dict) -> list[tuple]:
    """Block C's generated families by language, row-weighted over `per_family_test` (source `owned`)."""
    totals = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for e in report["per_family_test"]:
        if e["source"] == "owned":
            t = totals[e["lang"]]
            t[0] += e["rows"]
            for i, key in enumerate(("s1_accuracy", "s2_accuracy", "system_accuracy", "escalation_rate"), 1):
                t[i] += e[key] * e["rows"]
    out = []
    for lang, (n, s1, s2, system, escalated) in sorted(totals.items()):
        field = f"`per_family_test` ({{}}, row-weighted over source 'owned', lang '{lang}')"
        out += [(f"lang_{lang}_rows", n, "int", field.format("rows")),
                (f"lang_{lang}_s1", s1 / n, "pct", field.format("s1_accuracy")),
                (f"lang_{lang}_s2", s2 / n, "pct", field.format("s2_accuracy")),
                (f"lang_{lang}_system", system / n, "pct", field.format("system_accuracy")),
                (f"lang_{lang}_escalation", escalated / n, "pct", field.format("escalation_rate"))]
    return out


def stop_numbers(report: dict) -> list[tuple]:
    """Where the anytime policy stops on test rows: S1, a checkpoint before the full thought, or the full thought.

    Replays the policy at its dev-selected tau (tolerance 0.01) with the report's temperatures on
    the cascade file the report names (`inputs.cascade`); skipped when that file is absent. The
    replay's mean thought tokens must match the report's operating point, which ties the derived
    shares to the report.
    """
    cascade = Path(report["inputs"]["cascade"])
    point = _point(report, "anytime")
    if not cascade.exists() or point is None:
        return []
    anytime = report["anytime_test"]
    tau = point["tau"]
    temperatures = {c["t"]: c["temperature"] for c in anytime["per_checkpoint"]}
    scores = {r["id"]: r for r in report["row_scores"] if r["split"] == "test"}
    stops, tokens = {"s1": 0, "checkpoint": 0, "full": 0}, 0.0
    for line in cascade.read_text().splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        score = scores.get(pred["id"])
        if score is None:
            continue
        if score[gate_policy(report)] >= tau:
            stops["s1"] += 1
            continue
        for read in pred.get("s2_anytime", []):
            if read["t"] < pred["s2_think_tokens"] and max(_softmax(read["option_logits"], temperatures[read["t"]])) >= tau:
                stops["checkpoint"] += 1
                tokens += read["t"]
                break
        else:
            stops["full"] += 1
            tokens += pred["s2_think_tokens"]
    n = sum(stops.values())
    if n != len(scores) or abs(tokens / n - point["test_mean_think_tokens"]) > 1:
        raise ValueError(f"anytime replay on {cascade} does not reproduce the report's operating point "
                         f"({n} rows, {tokens / max(n, 1):.1f} mean thought tokens)")
    field = ("`anytime_test.dev_selected_operating_points` (anytime, tolerance 0.01, tau) replayed on "
             "`inputs.cascade` with `anytime_test.per_checkpoint` temperatures: {}")
    escalated = stops["checkpoint"] + stops["full"]
    return [("stop_s1", stops["s1"] / n, "pct", field.format("share of test rows answered by S1")),
            ("stop_checkpoint", stops["checkpoint"] / n, "pct",
             field.format("share of test rows answered at a checkpoint before the full thought")),
            ("stop_full", stops["full"] / n, "pct", field.format("share of test rows answered after the full thought")),
            ("stop_checkpoint_of_escalated", stops["checkpoint"] / escalated if escalated else None, "pct",
             field.format("share of escalated rows that stop at a checkpoint"))]


def _escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _source_name(source: str) -> str:
    return OWNED_NAMES.get(source) or SOURCE_NAMES.get(source, source)


BLOCK_LETTERS = {"jev": "A", "public": "B", "weak": "C", "other": "--"}


def write_per_source_tex(report: dict, path: Path) -> None:
    """Rows `block & source & lang & level & rows & S1 & S2 & gate & escalated \\\\` from `per_family_test`."""
    cells = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for e in report["per_family_test"]:
        cell = cells[(e["source"], e["lang"], e.get("difficulty"))]
        cell[0] += e["rows"]
        for i, field in enumerate(("s1_accuracy", "s2_accuracy", "system_accuracy", "escalation_rate"), 1):
            cell[i] += e[field] * e["rows"]
    order = {"jev": 0, "public": 1, "weak": 2, "other": 3}
    keys = sorted(cells, key=lambda k: (order[block_key(k[0])], _source_name(k[0]), k[1], k[2] or 0))
    lines = ["% Generated by benchmarks/make_figures.py from `per_family_test` (row-weighted over families). Do not edit."]
    previous = None
    for source, lang, level in keys:
        n, *sums = cells[(source, lang, level)]
        block = block_key(source)
        if previous is not None and block != previous:
            lines.append("\\midrule")
        previous = block
        values = " & ".join(_tex(v / n, "pct") for v in sums)
        lines.append(f"{BLOCK_LETTERS[block]} & {_escape(_source_name(source))} & {lang} & "
                     f"{'--' if level is None else level} & {n} & {values} \\\\")
    path.write_text("\n".join(lines) + "\n")


def _softmax(logits: list[float], temperature: float) -> list[float]:
    import math

    top = max(logits)
    weights = [math.exp((z - top) / temperature) for z in logits]
    return [w / sum(weights) for w in weights]


def _tex_text(text: str) -> str:
    return _escape(text).replace("$", "\\$").replace("#", "\\#")


def example_values(report: dict, rows: Path, row_id: str) -> list[tuple[str, str, str]]:
    """One test row traced through the cascade with the report's own temperatures and thresholds.

    Probabilities are raw decimals (they set bar heights); the stopping rule is the anytime policy's
    at its dev-selected tau (tolerance 0.01): S1 if its gate score reaches tau, else the first
    checkpoint whose calibrated confidence does, else the full thought.
    """
    row = next(json.loads(line) for line in rows.read_text().splitlines() if line.strip()
               and json.loads(line)["id"] == row_id)
    cascade = Path(report["inputs"]["cascade"])
    pred = next(json.loads(line) for line in cascade.read_text().splitlines() if line.strip()
                and json.loads(line)["id"] == row_id)
    score = next(r for r in report["row_scores"] if r["id"] == row_id)
    if score["split"] != "test":
        raise ValueError(f"{row_id} is not a test row")
    anytime = report["anytime_test"]
    tau = _point(report, "anytime")["tau"]
    letters = "ABCDEFGHIJKLMNOP"
    gold = row["label"]
    out = [("ex_id", _tex_text(row_id), "row id"), ("ex_state", _tex_text(row["state"]), "row state"),
           ("ex_question", _tex_text(row["question"]), "row question"),
           ("ex_gold", letters[gold], "row label"), ("ex_options", str(len(row["options"])), "option count"),
           ("ex_tau", f"{tau:.3f}", "`anytime_test.dev_selected_operating_points` (anytime, tolerance 0.01, tau)"),
           ("ex_think_tokens", str(pred["s2_think_tokens"]), "cascade `s2_think_tokens`")]
    for k, option in enumerate(row["options"]):
        out.append((f"ex_opt_{letters[k]}", _tex_text(option["description"]), "row option"))
    reads = [("s1", _softmax(pred["debiased_logits"], report["temperature_rot"]),
              "cascade `debiased_logits` / `temperature_rot`")]
    by_t = {read["t"]: read["option_logits"] for read in pred.get("s2_anytime", [])}
    for c in anytime["per_checkpoint"]:
        logits = by_t.get(c["t"]) if c["t"] < pred["s2_think_tokens"] else pred["s2_option_logits"]
        reads.append((f"t{c['t']}", _softmax(logits, c["temperature"]),
                      f"cascade `s2_anytime` (t = {c['t']}) / `anytime_test.per_checkpoint` temperature"))
    reads.append(("full", _softmax(pred["s2_option_logits"], anytime["full_thought"]["temperature"]),
                  "cascade `s2_option_logits` / `anytime_test.full_thought.temperature`"))
    stop = "full"
    if score["confidence_rot"] >= tau:
        stop = "s1"
    else:
        for name, probs, _ in reads[1:-1]:
            if int(name[1:]) < pred["s2_think_tokens"] and max(probs) >= tau:
                stop = name
                break
    for name, probs, field in reads:
        conf = score["confidence_rot"] if name == "s1" else max(probs)
        out += [(f"ex_{name}_p{letters[k]}", f"{p:.3f}", field) for k, p in enumerate(probs)]
        out += [(f"ex_{name}_conf", f"{conf:.2f}", field + " (max; S1: `row_scores.confidence_rot`)"),
                (f"ex_{name}_pick", letters[probs.index(max(probs))], field + " (argmax)")]
    out.append(("ex_stop", stop, "anytime stopping rule at ex_tau"))
    for c in anytime["per_checkpoint"]:
        out.append((f"curve_acc_t{c['t']}", f"{c['accuracy']:.4f}", f"`anytime_test.per_checkpoint` (t = {c['t']}, accuracy)"))
    out.append(("curve_acc_full", f"{anytime['full_thought']['accuracy']:.4f}", "`anytime_test.full_thought.accuracy`"))
    return out


def write_example_tex(report: dict, rows: Path, row_id: str, path: Path) -> None:
    lines = ["% Generated by benchmarks/make_figures.py: one test row for the overview figure. Do not edit."]
    lines += [f"\\tjset{{{key}}}{{{value}}} % {field}" for key, value, field in example_values(report, rows, row_id)]
    path.write_text("\n".join(lines) + "\n")


def write_data_tex(report: dict, manifest: Path, path: Path) -> None:
    """Rows `block & source & dev & test & languages & license \\\\` from the release manifest."""
    counts = defaultdict(lambda: {"dev": 0, "test": 0})
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            counts[row["source"]][row["split"]] += 1
    langs = defaultdict(set)
    for e in report["per_family_test"]:
        langs[e["source"]].add(e["lang"])
    order = {"jev": 0, "public": 1, "weak": 2, "other": 3}
    lines = [f"% Generated by benchmarks/make_figures.py from `{manifest.name}` (counts), `per_family_test` "
             "(languages), and benchmarks/DATA_SOURCES.md (licenses). Do not edit."]
    previous = None
    for source in sorted(counts, key=lambda s: (order[block_key(s)], _source_name(s))):
        block = block_key(source)
        if previous is not None and block != previous:
            lines.append("\\midrule")
        previous = block
        c = counts[source]
        lines.append(f"{BLOCK_LETTERS[block]} & {_escape(_source_name(source))} & {c['dev']} & {c['test']} & "
                     f"{', '.join(sorted(langs[source])) or '--'} & {LICENSES.get(source, '--')} \\\\")
    dev, test = sum(c["dev"] for c in counts.values()), sum(c["test"] for c in counts.values())
    lines += ["\\midrule", f"All & & {dev} & {test} & & \\\\"]
    path.write_text("\n".join(lines) + "\n")


def write_numbers_tex(report: dict, path: Path, latency: Path | None = None, jevbench: Path | None = None) -> None:
    """`\\tjset{key}{value}` lines for the technical report; each line names the report field it reads."""
    h = headline(report)
    point = "`anytime_test.dev_selected_operating_points` ({}, tolerance 0.01)"
    rows = [("rows_test", h["rows_test"], "int", "`rows.test`"),
            ("always_s1_accuracy", h["always_s1_accuracy"], "pct", "`baselines_test.always_s1_accuracy`"),
            ("always_s2_accuracy", h["always_s2_accuracy"], "pct", "`baselines_test.always_s2_accuracy`"),
            ("always_s2_mean_think_tokens", h["always_s2_mean_think_tokens"], "int",
             "`baselines_test.s2_mean_think_tokens`"),
            ("s1_wrong_s2_right", h["s1_wrong_s2_right"], "int", "`baselines_test.s1_wrong_s2_right`"),
            ("s1_right_s2_wrong", h["s1_right_s2_wrong"], "int", "`baselines_test.s1_right_s2_wrong`"),
            ("s1_ece", h["s1_ece"], "ece", "`s1_readouts_test.*.ece`")]
    for policy in ("gate", "anytime"):
        if f"{policy}_test_accuracy" in h:
            rows += [(f"{policy}_test_accuracy", h[f"{policy}_test_accuracy"], "pct", point.format(policy)),
                     (f"{policy}_mean_think_tokens", h[f"{policy}_mean_think_tokens"], "int", point.format(policy)),
                     (f"{policy}_thinking_share", h[f"{policy}_thinking_share"], "share",
                      point.format(policy) + " / `baselines_test.s2_mean_think_tokens`")]
    if "s1_ltt_coverage_at_5pct" in h:
        rows += [("s1_ltt_coverage_at_5pct", h["s1_ltt_coverage_at_5pct"], "pct",
                  "`s1_readouts_test.*.coverage_at_risk` (risk 0.05, ltt_test_coverage)"),
                 ("s1_ltt_test_risk_at_5pct", h["s1_ltt_test_risk_at_5pct"], "pct",
                  "`s1_readouts_test.*.coverage_at_risk` (risk 0.05, ltt_test_risk)")]
    for key, block in sorted(blocks(report).items()):
        field = "`per_family_test` ({}, row-weighted over block '{}', gate `per_family_test_gate`)"
        rows.append((f"block_{key}_rows", block["rows"], "int", field.format("rows", key)))
        for name, source, kind in (("s1", "s1_accuracy", "pct"), ("s2", "s2_accuracy", "pct"),
                                   ("system", "system_accuracy", "pct"), ("escalation", "escalation_rate", "pct")):
            rows.append((f"block_{key}_{name}", block[name], kind, field.format(source, key)))
    rows += [r for r in analysis_numbers(report) if r[0] not in {"s1_ltt_coverage_at_5pct", "s1_ltt_test_risk_at_5pct"}]
    if latency:
        rows += latency_numbers(latency)
    if jevbench:
        rows += jevbench_numbers(jevbench)
    lines = ["% Generated by benchmarks/make_figures.py from one evaluate_cascade report. Do not edit.",
             "% Each line: \\tjset{key}{value} % report field."]
    lines += [f"\\tjset{{{key}}}{{{_tex(value, kind)}}} % {field}" for key, value, kind, field in rows]
    path.write_text("\n".join(lines) + "\n")


def fig_pareto(plt, report: dict, label: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    base = report["baselines_test"]
    gate = gate_policy(report)
    curves = report["cost_curves_test"]
    series = []
    if "oracle" in curves:
        ax.plot(curves["oracle"]["mean_think_tokens"], curves["oracle"]["accuracy"], color=FAINT, lw=1.2, ls="--",
                drawstyle="steps-post", zorder=1)
        series.append(("Oracle", FAINT, curves["oracle"]))
    if "router_probe" in curves:
        series.append(("Pre-fork router", ROUTER, curves["router_probe"]))
    anytime = (report.get("anytime_test") or {}).get("curves", {}).get("anytime")
    if anytime:
        pairs = sorted(zip(anytime["mean_think_tokens"], anytime["accuracy"]))
        series.append(("ThinkJev anytime", ANYTIME, {"mean_think_tokens": [p[0] for p in pairs],
                                                    "accuracy": [p[1] for p in pairs]}))
    series.append(("ThinkJev gate", GATE, curves[gate]))
    for name, color, curve in series:
        x, y = curve["mean_think_tokens"], curve["accuracy"]
        if name == "Oracle":
            best = max(range(len(y)), key=lambda i: (y[i], -x[i]))
            ax.annotate(name, (x[best], y[best]), xytext=(4, 4), textcoords="offset points", fontsize=9, color=INK)
        else:
            ax.plot(x, y, color=color, zorder=3, label=name)
    # Every policy curve ends at always-S2, so end-of-curve labels collide; a legend stays readable.
    ax.legend(loc="lower right", fontsize=9)
    s2_tokens = base["s2_mean_think_tokens"]
    for name, x, y in (("Always S1", 0.0, base["always_s1_accuracy"]), ("Always S2", s2_tokens, base["always_s2_accuracy"])):
        ax.plot([x], [y], "o", color=MUTED, ms=7, zorder=4, mec=SURFACE, mew=2)
        ax.annotate(name, (x, y), xytext=(6, -12), textcoords="offset points", fontsize=9, color=MUTED)
    point = _point(report, "gate")
    if point:
        ax.plot([point["test_mean_think_tokens"]], [point["test_accuracy"]], "o", color=GATE, ms=9, zorder=5,
                mec=SURFACE, mew=2)
        share = point["test_mean_think_tokens"] / s2_tokens
        ax.annotate(f"dev-selected: {point['test_accuracy']:.1%} accuracy,\n{share:.0%} of always-S2 thinking",
                    (point["test_mean_think_tokens"], point["test_accuracy"]), xytext=(14, -34),
                    textcoords="offset points", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": FAINT, "lw": 0.8})
        ax.set_title(f"{label}: {point['test_accuracy']:.1%} accuracy with {share:.0%} of always-think's thought tokens")
    else:
        ax.set_title(f"{label}: accuracy against thinking spent")
    ax.set_xlabel("Mean thought tokens per decision (test)")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(left=-0.02 * s2_tokens)
    _percent(ax)
    fig.tight_layout()
    return fig


def _groups(report: dict) -> list[dict]:
    totals = defaultdict(lambda: {"rows": 0, "s1": 0.0, "s2": 0.0, "system": 0.0, "escalation": 0.0})
    for entry in report["per_family_test"]:
        owned = entry["source"] in OWNED
        key = (MULTI_STEP if owned else BLOCKS.get(entry["source"], "Other"),
               FAMILY_NAMES.get(entry["family"], entry["family"]) if owned
               else SOURCE_NAMES.get(entry["source"], entry["source"]))
        total, n = totals[key], entry["rows"]
        total["rows"] += n
        for field, source in (("s1", "s1_accuracy"), ("s2", "s2_accuracy"), ("system", "system_accuracy"),
                              ("escalation", "escalation_rate")):
            total[field] += entry.get(source, float("nan")) * n
    groups = [{"block": block, "name": name, "rows": t["rows"],
               **{field: t[field] / t["rows"] for field in ("s1", "s2", "system", "escalation")}}
              for (block, name), t in totals.items()]
    return sorted(groups, key=lambda g: (BLOCK_ORDER.index(g["block"]) if g["block"] in BLOCK_ORDER else 9, g["s1"]))


def fig_tasks(plt, report: dict, label: str):
    groups = _groups(report)
    lines, previous = [], None  # a header line before each block, then its groups
    for g in groups:
        if g["block"] != previous:
            lines.append(("header", g["block"]))
            previous = g["block"]
        lines.append(("group", g))
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(8.6, 0.3 * len(lines) + 1.7),
                                      gridspec_kw={"width_ratios": [3, 1.6], "wspace": 0.22})
    ys = list(range(len(lines)))[::-1]
    labels = []
    for y, (kind, value) in zip(ys, lines):
        if kind == "header":
            labels.append("")
            left.annotate(value, (0, y), xycoords=("axes fraction", "data"), xytext=(-6, 0),
                          textcoords="offset points", ha="right", va="center", fontsize=9.5, fontweight="semibold")
            continue
        g = value
        labels.append(f"{g['name']}  (n={g['rows']})")
        left.plot([g["s1"], g["s2"]], [y, y], color=GRID, lw=2.2, zorder=1)
        left.plot([g["s1"]], [y], "o", mfc=SURFACE, mec=MUTED, mew=1.4, zorder=2)
        left.plot([g["s2"]], [y], "o", color=MUTED, zorder=2)
        left.plot([g["system"]], [y], "D", color=GATE, ms=5.5, zorder=3)
        right.barh(y, g["escalation"], height=0.62, color=GATE)
        right.annotate(f"{g['escalation']:.0%}", (g["escalation"], y), xytext=(3, 0), textcoords="offset points",
                       va="center", fontsize=8, color=MUTED)
    left.set_yticks(ys, labels, fontsize=9)
    left.tick_params(axis="y", length=0)
    right.tick_params(axis="y", length=0)
    left.set_xlim(0, 1.02)
    right.set_xlim(0, 1.18)
    right.set_xticks([0, 0.5, 1.0])
    _percent(left, "x")
    _percent(right, "x")
    left.set_xlabel("Test accuracy")
    right.set_xlabel("Rows that thought")
    left.grid(axis="y", visible=False)
    right.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, label="S1 only"),
               plt.Line2D([], [], marker="o", ls="", color=MUTED, label="S2 always"),
               plt.Line2D([], [], marker="D", ls="", color=GATE, label="ThinkJev")]
    left.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    rho = _rank_correlation([1 - g["s1"] for g in groups], [g["escalation"] for g in groups])
    title = ("thinking goes where S1 is weak" if rho >= 0.5
             else "accuracy and escalation by task group")
    fig.suptitle(f"{label}: {title}", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.subplots_adjust(left=0.3)
    return fig


def fig_difficulty(plt, report: dict, label: str):
    cells = _difficulty_cells(report)
    if not cells:
        return None
    names = sorted(cells)
    columns = min(3, len(names))
    rows = -(-len(names) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns + 0.6, 2.0 * rows + 0.8), sharex=True,
                             sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        levels = sorted(cells[name])
        n = [cells[name][level][0] for level in levels]
        ax.plot(levels, [cells[name][level][1] / k for level, k in zip(levels, n)], "o-", color=GATE, ms=5)
        ax.plot(levels, [cells[name][level][2] / k for level, k in zip(levels, n)], "o--", color=FAINT, ms=4, lw=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(levels)
        _percent(ax)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", color=GATE, label="Rows that thought"),
               plt.Line2D([], [], marker="o", ls="--", color=FAINT, label="S1 accuracy")]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9)
    fig.supxlabel("Difficulty level", fontsize=10, color=MUTED)
    rising = _rising(cells)
    title = ("harder rows think more" if sum(r > 0 for r in rising) > len(names) / 2
             else "escalation by difficulty level")
    fig.suptitle(f"{label}: {title}", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig


def fig_anytime(plt, report: dict, label: str):
    anytime = report.get("anytime_test")
    if not anytime:
        return None
    points = anytime["per_checkpoint"] + [{**anytime["full_thought"], "t": "full"}]
    xs = list(range(len(points)))
    ticks = [str(p["t"]) for p in points]
    fig, (acc, ece) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    s1 = report["baselines_test"]["always_s1_accuracy"]
    acc.axhline(s1, color=FAINT, ls="--", lw=1.2)
    acc.annotate("S1 (no thinking mode)", (xs[-1], s1), xytext=(0, 4), textcoords="offset points", ha="right",
                 fontsize=8.5, color=MUTED)
    acc.plot(xs, [p["accuracy"] for p in points], "o-", color=ANYTIME)
    ece.plot(xs, [p["ece"] for p in points], "o-", color=ANYTIME)
    for ax, title in ((acc, "S2 accuracy at think budget t"), (ece, "Calibration error (ECE) at budget t")):
        ax.set_xticks(xs, ticks)
        ax.set_xlabel("Thought tokens allowed (t)")
        ax.set_title(title, fontsize=10)
    _percent(acc)
    ece.set_ylim(bottom=0)
    fig.suptitle(f"{label}: one run, every think budget (t = 0 is thinking mode with an empty thought)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig


def fig_risk(plt, report: dict, label: str):
    view = "rotation" if gate_policy(report) == "confidence_rot" else "raw"
    entries = report["s1_readouts_test"][view]["coverage_at_risk"]
    if not entries:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    top = max(e["risk"] for e in entries) * 1.25
    ax.plot([0, top], [0, top], color=FAINT, ls="--", lw=1.2)
    ax.annotate("realized = target", (top * 0.47, top * 0.51), fontsize=8.5, color=MUTED, rotation=38)
    certified = [e for e in entries if e["ltt_test_risk"] is not None and e["ltt_test_coverage"] > 0]
    for e in entries:
        if e not in certified:  # the threshold accepts nothing: no S1 answer is certified at this target
            ax.plot([e["risk"]], [0], "o", mfc=SURFACE, mec=MUTED, mew=1.4, ms=7, clip_on=False, zorder=3)
            ax.annotate("none certified", (e["risk"], 0), xytext=(0, 8), textcoords="offset points", rotation=90,
                        ha="center", va="bottom", fontsize=8, color=MUTED)
            continue
        interval = risk_interval(e["ltt_test_risk"], e["ltt_test_coverage"], report["rows"]["test"])
        if interval:
            ax.plot([e["risk"]] * 2, interval, color=GATE, lw=1.2, alpha=0.5, zorder=2)
        ax.plot([e["risk"]], [e["ltt_test_risk"]], "o", color=GATE, ms=8, mec=SURFACE, mew=2, clip_on=False, zorder=3)
        ax.annotate(f"{e['ltt_test_coverage']:.0%} answered by S1", (e["risk"], e["ltt_test_risk"]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlim(0, top)
    ax.set_ylim(-0.03 * top, top)
    _percent(ax, "x")
    _percent(ax)
    ax.set_xlabel("Target error rate")
    ax.set_ylabel("Realized test error rate")
    held = bool(certified) and all(e["ltt_test_risk"] <= e["risk"] for e in certified)
    title = ("target vs realized S1 error" if not held else "S1 error stays under every target"
             if len(certified) == len(entries) else "S1 error stays under every target it certifies")
    ax.set_title(f"{label}: {title}")
    fig.tight_layout()
    return fig


FIGURES = {"pareto": fig_pareto, "tasks": fig_tasks, "difficulty": fig_difficulty, "anytime": fig_anytime,
           "risk": fig_risk}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Model name shown in titles, e.g. Qwen3.5-4B")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the figures")
    parser.add_argument("--formats", default="png,svg,pdf", help="pdf is what the technical report includes")
    parser.add_argument("--manifest", type=Path, help="Release manifest; also writes data.tex")
    parser.add_argument("--latency", type=Path, help="Latency run (scorer JSONL); adds lat_* to numbers.tex")
    parser.add_argument("--jevbench", type=Path, help="evaluate_jevbench report; adds jb_* to numbers.tex")
    parser.add_argument("--example", help="Test row ID to trace in example.tex (needs --example-rows)")
    parser.add_argument("--example-rows", type=Path, help="Rows file holding the example's text")
    parser.add_argument("--regenerate", action="store_true",
                        help="Rewrite an existing output directory from the same report")
    args = parser.parse_args()
    if args.out.exists() and not args.regenerate:
        parser.error("Output directory must be new (pass --regenerate to rewrite it from the same report)")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    plt.rcParams["svg.hashsalt"] = "thinkjev"  # stable SVG ids, so a regenerated figure diffs clean
    metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}
    report = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=args.regenerate)
    written = []
    for name, build in FIGURES.items():
        fig = build(plt, report, args.label)
        if fig is None:
            continue
        for fmt in args.formats.split(","):
            path = args.out / f"{name}.{fmt}"
            fig.savefig(path, dpi=200, bbox_inches="tight", metadata=metadata.get(fmt))
            written.append(path)
        plt.close(fig)
    write_summary(report, args.label, args.out / "summary.md")
    write_numbers_tex(report, args.out / "numbers.tex", args.latency, args.jevbench)
    write_per_source_tex(report, args.out / "per_source.tex")
    written += [args.out / "summary.md", args.out / "numbers.tex", args.out / "per_source.tex"]
    if args.manifest:
        write_data_tex(report, args.manifest, args.out / "data.tex")
        written.append(args.out / "data.tex")
    if args.example:
        write_example_tex(report, args.example_rows, args.example, args.out / "example.tex")
        written.append(args.out / "example.tex")
    print("\n".join(str(path) for path in written))


if __name__ == "__main__":
    main()
