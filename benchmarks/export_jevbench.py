"""Export JevBench's public items to ThinkJev rows, for the exploratory JevBench check.

JevBench (https://github.com/fstandhartinger/jevbench, Benchmark Heaven, MIT) publishes 231
public decisions (`datasets/public/{easy,original,hard}.jsonl`) and, per system, the outcome on
every public item (`results/v1.2/jevbench-v1.2-per-task.json`). The GPU machines cannot reach
GitHub, so a checkout is copied there; it must hold a `COMMIT` file with the pinned commit:

    python benchmarks/export_jevbench.py --jevbench build/jevbench-src \
        --output build/jevbench/rows.jsonl --manifest benchmarks/manifests/jevbench-public.jsonl

Each item becomes the question JevBench's harness sends (`jevbench/adapters/base.py`
`build_question`: type, instructions, criteria), and that question becomes a row through
`thinkjev.api.question_row`, the mapping `thinkjev-serve` applies to the same request. So a row
here is what the server would score. Option IDs map back to JevBench labels as: choice, the
criteria key; noul, `true`/`false` to `yes`/`no`; score, the level index as a string.

Rows carry the item text and stay in `build/`; the manifest carries IDs, tiers, types, and a
hash of each row, and is committed. Outputs are create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thinkjev.api import question_row  # noqa: E402

JEVBENCH_COMMIT = "7625ecb1b075cfe4d09d34141d353325825eea1d"
FILES = ("easy", "original", "hard")  # JevBench calls the `original` cohort tier `standard`
TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}
NOUL_LABELS = {"true": "yes", "false": "no"}


def label_of(kind: str, option_id: str) -> str:
    """The JevBench label an option ID stands for."""
    return NOUL_LABELS[option_id] if kind == "noul" else option_id


def to_row(task: dict, tier: str) -> dict:
    question = {"type": task["question"]["type"], "instructions": task["question"]["instructions"]}
    if task["question"].get("criteria") is not None:
        question["criteria"] = task["question"]["criteria"]
    kind, row = question_row(task["id"], task["state"], question)
    labels = [label_of(kind, option["id"]) for option in row["options"]]
    if sorted(labels) != sorted(str(label) for label in task["labels"]):
        raise ValueError(f"{task['id']}: options {labels} do not match JevBench labels {task['labels']}")
    expected = str(task["expected"])
    if expected not in labels:
        raise ValueError(f"{task['id']}: expected {expected!r} is not an option")
    return {**row, "label": labels.index(expected), "jevbench_type": kind, "jevbench_labels": labels,
            "cascade_split": "test", "cascade_source": "jevbench", "family": task["family"],
            "group_id": task["group"], "tier": tier, "lang": "en"}


def row_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def export(checkout: Path) -> list[dict]:
    commit = (checkout / "COMMIT").read_text().strip()
    if commit != JEVBENCH_COMMIT:
        raise ValueError(f"{checkout}: JevBench commit {commit}, expected {JEVBENCH_COMMIT}")
    rows = []
    for name in FILES:
        for line in (checkout / "datasets" / "public" / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(to_row(json.loads(line), TIERS[name]))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate JevBench item IDs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jevbench", type=Path, required=True, help="JevBench checkout with a COMMIT file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.manifest):
        if path.exists():
            raise SystemExit(f"{path} exists; outputs are create-only")
    rows = export(args.jevbench)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    args.manifest.write_text("".join(json.dumps({"id": row["id"], "tier": row["tier"], "type": row["jevbench_type"],
                                                 "jevbench_commit": JEVBENCH_COMMIT, "row_sha256": row_hash(row)}) + "\n"
                                     for row in rows))
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
