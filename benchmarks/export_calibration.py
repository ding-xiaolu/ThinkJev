"""Bundled calibration from an evaluate_cascade report, for `thinkjev-serve --preset`.

Keeps only the report fields that `thinkjev.api.calibration_from_report` reads (S1/S2
temperatures, the dev-selected gate thresholds, and the anytime readout's checkpoints, taus, and
test curve), so the package ships a few kilobytes instead of the 10 MB report, and adds the
model, revision, and runtime settings the report was produced with. Create-only output.

    python benchmarks/export_calibration.py --report results/qwen3.5-4b-b4096-report-v2.json \
        --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
        --think-budget 4096 --output src/thinkjev/calibrations/qwen3.5-4b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GATES = ("confidence", "confidence_rot")


def bundled(report: dict) -> dict:
    """The subset of `report` that calibration_from_report reads, in the report's own schema."""
    anytime = report.get("anytime_test")
    kept = {
        "temperature": report["temperature"],
        "temperature_rot": report["temperature_rot"],
        "s2_temperature": report["s2_temperature"],
        "dev_selected_operating_points": [point for point in report["dev_selected_operating_points"]
                                          if point["policy"] in GATES],
    }
    if anytime:
        kept["anytime_test"] = {
            "gate": anytime["gate"],
            "per_checkpoint": [{key: c[key] for key in ("t", "temperature", "accuracy", "ece")}
                               for c in anytime["per_checkpoint"]],
            "full_thought": {key: anytime["full_thought"][key] for key in ("accuracy", "ece")},
            "dev_selected_operating_points": [point for point in anytime["dev_selected_operating_points"]
                                              if point["policy"] == "anytime"],
        }
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--think-budget", type=int, required=True)
    parser.add_argument("--no-rotations", action="store_true", help="The report's runs did not use rotations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; outputs are create-only")
    raw = args.report.read_bytes()
    out = {"model": args.model, "revision": args.revision,
           "settings": {"rotations": not args.no_rotations, "think_budget": args.think_budget},
           "report": args.report.name, "report_sha256": hashlib.sha256(raw).hexdigest(),
           **bundled(json.loads(raw))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
