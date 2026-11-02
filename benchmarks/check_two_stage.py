"""Check that the two-stage S2 readout reproduces the one-process scorer's S2 logits.

The one-process scorer (`thinkjev-score --s2 always --trace T`) records each thought's token
IDs. `thoughts` turns them into a vLLM-style thoughts file, `thinkjev-s2 readout` reads the
answer slots after those same thoughts, and `compare` checks the two S2 readouts agree: same
argmax on every row and logits within BF16 noise. Run on ~20 rows on a GPU before any
batch run; sampling itself differs between HF and vLLM, so only the readout is compared exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def thoughts(cascade: Path, trace: Path, output: Path, budget: int) -> None:
    traced = {record["id"]: record for record in map(json.loads, trace.read_text().splitlines()) if record}
    with output.open("x") as stream:
        for pred in map(json.loads, cascade.read_text().splitlines()):
            ids = traced[pred["id"]]["thought_ids"]
            # An unclosed thought (budget or end of turn) is closed with the early-stop sentence either way.
            ending = "closed" if pred["s2_closed_naturally"] else "budget"
            stream.write(json.dumps({"id": pred["id"], "seed_index": 0, "seed": pred["s2_seed"], "thought_ids": ids,
                                     "ending": ending, "think_budget": budget, "engine": {"name": "hf-trace"}}) + "\n")


def compare(one_process: Path, two_stage: Path, tolerance: float) -> int:
    reference = {pred["id"]: pred for pred in map(json.loads, one_process.read_text().splitlines())}
    worst, flips = 0.0, []
    for pred in map(json.loads, two_stage.read_text().splitlines()):
        a, b = reference[pred["id"]]["s2_option_logits"], pred["s2_option_logits"]
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
        if a.index(max(a)) != b.index(max(b)):
            flips.append(pred["id"])
    print(f"max |logit difference| {worst:.4f}; argmax flips: {len(flips)} {flips[:5]}")
    return 0 if worst <= tolerance and not flips else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("thoughts")
    make.add_argument("--cascade", type=Path, required=True)
    make.add_argument("--trace", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--think-budget", type=int, required=True)
    check = commands.add_parser("compare")
    check.add_argument("--one-process", type=Path, required=True)
    check.add_argument("--two-stage", type=Path, required=True)
    check.add_argument("--tolerance", type=float, default=0.5, help="BF16 logits after long contexts drift slightly")
    args = parser.parse_args()
    if args.command == "thoughts":
        thoughts(args.cascade, args.trace, args.output, args.think_budget)
    else:
        raise SystemExit(compare(args.one_process, args.two_stage, args.tolerance))


if __name__ == "__main__":
    main()
