"""Deterministic stratified sample of assembled rows (for latency runs and smoke tests).

Keeps up to N test rows per source (or per source and language), chosen by a hash of the row ID,
so the same command always picks the same rows. Create-only output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sample(rows: list[dict], per_stratum: int, split: str | None, keys: tuple[str, ...]) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        if split is None or row.get("cascade_split") == split:
            strata[tuple(row.get(key) for key in keys)].append(row)
    kept = []
    for stratum in sorted(strata, key=str):
        ranked = sorted(strata[stratum], key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        kept.extend(ranked[:per_stratum])
    order = {row["id"]: position for position, row in enumerate(rows)}
    return sorted(kept, key=lambda row: order[row["id"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, required=True)
    parser.add_argument("--split", default="test", help="dev, test, or all")
    parser.add_argument("--by", default="cascade_source", help="Comma-separated stratum fields")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    kept = sample(rows, args.per_stratum, None if args.split == "all" else args.split, tuple(args.by.split(",")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
    print(f"kept {len(kept)} of {len(rows)} rows")


if __name__ == "__main__":
    main()
