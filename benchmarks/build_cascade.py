"""Build the S1/S2 cascade evaluation: owned hard200 rows, dev/test split, router rows.

`hard` writes 200 programmatic decisions in the failure modes a single forward pass is
expected to miss (date arithmetic, counting, chained rules, stacked negation). Every
label is computed by code, not by a model. `assemble` merges labeled sources, assigns a
group-disjoint dev/test split, and writes the inference rows plus the matching
zero-training pre-router rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from pathlib import Path

GENERATOR = "hard200-v1"
SEED = 260924
PER_FAMILY = 50
DEV_PERCENT = 40
ROUTER_QUESTION = (
    "Before answering it, judge this decision task: can the criterion be decided reliably at a glance, "
    "without step-by-step work such as calculation, counting, date arithmetic, or chaining several rules?"
)
ROUTER_OPTIONS = [
    {"id": "glance", "description": "Yes: it can be decided reliably at a glance"},
    {"id": "work", "description": "No: it needs step-by-step work before it can be decided"},
]


def _row(family: str, index: int, state: str, question: str, options: list[dict], label: int, derivation: dict) -> dict:
    row_id = f"hard/{family}/{index:03d}"
    return {
        "id": row_id,
        "group_id": row_id,
        "family": family,
        "split": "test",
        "state": state,
        "question": question,
        "options": options,
        "label": label,
        "target_distribution": None,
        "provenance": {
            "source": "project-authored-programmatic",
            "generator": GENERATOR,
            "seed": SEED,
            "derivation": derivation,
            "rights": "Project authored; no copied external text",
        },
    }


def _date(value: dt.date) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def date_deadline(rng: random.Random, index: int) -> dict:
    issued = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(330))
    terms = rng.choice((15, 30, 45, 60, 90))
    due = issued + dt.timedelta(days=terms)
    late = index % 2 == 1
    offset = rng.randint(1, 6) if late else -rng.randint(0, 6)
    paid = due + dt.timedelta(days=offset)
    reminder = issued + dt.timedelta(days=rng.randint(5, terms - 3))
    state = (
        f"Invoice {rng.randint(1000, 9999)} was issued on {_date(issued)} with net-{terms} payment terms, "
        f"meaning payment is due {terms} calendar days after the issue date. A courtesy reminder was sent on "
        f"{_date(reminder)}. The customer's payment was received on {_date(paid)}. "
        "A payment is late only if it is received after the due date."
    )
    options = [
        {"id": "on_time", "description": "The payment was received on or before the due date"},
        {"id": "late", "description": "The payment was received after the due date"},
    ]
    derivation = {"issued": issued.isoformat(), "due": due.isoformat(), "paid": paid.isoformat()}
    return _row("date_deadline", index, state, "Was this payment late under the stated terms?", options, int(late), derivation)


def counting(rng: random.Random, index: int) -> dict:
    count = rng.randint(10, 15)
    tickets, matches = [], 0
    for number in rng.sample(range(1000, 9999), count):
        priority = rng.choice(("high", "high", "medium", "low"))
        status = rng.choice(("open", "open", "resolved", "waiting on customer"))
        team = rng.choice(("billing", "platform", "mobile", "identity"))
        matches += priority == "high" and status == "open"
        tickets.append(f"#{number}: priority {priority}, status {status}, team {team}")
    choices = sorted(rng.sample([value for value in range(max(0, matches - 3), matches + 4) if value != matches], 3) + [matches])
    options = [{"id": f"count_{value}", "description": f"Exactly {value}"} for value in choices]
    state = "Support queue snapshot:\n" + "\n".join(tickets)
    question = "How many tickets are both high priority and still open (status exactly 'open')?"
    return _row("counting", index, state, question, options, choices.index(matches), {"matches": matches})


def multihop_rules(rng: random.Random, index: int) -> dict:
    vendors = ["Acme Supply", "Borealis Parts", "Cobalt Office", "Dunmore Labs", "Everline Tools", "Fairview Print"]
    while True:
        limit = rng.choice((500, 1000, 2500))
        preapproved = sorted(rng.sample(vendors, 3))
        restricted = sorted(rng.sample([f"P-{value}" for value in range(10, 40)], 2))
        vendor = rng.choice(vendors)
        category = rng.choice(("hardware", "software", "services", "supplies"))
        project = rng.choice(restricted + [f"P-{value}" for value in range(10, 40)])
        amount = rng.choice((limit - rng.randint(1, 200), limit + rng.randint(1, 900)))
        exempt = vendor in preapproved and category != "hardware"
        needed = (amount > limit and not exempt) or project in restricted
        if needed == (index % 2 == 0):
            break
    state = (
        f"Purchasing policy. Rule 1: purchases above ${limit:,} require manager approval. "
        "Rule 2: purchases from a pre-approved vendor are exempt from Rule 1, except hardware purchases. "
        "Rule 3: any purchase charged to a restricted project requires manager approval regardless of Rules 1 and 2. "
        f"Pre-approved vendors: {', '.join(preapproved)}. Restricted projects: {', '.join(restricted)}.\n"
        f"Request: ${amount:,} of {category} from {vendor}, charged to project {project}."
    )
    options = [
        {"id": "required", "description": "Manager approval is required"},
        {"id": "not_required", "description": "Manager approval is not required"},
    ]
    derivation = {"over_limit": amount > limit, "exempt": exempt, "restricted": project in restricted}
    return _row("multihop_rules", index, state, "Under this policy, does the request require manager approval?",
                options, 0 if needed else 1, derivation)


def double_negation(rng: random.Random, index: int) -> dict:
    subjects = ["pallet 7", "the Leeds shipment", "container B-12", "the replacement pump", "batch 4471", "the north gate order"]
    actions = [("inspected", "not inspected"), ("signed for", "not signed for"), ("weighed", "not weighed"),
               ("sealed", "not sealed"), ("scanned", "not scanned")]
    subject, other = rng.sample(subjects, 2)
    positive, negative = rng.choice(actions)
    inner_negated = rng.random() < 0.5
    wraps = rng.choice((1, 2)) if not inner_negated else rng.choice((0, 1, 2))
    about_other = index % 5 == 4
    sentence = f"{other if about_other else subject} was {negative if inner_negated else positive}"
    for _ in range(wraps):
        sentence = f"{rng.choice(('it is not the case that', 'it is false that'))} {sentence}"
    state = f"Warehouse log, verified entry: {sentence[0].upper()}{sentence[1:]}. No other entry mentions {subject}."
    options = [
        {"id": "yes", "description": f"Yes, {subject} was {positive}"},
        {"id": "no", "description": f"No, {subject} was {negative}"},
        {"id": "insufficient", "description": "The log does not say"},
    ]
    negations = wraps + inner_negated
    label = 2 if about_other else (0 if negations % 2 == 0 else 1)
    derivation = {"negations": negations, "about_other_subject": about_other}
    return _row("double_negation", index, state, f"According to the log, was {subject} {positive}?", options, label, derivation)


def hard_rows() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for family in (date_deadline, counting, multihop_rules, double_negation):
        rows.extend(family(rng, index) for index in range(PER_FAMILY))
    return rows


def split_of(group_id: str) -> str:
    return "dev" if int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100 < DEV_PERCENT else "test"


def router_row(row: dict) -> dict:
    task = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [option["description"] for option in row["options"]],
    }
    return {"id": "router/" + row["id"], "state": task, "question": ROUTER_QUESTION, "options": ROUTER_OPTIONS}


def _limit(rows: list[dict], count: int) -> list[dict]:
    """Keep whole groups in hashed order until `count` rows (outcome-blind, deterministic)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    kept = []
    for group in sorted(groups, key=lambda key: hashlib.sha256(key.encode()).hexdigest()):
        if len(kept) + len(groups[group]) > count:
            break
        kept.extend(groups[group])
    return kept


def assemble(sources: list[tuple[str, Path]], limits: dict[str, int]) -> tuple[list[dict], list[dict]]:
    rows, manifest, seen = [], [], set()
    for name, path in sources:
        part = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        part = [row for row in part if isinstance(row.get("label"), int) and row.get("target_distribution") is None]
        if name in limits:
            part = _limit(part, limits[name])
        for row in part:
            if row["id"] in seen:
                raise ValueError(f"Duplicate row id across sources: {row['id']}")
            seen.add(row["id"])
            split = split_of(f"{name}/{row['group_id']}")
            rows.append({**row, "cascade_source": name, "cascade_split": split})
            manifest.append({"id": row["id"], "source": name, "group_id": row["group_id"],
                             "family": row["family"], "split": split})
    return rows, manifest


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    hard = commands.add_parser("hard", help="Write the owned hard200 fixture")
    hard.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("assemble", help="Merge labeled sources into dev/test cascade rows")
    build.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    build.add_argument("--limit", action="append", default=[], metavar="NAME=ROWS")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "hard":
        _write(args.output, hard_rows())
        print(f"wrote {len(hard_rows())} rows")
        return
    sources = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.source]
    limits = {item.split("=", 1)[0]: int(item.split("=", 1)[1]) for item in args.limit}
    rows, manifest = assemble(sources, limits)
    _write(args.manifest, manifest)
    _write(args.output_dir / "rows.jsonl", rows)
    _write(args.output_dir / "router.jsonl", [router_row(row) for row in rows])
    counts = {split: sum(row["cascade_split"] == split for row in rows) for split in ("dev", "test")}
    print(f"wrote {len(rows)} rows {counts}")


if __name__ == "__main__":
    main()
