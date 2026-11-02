"""Shared, dependency-free helpers for the public-source exporters.

`export_evalscope.py` and `export_hf.py` turn third-party datasets into ThinkJev rows
(the SemIf schema). Everything here is pure Python: no EvalScope, no `datasets`, no network,
so `tests/test_exporters.py` can exercise it in the main environment.

A row is:

    id, group_id, family, lang, state, question, options[{id, description}], label,
    target_distribution (always null here), split ("test" placeholder; the real
    dev/test split is assigned later by `build_cascade.py assemble`), provenance.

Rows that cannot be represented (too many options, missing or ambiguous gold, free-form
answers, images) are dropped and counted by reason, never silently. Every kept row passes
`thinkjev._semif.validate_row`. Outputs are create-only: the JSONL is opened with
mode "x", and its sidecar `<output>.meta.json` (counts, drops, SHA-256 of the JSONL bytes,
pins) is create-only too. Third-party text only ever goes to the given output path.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # data venvs do not install thinkjev (it pulls torch)
    sys.path.insert(0, str(ROOT / "src"))

from thinkjev._semif import LETTERS, validate_row  # noqa: E402

MAX_OPTIONS = len(LETTERS)  # 16
GENERIC_QUESTION = {
    "en": "Which option correctly answers the question?",
    "zh": "哪个选项正确回答了这个问题？",
}
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class Drop(Exception):
    """Raised by a converter when a source record cannot become a row; the message is the reason."""


def slug(text: str) -> str:
    """Lowercase ID component: spaces and slashes become underscores."""
    value = re.sub(r"[\s/]+", "_", str(text).strip().lower())
    if not value:
        raise ValueError("empty ID component")
    return value


def id_part(value) -> str:
    """Case-preserving ID component for native IDs: whitespace and slashes become underscores."""
    text = re.sub(r"[\s/]+", "_", str(value).strip())
    if not text:
        raise ValueError("empty ID component")
    return text


def sha_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_group(prefix: str, text: str) -> str:
    """Group ID for rows that share a passage: identical context text shares a group."""
    return f"{prefix}/ctx-{sha_key(text)[:16]}"


def label_from_target(target, count: int) -> int:
    """Map a gold answer (letter such as "B" or "(B)", or an int index) to an option index."""
    if isinstance(target, list):
        if len(target) != 1:
            raise Drop("multi_answer_gold")
        target = target[0]
    if isinstance(target, bool):
        raise Drop("unparseable_gold")
    if isinstance(target, int):
        index = target
    elif isinstance(target, str):
        cleaned = target.strip().strip("()").strip()
        if len(cleaned) != 1 or not cleaned.isalpha() or not cleaned.isupper():
            raise Drop("multi_answer_gold" if len(cleaned) > 1 and cleaned.isalpha() and cleaned.isupper()
                       else "unparseable_gold")
        index = ord(cleaned) - ord("A")
    else:
        raise Drop("missing_gold")
    if not 0 <= index < count:
        raise Drop("gold_out_of_range")
    return index


def make_row(*, row_id: str, group_id: str, family: str, lang: str, state, question: str,
             options: list[tuple[str, str]], label: int, provenance: dict) -> dict:
    """Build and validate one row; raise Drop for anything the schema cannot hold."""
    if lang not in ("en", "zh"):
        raise ValueError(f"unknown lang {lang!r}")
    if not isinstance(state, str) or not state.strip():
        raise Drop("empty_state")
    if not isinstance(question, str) or not question.strip():
        raise Drop("empty_question")
    if len(options) < 2:
        raise Drop("too_few_options")
    if len(options) > MAX_OPTIONS:
        raise Drop("too_many_options")
    ids = [option_id for option_id, _ in options]
    if len(set(ids)) != len(ids):
        raise Drop("duplicate_option_ids")
    if any(not isinstance(text, str) or not text.strip() for _, text in options):
        raise Drop("empty_option")
    if not isinstance(label, int) or not 0 <= label < len(options):
        raise Drop("gold_out_of_range")
    row = {
        "id": row_id,
        "group_id": group_id,
        "family": family,
        "lang": lang,
        "state": state,
        "question": question,
        "options": [{"id": option_id, "description": text} for option_id, text in options],
        "label": label,
        "target_distribution": None,
        "split": "test",
        "provenance": provenance,
    }
    validate_row(row)
    return row


def letter_options(texts: list[str]) -> list[tuple[str, str]]:
    """Letter IDs in the original order. More than 16 options cannot get letters: drop."""
    if len(texts) > MAX_OPTIONS:
        raise Drop("too_many_options")
    return [(LETTERS[index], str(text).strip()) for index, text in enumerate(texts)]


def sample_groups(rows: list[dict], cap: int | None) -> list[dict]:
    """Deterministic, outcome-blind cap: whole groups in SHA-256(group_id) order until `cap` rows.

    Kept rows stay in their original order. Same rule as `build_cascade._limit`.
    """
    if cap is None or len(rows) <= cap:
        return list(rows)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(index)
    keep: set[int] = set()
    for group in sorted(groups, key=sha_key):
        if len(keep) + len(groups[group]) > cap:
            break
        keep.update(groups[group])
    return [row for index, row in enumerate(rows) if index in keep]


class Tally:
    """Per-subset kept and dropped counts; drops keep their reason."""

    def __init__(self) -> None:
        self.seen: Counter = Counter()
        self.kept: Counter = Counter()
        self.capped: Counter = Counter()
        self.dropped: dict[str, Counter] = {}

    def drop(self, subset: str, reason: str) -> None:
        self.dropped.setdefault(subset, Counter())[reason] += 1

    def as_dict(self) -> dict:
        subsets = sorted(set(self.seen) | set(self.kept) | set(self.dropped))
        return {
            "rows_seen": sum(self.seen.values()),
            "rows_written": sum(self.kept.values()),
            "rows_capped": sum(self.capped.values()),
            "rows_dropped": sum(sum(counter.values()) for counter in self.dropped.values()),
            "drop_reasons": dict(sum(self.dropped.values(), Counter())),
            "subsets": {
                subset: {"seen": self.seen[subset], "written": self.kept[subset], "capped": self.capped[subset],
                         "dropped": dict(self.dropped.get(subset, {}))}
                for subset in subsets
            },
        }


def check_unique(rows: Iterable[dict]) -> None:
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate row id {row['id']}")
        seen.add(row["id"])


def write_rows(output: Path, rows: list[dict], meta: dict) -> dict:
    """Create-only write of the JSONL and its `.meta.json` sidecar; returns the final meta."""
    output = Path(output)
    meta_path = output.with_name(output.name + ".meta.json")
    for path in (output, meta_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; outputs are create-only, choose a new path")
    rows = list(rows)
    check_unique(rows)
    for row in rows:
        validate_row(row)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    meta = {**meta, "output": output.name, "rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    with meta_path.open("x") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return meta
