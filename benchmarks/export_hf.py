"""Export the routine-task sources (intent routing, escalation, judgment; not in EvalScope) to ThinkJev rows.

Loads Hugging Face datasets with the `datasets` library in the separate data environment
(the same one as `export_evalscope.py`, pinned by `benchmarks/requirements-data.txt`):

    build/venv-data/bin/python benchmarks/export_hf.py --source banking20 \
        --revision <40-char sha> --output build/public/banking20.jsonl

`--revision` is required and must be a 40-character commit SHA; the revisions of the release
runs are in benchmarks/DATA_SOURCES.md, and `benchmarks/resolve_revisions.py` prints the current
ones (`huggingface_hub.HfApi().dataset_info(name).sha`). Sources:

| source          | dataset (config)                 | split(s)                | revision        | license |
|-----------------|----------------------------------|-------------------------|------------------|---------|
| banking20       | mteb/banking77                   | train (label counts), test | DATA_SOURCES.md | CC-BY-4.0 |
| clinc_escalate  | clinc/clinc_oos (plus)           | test, validation, train | DATA_SOURCES.md | CC-BY-3.0 |
| clinc_intent    | clinc/clinc_oos (plus)           | test                    | DATA_SOURCES.md | CC-BY-3.0 |
| typed_decisions | LocalLLaMA/typed-decisions (all) | test                    | DATA_SOURCES.md | Apache-2.0 (verify card) |
| rewardbench2    | allenai/reward-bench-2           | test                    | DATA_SOURCES.md | ODC-BY-1.0 (verify card) |

Protocols. AnyJev (https://github.com/nokia-applied-research/AnyJev, read at commit
10d5db91dda38dbde74c6abc1c075ce6463723d1) is the comparison, so its task definitions are
reproduced where the 16-option cap allows:

- banking20: AnyJev `bench/tasks/banking.py`. Uses the `mteb/banking77` parquet mirror of
  PolyAI/banking77 (as AnyJev does; the PolyAI repo needs a loading script). The 20 intents
  that are most frequent in `train` (`Counter.most_common(20)`), sorted by label id, names
  with underscores turned into spaces; every `test` row with one of those labels. AnyJev
  scores 20-way; SemIf rows allow 16 options. DEVIATION: each row offers a 16-option
  candidate set, the gold intent plus the 15 non-gold intents with the smallest
  SHA-256(row id + intent), listed in AnyJev's order. Every row also records its AnyJev role
  (`Task.split(n_test=300, n_calib=200, seed=0)`, i.e. `random.Random(0).shuffle`) so
  results can be restricted to AnyJev's 300 test items. 16 of 20 options makes each row
  easier than AnyJev's, so accuracies are comparable in item set, not in absolute value.
- clinc_escalate: AnyJev `bench/tasks/supervisor.py` `clinc_escalate`, exactly: `clinc_oos`
  `plus`, items from test, validation, then train; option "Yes" (an agent can handle it,
  in scope) or "No" (out of scope); AnyJev's question text. Only AnyJev's 300 test and 200
  calibration items (seed 0) are exported, each with its role.
- clinc_intent: our own protocol (AnyJev has no CLINC intent-routing task). CLINC150 test
  queries; the options are the 15 intents of the gold intent's domain (official
  `domains.json` of clinc/oos-eval at 828f8093932c8fe6ca7936c3d2e52903b1c523de), so every
  distractor is a same-domain near miss. Out-of-scope queries are dropped here (counted)
  because they belong to clinc_escalate. Default cap 600 rows by SHA-256 of the row id.
- typed_decisions: AnyJev `bench/tasks/typed_decisions.py`. One row per (case, question);
  the case's JSON state is the row `state`; the five questions of a case share `group_id`.
  choice: options "key: description"; noul: "Is the following statement true? ..." with
  options "Yes: <true meaning>" / "No: <false meaning>"; score: the ordered levels. Gold is
  the teacher's hard label; its soft probabilities go to `provenance.teacher_probabilities`
  (`target_distribution` stays null). Gold is agreement with one teacher LLM, not a human.
- rewardbench2: pairwise preference judging. For each non-Ties prompt, the chosen response
  and one rejected response (picked by SHA-256 of the prompt id) become options A and B,
  with the chosen position set by SHA-256 so it is balanced and reproducible. Ties (several
  correct answers) are dropped and counted. Responses longer than `--max-response-chars`
  (default 4000) and prompts longer than `--max-prompt-chars` (default 8000) are cut; the
  original lengths are recorded in `provenance.truncated`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_rows import REVISION_RE, Drop, Tally, id_part, make_row, sample_groups, sha_key, write_rows  # noqa: E402

EXPORTER_VERSION = "export-hf-v1"
DATASETS_VERSION = "4.8.4"
ANYJEV_COMMIT = "10d5db91dda38dbde74c6abc1c075ce6463723d1"
ANYJEV_SEED, ANYJEV_N_TEST, ANYJEV_N_CALIB = 0, 300, 200
BANKING_K, BANKING_CANDIDATES = 20, 16
BANKING_QUESTION = "What is the customer's intent?"  # AnyJev banking20
CLINC_ESCALATE_QUESTION = (  # AnyJev bench/tasks/supervisor.py ESCALATE_TEXT (Apache-2.0)
    "A user message has arrived at a task assistant that covers banking, credit "
    "cards, travel, work, home, auto, dining, utilities, small talk and its own "
    "settings. Can one of those agents handle this message?"
)
CLINC_INTENT_QUESTION = "What is the user's intent?"
PREFERENCE_QUESTION = "Which response is the better reply to the user's prompt?"
TRUNCATION_MARK = " [truncated]"

# clinc/oos-eval data/domains.json at 828f8093932c8fe6ca7936c3d2e52903b1c523de (CC-BY-3.0).
CLINC_DOMAINS = {
    "banking": ["freeze_account", "routing", "pin_change", "bill_due", "pay_bill", "account_blocked",
                "interest_rate", "min_payment", "bill_balance", "transfer", "order_checks", "balance",
                "spending_history", "transactions", "report_fraud"],
    "credit_cards": ["replacement_card_duration", "expiration_date", "damaged_card", "improve_credit_score",
                     "report_lost_card", "card_declined", "credit_limit_change", "apr", "redeem_rewards",
                     "credit_limit", "rewards_balance", "application_status", "credit_score", "new_card",
                     "international_fees"],
    "kitchen_and_dining": ["food_last", "confirm_reservation", "how_busy", "ingredients_list", "calories",
                           "nutrition_info", "recipe", "restaurant_reviews", "restaurant_reservation",
                           "meal_suggestion", "restaurant_suggestion", "cancel_reservation",
                           "ingredient_substitution", "cook_time", "accept_reservations"],
    "home": ["what_song", "play_music", "todo_list_update", "reminder", "reminder_update", "calendar_update",
             "order_status", "update_playlist", "shopping_list", "calendar", "next_song", "order", "todo_list",
             "shopping_list_update", "smart_home"],
    "auto_and_commute": ["current_location", "oil_change_when", "oil_change_how", "uber", "traffic",
                         "tire_pressure", "schedule_maintenance", "gas", "mpg", "distance", "directions",
                         "last_maintenance", "gas_type", "tire_change", "jump_start"],
    "travel": ["plug_type", "travel_notification", "translate", "flight_status", "international_visa", "timezone",
               "exchange_rate", "travel_suggestion", "travel_alert", "vaccines", "lost_luggage", "book_flight",
               "book_hotel", "carry_on", "car_rental"],
    "utility": ["weather", "alarm", "date", "find_phone", "share_location", "timer", "make_call", "calculator",
                "definition", "measurement_conversion", "flip_coin", "spelling", "time", "roll_dice", "text"],
    "work": ["pto_request_status", "next_holiday", "insurance_change", "insurance", "meeting_schedule", "payday",
             "taxes", "income", "rollover_401k", "pto_balance", "pto_request", "w2", "schedule_meeting",
             "direct_deposit", "pto_used"],
    "small_talk": ["who_made_you", "meaning_of_life", "who_do_you_work_for", "do_you_have_pets",
                   "what_are_your_hobbies", "fun_fact", "what_is_your_name", "where_are_you_from", "goodbye",
                   "thank_you", "greeting", "tell_joke", "are_you_a_bot", "how_old_are_you", "what_can_i_ask_you"],
    "meta": ["change_speed", "user_name", "whisper_mode", "yes", "change_volume", "no", "change_language", "repeat",
             "change_accent", "cancel", "sync_device", "change_user_name", "change_ai_name", "reset_settings",
             "maybe"],
}
CLINC_DOMAIN_OF = {intent: domain for domain, intents in CLINC_DOMAINS.items() for intent in intents}

SOURCES = {
    "banking20": {"dataset": "mteb/banking77", "config": None, "splits": ("train", "test"), "license": "CC-BY-4.0",
                  "upstream": "https://huggingface.co/datasets/PolyAI/banking77 (mteb parquet mirror)", "cap": None},
    "clinc_escalate": {"dataset": "clinc/clinc_oos", "config": "plus", "splits": ("test", "validation", "train"),
                       "license": "CC-BY-3.0", "upstream": "https://huggingface.co/datasets/clinc/clinc_oos",
                       "cap": None},
    "clinc_intent": {"dataset": "clinc/clinc_oos", "config": "plus", "splits": ("test",), "license": "CC-BY-3.0",
                     "upstream": "https://huggingface.co/datasets/clinc/clinc_oos", "cap": 600},
    "typed_decisions": {"dataset": "LocalLLaMA/typed-decisions", "config": "all", "splits": ("test",),
                        "license": "Apache-2.0 (per AnyJev THIRD_PARTY.md; verify the dataset card)",
                        "upstream": "https://huggingface.co/datasets/LocalLLaMA/typed-decisions", "cap": None},
    "rewardbench2": {"dataset": "allenai/reward-bench-2", "config": None, "splits": ("test",),
                     "license": "ODC-BY-1.0 (verify the dataset card); responses are model outputs",
                     "upstream": "https://huggingface.co/datasets/allenai/reward-bench-2", "cap": 100},
}


def pretty(name: str) -> str:
    return name.replace("_", " ")


def anyjev_roles(count: int, n_test: int = ANYJEV_N_TEST, n_calib: int = ANYJEV_N_CALIB,
                 seed: int = ANYJEV_SEED) -> list[str]:
    """AnyJev `Task.split`: shuffle item indices with random.Random(seed); test first, then calib."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    roles = ["unused"] * count
    for position, index in enumerate(order[:n_test + n_calib]):
        roles[index] = "test" if position < n_test else "calib"
    return roles


def _provenance(source: str, revision: str, row_ref: dict, **extra) -> dict:
    spec = SOURCES[source]
    return {"source": f"hf:{spec['dataset']}", "upstream": spec["upstream"], "revision": revision,
            "config": spec["config"], "license": spec["license"], "exporter": EXPORTER_VERSION,
            "row_ref": row_ref, **extra}


def banking20_rows(train: list[dict], test: list[dict], *, revision: str, tally: Tally,
                   k: int = BANKING_K, candidates: int = BANKING_CANDIDATES) -> list[dict]:
    names = {int(row["label"]): row["label_text"] for row in train}
    top = sorted(label for label, _ in Counter(int(row["label"]) for row in train).most_common(k))
    items = [(index, row) for index, row in enumerate(test) if int(row["label"]) in top]
    roles = anyjev_roles(len(items))
    tally.seen["test"] += len(test)
    if len(test) > len(items):
        tally.dropped.setdefault("test", Counter())[f"label_outside_top{k}"] += len(test) - len(items)
    rows = []
    for position, (index, row) in enumerate(items):
        gold = int(row["label"])
        row_id = f"banking77/banking{k}/{index}"
        pool = [label for label in top if label != gold]
        keep = set(sorted(pool, key=lambda label: sha_key(f"{row_id}|{names[label]}"))[:candidates - 1]) | {gold}
        offered = [label for label in top if label in keep]
        try:
            rows.append(make_row(
                row_id=row_id, group_id=row_id, family=f"banking77/banking{k}", lang="en", state=row["text"],
                question=BANKING_QUESTION, options=[(names[label], pretty(names[label])) for label in offered],
                label=offered.index(gold),
                provenance=_provenance("banking20", revision, {"split": "test", "index": index},
                                       protocol=f"anyjev-banking{k}-cand{len(offered)}", anyjev_commit=ANYJEV_COMMIT,
                                       anyjev_item=position, anyjev_role=roles[position]),
            ))
        except Drop as reason:
            tally.drop("test", str(reason))
    tally.kept["test"] += len(rows)
    return rows


def clinc_escalate_rows(splits: dict[str, list[dict]], intent_names: list[str], *, revision: str,
                        tally: Tally) -> list[dict]:
    oos = intent_names.index("oos")
    items = [(split, index, row) for split in ("test", "validation", "train")
             for index, row in enumerate(splits[split])]
    roles = anyjev_roles(len(items))
    rows = []
    for position, (split, index, row) in enumerate(items):
        tally.seen[split] += 1
        if roles[position] == "unused":
            tally.capped[split] += 1
            continue
        row_id = f"clinc150/escalate/{split}-{index}"
        try:
            rows.append(make_row(
                row_id=row_id, group_id=row_id, family="clinc150/escalate", lang="en", state=row["text"],
                question=CLINC_ESCALATE_QUESTION, options=[("yes", "Yes"), ("no", "No")],
                label=1 if int(row["intent"]) == oos else 0,
                provenance=_provenance("clinc_escalate", revision, {"split": split, "index": index},
                                       protocol="anyjev-clinc_escalate", anyjev_commit=ANYJEV_COMMIT,
                                       anyjev_item=position, anyjev_role=roles[position]),
            ))
            tally.kept[split] += 1
        except Drop as reason:
            tally.drop(split, str(reason))
    return rows


def clinc_intent_rows(test: list[dict], intent_names: list[str], *, revision: str, tally: Tally,
                      cap: int | None) -> list[dict]:
    rows = []
    for index, row in enumerate(test):
        tally.seen["test"] += 1
        name = intent_names[int(row["intent"])]
        if name == "oos":
            tally.drop("test", "out_of_scope_in_intent_task")
            continue
        domain = CLINC_DOMAIN_OF.get(name)
        if domain is None:
            tally.drop("test", "intent_not_in_domain_map")
            continue
        intents = CLINC_DOMAINS[domain]
        row_id = f"clinc150/intent/{index}"
        try:
            rows.append(make_row(
                row_id=row_id, group_id=row_id, family=f"clinc150/{domain}", lang="en", state=row["text"],
                question=CLINC_INTENT_QUESTION, options=[(intent, pretty(intent)) for intent in intents],
                label=intents.index(name),
                provenance=_provenance("clinc_intent", revision, {"split": "test", "index": index},
                                       protocol="clinc-domain15"),
            ))
        except Drop as reason:
            tally.drop("test", str(reason))
    kept = sample_groups(rows, cap)
    tally.capped["test"] += len(rows) - len(kept)
    tally.kept["test"] += len(kept)
    return kept


def _load_json(value):
    return json.loads(value) if isinstance(value, str) else value


def typed_question(qname: str, spec: dict) -> tuple[str, list[tuple[str, str]], list[str]]:
    """(question text, options, dataset label keys in option order), mirroring AnyJev's build_question."""
    kind, instructions, criteria = spec["type"], spec["instructions"], spec.get("criteria")
    if kind == "choice":
        keys = list(criteria.keys()) if isinstance(criteria, dict) else [str(c) for c in criteria]
        texts = [f"{key}: {criteria[key]}" for key in keys] if isinstance(criteria, dict) else keys
        return instructions, list(zip(keys, texts)), keys
    if kind == "noul":
        meanings = criteria if isinstance(criteria, dict) else {}
        yes = f"Yes: {meanings['true']}" if meanings.get("true") else "Yes"
        no = f"No: {meanings['false']}" if meanings.get("false") else "No"
        return f"Is the following statement true? {instructions}", [("true", yes), ("false", no)], ["true", "false"]
    if kind == "score":
        levels = list(criteria.values()) if isinstance(criteria, dict) else [str(c) for c in criteria]
        keys = [str(index) for index in range(len(levels))]
        return instructions, [(key, str(level)) for key, level in zip(keys, levels)], keys
    raise Drop("unknown_question_type")


def typed_decision_rows(records: list[dict], *, revision: str, tally: Tally, split: str = "test") -> list[dict]:
    rows = []
    for record in records:
        workflow = record["workflow"]
        case = f"typed-decisions/{id_part(workflow)}/{id_part(record['id'])}"
        state = _load_json(record["state"])
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        questions, gold = _load_json(record["questions"]), _load_json(record["gold"])
        for qname, spec in questions.items():
            tally.seen[workflow] += 1
            try:
                question, options, keys = typed_question(qname, spec)
                answer = gold.get(qname)
                if not answer or "label" not in answer:
                    raise Drop("missing_gold")
                if str(answer["label"]) not in keys:
                    raise Drop("gold_not_in_options")
                probabilities = answer.get("probabilities") or {}
                extra = {"question_type": spec["type"],
                         "teacher_probabilities": [float(probabilities.get(key, 0.0)) for key in keys]}
                if spec["type"] == "score" and "score" in answer:
                    extra["teacher_score"] = float(answer["score"])
                rows.append(make_row(
                    row_id=f"{case}/{id_part(qname)}", group_id=case, family=f"typed-decisions/{workflow}/{qname}",
                    lang="en", state=state_text, question=question, options=options,
                    label=keys.index(str(answer["label"])),
                    provenance=_provenance("typed_decisions", revision,
                                           {"split": split, "case_id": record["id"], "question": qname},
                                           anyjev_commit=ANYJEV_COMMIT, gold="teacher-llm", **extra),
                ))
                tally.kept[workflow] += 1
            except Drop as reason:
                tally.drop(workflow, str(reason))
    return rows


def truncate(text: str, limit: int) -> tuple[str, int | None]:
    """Cut to `limit` characters; returns the text and the original length when cut."""
    if len(text) <= limit:
        return text, None
    return text[:limit].rstrip() + TRUNCATION_MARK, len(text)


def preference_rows(records: list[dict], *, revision: str, tally: Tally, cap: int | None,
                    max_response_chars: int = 4000, max_prompt_chars: int = 8000) -> list[dict]:
    by_subset: dict[str, list[dict]] = {}
    for index, record in enumerate(records):
        subset = str(record.get("subset") or "default")
        tally.seen[subset] += 1
        chosen, rejected = record.get("chosen"), record.get("rejected")
        chosen = [chosen] if isinstance(chosen, str) else list(chosen or [])
        rejected = [rejected] if isinstance(rejected, str) else list(rejected or [])
        native = record.get("id", index)
        row_id = f"rewardbench2/{slug_subset(subset)}/{id_part(native)}"
        try:
            if subset.lower() == "ties":
                raise Drop("ties_multiple_correct")
            if len(chosen) != 1:
                raise Drop("not_exactly_one_chosen")
            if not rejected:
                raise Drop("no_rejected")
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                raise Drop("non_text_prompt")
            pick = int(sha_key(f"{row_id}|rejected"), 16) % len(rejected)
            chosen_text, chosen_len = truncate(str(chosen[0]), max_response_chars)
            rejected_text, rejected_len = truncate(str(rejected[pick]), max_response_chars)
            state, prompt_len = truncate(prompt, max_prompt_chars)
            chosen_first = int(sha_key(f"{row_id}|position"), 16) % 2 == 0
            pair = [chosen_text, rejected_text] if chosen_first else [rejected_text, chosen_text]
            truncated = {key: value for key, value in (("prompt", prompt_len), ("chosen", chosen_len),
                                                         ("rejected", rejected_len)) if value is not None}
            row = make_row(
                row_id=row_id, group_id=row_id, family=f"rewardbench2/{slug_subset(subset)}", lang="en",
                state=state, question=PREFERENCE_QUESTION, options=[("A", pair[0]), ("B", pair[1])],
                label=0 if chosen_first else 1,
                provenance=_provenance("rewardbench2", revision, {"split": "test", "index": index, "id": native},
                                       rejected_index=pick, truncated=truncated or None,
                                       max_response_chars=max_response_chars, max_prompt_chars=max_prompt_chars),
            )
            by_subset.setdefault(subset, []).append(row)
        except Drop as reason:
            tally.drop(subset, str(reason))
    rows = []
    for subset, part in by_subset.items():
        kept = sample_groups(part, cap)
        tally.capped[subset] += len(part) - len(kept)
        tally.kept[subset] += len(kept)
        rows.extend(kept)
    return rows


def slug_subset(subset: str) -> str:
    return id_part(subset).lower()


def load_hf(source: str, revision: str, cache_dir: Path,
            split_override: str | None = None) -> tuple[dict[str, list[dict]], dict]:
    """Download the pinned splits; returns plain row dicts plus ClassLabel names. Network.

    `split_override` loads another upstream split for a single-split source, stored under "test".
    """
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "hf-datasets"))
    import datasets

    if datasets.__version__ != DATASETS_VERSION:
        raise SystemExit(f"need datasets=={DATASETS_VERSION}, found {datasets.__version__}")
    spec = SOURCES[source]
    splits, labels = {}, {}
    for split in spec["splits"]:
        data = datasets.load_dataset(spec["dataset"], spec["config"], split=split_override or split,
                                     revision=revision, cache_dir=str(cache_dir / "hf-datasets"))
        for name, feature in data.features.items():
            if hasattr(feature, "names"):
                labels[name] = list(feature.names)
        splits[split] = [dict(row) for row in data]
    return splits, labels


def convert(source: str, splits: dict[str, list[dict]], labels: dict, *, revision: str, tally: Tally,
            cap: int | None, args=None) -> list[dict]:
    if source == "banking20":
        return banking20_rows(splits["train"], splits["test"], revision=revision, tally=tally)
    if source.startswith("clinc") and "intent" not in labels:
        raise SystemExit("clinc_oos loaded without ClassLabel names for `intent`; check the dataset revision")
    if source == "clinc_escalate":
        return clinc_escalate_rows(splits, labels["intent"], revision=revision, tally=tally)
    if source == "clinc_intent":
        return clinc_intent_rows(splits["test"], labels["intent"], revision=revision, tally=tally, cap=cap)
    if source == "typed_decisions":
        return typed_decision_rows(splits["test"], revision=revision, tally=tally,
                                   split=getattr(args, "split", None) or "test")
    if source == "rewardbench2":
        return preference_rows(splits["test"], revision=revision, tally=tally, cap=cap,
                               max_response_chars=getattr(args, "max_response_chars", 4000),
                               max_prompt_chars=getattr(args, "max_prompt_chars", 8000))
    raise KeyError(source)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--revision", required=True, help="40-character dataset commit SHA")
    parser.add_argument("--output", type=Path, required=True, help="create-only JSONL under build/")
    parser.add_argument("--cache-dir", type=Path, default=Path("build/cache"))
    parser.add_argument("--limit", type=int, help="override the default cap (per subset); 0 means no cap")
    parser.add_argument("--split", help="load this upstream split instead (typed_decisions, rewardbench2 only)")
    parser.add_argument("--max-response-chars", type=int, default=4000)
    parser.add_argument("--max-prompt-chars", type=int, default=8000)
    args = parser.parse_args(argv)
    if not REVISION_RE.match(args.revision):
        parser.error("--revision must be a 40-character lowercase hex commit SHA (see resolve_revisions.py)")
    for path in (args.output, args.output.with_name(args.output.name + ".meta.json")):
        if path.exists():
            raise SystemExit(f"{path} exists; outputs are create-only")
    spec = SOURCES[args.source]
    cap = spec["cap"] if args.limit is None else (args.limit or None)
    if args.split and len(spec["splits"]) != 1:
        parser.error("--split applies only to single-split sources")
    splits, labels = load_hf(args.source, args.revision, args.cache_dir, args.split)
    tally = Tally()
    rows = convert(args.source, splits, labels, revision=args.revision, tally=tally, cap=cap, args=args)
    meta = write_rows(args.output, rows, {
        "exporter": EXPORTER_VERSION, "source": args.source, "dataset": spec["dataset"], "config": spec["config"],
        "revision": args.revision, "split": args.split or list(spec["splits"]), "datasets": DATASETS_VERSION, "license": spec["license"], "cap": cap,
        "anyjev_commit": ANYJEV_COMMIT, **tally.as_dict(),
    })
    print(f"wrote {meta['rows']} rows to {args.output} (dropped {meta['rows_dropped']}: {meta['drop_reasons']}; "
          f"sha256 {meta['sha256']})")


if __name__ == "__main__":
    main()
