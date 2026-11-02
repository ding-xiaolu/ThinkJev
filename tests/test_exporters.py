"""Public-source exporters: pure conversion on fake EvalScope Samples and fake dataset rows.

No network, and neither evalscope nor datasets is imported (the last test runs EvalScope's
real adapter path offline only when evalscope is installed, i.e. in the data environment).
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import export_evalscope as es  # noqa: E402
import export_hf as eh  # noqa: E402
import public_rows as pr  # noqa: E402

from thinkjev._semif import validate_row  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"


def test_exporters_do_not_import_data_libraries():
    for module in (es, eh, pr):
        source = Path(module.__file__).read_text()
        top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert not any("evalscope" in line or "datasets" in line for line in top_level), module.__name__


def sample(input, choices=None, target="A", raw=None, **metadata):
    meta = dict(metadata)
    if raw is not None:
        meta[es.RAW_KEY] = raw
    return SimpleNamespace(input=input, choices=choices, target=target, metadata=meta, subset_key=None, id=0)


def convert(source_name, subset, samples, cap=None):
    tally = pr.Tally()
    rows = es.convert_samples(es.SOURCES[source_name], subset, samples, dataset_id="org/data", revision=None,
                              hub="modelscope", cap=cap, tally=tally)
    return rows, tally


def assert_schema(rows, lang):
    for row in rows:
        validate_row(row)
        assert set(row) == {"id", "group_id", "family", "lang", "state", "question", "options", "label",
                            "target_distribution", "split", "provenance"}
        assert row["lang"] == lang and row["split"] == "test" and row["target_distribution"] is None
        assert 2 <= len(row["options"]) <= 16 and 0 <= row["label"] < len(row["options"])
        assert {"source", "upstream", "revision", "license", "row_ref"} <= row["provenance"].keys()


BBH_INPUT = ("Which sentence has the correct adjective order:\nOptions:\n(A) old little red car\n"
             "(B) little old red car\n(C) red old little car")


def test_bbh_options_leave_the_stem_and_map_the_gold_letter():
    many = "Q?\nOptions:\n" + "\n".join(f"({chr(65 + i)}) item {i}" for i in range(18))
    rows, tally = convert("bbh", "hyperbaton", [
        sample(BBH_INPUT, target="B"),
        sample(many, target="C"),
        sample("No options at all", target="valid"),
    ])
    assert_schema(rows, "en")
    (row,) = rows
    assert row["state"] == "Which sentence has the correct adjective order:"
    assert "Options" not in row["state"] and "(A)" not in row["state"]
    assert row["question"] == pr.GENERIC_QUESTION["en"]
    assert [o["description"] for o in row["options"]] == ["old little red car", "little old red car", "red old little car"]
    assert [o["id"] for o in row["options"]] == ["A", "B", "C"] and row["label"] == 1
    assert row["id"] == "bbh/hyperbaton/0" and row["group_id"] == row["id"] and row["family"] == "bbh/hyperbaton"
    assert tally.dropped["hyperbaton"] == Counter({"too_many_options": 1, "no_options_block": 1})
    assert tally.as_dict()["rows_dropped"] == 2 and tally.as_dict()["rows_written"] == 1


def test_bbh_multiple_choice_list_matches_evalscope_and_excludes_free_form():
    assert len(es.BBH_MULTIPLE_CHOICE) == 17 and "boolean_expressions" not in es.BBH_MULTIPLE_CHOICE


def test_mcq_uses_raw_stem_native_id_and_keeps_option_order():
    options = [f"opt {i}" for i in range(10)]
    raw = {"question": "What is 2+2?", "question_id": 70, "options": options, "src": "ori_mmlu-math"}
    rendered = "Answer the following...\nQuestion:\nWhat is 2+2?\nOptions:\nA) opt 0 ..."
    rows, _ = convert("mmlu_pro", "math", [sample(rendered, options, "J", raw=raw)])
    assert_schema(rows, "en")
    (row,) = rows
    assert row["state"] == "What is 2+2?" and row["id"] == "mmlu_pro/math/70"
    assert [o["description"] for o in row["options"]] == options and row["label"] == 9
    assert row["provenance"]["row_ref"] == {"subset": "math", "index": 0, "native_id": "70"}
    assert row["provenance"]["evalscope"] == es.EVALSCOPE_VERSION


def test_gpqa_keeps_case_sensitive_record_ids():
    raw = {"Question": "Q", "Record ID": "recAbC/12"}
    rows, _ = convert("gpqa_diamond", "default", [sample("Q", ["a", "b", "c", "d"], "D", raw=raw)])
    assert rows[0]["id"] == "gpqa_diamond/default/recAbC_12" and rows[0]["label"] == 3


def test_gold_problems_are_dropped_and_counted():
    raw = {"question": "题目", "id": 3}
    rows, tally = convert("ceval", "computer_network", [
        sample("题目", ["甲", "乙", "丙", "丁"], "E", raw=raw),
        sample("题目", ["甲", "乙", "丙", "丁"], "AB", raw={"question": "题目", "id": 4}),
        sample("题目", ["甲", "乙", "丙", "丁"], "", raw={"question": "题目", "id": 5}),
        sample("题目", None, "A", raw={"question": "题目", "id": 6}),
        sample("题目", ["甲", "乙", "丙", "丁"], "B", raw={"question": "题目", "id": 7}),
        sample("题目", ["甲", "乙", "丙", "丁"], "C", raw={"question": "题目", "id": 7}),
    ])
    assert tally.dropped["computer_network"] == Counter({
        "gold_out_of_range": 1, "multi_answer_gold": 1, "unparseable_gold": 1, "no_choices": 1,
        "duplicate_native_id": 1})
    assert_schema(rows, "zh")
    assert [row["id"] for row in rows] == ["ceval/computer_network/7"]
    assert rows[0]["question"] == pr.GENERIC_QUESTION["zh"] and rows[0]["label"] == 1


def test_label_from_target_forms():
    assert pr.label_from_target("(C)", 4) == 2 and pr.label_from_target(["B"], 4) == 1
    assert pr.label_from_target(0, 2) == 0
    for bad in (["A", "B"], True, None, "P", "yes"):
        with pytest.raises(pr.Drop):
            pr.label_from_target(bad, 4)


def test_context_rows_share_groups_and_keep_choices_out_of_state():
    shared = "All birds fly. Tweety is a bird."
    samples = [
        sample(f"{shared}\nDoes Tweety fly?", ["Yes", "No"], "A", raw={"context": shared, "question": "Does Tweety fly?"}),
        sample(f"{shared}\nIs Tweety a bird?", ["Yes", "No"], "A", raw={"context": shared, "question": "Is Tweety a bird?"}),
        sample("Other.\nQ?", ["Yes", "No"], "B", raw={"context": "Other.", "question": "Q?"}),
    ]
    rows, _ = convert("logiqa_en", "default", samples)
    assert_schema(rows, "en")
    assert rows[0]["group_id"] == rows[1]["group_id"] != rows[2]["group_id"]
    assert rows[0]["state"] == shared and rows[0]["question"] == "Does Tweety fly?"
    assert all("Yes" not in row["state"] for row in rows)


def test_logiqa_zh_strips_native_letter_prefixes():
    raw = {"passage": "所有的鸟都会飞。", "question": "以下哪项为真？"}
    options = ["(A)鸟会飞", "(B)鸟不会飞", "(C)无法判断", "(D)以上都不对"]
    rows, _ = convert("logiqa_zh", "logiqa-zh", [sample("x", options, "A", raw=raw)])
    assert_schema(rows, "zh")
    assert [o["description"] for o in rows[0]["options"]] == ["鸟会飞", "鸟不会飞", "无法判断", "以上都不对"]
    assert es.strip_option_prefix("A. 甲", 0) == "甲" and es.strip_option_prefix("(B) x", 0) == "(B) x"


def test_sampling_is_deterministic_group_whole_and_order_free():
    samples = [sample(f"s{i}", ["x", "y"], "A", raw={"narrative": f"ctx{i // 3}", "question": f"q{i}"})
               for i in range(60)]
    first, tally = convert("musr", "murder_mysteries", samples, cap=20)
    second, _ = convert("musr", "murder_mysteries", samples, cap=20)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) <= 20 and tally.capped["murder_mysteries"] == 60 - len(first)
    groups = Counter(row["group_id"] for row in first)
    assert set(groups.values()) == {3}
    shuffled = pr.sample_groups(list(reversed(convert("musr", "murder_mysteries", samples)[0])), 20)
    assert {row["id"] for row in shuffled} == {row["id"] for row in first}


def fake_loader(source, *, revision, hub, cache_dir, subsets):
    yield "hyperbaton", [sample(BBH_INPUT, target="A"), sample("free form", target="x")], "evalscope/bbh"
    yield "snarks", [sample("Which is sarcastic?\nOptions:\n(A) a\n(B) b", target="(B)")], "evalscope/bbh"


def test_export_writes_only_the_output_and_its_meta(tmp_path):
    output = tmp_path / "out" / "bbh.jsonl"
    meta = es.export(es.SOURCES["bbh"], output, revision="v1", hub="modelscope", cache_dir=tmp_path / "cache",
                     cap=None, subsets=None, loader=fake_loader)
    files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert files == ["out/bbh.jsonl", "out/bbh.jsonl.meta.json"]
    payload = output.read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    assert_schema(rows, "en")
    saved = json.loads((tmp_path / "out/bbh.jsonl.meta.json").read_text())
    assert saved == meta and saved["sha256"] == hashlib.sha256(payload).hexdigest()
    assert saved["rows"] == 2 and saved["drop_reasons"] == {"no_options_block": 1}
    assert saved["evalscope"] == es.EVALSCOPE_VERSION and saved["revision"] == "v1"
    with pytest.raises(FileExistsError):
        es.export(es.SOURCES["bbh"], output, revision="v1", hub="modelscope", cache_dir=tmp_path / "cache",
                  cap=None, subsets=None, loader=fake_loader)


def test_evalscope_cli_refuses_skipped_sources(tmp_path):
    with pytest.raises(SystemExit, match="grid_mode"):
        es.main(["--source", "zebralogicbench", "--output", str(tmp_path / "z.jsonl")])
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------- Hugging Face sources


def anyjev_split(items, n_test, n_calib, seed):
    """Verbatim logic of AnyJev bench/tasks/base.py Task.split."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n_test]], [items[i] for i in idx[n_test:n_test + n_calib]]


def test_anyjev_roles_match_anyjev_task_split():
    roles = eh.anyjev_roles(800)
    test, calib = anyjev_split(list(range(800)), 300, 200, 0)
    assert [i for i, role in enumerate(roles) if role == "test"] == sorted(test)
    assert [i for i, role in enumerate(roles) if role == "calib"] == sorted(calib)
    assert Counter(roles) == {"test": 300, "calib": 200, "unused": 300}


def banking_fixture():
    names = {label: f"intent_{label:02d}_name" for label in range(25)}
    train = [{"text": "t", "label": label, "label_text": names[label]}
             for label in range(25) for _ in range(100 - label)]  # label 0 most frequent
    test = [{"text": f"query {i}", "label": i % 25, "label_text": names[i % 25]} for i in range(250)]
    return train, test, names


def test_banking20_follows_anyjev_with_16_candidates():
    train, test, names = banking_fixture()
    tally = pr.Tally()
    rows = eh.banking20_rows(train, test, revision=REV, tally=tally)
    top = sorted(label for label, _ in Counter(row["label"] for row in train).most_common(20))
    assert top == list(range(20))
    assert len(rows) == 200 and tally.as_dict()["drop_reasons"] == {"label_outside_top20": 50}
    assert_schema(rows, "en")
    order = [names[label] for label in top]
    for row in rows:
        ids = [o["id"] for o in row["options"]]
        assert len(ids) == 16 and ids == sorted(ids, key=order.index)
        index = row["provenance"]["row_ref"]["index"]
        assert ids[row["label"]] == test[index]["label_text"]
        assert row["options"][row["label"]]["description"] == test[index]["label_text"].replace("_", " ")
        assert row["provenance"]["anyjev_role"] in {"test", "calib", "unused"}
    again = eh.banking20_rows(train, test, revision=REV, tally=pr.Tally())
    assert again == rows
    assert len({tuple(o["id"] for o in row["options"]) for row in rows}) > 1  # candidates vary by row


def test_clinc_escalate_exports_anyjev_items_only():
    names = ["oos", "balance", "transfer"]
    splits = {split: [{"text": f"{split} {i}", "intent": i % 3} for i in range(300)]
              for split in ("test", "validation", "train")}
    tally = pr.Tally()
    rows = eh.clinc_escalate_rows(splits, names, revision=REV, tally=tally)
    assert_schema(rows, "en")
    assert Counter(row["provenance"]["anyjev_role"] for row in rows) == {"test": 300, "calib": 200}
    assert tally.as_dict()["rows_capped"] == 400
    for row in rows:
        split, index = row["provenance"]["row_ref"]["split"], row["provenance"]["row_ref"]["index"]
        is_oos = splits[split][index]["intent"] == 0
        assert row["options"][row["label"]]["description"] == ("No" if is_oos else "Yes")
    assert rows[0]["question"] == eh.CLINC_ESCALATE_QUESTION


def test_clinc_domain_map_is_complete_and_intent_rows_use_it():
    intents = [intent for values in eh.CLINC_DOMAINS.values() for intent in values]
    assert len(eh.CLINC_DOMAINS) == 10 and len(intents) == 150 == len(set(intents))
    names = ["oos", "balance", "weather", "not_a_clinc_intent"]
    test = [{"text": f"q{i}", "intent": i % 4} for i in range(40)]
    tally = pr.Tally()
    rows = eh.clinc_intent_rows(test, names, revision=REV, tally=tally, cap=None)
    assert_schema(rows, "en")
    assert tally.dropped["test"] == Counter({"out_of_scope_in_intent_task": 10, "intent_not_in_domain_map": 10})
    for row in rows:
        assert len(row["options"]) == 15
        gold = names[test[row["provenance"]["row_ref"]["index"]]["intent"]]
        assert row["options"][row["label"]]["id"] == gold
        assert row["family"] == f"clinc150/{eh.CLINC_DOMAIN_OF[gold]}"
    capped = eh.clinc_intent_rows(test, names, revision=REV, tally=pr.Tally(), cap=7)
    assert len(capped) == 7


def typed_record(case="c1", label="refund"):
    questions = {
        "action": {"type": "choice", "instructions": "What should happen next?",
                   "criteria": {"refund": "Issue a refund", "escalate": "Send to a human", "close": "Close it"}},
        "needs_human": {"type": "noul", "instructions": "A human must review this.",
                        "criteria": {"true": "review needed", "false": "no review"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": {"0": "low", "1": "mid", "2": "high"}},
    }
    gold = {
        "action": {"label": label, "probabilities": {"refund": 0.7, "escalate": 0.2, "close": 0.1}},
        "needs_human": {"label": "false", "probabilities": {"true": 0.4, "false": 0.6}},
        "urgency": {"label": 2, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "score": 1.6},
    }
    return {"id": case, "workflow": "customer_service", "state": json.dumps({"ticket": "My card was charged twice"}),
            "questions": json.dumps(questions), "gold": json.dumps(gold)}


def test_typed_decisions_rows_per_question_share_the_case_group():
    tally = pr.Tally()
    rows = eh.typed_decision_rows([typed_record(), typed_record("c2", label="unknown")], revision=REV, tally=tally)
    assert_schema(rows, "en")
    assert tally.dropped["customer_service"] == Counter({"gold_not_in_options": 1}) and len(rows) == 5
    by_id = {row["id"]: row for row in rows}
    action = by_id["typed-decisions/customer_service/c1/action"]
    assert action["options"][action["label"]]["description"] == "refund: Issue a refund"
    assert action["provenance"]["teacher_probabilities"] == [0.7, 0.2, 0.1]
    noul = by_id["typed-decisions/customer_service/c1/needs_human"]
    assert noul["question"].startswith("Is the following statement true?")
    assert [o["description"] for o in noul["options"]] == ["Yes: review needed", "No: no review"] and noul["label"] == 1
    score = by_id["typed-decisions/customer_service/c1/urgency"]
    assert score["label"] == 2 and score["provenance"]["teacher_score"] == 1.6
    assert {row["group_id"] for row in rows if "/c1/" in row["id"]} == {"typed-decisions/customer_service/c1"}
    assert json.loads(action["state"]) == {"ticket": "My card was charged twice"}


def test_preference_rows_pick_balanced_positions_truncate_and_drop_ties():
    records = [{"id": str(i), "prompt": f"prompt {i}", "chosen": [f"good {i}"],
                "rejected": [f"bad {i} a", f"bad {i} b", f"bad {i} c"], "subset": "Focus"} for i in range(200)]
    records.append({"id": "long", "prompt": "p" * 50, "chosen": ["g" * 30], "rejected": ["r"], "subset": "Math"})
    records.append({"id": "tie", "prompt": "p", "chosen": ["a", "b"], "rejected": ["c"], "subset": "Ties"})
    tally = pr.Tally()
    rows = eh.preference_rows(records, revision=REV, tally=tally, cap=None, max_response_chars=20,
                              max_prompt_chars=40)
    assert_schema(rows, "en")
    assert tally.dropped == {"Ties": Counter({"ties_multiple_correct": 1})}
    for row in rows:
        assert row["options"][row["label"]]["description"].startswith(("good", "g"))
        assert [o["id"] for o in row["options"]] == ["A", "B"]
    assert 70 < Counter(row["label"] for row in rows)[0] < 130
    long = next(row for row in rows if row["id"] == "rewardbench2/math/long")
    assert long["provenance"]["truncated"] == {"prompt": 50, "chosen": 30}
    assert long["state"].endswith(eh.TRUNCATION_MARK) and len(long["state"]) == 40 + len(eh.TRUNCATION_MARK)
    assert eh.preference_rows(records, revision=REV, tally=pr.Tally(), cap=None, max_response_chars=20,
                              max_prompt_chars=40) == rows
    capped = eh.preference_rows(records, revision=REV, tally=pr.Tally(), cap=50)
    assert Counter(row["family"] for row in capped) == {"rewardbench2/focus": 50, "rewardbench2/math": 1}


def test_hf_cli_requires_a_full_revision_before_any_download(tmp_path):
    with pytest.raises(SystemExit):
        eh.main(["--source", "banking20", "--output", str(tmp_path / "b.jsonl")])
    with pytest.raises(SystemExit):
        eh.main(["--source", "banking20", "--revision", "main", "--output", str(tmp_path / "b.jsonl")])
    assert not any(tmp_path.iterdir())


def test_write_rows_is_create_only_and_validates(tmp_path):
    row = pr.make_row(row_id="x/1", group_id="x/1", family="x", lang="en", state="s", question="q",
                      options=[("A", "a"), ("B", "b")], label=0, provenance={})
    pr.write_rows(tmp_path / "rows.jsonl", [row], {})
    with pytest.raises(FileExistsError):
        pr.write_rows(tmp_path / "rows.jsonl", [row], {})
    with pytest.raises(ValueError):
        pr.write_rows(tmp_path / "dup.jsonl", [row, row], {})
    with pytest.raises(pr.Drop, match="too_many_options"):
        pr.make_row(row_id="x/2", group_id="x/2", family="x", lang="en", state="s", question="q",
                    options=[(str(i), "o") for i in range(17)], label=0, provenance={})


# ---------------------------------------------------------------- real EvalScope path, offline


def test_evalscope_adapter_path_offline(monkeypatch, tmp_path):
    """EvalScope's registry, adapter, loader, and record_to_sample on fake records (data env only)."""
    pytest.importorskip("evalscope")
    datasets = pytest.importorskip("datasets")
    import evalscope.api.dataset.loader as loader

    ten = [f"choice {i}" for i in range(10)]
    records = {
        ("evalscope/bbh", "hyperbaton", "test"): [{"input": BBH_INPUT, "target": "(B)"}],
        ("TIGER-Lab/MMLU-Pro", "default", "test"): [
            {"question_id": 70, "question": "What is 2+2?", "options": ten, "answer": "C", "answer_index": 2,
             "cot_content": "", "category": "math", "src": "ori_mmlu-high_school_mathematics"}],
        ("AI-ModelScope/gpqa_diamond", "default", "train"): [
            {"Question": "Invented question?", "Correct Answer": "right", "Incorrect Answer 1": "wrong one",
             "Incorrect Answer 2": "wrong two\n", "Incorrect Answer 3": "wrong three", "Record ID": "recXyZ"}],
        ("m-a-p/SuperGPQA", "default", "train"): [
            {"uuid": "ab12", "question": "Q?", "options": ten, "answer": "choice 3", "answer_letter": "D",
             "discipline": "Science", "field": "Mathematics", "subfield": "Algebra", "difficulty": "hard",
             "is_calculation": True}],
        ("extraordinarylab/logiqa", "default", "test"): [
            {"context": "All birds fly.", "question": "Which is true?", "choices": ["a", "b", "c", "d"], "answer": "C"}],
        ("AI-ModelScope/MuSR", "default", "murder_mysteries"): [
            {"narrative": "A long story.", "question": "Who did it?", "choices": "['Ann', 'Bob']", "answer_index": 1}],
        ("evalscope/ceval", "computer_network", "val"): [
            {"id": 0, "question": "网络层协议是？", "A": "IP", "B": "TCP", "C": "HTTP", "D": "FTP", "answer": "A",
             "explanation": ""}],
        ("evalscope/cmmlu", "default", "test"): [
            {"question": "虚构的题目", "choices": ["(A) 选项一", "(B) 选项二", "(C) 选项三", "(D) 选项四"],
             "answer": "(B)", "category": "agronomy"}],
        ("opencompass/agieval", "logiqa-zh", "test"): [
            {"passage": "所有的鸟都会飞。", "question": "以下哪项为真？", "options": ["(A)甲", "(B)乙", "(C)丙", "(D)丁"],
             "label": "B"}],
    }
    loaded = []

    class FakeHub:
        def __init__(self, data_id_or_path, **kwargs):
            self.data_id = data_id_or_path

        def load(self, split, subset="default", **kwargs):
            loaded.append((self.data_id, subset, split))
            return datasets.Dataset.from_list(records[(self.data_id, subset, split)])

    monkeypatch.setattr(loader, "DatasetHub", FakeHub)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "cache" / "modelscope"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache" / "hf-datasets"))
    checks = {
        "bbh": ("hyperbaton", lambda r: r["label"] == 1 and "Options" not in r["state"]),
        "mmlu_pro": ("math", lambda r: r["id"] == "mmlu_pro/math/70" and r["state"] == "What is 2+2?" and r["label"] == 2),
        "gpqa_diamond": ("default", lambda r: r["id"] == "gpqa_diamond/default/recXyZ"
                         and r["options"][r["label"]]["description"] == "right"),
        "super_gpqa": ("Mathematics", lambda r: r["id"] == "super_gpqa/mathematics/ab12" and r["label"] == 3
                       and len(r["options"]) == 10 and r["provenance"]["extra"]["difficulty"] == "hard"),
        "logiqa_en": ("default", lambda r: r["state"] == "All birds fly." and r["question"] == "Which is true?"
                      and r["label"] == 2),
        "musr": ("murder_mysteries", lambda r: r["state"] == "A long story." and r["label"] == 1),
        "ceval": ("computer_network", lambda r: r["state"] == "网络层协议是？" and r["id"] == "ceval/computer_network/0"),
        "cmmlu": ("agronomy", lambda r: r["options"][1]["description"] == "选项二" and r["label"] == 1),
        "logiqa_zh": ("logiqa-zh", lambda r: [o["description"] for o in r["options"]] == ["甲", "乙", "丙", "丁"]
                      and r["label"] == 1 and r["state"] == "所有的鸟都会飞。"),
    }
    assert set(checks) == set(es.SOURCES)
    for name, (subset, check) in checks.items():
        output = tmp_path / f"{name}.jsonl"
        meta = es.export(es.SOURCES[name], output, revision=None, hub="modelscope", cache_dir=tmp_path / "cache",
                         cap=None, subsets=(subset,))
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert meta["rows"] == 1 and check(rows[0]), (name, rows, meta)
        assert_schema(rows, es.SOURCES[name].lang)
    assert len(loaded) == len(checks)


def test_hf_main_offline(monkeypatch, tmp_path):
    """export_hf.main with datasets.load_dataset patched to local fakes (data env only)."""
    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache" / "hf-datasets"))
    train, test, _ = banking_fixture()
    intents = datasets.Features({"text": datasets.Value("string"),
                                 "intent": datasets.ClassLabel(names=["oos", "balance", "weather"])})
    fakes = {
        ("mteb/banking77", "train"): datasets.Dataset.from_list(train),
        ("mteb/banking77", "test"): datasets.Dataset.from_list(test),
        ("clinc/clinc_oos", "test"): datasets.Dataset.from_list(
            [{"text": f"q{i}", "intent": i % 3} for i in range(30)], features=intents),
    }
    calls = []

    def fake_load(name, config=None, *, split, revision, cache_dir):
        calls.append((name, config, split, revision))
        return fakes[(name, split)]

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    eh.main(["--source", "banking20", "--revision", REV, "--output", str(tmp_path / "out/banking20.jsonl"),
             "--cache-dir", str(tmp_path / "cache")])
    eh.main(["--source", "clinc_intent", "--revision", REV, "--output", str(tmp_path / "out/clinc.jsonl"),
             "--cache-dir", str(tmp_path / "cache")])
    assert calls == [("mteb/banking77", None, "train", REV), ("mteb/banking77", None, "test", REV),
                     ("clinc/clinc_oos", "plus", "test", REV)]
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "banking20.jsonl", "banking20.jsonl.meta.json", "clinc.jsonl", "clinc.jsonl.meta.json"]
    meta = json.loads((tmp_path / "out/clinc.jsonl.meta.json").read_text())
    assert meta["rows"] == 20 and meta["drop_reasons"] == {"out_of_scope_in_intent_task": 10}
    assert meta["revision"] == REV and meta["datasets"] == eh.DATASETS_VERSION
