"""Export public benchmarks to ThinkJev rows through EvalScope's own dataset adapters.

The English reasoning and Chinese benchmarks of the release split.
We do not maintain per-dataset download code: EvalScope's registered adapter downloads the
dataset (ModelScope by default, or Hugging Face) and its `record_to_sample` parses it. We
call `adapter.load()`, which loads raw `Sample`s without rendering any prompt and without
running a model, and we wrap `record_to_sample` so every Sample also carries the raw record.
The raw record gives the clean stem, context, and native ID, so choices are never
duplicated inside `state`.

Runs in a separate data environment, never in the scorer environment:

    python3 -m venv build/venv-data
    build/venv-data/bin/pip install --no-cache-dir -r benchmarks/requirements-data.txt  # evalscope==1.12.0
    build/venv-data/bin/python benchmarks/export_evalscope.py --source ceval \
        --output build/public/ceval.jsonl

Sources (defaults give about 4.6k English and 3.3k Chinese rows before the dev/test split):

| source        | EvalScope benchmark | subsets                              | cap/subset | lang |
|---------------|---------------------|--------------------------------------|-----------:|------|
| bbh           | bbh                 | the 17 multiple-choice tasks         | 60         | en   |
| mmlu_pro      | mmlu_pro            | math, physics, engineering, chemistry, computer science; law, history (knowledge contrast) | 170 | en |
| gpqa_diamond  | gpqa_diamond        | all 198                              | none       | en   |
| super_gpqa    | super_gpqa          | all 72 fields                        | 15         | en   |
| logiqa_en     | logi_qa             | test                                 | 500        | en   |
| musr          | musr                | murder_mysteries, object_placements, team_allocation | 200 | en |
| ceval         | ceval               | all 52 subjects, `val` split (test answers are hidden) | none | zh |
| cmmlu         | cmmlu               | all 67 subjects (test answers are public) | 20    | zh   |
| logiqa_zh     | agieval             | logiqa-zh                            | none       | zh   |

Not exported: `zebralogicbench`. EvalScope's adapter loads `grid_mode`, whose gold is a
full JSON grid (free-form), so it has no option list. Upstream's gated `mc_mode` config is
multiple choice but EvalScope's `record_to_sample` cannot parse it; add a dedicated
exporter if it is wanted.

Per-dataset choices of `state` and `question`:
- Plain MCQ (mmlu_pro, gpqa_diamond, super_gpqa, ceval, cmmlu, bbh): `state` is the question
  stem and `question` is the generic "Which option correctly answers the question?" (Chinese
  rows: "哪个选项正确回答了这个问题？"). BBH's input embeds an "Options:" block; it is parsed
  into options and removed from the stem.
- Context + question (logiqa_en, logiqa_zh, musr): `state` is the passage or narrative,
  `question` is the dataset's own question. Rows that share a passage share `group_id`.
- Options keep EvalScope's order. For GPQA that order is EvalScope's deterministic,
  question-hashed shuffle (the upstream records have no order: one correct and three
  incorrect answers); for every other source it is the dataset's own order.

Pinning: pass `--revision` (a dataset commit/tag accepted by the hub) to pin the upstream.
Either way the sidecar `<output>.meta.json` records the EvalScope version, dataset ID, hub,
revision, per-subset counts, drop reasons, and the SHA-256 of the exported JSONL, so a
rebuild can be checked byte for byte. Caps sample whole groups in SHA-256(group_id) order
(`--limit-per-subset` overrides the default; 0 means no cap).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_rows import (  # noqa: E402
    GENERIC_QUESTION,
    Drop,
    Tally,
    id_part,
    label_from_target,
    letter_options,
    make_row,
    sample_groups,
    slug,
    text_group,
    write_rows,
)

EXPORTER_VERSION = "export-evalscope-v1"
EVALSCOPE_VERSION = "1.12.0"
RAW_KEY = "_thinkjev_raw_record"

# EvalScope 1.12.0 `benchmarks/bbh/bbh_adapter.py` MULTIPLE_CHOICE_LIST (free-form tasks are
# not multiple choice). reasoning_about_colored_objects lists 18 options, so its rows drop
# as too_many_options and are counted.
BBH_MULTIPLE_CHOICE = (
    "temporal_sequences", "disambiguation_qa", "date_understanding", "tracking_shuffled_objects_three_objects",
    "penguins_in_a_table", "geometric_shapes", "snarks", "ruin_names", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_five_objects", "logical_deduction_three_objects", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "movie_recommendation",
    "salient_translation_error_detection", "reasoning_about_colored_objects",
)
MMLU_PRO_SUBJECTS = ("math", "physics", "engineering", "chemistry", "computer science", "law", "history")


@dataclass
class Converted:
    state: str
    question: str
    options: list[str]
    label: int
    native_id: str | None = None
    context: str | None = None  # rows with the same context share a group
    extra: dict = field(default_factory=dict)


def _raw(sample) -> dict:
    return (getattr(sample, "metadata", None) or {}).get(RAW_KEY) or {}


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _choices(sample) -> list[str]:
    choices = getattr(sample, "choices", None)
    if not choices:
        raise Drop("no_choices")
    return [str(choice) for choice in choices]


def _stem_row(lang: str, stem: str, sample, native_id=None, extra=None) -> Converted:
    options = _choices(sample)
    if len(options) > 16:
        raise Drop("too_many_options")
    return Converted(stem, GENERIC_QUESTION[lang], options, label_from_target(sample.target, len(options)),
                     native_id=native_id, extra=extra or {})


def _clean_input(sample) -> str:
    value = sample.input
    if not isinstance(value, str):
        raise Drop("non_text_input")
    return value.strip()


OPTION_LINE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def split_bbh(text: str) -> tuple[str, list[str]]:
    """Split a BBH input into its stem and the options listed after the last "Options:"."""
    position = text.rfind("Options:")
    if position < 0:
        raise Drop("no_options_block")
    stem = text[:position].rstrip()
    options: list[str] = []
    for line in text[position + len("Options:"):].splitlines():
        line = line.strip()
        if not line:
            continue
        match = OPTION_LINE.match(line)
        if match:
            if match.group(1) != chr(ord("A") + len(options)):
                raise Drop("unexpected_option_letters")
            options.append(match.group(2).strip())
        elif options:
            options[-1] = f"{options[-1]}\n{line}"
        else:
            raise Drop("unparseable_options")
    if not stem:
        raise Drop("empty_state")
    return stem, options


def convert_bbh(sample, subset: str) -> Converted:
    stem, options = split_bbh(_clean_input(sample))
    if len(options) > 16:
        raise Drop("too_many_options")
    return Converted(stem, GENERIC_QUESTION["en"], options, label_from_target(sample.target, len(options)))


def convert_mmlu_pro(sample, subset: str) -> Converted:
    raw = _raw(sample)
    stem = _text(raw.get("question")) or _clean_input(sample)
    native = raw.get("question_id", (sample.metadata or {}).get("question_id"))
    return _stem_row("en", stem, sample, native_id=None if native is None else str(native),
                     extra={"src": raw.get("src")} if raw.get("src") else None)


def convert_gpqa(sample, subset: str) -> Converted:
    raw = _raw(sample)
    stem = _text(raw.get("Question")) or _clean_input(sample)
    native = _text(raw.get("Record ID")) or None
    extra = {key: raw[key] for key in ("High-level domain", "Subdomain") if raw.get(key)}
    return _stem_row("en", stem, sample, native_id=native, extra=extra)


def convert_super_gpqa(sample, subset: str) -> Converted:
    raw = _raw(sample)
    stem = _text(raw.get("question")) or _clean_input(sample)
    native = _text(raw.get("uuid")) or _text((sample.metadata or {}).get("uuid")) or None
    extra = {key: raw[key] for key in ("discipline", "field", "subfield", "difficulty", "is_calculation")
             if raw.get(key) is not None}
    return _stem_row("en", stem, sample, native_id=native, extra=extra)


def convert_context_question(lang: str, context_key: str, question_key: str):
    def convert(sample, subset: str) -> Converted:
        raw = _raw(sample)
        context, question = _text(raw.get(context_key)), _text(raw.get(question_key))
        if not raw:
            raise Drop("missing_raw_record")
        options = _choices(sample)
        if len(options) > 16:
            raise Drop("too_many_options")
        label = label_from_target(sample.target, len(options))
        if not context:  # no passage: behave like a plain MCQ
            return Converted(question, GENERIC_QUESTION[lang], options, label)
        if not question:
            raise Drop("empty_question")
        return Converted(context, question, options, label, context=context)
    return convert


AGIEVAL_PREFIX = re.compile(r"^\s*[\(（]?([A-Z])[\)）]?\s*[\.．、:：]?\s*")


def strip_option_prefix(text: str, index: int) -> str:
    """Strip a leading "(A)" / "A." label when it matches the option's position."""
    match = AGIEVAL_PREFIX.match(text)
    if match and match.group(1) == chr(ord("A") + index) and match.end() < len(text):
        return text[match.end():].strip()
    return text.strip()


def convert_logiqa_zh(sample, subset: str) -> Converted:
    converted = convert_context_question("zh", "passage", "question")(sample, subset)
    converted.options = [strip_option_prefix(option, index) for index, option in enumerate(converted.options)]
    return converted


def convert_ceval(sample, subset: str) -> Converted:
    raw = _raw(sample)
    stem = _text(raw.get("question")) or _clean_input(sample)
    native = raw.get("id", (sample.metadata or {}).get("id"))
    return _stem_row("zh", stem, sample, native_id=None if native in (None, "") else str(native))


def convert_cmmlu(sample, subset: str) -> Converted:
    raw = _raw(sample)
    return _stem_row("zh", _text(raw.get("question")) or _clean_input(sample), sample)


@dataclass(frozen=True)
class Source:
    name: str
    benchmark: str
    lang: str
    license: str
    convert: Callable[[Any, str], Converted]
    subsets: tuple[str, ...] | None = None  # None: the adapter's own subset list
    cap: int | None = None
    eval_split: str | None = None


SOURCES = {
    source.name: source
    for source in (
        Source("bbh", "bbh", "en", "MIT (BIG-Bench Hard)", convert_bbh, BBH_MULTIPLE_CHOICE, 60),
        Source("mmlu_pro", "mmlu_pro", "en", "MIT", convert_mmlu_pro, MMLU_PRO_SUBJECTS, 170),
        Source("gpqa_diamond", "gpqa_diamond", "en", "CC-BY-4.0; do not reveal examples online",
               convert_gpqa),
        Source("super_gpqa", "super_gpqa", "en", "ODC-BY-1.0", convert_super_gpqa, None, 15),
        Source("logiqa_en", "logi_qa", "en", "LogiQA: no license file upstream; research use",
               convert_context_question("en", "context", "question"), None, 500),
        Source("musr", "musr", "en", "CC-BY-4.0", convert_context_question("en", "narrative", "question"),
               ("murder_mysteries", "object_placements", "team_allocation"), 200),
        Source("ceval", "ceval", "zh", "CC-BY-NC-SA-4.0", convert_ceval, None, None, "val"),
        Source("cmmlu", "cmmlu", "zh", "CC-BY-NC-4.0", convert_cmmlu, None, 20),
        Source("logiqa_zh", "agieval", "zh", "AGIEval MIT; LogiQA text: no license file upstream; research use",
               convert_logiqa_zh, ("logiqa-zh",)),
    )
}
SKIPPED = {
    "zebralogicbench": "EvalScope's adapter loads grid_mode, whose gold is a JSON grid (free-form), not options",
}


def convert_samples(source: Source, subset: str, samples: Iterable, *, dataset_id: str, revision: str | None,
                    hub: str, cap: int | None, tally: Tally) -> list[dict]:
    """Pure conversion of one subset's EvalScope Samples into capped, validated rows."""
    subset_slug = slug(subset) if subset else "default"
    prefix = f"{source.name}/{subset_slug}"
    rows, used = [], set()
    for index, sample in enumerate(samples):
        tally.seen[subset] += 1
        try:
            converted = source.convert(sample, subset)
            row_id = f"{prefix}/{id_part(converted.native_id) if converted.native_id else index}"
            if row_id in used:
                raise Drop("duplicate_native_id")
            used.add(row_id)
            provenance = {
                "source": f"evalscope:{source.benchmark}",
                "upstream": dataset_id,
                "revision": revision,
                "hub": hub,
                "evalscope": EVALSCOPE_VERSION,
                "exporter": EXPORTER_VERSION,
                "license": source.license,
                "row_ref": {"subset": subset, "index": index, "native_id": converted.native_id},
            }
            if converted.extra:
                provenance["extra"] = converted.extra
            rows.append(make_row(
                row_id=row_id,
                group_id=text_group(prefix, converted.context) if converted.context else row_id,
                family=prefix, lang=source.lang, state=converted.state, question=converted.question,
                options=letter_options(converted.options), label=converted.label, provenance=provenance,
            ))
        except Drop as reason:
            tally.drop(subset, str(reason))
    kept = sample_groups(rows, cap)
    tally.capped[subset] += len(rows) - len(kept)
    tally.kept[subset] += len(kept)
    return kept


def load_evalscope(source: Source, *, revision: str | None, hub: str, cache_dir: Path,
                   subsets: tuple[str, ...] | None):
    """Yield (subset, samples, dataset_id) from EvalScope's adapter; network, no model."""
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir / "modelscope"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "hf-datasets"))
    import evalscope
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    if evalscope.__version__ != EVALSCOPE_VERSION:
        raise SystemExit(f"need evalscope=={EVALSCOPE_VERSION}, found {evalscope.__version__}")
    args: dict[str, Any] = {"few_shot_num": 0}
    if subsets:
        args["subset_list"] = list(subsets)
    if revision:
        args["dataset_revision"] = revision
    if source.eval_split:
        args["eval_split"] = source.eval_split
    config = TaskConfig(datasets=[source.benchmark], dataset_args={source.benchmark: args},
                        dataset_hub=hub, dataset_dir=str(cache_dir / "evalscope"))
    adapter = get_benchmark(source.benchmark, config)
    original = adapter.record_to_sample

    def capture(record):
        produced = original(record)
        for sample in produced if isinstance(produced, list) else [produced]:
            sample.metadata = {**(sample.metadata or {}), RAW_KEY: dict(record)}
        return produced

    adapter.record_to_sample = capture  # the loader reads this attribute at load time
    test_dataset, _ = adapter.load()  # raw Samples: no prompt rendering, no few-shot, no model
    for subset, dataset in test_dataset.items():
        yield subset, list(dataset), adapter.dataset_id


def export(source: Source, output: Path, *, revision: str | None, hub: str, cache_dir: Path,
           cap: int | None, subsets: tuple[str, ...] | None, loader=load_evalscope) -> dict:
    if output.exists() or output.with_name(output.name + ".meta.json").exists():
        raise FileExistsError(f"{output} or its meta exists; outputs are create-only")
    tally, rows, dataset_ids = Tally(), [], set()
    for subset, samples, dataset_id in loader(source, revision=revision, hub=hub, cache_dir=cache_dir,
                                              subsets=subsets):
        dataset_ids.add(dataset_id)
        rows.extend(convert_samples(source, subset, samples, dataset_id=dataset_id, revision=revision, hub=hub,
                                    cap=cap, tally=tally))
    meta = {
        "exporter": EXPORTER_VERSION,
        "source": source.name,
        "evalscope_benchmark": source.benchmark,
        "evalscope": EVALSCOPE_VERSION,
        "upstream": sorted(dataset_ids),
        "hub": hub,
        "revision": revision,
        "license": source.license,
        "lang": source.lang,
        "cap_per_subset": cap,
        "eval_split": source.eval_split,
        **tally.as_dict(),
    }
    return write_rows(output, rows, meta)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=sorted(SOURCES) + sorted(SKIPPED))
    parser.add_argument("--output", type=Path, required=True, help="create-only JSONL under build/")
    parser.add_argument("--revision", help="pin the upstream dataset revision (hub commit or tag)")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "huggingface"])
    parser.add_argument("--cache-dir", type=Path, default=Path("build/cache"))
    parser.add_argument("--limit-per-subset", type=int, help="override the default cap; 0 means no cap")
    parser.add_argument("--subsets", help="comma-separated override of the default subsets")
    args = parser.parse_args(argv)
    if args.source in SKIPPED:
        raise SystemExit(f"{args.source} is not exported: {SKIPPED[args.source]}")
    source = SOURCES[args.source]
    cap = source.cap if args.limit_per_subset is None else (args.limit_per_subset or None)
    subsets = tuple(s.strip() for s in args.subsets.split(",")) if args.subsets else source.subsets
    meta = export(source, args.output, revision=args.revision, hub=args.hub, cache_dir=args.cache_dir,
                  cap=cap, subsets=subsets)
    print(f"wrote {meta['rows']} rows to {args.output} (dropped {meta['rows_dropped']}: {meta['drop_reasons']}; "
          f"sha256 {meta['sha256']})")


if __name__ == "__main__":
    main()
