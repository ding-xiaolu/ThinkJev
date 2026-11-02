"""Create-only JSONL scorer for the S1/S2 self-cascade and its plain S1 readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._semif import load_causal_model, validate_row
from ._semif import score as direct_score
from .cascade import CascadeScorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cascade", "direct"), default="cascade",
                        help="cascade: S1 and S2 over a shared prefix; direct: SemIf's plain S1 readout (router rows)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto",
                        help="Torch GPU (auto prefers CUDA, then Apple MPS)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Model precision; changing it can change option scores")
    parser.add_argument("--think-budget", type=int, default=1024, help="S2 thinking-token budget")
    parser.add_argument("--s2", choices=("always", "gated", "never"), default="always",
                        help="Run S2 on every row (evaluation), only below --threshold, or never "
                             "(S1 only; batch S2 then comes from thinkjev-s2)")
    parser.add_argument("--temperature", type=float, default=1.0, help="S1 calibration temperature")
    parser.add_argument("--threshold", type=float, help="Escalate when calibrated S1 confidence is below this")
    parser.add_argument("--trace", type=Path, help="Create-only JSONL of S2 thoughts, kept out of --output")
    parser.add_argument("--rotations", action="store_true",
                        help="Also read S1 under every cyclic option shift and record the debiased mean")
    parser.add_argument("--features", type=Path,
                        help="Create-only file of S1 hidden states, one np.save record per output row")
    parser.add_argument("--feature-layers", default="16,20,24,28,32",
                        help="Comma-separated hidden-state indices for --features (0 is the embedding)")
    parser.add_argument("--checkpoints",
                        help="Comma-separated thought lengths t at which the answer slots are also read on "
                             "a copy of the cache (the budget-t answer), written per row as s2_anytime")
    parser.add_argument("--prefork-features", type=Path,
                        help="Create-only file of hidden states at the end of the shared prefix, before the "
                             "answer (pre-router probe input), one np.save record per output row; needs "
                             "--rotations and --features")
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.mode == "direct" and (args.s2 != "always" or args.threshold is not None or args.trace
                                  or args.rotations or args.features or args.prefork_features):
        parser.error("--s2 gated/never, --threshold, --trace, --rotations, --features, and --prefork-features "
                     "require --mode cascade")
    if args.prefork_features is not None and not (args.rotations and args.features):
        parser.error("--prefork-features requires --rotations and --features")
    if args.prefork_features is not None and args.prefork_features.exists():
        parser.error("Pre-fork features must be new")
    if args.s2 == "gated" and args.threshold is None:
        parser.error("--s2 gated requires --threshold")
    if args.s2 == "always" and args.threshold is not None:
        parser.error("--threshold requires --s2 gated")
    if args.trace is not None and args.trace.exists():
        parser.error("Trace must be new")
    if args.features is not None and args.features.exists():
        parser.error("Features must be new")
    try:
        layers = tuple(int(value) for value in args.feature_layers.split(",")) if args.features else ()
    except ValueError:
        parser.error("--feature-layers must be comma-separated integers")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.device, args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "direct":
        with args.output.open("x") as destination:
            for row in rows:
                result = direct_score(model, tokenizer, row, metadata, args.max_tokens)
                destination.write(json.dumps(result, allow_nan=False) + "\n")
                destination.flush()
        return
    scorer = CascadeScorer(model, tokenizer, metadata, args.max_tokens, args.think_budget,
                           args.s2, args.temperature, args.threshold, args.rotations, layers,
                           prefork_features=args.prefork_features is not None,
                           checkpoints=tuple(parse_checkpoints(parser, args)))
    with args.output.open("x") as destination:
        trace = args.trace.open("x") if args.trace else None
        features = args.features.open("xb") if args.features else None
        prefork = args.prefork_features.open("xb") if args.prefork_features else None
        try:
            for row in rows:
                result = scorer.score(row)
                thought = result.pop("s2_thought", None)
                thought_ids = result.pop("s2_thought_ids", None)
                hidden = result.pop("s1_features", None)
                before = result.pop("prefork_features", None)
                if features:
                    import numpy as np

                    # Written before the row so every output row has its feature records.
                    np.save(features, hidden, allow_pickle=False)
                    features.flush()
                    if prefork:
                        np.save(prefork, before, allow_pickle=False)
                        prefork.flush()
                destination.write(json.dumps(result, allow_nan=False) + "\n")
                destination.flush()
                if trace and thought is not None:
                    trace.write(json.dumps({"id": row["id"], "thought": thought, "thought_ids": thought_ids},
                                           ensure_ascii=False) + "\n")
                    trace.flush()
        finally:
            for stream in (trace, features, prefork):
                if stream:
                    stream.close()


def parse_checkpoints(parser, args) -> list[int]:
    """`--checkpoints 0,128,256` as a sorted list of distinct nonnegative thought lengths ([] when unset)."""
    if not getattr(args, "checkpoints", None):
        return []
    try:
        checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    except ValueError:
        parser.error("--checkpoints must be comma-separated integers")
    if checkpoints[0] < 0:
        parser.error("--checkpoints must be nonnegative")
    return checkpoints


def s2_main() -> None:
    """Two-stage batch S2: write thinking prompts for vLLM, then read the answer slots after the thoughts."""
    from . import twostage

    parser = argparse.ArgumentParser(description=s2_main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prompts = commands.add_parser("prompts", help="Token IDs of each row's thinking prompt (tokenizer only, no GPU)")
    readout = commands.add_parser("readout", help="S2 answer-slot logits after vLLM thoughts, merged into S1 rows")
    for command in (prompts, readout):
        command.add_argument("--model", required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--input", type=Path, required=True, help="The rows given to thinkjev-score")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-tokens", type=int, default=4096)
    readout.add_argument("--s1", type=Path, required=True, help="thinkjev-score --s2 never output")
    readout.add_argument("--thoughts", type=Path, required=True, help="benchmarks/vllm_think.py output")
    readout.add_argument("--prompts", type=Path, required=True, help="The prompts the thoughts were sampled from")
    readout.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    readout.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    readout.add_argument("--checkpoints",
                         help="Comma-separated thought lengths for anytime readouts of the seed-0 thought, "
                              "e.g. 0,128,256,512,1024,2048 (each is the answer under that think budget)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    checkpoints = parse_checkpoints(parser, args) if getattr(args, "checkpoints", None) else None
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "prompts":
        tokenizer = twostage.load_tokenizer(args.model, args.revision)
        with args.output.open("x") as destination:
            for row in rows:
                destination.write(json.dumps(twostage.think_prompt(tokenizer, row, args.max_tokens)) + "\n")
        return
    s1 = {record["id"]: record for record in map(json.loads, args.s1.read_text().splitlines()) if record}
    hashes = {record["id"]: record["s2_prompt_sha256"]
              for record in map(json.loads, args.prompts.read_text().splitlines()) if record}
    thoughts = twostage.load_thoughts(args.thoughts)
    missing = [row["id"] for row in rows if row["id"] not in s1 or row["id"] not in thoughts]
    if missing:
        parser.error(f"{len(missing)} rows lack an S1 result or a thought, for example {missing[0]}")
    model, tokenizer, _ = load_causal_model(args.model, args.revision, args.device, args.dtype)
    with args.output.open("x") as destination:
        for row in rows:
            if s1[row["id"]]["s2_prompt_sha256"] != hashes[row["id"]]:
                raise ValueError(f"{row['id']}: S1 run and thinking prompts rendered different S2 prompts")
            reads, anytime = [], None
            for thought in thoughts[row["id"]]:
                if checkpoints is not None and thought["seed_index"] == 0:
                    logits, anytime = twostage.read_s2_anytime(model, tokenizer, row, thought, args.max_tokens,
                                                               checkpoints)
                else:
                    logits = twostage.read_s2(model, tokenizer, row, thought, args.max_tokens)
                reads.append((thought, logits))
            merged = twostage.merge(s1[row["id"]], reads, anytime)
            destination.write(json.dumps(merged, allow_nan=False) + "\n")
            destination.flush()


if __name__ == "__main__":
    main()
