"""Batch-generate S2 thoughts with vLLM from pre-tokenized thinking prompts.

Runs in its own environment (vLLM pins its own torch). It imports only vLLM and the standard
library, and never tokenizes text: `thinkjev-s2 prompts` writes the exact token IDs of
the shared prefix plus the thinking suffix, and `thinkjev-s2 readout` reads the answer
slots after the thought in the main environment. This script only samples the thought.

Sampling matches the scorer's HF path: Qwen's thinking settings, a per-row seed, at most
`--think-budget` thought tokens, stopping at `</think>` or an end-of-turn token. Each output
line records the thought tokens without the stop token, and how the thought ended:

- `closed`: the model emitted `</think>`;
- `budget`: the budget ran out (readout appends the early-stop sentence);
- `stopped`: the model ended its turn without closing the thought (readout also appends the
  early-stop sentence, matching the HF path).

Expose one GPU per process (CUDA_VISIBLE_DEVICES) and shard rows for more GPUs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def classify(tokens: list[int], think_end: int, stop_ids: set[int], budget: int) -> tuple[list[int], str]:
    """Strip a trailing stop token and name how the thought ended."""
    if tokens and tokens[-1] == think_end:
        return tokens[:-1], "closed"
    if tokens and tokens[-1] in stop_ids:
        return tokens[:-1], "stopped"
    if len(tokens) >= budget:
        return tokens[:budget], "budget"
    # vLLM can also end on the model's max length; treat it as an unclosed thought.
    return tokens, "stopped"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Same checkpoint (local path or repo) as the prompts")
    parser.add_argument("--revision", required=True, help="Pinned 40-character commit for a remote model")
    parser.add_argument("--prompts", type=Path, required=True, help="thinkjev-s2 prompts output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--think-budget", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Thoughts per row; seed k is the row seed plus k (k = 0 matches the HF path)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if args.think_budget < 1 or args.seeds < 1:
        parser.error("Think budget and seeds must be positive")

    from vllm import LLM, SamplingParams, __version__
    from vllm.inputs import TokensPrompt

    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if not prompts:
        parser.error("Prompts are empty")
    longest = max(len(item["prompt_ids"]) for item in prompts)
    llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision, dtype=args.dtype, seed=0,
              max_model_len=longest + args.think_budget + 8,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=True, trust_remote_code=False)
    requests, params, keys = [], [], []
    for item in prompts:
        stops = [item["think_end"], *item["stop_ids"]]
        for k in range(args.seeds):
            requests.append(TokensPrompt(prompt_token_ids=item["prompt_ids"]))
            params.append(SamplingParams(temperature=item["sampling"]["temperature"], top_p=item["sampling"]["top_p"],
                                         top_k=item["sampling"]["top_k"], max_tokens=args.think_budget,
                                         stop_token_ids=stops, seed=item["seed"] + k, detokenize=False,
                                         # Stop only on the scorer's stop set, exactly as the HF path does.
                                         ignore_eos=True))
            keys.append((item, k))
    started = time.perf_counter()
    outputs = llm.generate(requests, params, use_tqdm=True)
    seconds = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for (item, k), output in zip(keys, outputs):
            completion = output.outputs[0]
            tokens, ending = classify(list(completion.token_ids), item["think_end"], set(item["stop_ids"]),
                                      args.think_budget)
            stream.write(json.dumps({
                "id": item["id"],
                "seed_index": k,
                "seed": item["seed"] + k,
                "thought_ids": tokens,
                "ending": ending,
                "think_budget": args.think_budget,
                "engine": {"name": "vllm", "version": __version__, "model": args.model, "revision": args.revision,
                           "dtype": args.dtype},
            }) + "\n")
    print(f"{len(outputs)} thoughts in {seconds:.0f} s")


if __name__ == "__main__":
    main()
