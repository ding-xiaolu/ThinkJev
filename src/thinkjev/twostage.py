"""Two-stage S2 for batch evaluation: vLLM samples the thought, the HF model reads the answer slots.

The one-process scorer (`cascade.CascadeScorer`) samples thoughts token by token, which is right
for latency measurements and far too slow to label thousands of rows. For labels, S2 is split:

1. `prompts` (tokenizer only): the exact token IDs of the shared prefix plus the thinking
   suffix, the stop tokens, the per-row seed, and the sampling settings.
2. `benchmarks/vllm_think.py` (separate vLLM environment): samples the thought from those IDs.
3. `readout` (HF, one GPU): one forward over prefix + thinking suffix + thought + closing, then
   the same answer-slot logits as the one-process path, merged into the S1 output row.

No generated text is parsed at any stage, and the closing tokens are the same as in
`CascadeScorer._think`: `</think>` plus the answer tail when the thought closed, otherwise
Qwen's early-stop sentence.

Anytime readout (`readout --checkpoints`): the thought is fed in chunks on one cache, and at
each checkpoint t a copy of the cache gets the early-stop sentence and is read. That read is the
answer the model gives under a think budget of t, because a budget-t thought is exactly the first
t tokens of this one. Checkpoints at or past the thought's length are not read; the final S2
readout covers them. With checkpoints, the final readout also comes from the chunked cache.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ._semif import _cached_forward, _forward, softmax, synchronize
from .cascade import SAMPLING, fork, row_seed, stop_token_ids

ENGINE = "vllm-thought+hf-readout-v1"


def load_tokenizer(source: str, revision: str):
    """The tokenizer alone, with the same pinning rules as `_semif.load_causal_model`."""
    import re

    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    return transformers.AutoTokenizer.from_pretrained(
        source, revision=None if local else revision, local_files_only=local, trust_remote_code=False)


def think_prompt(tokenizer, row: dict, max_tokens: int) -> dict:
    parts = fork(tokenizer, row, max_tokens)
    return {
        "id": row["id"],
        "prompt_ids": parts["prefix"] + parts["s2_suffix"],
        "think_end": parts["think_end"],
        "stop_ids": sorted(stop_token_ids(tokenizer)),
        "seed": row_seed(row["id"]),
        "sampling": SAMPLING,
        "s2_prompt_sha256": parts["s2_prompt_sha256"],
    }


def closing(parts: dict, ending: str) -> list[int]:
    if ending == "closed":
        return [parts["think_end"], *parts["answer_tail"]]
    if ending in {"budget", "stopped"}:
        return parts["early_stop"]
    raise ValueError(f"Unknown thought ending {ending!r}")


def readout_ids(parts: dict, thought: dict) -> list[int]:
    return parts["prefix"] + parts["s2_suffix"] + list(thought["thought_ids"]) + closing(parts, thought["ending"])


def read_s2(model, tokenizer, row: dict, thought: dict, max_tokens: int) -> list[float]:
    """Answer-slot logits after one sampled thought."""
    import torch

    parts = fork(tokenizer, row, max_tokens)
    ids = readout_ids(parts, thought)
    device = next(model.parameters()).device
    with torch.inference_mode():
        vocabulary = _forward(model, {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        })[0].float()
    synchronize(device)
    return vocabulary[parts["slots"]].cpu().tolist()


def read_s2_anytime(model, tokenizer, row: dict, thought: dict, max_tokens: int,
                    checkpoints: list[int]) -> tuple[list[float], list[dict]]:
    """Final answer-slot logits and the early-stopped readout at every checkpoint shorter than the thought."""
    import torch

    parts = fork(tokenizer, row, max_tokens)
    thought_ids = list(thought["thought_ids"])
    marks = sorted({t for t in checkpoints if 0 <= t < len(thought_ids)})
    device = next(model.parameters()).device

    def step(ids: list[int], cache, position: int):
        output = _cached_forward(model, {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, position + len(ids)), dtype=torch.long, device=device),
            "past_key_values": cache,
        })
        return output.logits[0, -1, :].float(), output.past_key_values, position + len(ids)

    reads, consumed = [], 0
    with torch.inference_mode():
        _, cache, position = step(parts["prefix"] + parts["s2_suffix"], None, 0)
        for mark in marks:
            if mark > consumed:
                _, cache, position = step(thought_ids[consumed:mark], cache, position)
                consumed = mark
            vocabulary, _, _ = step(parts["early_stop"], copy.deepcopy(cache), position)
            reads.append({"t": mark, "option_logits": vocabulary[parts["slots"]].cpu().tolist()})
        if consumed < len(thought_ids):
            _, cache, position = step(thought_ids[consumed:], cache, position)
        vocabulary, _, _ = step(closing(parts, thought["ending"]), cache, position)
        final = vocabulary[parts["slots"]].cpu().tolist()
    synchronize(device)
    return final, reads


def merge(s1: dict, reads: list[tuple[dict, list[float]]], anytime: list[dict] | None = None) -> dict:
    """Add S2 fields to an S1 output row; seed index 0 is the primary S2 answer.

    `anytime` holds the seed-0 thought's checkpoint readouts (`read_s2_anytime`), if any.
    """
    reads = sorted(reads, key=lambda pair: pair[0]["seed_index"])
    if not reads or reads[0][0]["seed_index"] != 0:
        raise ValueError(f"{s1['id']}: missing the seed-0 thought")
    thought, logits = reads[0]
    merged = {
        **s1,
        "escalated": True,
        "s2_option_logits": logits,
        "s2_probabilities": softmax(logits),
        "s2_think_tokens": len(thought["thought_ids"]),
        "s2_closed_naturally": thought["ending"] == "closed",
        "s2_ending": thought["ending"],
        "s2_seed": thought["seed"],
        "s2_seconds": None,
        "s2_engine": {"name": ENGINE, "thought": thought["engine"], "think_budget": thought["think_budget"],
                      "readout": "one-shot" if anytime is None else "chunked"},
    }
    if anytime is not None:
        merged["s2_anytime"] = anytime
    if len(reads) > 1:
        merged["s2_samples"] = [
            {"seed": t["seed"], "option_logits": l, "think_tokens": len(t["thought_ids"]), "ending": t["ending"]}
            for t, l in reads
        ]
    return merged


def load_thoughts(path: Path) -> dict[str, list[dict]]:
    thoughts: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            thoughts.setdefault(record["id"], []).append(record)
    return thoughts
