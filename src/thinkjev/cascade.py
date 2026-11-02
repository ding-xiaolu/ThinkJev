"""Zero-training S1/S2 self-cascade over one shared prompt prefix.

S1 is the direct option readout. S2 is the same model in thinking mode. Both prompts are
rendered from the same messages, so they share every token up to the thinking fork; that
prefix is prefilled once and snapshotted (hybrid linear-attention layers keep recurrent
state, which cannot be rolled back, only copied). S1 reads option logits on a copy; S2
continues thinking on the original and, once thinking closes or its budget runs out,
reads the same option logits. Neither path parses generated text.

With rotations on, S1 is also read under every cyclic shift of the option list and the
shifts are combined by a geometric mean (mean log-probability), which removes any position
bias that is additive in logit space (cyclic-shift marginalization, Zheng et al. 2024, as
used by AnyJev's L0). The shifts share every token before the option list, so that shorter
prefix is prefilled once and each shift costs one short suffix forward.

With checkpoints, the thought is also read at every checkpoint t it passes: a copy of the cache
after the first t thought tokens receives Qwen's early-stop sentence and the same option logits
are read. That read is the answer the model gives under a think budget of t for this sampled
thought (the batch readout in twostage.py does the same on a replayed thought). Sampling is not
touched, so a run with checkpoints emits the same thought and final logits as one without.
`score(row, on_event=...)` reports S1, thinking progress, and each checkpoint read as they happen,
which the server streams to its demo page.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time

from ._semif import PROMPT_VERSION, _cached_forward, digest, direct_messages, encode_prompt, softmax, synchronize

CASCADE_VERSION = "thinkjev-v3"
# Qwen's documented wording for closing an over-budget thought before answering.
EARLY_STOP = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
    "</think>\n\n"
)
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
THINKING_EVENT_EVERY = 32  # thought tokens between "thinking" progress events


def s2_template_kwargs() -> dict:
    """Extra chat-template arguments for the thinking prompt, from THINKJEV_CHAT_TEMPLATE_KWARGS (JSON).

    Empty by default. Some templates put text before the user turn only in thinking mode (Qwen3.8
    adds a reasoning-effort system note unless reasoning_effort is "medium"), which leaves no shared
    decision prefix; this passes the argument that keeps the two prompts aligned. The rendered
    prompt is recorded through s2_prompt_sha256.
    """
    raw = os.environ.get("THINKJEV_CHAT_TEMPLATE_KWARGS", "")
    kwargs = json.loads(raw) if raw else {}
    if not isinstance(kwargs, dict) or "enable_thinking" in kwargs:
        raise ValueError("THINKJEV_CHAT_TEMPLATE_KWARGS must be a JSON object without enable_thinking")
    return kwargs


def fork(tokenizer, row: dict, max_tokens: int) -> dict:
    """Split the S1 and S2 prompts at their longest common token prefix."""
    s1_ids, slots, s1_hash = encode_prompt(tokenizer, row, max_tokens)
    s2_prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=True,
        **s2_template_kwargs()
    )
    s2_ids = tokenizer.encode(s2_prompt, add_special_tokens=False)
    shared = 0
    while shared < min(len(s1_ids), len(s2_ids)) and s1_ids[shared] == s2_ids[shared]:
        shared += 1
    if shared == len(s1_ids) or shared == len(s2_ids):
        raise ValueError("Thinking and direct prompts do not diverge; the template has no thinking switch")
    if direct_messages(row)[-1]["content"] not in tokenizer.decode(s1_ids[:shared]):
        raise ValueError("The prompts diverge before the user turn ends; no shared decision prefix")
    think_end = tokenizer.encode("</think>", add_special_tokens=False)
    s1_suffix = s1_ids[shared:]
    if len(think_end) != 1 or think_end[0] not in s1_suffix:
        raise ValueError("Direct prompt does not close an empty thinking block with one </think> token")
    early_stop = tokenizer.encode(EARLY_STOP, add_special_tokens=False)
    if think_end[0] not in early_stop:
        raise ValueError("Early-stop text does not tokenize to a </think> token")
    return {
        "prefix": s1_ids[:shared],
        "s1_suffix": s1_suffix,
        "s2_suffix": s2_ids[shared:],
        "answer_tail": s1_suffix[s1_suffix.index(think_end[0]) + 1 :],
        "think_end": think_end[0],
        "early_stop": early_stop,
        "slots": slots,
        "s1_prompt_sha256": s1_hash,
        "s2_prompt_sha256": digest(s2_prompt),
    }


def _common_prefix(sequences: list[list[int]]) -> int:
    shared = 0
    while all(shared < len(ids) for ids in sequences) and len({ids[shared] for ids in sequences}) == 1:
        shared += 1
    return shared


def rotated_row(row: dict, shift: int) -> dict:
    """Show option `shift` first; option IDs travel with their descriptions."""
    options = row["options"]
    return {**row, "options": options[shift:] + options[:shift]}


def rotation_parts(tokenizer, row: dict, max_tokens: int, fork_prefix: list[int]) -> dict:
    """Split the S1 prompts of every cyclic shift at their longest common token prefix."""
    encoded = [encode_prompt(tokenizer, rotated_row(row, shift), max_tokens)
               for shift in range(len(row["options"]))]
    shared = _common_prefix([ids for ids, _, _ in encoded])
    if shared >= len(fork_prefix) or encoded[0][0][: len(fork_prefix)] != fork_prefix:
        raise ValueError("Option shifts must diverge before the thinking fork")
    return {
        "prefix": fork_prefix[:shared],
        "fork_tail": fork_prefix[shared:],
        "shift_suffixes": [ids[shared:] for ids, _, _ in encoded],
        "shift_prompt_sha256": [prompt_hash for _, _, prompt_hash in encoded],
    }


def unshift(logits: list[float], shift: int) -> list[float]:
    """Map slot-order logits of a shifted prompt back to the row's option order."""
    count = len(logits)
    return [logits[(index - shift) % count] for index in range(count)]


def debiased_logits(rotation_logits: list[list[float]]) -> list[float]:
    """Mean log-probability over shifts; softmax of it is the renormalized geometric mean."""
    logs = []
    for row in rotation_logits:
        top = max(row)
        normalizer = top + math.log(sum(math.exp(value - top) for value in row))
        logs.append([value - normalizer for value in row])
    return [sum(column) / len(logs) for column in zip(*logs)]


def calibrated_confidence(logits: list[float], temperature: float) -> float:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return max(softmax([value / temperature for value in logits]))


def row_seed(row_id: str) -> int:
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16)


def stop_token_ids(tokenizer) -> set[int]:
    """End-of-sequence and end-of-turn tokens that end a thought without closing it."""
    end_turn = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    return ({tokenizer.eos_token_id} | set(end_turn[:1] if len(end_turn) == 1 else [])) - {None}


class CascadeScorer:
    """Score S1 on every row; run S2 always, only below a confidence threshold, or never.

    `never` is for batch evaluation, where S2 comes from the two-stage path in `twostage.py`.
    `prefork_features` also records hidden states at the end of the shared prefix, before the
    model has started answering (the pre-router probe's input); it needs rotations, whose
    short fork-tail forward ends exactly there.
    """

    def __init__(self, model, tokenizer, metadata: dict, max_tokens: int = 4096, think_budget: int = 1024,
                 policy: str = "always", temperature: float = 1.0, threshold: float | None = None,
                 rotations: bool = False, feature_layers: tuple[int, ...] = (), prefork_features: bool = False,
                 checkpoints: tuple[int, ...] = ()):
        if policy not in {"always", "gated", "never"}:
            raise ValueError("Policy must be always, gated, or never")
        if prefork_features and not (rotations and feature_layers):
            raise ValueError("Pre-fork features need rotations and feature layers")
        if policy == "gated" and (threshold is None or not 0 < threshold <= 1):
            raise ValueError("Gated policy needs a threshold in (0, 1]")
        if think_budget < 1:
            raise ValueError("Think budget must be positive")
        if any(layer < 0 for layer in feature_layers) or len(set(feature_layers)) != len(feature_layers):
            raise ValueError("Feature layers must be distinct nonnegative hidden-state indices")
        if any(t < 0 for t in checkpoints) or len(set(checkpoints)) != len(checkpoints):
            raise ValueError("Checkpoints must be distinct nonnegative thought lengths")
        calibrated_confidence([0.0, 0.0], temperature)
        self.checkpoints = tuple(sorted(checkpoints))
        self.model, self.tokenizer = model, tokenizer
        self.metadata = {**metadata, "serving_config": "shared-thinking-fork-prefix-cache-v1"}
        self.max_tokens, self.think_budget = max_tokens, think_budget
        self.policy, self.temperature, self.threshold = policy, temperature, threshold
        self.rotations, self.feature_layers = rotations, tuple(feature_layers)
        self.prefork_features = prefork_features
        self.device = next(model.parameters()).device
        self.stop_ids = stop_token_ids(tokenizer)

    def _run(self, ids: list[int], cache, total: int, **extra):
        import torch

        return _cached_forward(self.model, {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.ones((1, total), dtype=torch.long, device=self.device),
            "past_key_values": cache,
        }, **extra)

    def _forward(self, ids: list[int], cache, total: int):
        output = self._run(ids, cache, total)
        return output.logits[0, -1, :].float(), output.past_key_values

    def _features(self, output):
        """Last-position hidden states of the S1 read at the requested layers, as float16."""
        import torch

        states = output.hidden_states
        if states is None or max(self.feature_layers) >= len(states):
            raise ValueError(f"Model returned {0 if states is None else len(states)} hidden states; "
                             f"cannot read layers {self.feature_layers}")
        return torch.stack([states[layer][0, -1, :] for layer in self.feature_layers]).to(torch.float16).cpu().numpy()

    def _sample(self, logits, generator) -> int:
        import torch

        values, indices = logits.topk(SAMPLING["top_k"])
        probs = torch.softmax(values.cpu() / SAMPLING["temperature"], -1)
        keep = probs.cumsum(-1) - probs < SAMPLING["top_p"]
        probs = probs * keep
        choice = torch.multinomial(probs / probs.sum(), 1, generator=generator)
        return int(indices.cpu()[choice])

    def _think(self, parts: dict, cache, position: int, seed: int, on_event=None) -> dict:
        """Sample a thought; every token kept in `generated` has been fed to the cache.

        A checkpoint t is read only when the thought goes on past t tokens (as twostage.read_s2_anytime
        reads a replayed thought at t < len(thought)); a thought that ends at or before t is covered
        by the final read. The read runs on a copy of the cache and never touches the sampler.
        """
        import torch

        generator = torch.Generator().manual_seed(seed)
        ids, generated, closed, reads = parts["s2_suffix"], [], False, []
        marks = {t for t in self.checkpoints if t < self.think_budget}
        while True:
            position += len(ids)
            logits, cache = self._forward(ids, cache, position)
            if len(generated) == self.think_budget:
                break
            token = self._sample(logits, generator)
            if token == parts["think_end"]:
                closed = True
                break
            if token in self.stop_ids:
                break
            if len(generated) in marks:
                early = parts["early_stop"]
                vocabulary, _ = self._forward(early, copy.deepcopy(cache), position + len(early))
                reads.append({"t": len(generated), "option_logits": vocabulary[parts["slots"]].cpu().tolist()})
                if on_event:
                    on_event({"type": "checkpoint", **reads[-1]})
            generated.append(token)
            ids = [token]
            if on_event and len(generated) % THINKING_EVENT_EVERY == 0:
                on_event({"type": "thinking", "tokens": len(generated)})
        closing = [parts["think_end"], *parts["answer_tail"]] if closed else parts["early_stop"]
        logits, _ = self._forward(closing, cache, position + len(closing))
        return {"logits": logits, "tokens": generated, "closed": closed, "reads": reads}

    def score(self, row: dict, on_event=None) -> dict:
        """Score one row. `on_event`, if given, receives dicts as the row is processed: `s1` (raw and
        debiased logits, confidence, whether it escalates), `thinking` (tokens so far), and
        `checkpoint` (t and the option logits read there); the returned result holds everything."""
        started = time.perf_counter()
        parts = fork(self.tokenizer, row, self.max_tokens)
        shifts = rotation_parts(self.tokenizer, row, self.max_tokens, parts["prefix"]) if self.rotations else None
        sync = lambda: synchronize(self.device)
        import torch

        with torch.inference_mode():
            sync()
            mark = time.perf_counter()
            rotation_seconds = 0.0
            if shifts:
                # Prefill up to the option list once; the thinking fork continues on a copy.
                _, options_cache = self._forward(shifts["prefix"], None, len(shifts["prefix"]))
                sync()
                prefill_seconds = time.perf_counter() - mark
                mark = time.perf_counter()
                cache = copy.deepcopy(options_cache)
                sync()
                rotation_seconds += time.perf_counter() - mark
                mark = time.perf_counter()
                if self.prefork_features:
                    tail = self._run(shifts["fork_tail"], cache, len(parts["prefix"]), output_hidden_states=True)
                    prefork = self._features(tail)
                    cache = tail.past_key_values
                    del tail
                else:
                    _, cache = self._forward(shifts["fork_tail"], cache, len(parts["prefix"]))
                sync()
                prefill_seconds += time.perf_counter() - mark
            else:
                _, cache = self._forward(parts["prefix"], None, len(parts["prefix"]))
                sync()
                prefill_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            branch = copy.deepcopy(cache)
            sync()
            copy_seconds = time.perf_counter() - mark
            mark = time.perf_counter()
            total = len(parts["prefix"]) + len(parts["s1_suffix"])
            extra = {"output_hidden_states": True} if self.feature_layers else {}
            output = self._run(parts["s1_suffix"], branch, total, **extra)
            s1_logits = output.logits[0, -1, :].float()[parts["slots"]].cpu().tolist()
            features = self._features(output) if self.feature_layers else None
            sync()
            s1_seconds = time.perf_counter() - mark
            del output, branch
            rotation_logits = [s1_logits]
            if shifts:
                mark = time.perf_counter()
                for shift in range(1, len(s1_logits)):
                    suffix = shifts["shift_suffixes"][shift]
                    vocabulary, _ = self._forward(suffix, copy.deepcopy(options_cache),
                                                  len(shifts["prefix"]) + len(suffix))
                    rotation_logits.append(unshift(vocabulary[parts["slots"]].cpu().tolist(), shift))
                sync()
                rotation_seconds += time.perf_counter() - mark
                del options_cache
            debiased = debiased_logits(rotation_logits) if shifts else None
            confidence = calibrated_confidence(debiased or s1_logits, self.temperature)
            escalate = self.policy == "always" or (self.policy == "gated" and confidence < self.threshold)
            if on_event:
                on_event({"type": "s1", "option_logits": s1_logits, "debiased_logits": debiased,
                          "s1_confidence": confidence, "escalated": escalate,
                          "seconds": time.perf_counter() - started})
            result = {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "option_logits": s1_logits,
                "probabilities": softmax(s1_logits),
                "s1_confidence": confidence,
                "escalated": escalate,
                "prefix_tokens": len(parts["prefix"]),
                "s1_suffix_tokens": len(parts["s1_suffix"]),
                "prefill_seconds": prefill_seconds,
                "copy_seconds": copy_seconds,
                "s1_seconds": s1_seconds,
                "prompt_sha256": parts["s1_prompt_sha256"],
                "s2_prompt_sha256": parts["s2_prompt_sha256"],
                "prompt_version": PROMPT_VERSION,
                "cascade_version": CASCADE_VERSION,
                "cascade_config": {
                    "policy": self.policy,
                    "temperature": self.temperature,
                    "threshold": self.threshold,
                    "think_budget": self.think_budget,
                    "sampling": SAMPLING,
                    "rotations": self.rotations,
                    "feature_layers": list(self.feature_layers),
                    "prefork_features": self.prefork_features,
                    "checkpoints": list(self.checkpoints),
                },
                "model": self.metadata,
                "readout": "S1 and S2 both read declared answer-slot logits; no generated text is parsed",
                "probability_status": "S1 confidence uses the supplied temperature; raw probabilities are uncalibrated",
            }
            if shifts:
                result.update({
                    "rotation_option_logits": rotation_logits,
                    "debiased_logits": debiased,
                    "debiased_probabilities": softmax(debiased),
                    "s1_confidence_source": "debiased_logits",
                    "options_prefix_tokens": len(shifts["prefix"]),
                    "rotation_seconds": rotation_seconds,
                    "shift_prompt_sha256": shifts["shift_prompt_sha256"],
                })
            if features is not None:
                result["s1_features"] = features
            if self.prefork_features:
                result["prefork_features"] = prefork
            if escalate:
                seed = row_seed(row["id"])
                sync()
                mark = time.perf_counter()
                thought = self._think(parts, cache, len(parts["prefix"]), seed, on_event)
                s2_logits = thought["logits"][parts["slots"]].cpu().tolist()
                sync()
                result.update({
                    "s2_option_logits": s2_logits,
                    "s2_probabilities": softmax(s2_logits),
                    "s2_think_tokens": len(thought["tokens"]),
                    "s2_closed_naturally": thought["closed"],
                    "s2_seed": seed,
                    "s2_seconds": time.perf_counter() - mark,
                    "s2_thought": self.tokenizer.decode(thought["tokens"]),
                    "s2_thought_ids": thought["tokens"],
                })
                if self.checkpoints:
                    result["s2_anytime"] = thought["reads"]
            del cache
        sync()
        result["total_seconds"] = time.perf_counter() - started
        return result
