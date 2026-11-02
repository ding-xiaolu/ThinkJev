"""ThinkJev: fork, decode accounting, hard200 labels, and offline policy math; no weights needed."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from thinkjev import cascade
from thinkjev.cli import main

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import build_cascade  # noqa: E402
import evaluate_cascade  # noqa: E402

SPECIALS = ["<|im_end|>", "</think>", "<think>", "\n\n"]


class Tokenizer:
    """Greedy special-token tokenizer that merges a double newline like BPE does."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            special = next((s for s in SPECIALS if text.startswith(s, index)), None)
            ids.append(SPECIALS.index(special) if special else 100 + ord(text[index]))
            index += len(special) if special else 1
        return ids

    def decode(self, ids):
        return "".join(SPECIALS[i] if i < len(SPECIALS) else chr(i - 100) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        text = "".join(f"[{m['role']}]{m['content']}<|im_end|>" for m in messages) + "[assistant]"
        return text + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


ROW = {"id": "r", "state": "s", "question": "q?", "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]}


def test_fork_splits_at_think_token_not_at_text_boundary():
    tokenizer = Tokenizer()
    parts = cascade.fork(tokenizer, ROW, 4096)
    assert parts["prefix"][-1] == SPECIALS.index("<think>")
    assert parts["s1_suffix"] == tokenizer.encode("\n\n</think>\n\n")
    assert parts["s2_suffix"] == tokenizer.encode("\n")
    assert parts["answer_tail"] == tokenizer.encode("\n\n")
    assert parts["think_end"] in parts["early_stop"]


class EffortTokenizer(Tokenizer):
    """Qwen3.8-style template: thinking mode adds a system note unless reasoning_effort is "medium"."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking, reasoning_effort="xhigh"):
        text = super().apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking)
        return ("[system]Reasoning effort is set to xhigh.<|im_end|>" + text
                if enable_thinking and reasoning_effort != "medium" else text)


def test_template_kwargs_restore_the_shared_prefix(monkeypatch):
    monkeypatch.delenv("THINKJEV_CHAT_TEMPLATE_KWARGS", raising=False)
    plain = cascade.fork(Tokenizer(), ROW, 4096)
    with pytest.raises(ValueError, match="diverge before the user turn ends"):
        cascade.fork(EffortTokenizer(), ROW, 4096)
    monkeypatch.setenv("THINKJEV_CHAT_TEMPLATE_KWARGS", '{"reasoning_effort": "medium"}')
    parts = cascade.fork(EffortTokenizer(), ROW, 4096)
    assert {k: parts[k] for k in ("prefix", "s1_suffix", "s2_suffix")} == {k: plain[k] for k in ("prefix", "s1_suffix", "s2_suffix")}
    monkeypatch.setenv("THINKJEV_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    with pytest.raises(ValueError, match="without enable_thinking"):
        cascade.fork(EffortTokenizer(), ROW, 4096)


class ScriptedModel:
    """Feeds a scripted token stream and records every token the cache has seen."""

    def __init__(self, script):
        import torch

        self.script, self.weight = list(script), torch.zeros(1)

    def parameters(self):
        yield self.weight

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, return_dict=True, logits_to_keep=1):
        import torch

        cache = past_key_values if past_key_values is not None else SimpleNamespace(seen=[])
        cache.seen += input_ids[0].tolist()
        assert attention_mask.shape[1] == len(cache.seen)
        logits = torch.full((1, 1, 400), -1e9)
        logits[0, 0, self.script.pop(0) if self.script else 1] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=cache)

    __call__ = forward


@pytest.mark.parametrize("script,budget,closed,generated", [
    ([200, 201, 1], 8, True, 2),      # closes its thought naturally
    ([200] * 10, 3, False, 3),        # runs out of budget
    ([200, 0], 8, False, 1),          # ends the turn without closing
])
def test_think_feeds_every_kept_token_then_closes(script, budget, closed, generated):
    tokenizer = Tokenizer()
    parts = cascade.fork(tokenizer, ROW, 4096)
    model = ScriptedModel(script)
    scorer = cascade.CascadeScorer(model, tokenizer, {}, think_budget=budget)
    cache = SimpleNamespace(seen=list(parts["prefix"]))
    thought = scorer._think(parts, cache, len(parts["prefix"]), seed=1)
    assert thought["closed"] is closed and len(thought["tokens"]) == generated
    closing = [parts["think_end"], *parts["answer_tail"]] if closed else parts["early_stop"]
    assert cache.seen == parts["prefix"] + parts["s2_suffix"] + thought["tokens"] + closing


def test_gated_policy_requires_threshold():
    with pytest.raises(ValueError):
        cascade.CascadeScorer(ScriptedModel([]), Tokenizer(), {}, policy="gated")


def test_hard200_is_reproducible_and_labels_follow_the_state():
    committed = [json.loads(line) for line in (ROOT / "benchmarks/data/hard200.jsonl").read_text().splitlines()]
    assert committed == build_cascade.hard_rows()
    for row in committed:
        answer = row["options"][row["label"]]["id"]
        if row["family"] == "counting":
            count = len(re.findall(r"priority high, status open,", row["state"]))
            assert answer == f"count_{count}"
        elif row["family"] == "date_deadline":
            facts = row["provenance"]["derivation"]
            assert (answer == "late") == (facts["paid"] > facts["due"])


def test_assemble_keeps_groups_in_one_split_and_limits_whole_groups(tmp_path):
    rows = [{"id": f"{g}-{i}", "group_id": g, "family": "f", "label": 0, "target_distribution": None}
            for g in "abcdefghijklmnopqrst" for i in range(3)]
    rows.append({"id": "soft", "group_id": "z", "family": "f", "label": 0, "target_distribution": [0.5, 0.5]})
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    merged, manifest = build_cascade.assemble([("x", path)], {})
    assert len(merged) == 60 and {row["id"] for row in manifest} == {row["id"] for row in merged}
    splits = {}
    for row in merged:
        splits.setdefault(row["group_id"], set()).add(row["cascade_split"])
    assert all(len(values) == 1 for values in splits.values()) and {"dev", "test"} <= set().union(*splits.values())
    limited, _ = build_cascade.assemble([("x", path)], {"x": 10})
    assert len(limited) == 9


def test_curve_endpoints_ties_and_oracle():
    scores = np.array([0.9, 0.9, 0.5, 0.1])
    s1 = np.array([1.0, 0.0, 1.0, 0.0])
    s2 = np.array([1.0, 1.0, 0.0, 1.0])
    coverage, accuracy, _ = evaluate_cascade.curve(scores, s1, s2)
    assert coverage.tolist() == [0.0, 0.5, 0.75, 1.0]  # the tied pair is accepted together
    assert accuracy[0] == s2.mean() and accuracy[-1] == s1.mean()
    oracle = 1 - (1 - s1) * s2
    _, best, _ = evaluate_cascade.curve(oracle, s1, s2)
    assert best.max() == np.maximum(s1, s2).mean()


def test_supervised_router_stays_finite_on_long_rows():
    rng = np.random.default_rng(0)
    texts = [" ".join(f"w{rng.integers(5000)}" for _ in range(800)) for _ in range(40)]
    x = np.stack([evaluate_cascade._features(text) for text in texts])
    y = (np.arange(40) % 2).astype(float)
    scores = evaluate_cascade.predict_logistic(evaluate_cascade.fit_logistic(x, y), x)
    assert np.all(np.isfinite(scores)) and ((scores > 0.5) == y).mean() > 0.9


def test_operating_point_is_chosen_on_dev_and_applied_to_test():
    def items(split, scores, s1, s2):
        return [{"split": split, "confidence": c, "s1_correct": a, "s2_correct": b, "think_tokens": 10,
                 "prefill": 1.0, "copy": 0.0, "s1_seconds": 0.1, "s2_seconds": 5.0} for c, a, b in zip(scores, s1, s2)]
    dev = items("dev", [0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0], [1, 1, 1, 1])
    test = items("test", [0.95, 0.85, 0.5, 0.1], [1, 1, 1, 0], [1, 1, 1, 1])
    point = evaluate_cascade.operating_points(dev, test, ["confidence"])[0]
    assert point["threshold"] == 0.8
    assert point["test_coverage"] == 0.5
    assert point["mean_seconds_shared_prefix"] == pytest.approx(1.1 + 0.5 * 5.0)


@pytest.mark.parametrize("extra,message", [
    (["--s2", "gated"], "requires --threshold"),
    (["--threshold", "0.9"], "requires --s2 gated"),
])
def test_cascade_cli_rejects_bad_combinations(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["thinkjev-score", "--model", "m", "--revision", "r", "--input", "missing.jsonl",
                                      "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit):
        main()
    assert message in capsys.readouterr().err


class BiasedModel:
    """Scores only answer positions: fixed slot logits, so any preference is pure position bias."""

    def __init__(self, bias):
        import torch

        self.bias, self.weight, self.seen = bias, torch.zeros(1), []

    def parameters(self):
        yield self.weight

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, return_dict=True,
                logits_to_keep=1, output_hidden_states=False):
        import torch

        cache = past_key_values if past_key_values is not None else SimpleNamespace(seen=[])
        cache.seen += input_ids[0].tolist()
        assert attention_mask.shape[1] == len(cache.seen)
        self.seen.append(list(cache.seen))
        logits = torch.zeros((1, 1, 400))
        for index, value in enumerate(self.bias):
            logits[0, 0, 100 + ord("A") + index] = value
        hidden = tuple(torch.full((1, input_ids.shape[1], 4), float(layer)) for layer in range(3)) if output_hidden_states else None
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=hidden)

    __call__ = forward


ROW3 = {**ROW, "options": [{"id": x, "description": x.upper()} for x in "abc"]}


def test_rotation_parts_share_the_prefix_before_the_options():
    tokenizer = Tokenizer()
    parts = cascade.fork(tokenizer, ROW3, 4096)
    shifts = cascade.rotation_parts(tokenizer, ROW3, 4096, parts["prefix"])
    assert shifts["prefix"] + shifts["fork_tail"] == parts["prefix"]
    # Letters are positional, so the shared prefix runs up to the first option description.
    assert tokenizer.decode(shifts["prefix"]).endswith('"options": [{"letter": "A", "description": "')
    for shift, suffix in enumerate(shifts["shift_suffixes"]):
        ids, _, _ = cascade.encode_prompt(tokenizer, cascade.rotated_row(ROW3, shift), 4096)
        assert ids == shifts["prefix"] + suffix
    assert shifts["shift_suffixes"][0] == shifts["fork_tail"] + parts["s1_suffix"]


def test_rotations_cancel_pure_position_bias_and_reuse_the_options_prefix():
    tokenizer = Tokenizer()
    model = BiasedModel([3.0, 1.0, 0.0])
    scorer = cascade.CascadeScorer(model, tokenizer, {}, policy="gated", threshold=1e-6,
                                   rotations=True, feature_layers=(0, 2))
    result = scorer.score(ROW3)
    assert result["option_logits"] == [3.0, 1.0, 0.0]
    # Option b shown first under shift 1, so it inherits slot A's logit.
    assert result["rotation_option_logits"] == [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 3.0]]
    assert result["debiased_probabilities"] == pytest.approx([1 / 3] * 3)
    assert result["s1_confidence"] == pytest.approx(1 / 3) and not result["escalated"]
    assert result["s1_features"].shape == (2, 4) and result["s1_features"][1].tolist() == [2.0] * 4
    parts = cascade.fork(tokenizer, ROW3, 4096)
    shifts = cascade.rotation_parts(tokenizer, ROW3, 4096, parts["prefix"])
    assert model.seen[0] == shifts["prefix"]  # the one full prefill
    assert model.seen[1] == parts["prefix"]
    assert model.seen[2] == parts["prefix"] + parts["s1_suffix"]
    assert model.seen[3:] == [shifts["prefix"] + suffix for suffix in shifts["shift_suffixes"][1:]]


def test_without_rotations_the_v1_fields_and_forwards_are_unchanged():
    tokenizer = Tokenizer()
    model = BiasedModel([3.0, 1.0, 0.0])
    result = cascade.CascadeScorer(model, tokenizer, {}, policy="gated", threshold=1e-6).score(ROW3)
    assert not {"rotation_option_logits", "debiased_logits", "s1_features"} & result.keys()
    parts = cascade.fork(tokenizer, ROW3, 4096)
    assert model.seen == [parts["prefix"], parts["prefix"] + parts["s1_suffix"]]
    assert result["s1_confidence"] == pytest.approx(max(cascade.softmax([3.0, 1.0, 0.0])))


def test_debiased_logits_are_mean_log_probabilities():
    rows = [[2.0, 0.0], [0.0, 1.0]]
    logs = np.log([cascade.softmax(row) for row in rows]).mean(0)
    assert cascade.debiased_logits(rows) == pytest.approx(logs.tolist())
    assert cascade.debiased_logits([[1000.0, 0.0]])[1] == pytest.approx(-1000.0)


def test_features_round_trip_in_output_order(tmp_path):
    path = tmp_path / "features.npy"
    arrays = [np.full((2, 3), index, dtype=np.float16) for index in range(3)]
    with path.open("xb") as stream:
        for array in arrays:
            np.save(stream, array, allow_pickle=False)
    loaded = evaluate_cascade.read_features(path)
    assert [a.tolist() for a in loaded] == [a.tolist() for a in arrays]


def test_probe_selects_the_informative_layer_without_touching_test():
    rng = np.random.default_rng(1)

    def items(split, count):
        out = []
        for index in range(count):
            correct = index % 2
            noise = rng.normal(size=(3, 32))
            noise[1, :4] += 3.0 * (2 * correct - 1)
            out.append({"split": split, "group": f"{split}{index // 2}", "features": noise, "s1_correct": correct})
        return out

    dev, test = items("dev", 120), items("test", 80)
    selection = evaluate_cascade.probe_scores(dev, test)
    assert selection["layer"] == 1 and selection["dev_cv_auroc"] > 0.9
    scores = np.array([item["probe"] for item in test])
    assert evaluate_cascade.auroc(scores, np.array([item["s1_correct"] for item in test])) > 0.9


def test_rotation_policy_serves_the_debiased_answer():
    items = [{"confidence_rot": 0.9, "s1_correct": 0, "s1_rot_correct": 1, "s2_correct": 0}]
    _, s1, _ = evaluate_cascade._arrays(items, "confidence_rot")
    assert s1.tolist() == [1.0]
    _, s1, _ = evaluate_cascade._arrays([{**items[0], "confidence": 0.9}], "confidence")
    assert s1.tolist() == [0.0]


def test_risk_threshold_accepts_the_largest_prefix_within_risk():
    scores = np.array([0.99, 0.95, 0.9, 0.9, 0.6])
    correct = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert evaluate_cascade.risk_threshold(scores, correct, 0.0) == 0.95   # the tied pair would add an error
    assert evaluate_cascade.risk_threshold(scores, correct, 0.25) == 0.9
    assert evaluate_cascade.risk_threshold(np.array([0.9]), np.array([0.0]), 0.05) == float("inf")


@pytest.mark.parametrize("extra,message", [
    (["--mode", "direct", "--rotations"], "require --mode cascade"),
    (["--mode", "direct", "--features", "f.npy"], "require --mode cascade"),
    (["--mode", "cascade", "--features", "EXISTING"], "Features must be new"),
    (["--mode", "cascade", "--features", "new.npy", "--feature-layers", "a,b"], "comma-separated integers"),
])
def test_rotation_and_feature_cli_checks(tmp_path, monkeypatch, capsys, extra, message):
    existing = tmp_path / "existing.npy"
    existing.write_bytes(b"")
    extra = [str(existing) if value == "EXISTING" else value for value in extra]
    monkeypatch.setattr(sys, "argv", ["thinkjev-score", "--model", "m", "--revision", "r", "--input", "missing.jsonl",
                                      "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit):
        main()
    assert message in capsys.readouterr().err


class PositionModel(BiasedModel):
    """Hidden states carry the cache length, so each feature record names the forward it came from."""

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, return_dict=True,
                logits_to_keep=1, output_hidden_states=False):
        import torch

        output = super().forward(input_ids, attention_mask, past_key_values, use_cache, return_dict,
                                 logits_to_keep, output_hidden_states)
        if output_hidden_states:
            seen = float(len(output.past_key_values.seen))
            output.hidden_states = tuple(torch.full((1, input_ids.shape[1], 4), seen) for _ in range(3))
        return output

    __call__ = forward


def test_prefork_features_come_from_the_end_of_the_shared_prefix():
    tokenizer = Tokenizer()
    scorer = cascade.CascadeScorer(PositionModel([3.0, 1.0, 0.0]), tokenizer, {}, policy="never",
                                   rotations=True, feature_layers=(0, 2), prefork_features=True)
    result = scorer.score(ROW3)
    parts = cascade.fork(tokenizer, ROW3, 4096)
    assert result["prefork_features"].tolist() == [[len(parts["prefix"])] * 4] * 2
    assert result["s1_features"].tolist() == [[len(parts["prefix"]) + len(parts["s1_suffix"])] * 4] * 2
    assert not result["escalated"] and "s2_option_logits" not in result
    with pytest.raises(ValueError):
        cascade.CascadeScorer(PositionModel([0.0]), tokenizer, {}, policy="never", feature_layers=(0,),
                              prefork_features=True)


@pytest.mark.parametrize("script,budget,ending", [
    ([200, 201, 1], 8, "closed"), ([200] * 10, 3, "budget"), ([200, 0], 8, "stopped"),
])
def test_two_stage_readout_sees_exactly_the_one_process_sequence(script, budget, ending):
    import vllm_think
    from thinkjev import twostage

    tokenizer = Tokenizer()
    parts = cascade.fork(tokenizer, ROW, 4096)
    model = ScriptedModel(script)
    cache = SimpleNamespace(seen=list(parts["prefix"]))
    cascade.CascadeScorer(model, tokenizer, {}, think_budget=budget)._think(parts, cache, len(parts["prefix"]), seed=1)
    prompt = twostage.think_prompt(tokenizer, ROW, 4096)
    assert prompt["prompt_ids"] == parts["prefix"] + parts["s2_suffix"]
    # vLLM returns the stop token at the end of its output; the budget case has none.
    sampled = script[:budget] if ending == "budget" else script[:script.index(1 if ending == "closed" else 0) + 1]
    tokens, found = vllm_think.classify(sampled, prompt["think_end"], set(prompt["stop_ids"]), budget)
    assert found == ending
    thought = {"thought_ids": tokens, "ending": found}
    assert twostage.readout_ids(parts, thought) == cache.seen


def test_two_stage_reads_the_answer_slots_and_merges_seeds():
    from thinkjev import twostage

    tokenizer = Tokenizer()
    engine = {"name": "vllm"}
    thoughts = [{"id": "r", "seed_index": k, "seed": 5 + k, "thought_ids": [200] * k, "ending": "closed",
                 "think_budget": 8, "engine": engine} for k in (1, 0)]
    reads = [(t, twostage.read_s2(BiasedModel([2.0, 0.5]), tokenizer, ROW, t, 4096)) for t in thoughts]
    assert reads[0][1] == [2.0, 0.5]
    merged = twostage.merge({"id": "r", "option_logits": [0.0, 1.0]}, reads)
    assert merged["s2_seed"] == 5 and merged["s2_think_tokens"] == 0 and merged["s2_closed_naturally"]
    assert merged["s2_seconds"] is None and [s["seed"] for s in merged["s2_samples"]] == [5, 6]
    with pytest.raises(ValueError):
        twostage.merge({"id": "r"}, reads[:1])


def test_clopper_pearson_and_ltt_thresholds():
    assert evaluate_cascade.clopper_pearson_upper(0, 59, 0.05) == pytest.approx(1 - 0.05 ** (1 / 59), abs=1e-6)
    scores = np.linspace(1, 0, 400)
    groups = [f"g{i}" for i in range(400)]
    everything_right = np.ones(400)
    assert evaluate_cascade.ltt_threshold(scores, everything_right, groups, 0.05) == 0.0
    # Errors only below 0.5: the guarantee stops before them, and fewer groups mean a stricter bound.
    lower_half_wrong = (scores >= 0.5).astype(float)
    loose = evaluate_cascade.ltt_threshold(scores, lower_half_wrong, groups, 0.05)
    accepted = scores >= loose
    assert 0.45 < loose < 0.6 and 1 - lower_half_wrong[accepted].mean() < 0.05
    paired = evaluate_cascade.ltt_threshold(scores, lower_half_wrong, [f"g{i // 2}" for i in range(400)], 0.05)
    assert paired > loose
    # 50 groups can never certify 5% risk (zero errors needs 59).
    assert evaluate_cascade.ltt_threshold(scores, everything_right, [f"g{i // 8}" for i in range(400)], 0.05) == float("inf")
    assert evaluate_cascade.ltt_threshold(scores, np.zeros(400), groups, 0.05) == float("inf")


def _synthetic_run(tmp_path, rows=240):
    rng = np.random.default_rng(3)
    gold, cascade_rows, jev, prefork = [], [], [], []
    for index in range(rows):
        truth = int(rng.integers(3))
        margin = rng.normal(1.5, 1.5)
        s1 = rng.normal(0, 0.3, 3)
        s1[truth if margin > 0 else (truth + 1) % 3] += abs(margin)
        s2 = rng.normal(0, 0.3, 3)
        s2[truth if rng.random() < 0.85 else (truth + 2) % 3] += 3
        ids = ["a", "b", "c"]
        gold.append({"id": f"r{index}", "group_id": f"g{index // 2}", "family": f"f{index % 2}", "lang": ["en", "zh"][index % 2],
                     "difficulty": 1 + index % 4, "cascade_split": "dev" if index % 5 < 2 else "test",
                     "cascade_source": "owned", "state": "s", "question": "q",
                     "options": [{"id": x, "description": x} for x in ids], "label": truth})
        think = int(rng.integers(40, 300))
        # Checkpoint reads drift from the S1 read toward the final S2 read as the thought grows.
        anytime = [{"t": t, "option_logits": ((1 - t / think) * s1 + t / think * s2).tolist()}
                   for t in (0, 64, 128, 256) if t < think]
        cascade_rows.append({"id": f"r{index}", "option_ids": ids, "option_logits": s1.tolist(),
                             "s2_option_logits": s2.tolist(), "s2_think_tokens": think, "s2_closed_naturally": True,
                             "s2_seconds": None, "prefill_seconds": 0.01, "copy_seconds": 0.0, "s1_seconds": 0.01,
                             "s2_anytime": anytime})
        p = np.exp(s1 * 0.8) / np.exp(s1 * 0.8).sum()
        jev.append({"id": f"r{index}", "option_ids": ids, "option_logits": np.log(p).tolist()})
        prefork.append(np.stack([np.r_[margin, rng.normal(size=3)]] * 2).astype(np.float16))
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("rows", "cascade", "jev")}
    for name, data in (("rows", gold), ("cascade", cascade_rows), ("jev", jev)):
        paths[name].write_text("".join(json.dumps(item) + "\n" for item in data))
    paths["prefork"] = tmp_path / "prefork.npy"
    with paths["prefork"].open("wb") as stream:
        for array in prefork:
            np.save(stream, array)
    return paths


def test_evaluator_runs_end_to_end_on_two_stage_rows_with_externals(tmp_path, monkeypatch):
    paths = _synthetic_run(tmp_path)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_cascade.py", "--rows", str(paths["rows"]), "--cascade", str(paths["cascade"]),
                                      "--prefork-features", str(paths["prefork"]), "--external", f"jev={paths['jev']}",
                                      "--report", str(report), "--bootstrap", "20"])
    evaluate_cascade.main()
    result = json.loads(report.read_text())
    assert {"confidence", "router_probe", "jev_confidence", "oracle"} <= set(result["policies"])
    assert result["baselines_test"]["always_s2_mean_seconds"] is None
    assert "jev" in result["s1_readouts_test"] and "ece_t1" in result["s1_readouts_test"]["jev"]
    risk = result["s1_readouts_test"]["raw"]["coverage_at_risk"][-1]
    assert {"ltt_dev_threshold", "ltt_test_coverage"} <= risk.keys()
    assert {entry["lang"] for entry in result["per_family_test"]} == {"en", "zh"}
    assert all(0 <= entry["escalation_rate"] <= 1 for entry in result["per_family_test"])
    areas = result["normalized_area_test"]
    assert areas["primary"] == "confidence" and areas["area"]["confidence"] is not None
    assert "jev_confidence" in areas["difference_vs_primary"]
    anytime = result["anytime_test"]
    assert anytime["checkpoints"] == [0, 64, 128, 256]
    assert [entry["t"] for entry in anytime["per_checkpoint"]] == [0, 64, 128, 256]
    # tau = inf accepts nothing early: both policies are always-S2 at full thought cost.
    for name in ("anytime", "gate"):
        curve = anytime["curves"][name]
        assert curve["tau"][-1] == float("inf")
        assert curve["accuracy"][-1] == pytest.approx(result["baselines_test"]["always_s2_accuracy"])
    # At the same tau, stopping early never spends more thought than the gate.
    assert all(a <= g + 1e-9 for a, g in zip(anytime["curves"]["anytime"]["mean_think_tokens"],
                                             anytime["curves"]["gate"]["mean_think_tokens"]))
    assert {point["policy"] for point in anytime["dev_selected_operating_points"]} == {"anytime", "gate"}


def test_normalized_area_is_one_for_oracle_and_zero_for_constant_scores():
    rng = np.random.default_rng(0)
    s1 = (rng.random(300) < 0.7).astype(float)
    s2 = (rng.random(300) < 0.8).astype(float)
    assert evaluate_cascade.normalized_area(s1 - s2, s1, s2) == pytest.approx(1.0)
    assert evaluate_cascade.normalized_area(np.zeros(300), s1, s2) == pytest.approx(0.0, abs=1e-9)
