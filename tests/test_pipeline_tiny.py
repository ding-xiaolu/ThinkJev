"""End-to-end run of the real CLIs on a tiny random hybrid (Gated DeltaNet + attention) model.

A byte-level tokenizer with a Qwen3.5-style chat template and a 4-layer random Qwen3.5 text model
exercise the real transformers code paths: the shared-prefix fork, recurrent-state copies,
rotations, hidden states, two-stage prompts and readout, and the evaluator on the merged output.
Thoughts are scripted instead of sampled by vLLM. Accuracy is meaningless; only plumbing is tested.
Runs on CPU, or on CUDA when flash-linear-attention is installed: transformers then uses its Triton
kernels for every Gated DeltaNet call, and they cannot read CPU tensors.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import evaluate_cascade  # noqa: E402

from thinkjev import _semif, cli  # noqa: E402

# Newlines are written as expressions because transformers renders templates with trim_blocks.
TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}{{ '\\n</think>\\n\\n' }}{% endif %}{% endif %}"
)
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def build_model(directory: Path) -> None:
    alphabet = tokenizers.pre_tokenizers.ByteLevel.alphabet()
    vocab = {char: index for index, char in enumerate(sorted(alphabet))}
    # Like Qwen3.5's tokenizer, a double newline is one token, so "<think>\n" is not a token prefix of
    # "<think>\n\n</think>" and the thinking fork falls inside the text boundary.
    vocab["ĊĊ"] = len(vocab)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[("Ċ", "Ċ")]))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIALS)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|im_end|>",
                                                chat_template=TEMPLATE)
    fast.save_pretrained(directory)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=len(vocab) + len(SPECIALS), hidden_size=64, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        eos_token_id=fast.eos_token_id,
    )
    torch.manual_seed(0)
    transformers.Qwen3_5ForCausalLM(config).to(torch.bfloat16).save_pretrained(directory)


def pin_device(monkeypatch) -> None:
    if importlib.util.find_spec("fla") is None:
        monkeypatch.setattr(_semif, "resolve_device", lambda device="auto": torch.device("cpu"))
        monkeypatch.setattr(_semif, "synchronize", lambda device: None)
    elif torch.cuda.is_available():
        monkeypatch.setattr(_semif, "resolve_device", lambda device="auto": torch.device("cuda"))
    else:
        pytest.skip("flash-linear-attention is installed, so transformers needs CUDA for Gated DeltaNet")


def run(monkeypatch, entry, *argv):
    monkeypatch.setattr(sys, "argv", ["prog", *map(str, argv)])
    entry()


def test_real_pipeline_on_a_tiny_hybrid_model(tmp_path, monkeypatch):
    model = tmp_path / "tiny"
    build_model(model)
    pin_device(monkeypatch)
    rows = []
    for index in range(20):
        options = [{"id": f"o{k}", "description": f"option {k} for row {index}"} for k in range(2 + index % 3)]
        rows.append({"id": f"r{index}", "group_id": f"g{index}", "family": "tiny", "lang": "en", "difficulty": 1,
                     "cascade_source": "tiny", "cascade_split": "dev" if index % 2 else "test",
                     "state": {"n": index}, "question": "Pick one.", "options": options, "label": index % 2})
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    common = ["--model", model, "--revision", "local-test"]

    run(monkeypatch, cli.main, *common, "--input", rows_path, "--output", tmp_path / "s1.jsonl", "--s2", "never",
        "--rotations", "--features", tmp_path / "f.npy", "--prefork-features", tmp_path / "p.npy",
        "--feature-layers", "2,4", "--dtype", "float32")
    run(monkeypatch, cli.s2_main, "prompts", *common, "--input", rows_path, "--output", tmp_path / "prompts.jsonl")
    prompts = [json.loads(line) for line in (tmp_path / "prompts.jsonl").read_text().splitlines()]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    assert prompts[0]["think_end"] == tokenizer.convert_tokens_to_ids("</think>")
    # Scripted thoughts in place of vLLM: one closed, one over budget, per row alternately.
    word = tokenizer.encode("step", add_special_tokens=False)
    with (tmp_path / "thoughts.jsonl").open("x") as stream:
        for index, prompt in enumerate(prompts):
            stream.write(json.dumps({"id": prompt["id"], "seed_index": 0, "seed": prompt["seed"],
                                     "thought_ids": word * (1 + index % 4), "ending": ["closed", "budget"][index % 2],
                                     "think_budget": 16, "engine": {"name": "scripted"}}) + "\n")
    run(monkeypatch, cli.s2_main, "readout", *common, "--input", rows_path, "--output", tmp_path / "cascade.jsonl",
        "--s1", tmp_path / "s1.jsonl", "--thoughts", tmp_path / "thoughts.jsonl", "--prompts", tmp_path / "prompts.jsonl",
        "--dtype", "float32")
    merged = [json.loads(line) for line in (tmp_path / "cascade.jsonl").read_text().splitlines()]
    assert [pred["id"] for pred in merged] == [row["id"] for row in rows]
    assert all(len(pred["s2_option_logits"]) == len(row["options"]) for pred, row in zip(merged, rows))
    assert evaluate_cascade.read_features(tmp_path / "p.npy")[0].shape == (2, 64)

    # The one-process scorer on the same thought must read the same S2 logits.
    from thinkjev import cascade, twostage

    loaded, tok, meta = _semif.load_causal_model(str(model), "local-test", "auto", "float32")
    parts = cascade.fork(tok, rows[0], 4096)
    scorer = cascade.CascadeScorer(loaded, tok, meta, think_budget=16)
    with torch.inference_mode():
        _, cache = scorer._forward(parts["prefix"], None, len(parts["prefix"]))
        position = len(parts["prefix"]) + len(parts["s2_suffix"])
        _, cache = scorer._forward(parts["s2_suffix"], cache, position)
        _, cache = scorer._forward(word, cache, position + len(word))
        closing = twostage.closing(parts, "closed")
        logits, _ = scorer._forward(closing, cache, position + len(word) + len(closing))
    np.testing.assert_allclose(logits[parts["slots"]].tolist(), merged[0]["s2_option_logits"], atol=1e-3)

    # Anytime readout: the chunked final read matches the one-shot read, and the read at t matches
    # a one-shot read of the thought cut at t and closed as over budget (the budget-t answer).
    run(monkeypatch, cli.s2_main, "readout", *common, "--input", rows_path, "--output", tmp_path / "anytime.jsonl",
        "--s1", tmp_path / "s1.jsonl", "--thoughts", tmp_path / "thoughts.jsonl", "--prompts", tmp_path / "prompts.jsonl",
        "--dtype", "float32", "--checkpoints", "0,2,5,8")
    chunked = [json.loads(line) for line in (tmp_path / "anytime.jsonl").read_text().splitlines()]
    thoughts = twostage.load_thoughts(tmp_path / "thoughts.jsonl")
    for pred, plain, row in zip(chunked, merged, rows):
        np.testing.assert_allclose(pred["s2_option_logits"], plain["s2_option_logits"], atol=1e-3)
        ids = thoughts[row["id"]][0]["thought_ids"]
        assert [read["t"] for read in pred["s2_anytime"]] == [t for t in (0, 2, 5, 8) if t < len(ids)]
        for read in pred["s2_anytime"][:2]:
            cut = {"thought_ids": ids[:read["t"]], "ending": "budget"}
            np.testing.assert_allclose(read["option_logits"], twostage.read_s2(loaded, tok, row, cut, 4096), atol=1e-3)

    # The one-process scorer reads the same checkpoints while it thinks: on its own sampled thought,
    # its reads equal the batch readout's reads on that thought replayed, and the thought itself is
    # the same as without checkpoints.
    run(monkeypatch, cli.main, *common, "--input", rows_path, "--output", tmp_path / "live.jsonl", "--s2", "always",
        "--think-budget", "6", "--trace", tmp_path / "live-thoughts.jsonl", "--dtype", "float32",
        "--checkpoints", "0,2,5,8")
    run(monkeypatch, cli.main, *common, "--input", rows_path, "--output", tmp_path / "plain.jsonl", "--s2", "always",
        "--think-budget", "6", "--trace", tmp_path / "plain-thoughts.jsonl", "--dtype", "float32")
    live = [json.loads(line) for line in (tmp_path / "live.jsonl").read_text().splitlines()]
    plain = [json.loads(line) for line in (tmp_path / "plain.jsonl").read_text().splitlines()]
    live_thoughts = [json.loads(line) for line in (tmp_path / "live-thoughts.jsonl").read_text().splitlines()]
    plain_thoughts = [json.loads(line) for line in (tmp_path / "plain-thoughts.jsonl").read_text().splitlines()]
    assert [t["thought_ids"] for t in live_thoughts] == [t["thought_ids"] for t in plain_thoughts]
    assert all(a["s2_option_logits"] == b["s2_option_logits"] for a, b in zip(live, plain))
    seen_checkpoint = False
    for pred, thought, row in zip(live, live_thoughts, rows):
        ids = thought["thought_ids"]
        assert [read["t"] for read in pred["s2_anytime"]] == [t for t in (0, 2, 5) if t < len(ids)]
        replay = {"thought_ids": ids, "ending": "closed" if pred["s2_closed_naturally"] else "budget"}
        _, reads = twostage.read_s2_anytime(loaded, tok, row, replay, 4096, [0, 2, 5, 8])
        for mine, theirs in zip(pred["s2_anytime"], reads):
            assert mine["t"] == theirs["t"]
            np.testing.assert_allclose(mine["option_logits"], theirs["option_logits"], atol=1e-3)
            seen_checkpoint = True
    assert seen_checkpoint

    report = tmp_path / "report.json"
    run(monkeypatch, evaluate_cascade.main, "--rows", rows_path, "--cascade", tmp_path / "anytime.jsonl",
        "--features", tmp_path / "f.npy", "--report", report, "--bootstrap", "5")
    result = json.loads(report.read_text())
    assert result["rows"] == {"dev": 10, "test": 10}
    assert result["anytime_test"]["checkpoints"] == [0, 2, 5, 8]


def test_server_answers_jev_requests_with_a_tiny_hybrid_model(tmp_path, monkeypatch):
    import threading
    import urllib.request

    from thinkjev import ThinkJev, serve

    model = tmp_path / "tiny"
    build_model(model)
    pin_device(monkeypatch)
    engine = ThinkJev.load(str(model), "local-test", dtype="float32", think_budget=4, threshold=1.0, rotations=True,
                           checkpoints=(0, 2))
    server = serve.Server(("127.0.0.1", 0), serve.make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    # Talk to the local server directly even when HTTP(S)_PROXY is set in the environment.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        body = {"state": {"ticket": "charged twice"}, "model": "jev-latest", "thinkjev": {"include_thought": True},
                "questions": {"dept": {"type": "choice", "instructions": "Which team?",
                                       "criteria": {"billing": "Payments", "tech": "Bugs", "sales": None}},
                              "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                              "mood": {"type": "score", "instructions": "How angry?", "criteria": ["Calm", "Angry"]}}}
        request = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        out = json.loads(opener.open(request).read())
        assert set(out["answers"]) == {"dept", "urgent", "mood"} and out["model"] == "thinkjev-tiny"
        assert sum(out["answers"]["dept"]["probabilities"].values()) == pytest.approx(1.0)
        assert 0 <= out["answers"]["urgent"]["noul"] <= 1 and 0 <= out["answers"]["mood"]["score"] <= 1
        # Threshold 1.0: a random model is never sure, so every question thinks (at most 4 tokens).
        assert all(trace["system"] == "s2" and trace["think_tokens"] <= 4 for trace in out["thinkjev"].values())
        assert out["usage"]["output_tokens"] == sum(trace["think_tokens"] for trace in out["thinkjev"].values())
        assert all(trace["anytime_tau"] == 1.0 and trace["anytime_stop"] == "full" for trace in out["thinkjev"].values())
        page = opener.open(url + "/").read().decode()
        assert "/v1/systemone" in page and "/v1/thinkjev/info" in page
        info = json.loads(opener.open(url + "/v1/thinkjev/info").read())
        assert info["checkpoints"] == [0, 2] and info["think_budget"] == 4 and info["model"] == "thinkjev-tiny"
        # The streamed form of the same request: S1, then checkpoint reads as the thought passes them.
        streamed = {**body, "questions": {"dept": body["questions"]["dept"]}, "thinkjev": {"stream": True}}
        request = urllib.request.Request(url + "/v1/systemone", json.dumps(streamed).encode(),
                                         {"Content-Type": "application/json"})
        with opener.open(request) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            events = [json.loads(line[5:]) for line in response.read().decode().split("\n") if line.startswith("data:")]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "s1" and kinds[-2:] == ["answer", "done"] and events[0]["escalate"] is True
        checkpoints = [e for e in events if e["type"] == "checkpoint"]
        assert [e["t"] for e in checkpoints] == [t for t in (0, 2) if t < events[-2]["trace"]["think_tokens"]]
        assert all(abs(sum(e["probabilities"].values()) - 1) < 1e-6 for e in checkpoints)
        assert events[-1]["response"]["answers"]["dept"]["choice"] in {"billing", "tech", "sales"}
        bad = urllib.request.Request(url + "/v1/systemone", b'{"state": "s"}', {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(bad)
        assert error.value.code == 422
    finally:
        server.shutdown()
