"""Jev-format questions and answers: request mapping, Jev's documented confidence formulas, the HTTP
handler, and one round trip through TypeSafe's own SDK against a mock transport; no weights needed."""

import json

import pytest

from thinkjev import api, serve


def test_question_rows_cover_the_three_types():
    kind, row = api.question_row("dept", {"t": "x"}, {"type": "choice", "instructions": "Which team?",
                                                       "criteria": {"billing": "Payments", "tech": None}})
    assert kind == "choice" and [o["id"] for o in row["options"]] == ["billing", "tech"]
    assert row["options"][1]["description"] == "tech" and row["state"] == {"t": "x"}
    kind, row = api.question_row("urgent", "s", {"type": "noul", "instructions": "Urgent?"})
    assert kind == "noul" and [o["id"] for o in row["options"]] == ["true", "false"]
    kind, row = api.question_row("mood", "s", {"type": "score", "instructions": {"q": "How angry?"},
                                               "criteria": ["Calm", "Frustrated", "Very angry"]})
    assert kind == "score" and [o["id"] for o in row["options"]] == ["0", "1", "2"]
    assert json.loads(row["question"]) == {"q": "How angry?"}
    for bad in ({"type": "choice", "instructions": "q", "criteria": {"only": None}},
                {"type": "score", "instructions": "q", "criteria": ["one"]},
                {"type": "noul", "instructions": "q", "criteria": {"maybe": "x"}},
                {"type": "rank", "instructions": "q"},
                {"type": "noul"},
                {"type": "choice", "instructions": "q", "criteria": {str(i): None for i in range(17)}}):
        with pytest.raises(api.RequestError):
            api.question_row("bad", "s", bad)


def test_confidence_matches_typesafe_documented_examples():
    # Choice: (3 * p_max - 1) / 2 with three options.
    assert api.choice_confidence([0.88, 0.12, 0.0]) == pytest.approx((3 * 0.88 - 1) / 2)
    assert api.choice_confidence([0.5, 0.5]) == 0.0
    # Score: (0, 0.57, 0.43) -> 1 - 0.43 / (2/3); adjacent split 0.25; opposite ends 0.
    assert api.score_confidence([0.0, 0.57, 0.43]) == pytest.approx(1 - 0.43 / (2 / 3))
    assert api.score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert api.score_confidence([0.5, 0.0, 0.5]) == 0.0
    # Five levels: an even spread from the middle averages 1.2 levels.
    assert api.score_confidence([0.2] * 5) == pytest.approx(1 - 1.2 / 1.2)


def test_answer_shapes():
    _, row = api.question_row("m", "s", {"type": "score", "instructions": "q", "criteria": ["a", "b", "c"]})
    out = api.answer("score", row, [0.0, 0.95, 0.05])
    assert out["score"] == pytest.approx(1.05) and out["legend"] == {"0": "a", "1": "b", "2": "c"}
    _, row = api.question_row("n", "s", {"type": "noul", "instructions": "q"})
    assert api.answer("noul", row, [0.9, 0.1]) == {"type": "noul", "noul": 0.9}


class FakeEngine(api.ThinkJev):
    """ThinkJev with a scripted scorer: S1 is unsure on rows whose state says so."""

    def __init__(self):
        self.name, self.s1_temperature, self.s2_temperature, self.threshold = "thinkjev-fake", 1.0, 1.0, 0.8
        self.metadata, self.checkpoints, self.anytime_tau = {}, (0, 16), 0.9
        self.checkpoint_temperatures, self.reference_curve = {0: 1.0, 16: 1.0}, []
        import threading

        self.lock = threading.Lock()
        engine = self

        class Scorer:
            policy, think_budget, rotations = "gated", 42, False

            def score(self, row, on_event=None):
                n = len(row["options"])
                unsure = "hard" in json.dumps(row["state"])
                s1 = [0.0] * n if unsure else [4.0] + [0.0] * (n - 1)
                escalate = self.policy == "always" or (self.policy == "gated" and unsure)
                if on_event:
                    on_event({"type": "s1", "option_logits": s1, "debiased_logits": None,
                              "s1_confidence": 1 / n if unsure else 0.9, "escalated": escalate, "seconds": 0.01})
                result = {"option_logits": s1, "s1_confidence": 1 / n if unsure else 0.9, "escalated": escalate,
                          "prefix_tokens": 10, "s1_suffix_tokens": 3, "total_seconds": 0.05}
                if escalate:
                    # Unsure at t = 0, sure of the last option by t = 16 (the anytime policy stops there).
                    reads = [{"t": 0, "option_logits": [0.0] * n}, {"t": 16, "option_logits": [0.0] * (n - 1) + [4.0]}]
                    for read in reads:
                        if on_event:
                            on_event({"type": "checkpoint", **read})
                    if on_event:
                        on_event({"type": "thinking", "tokens": 32})
                    result.update({"s2_option_logits": [0.0] * (n - 1) + [5.0], "s2_think_tokens": 42,
                                   "s2_closed_naturally": True, "s2_thought": "thinking", "s2_anytime": reads})
                return result

        engine.scorer = Scorer()


def test_system_one_escalates_only_unsure_questions():
    engine = FakeEngine()
    body = {"state": {"case": "hard"}, "model": "jev-latest", "questions": {
        "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None, "c": None}}}}
    out = serve.handle_systemone(engine, body)
    assert out["answers"]["dept"]["choice"] == "c" and out["thinkjev"]["dept"]["system"] == "s2"
    assert out["usage"] == {"input_tokens": 13, "output_tokens": 42} and "thought" not in out["thinkjev"]["dept"]
    out = serve.handle_systemone(engine, {**body, "state": "easy", "thinkjev": {"include_thought": True}})
    assert out["answers"]["dept"]["choice"] == "a" and out["thinkjev"]["dept"]["system"] == "s1"
    out = serve.handle_systemone(engine, {**body, "thinkjev": {"think": "never"}})
    assert out["thinkjev"]["dept"]["system"] == "s1"
    out = serve.handle_systemone(engine, body, think_default="never")  # a client without the thinkjev field
    assert out["thinkjev"]["dept"]["system"] == "s1"
    for bad in ({"state": "s"}, {**body, "thinkjev": {"think": "maybe"}}, {**body, "questions": {}}):
        with pytest.raises(api.RequestError):
            serve.handle_systemone(engine, bad)


def test_streamed_events_arrive_in_order_and_end_with_the_response():
    engine = FakeEngine()
    body = {"state": {"case": "hard"}, "thinkjev": {"stream": True, "include_thought": True}, "questions": {
        "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None, "c": None}}}}
    events = list(serve.systemone_events(engine, body))
    assert [e["type"] for e in events] == ["s1", "checkpoint", "checkpoint", "thinking", "answer", "done"]
    assert all(e["question"] == "dept" for e in events[:-1])
    assert events[0]["escalate"] is True and events[0]["threshold"] == 0.8
    first, second = events[1], events[2]
    assert first["t"] == 0 and first["confidence"] == pytest.approx(1 / 3) and first["anytime_stop"] is False
    assert second["t"] == 16 and second["confidence"] > 0.9 and second["anytime_stop"] is True
    trace = events[4]["trace"]
    assert trace["anytime_stop"] == 16 and trace["thought"] == "thinking" and len(trace["anytime"]) == 2
    assert events[5]["response"]["answers"]["dept"]["choice"] == "c"
    assert events[5]["response"]["thinkjev"]["dept"]["anytime_tau"] == 0.9
    # An unsure-free question emits no checkpoint; the plain endpoint still carries the anytime trace.
    easy = serve.handle_systemone(engine, {**body, "state": "easy"})
    assert "anytime" not in easy["thinkjev"]["dept"]
    assert [e["type"] for e in serve.systemone_events(engine, {**body, "state": "easy"})] == ["s1", "answer", "done"]
    with pytest.raises(api.RequestError):
        list(serve.systemone_events(engine, {"state": "s"}))
    info = engine.info()
    assert info["checkpoints"] == [0, 16] and info["anytime_tau"] == 0.9 and info["model"] == "thinkjev-fake"


def test_typesafe_sdk_reads_our_response(monkeypatch):
    httpx2 = pytest.importorskip("httpx2")
    sdk = pytest.importorskip("typesafe_sdk")
    engine, seen = FakeEngine(), []

    def handler(request):
        seen.append(request.url.path)
        return httpx2.Response(200, json=serve.handle_systemone(engine, json.loads(request.content)))

    client = sdk.TypeSafeClient(api_key="local", transport=httpx2.MockTransport(handler))
    response = client.system_one(
        state={"ticket": "hard case"},
        questions={"dept": sdk.Choice(instructions="Which team?", criteria={"billing": None, "tech": None}),
                   "urgent": sdk.Noul(instructions="Is it urgent?"),
                   "mood": sdk.Score(instructions="How angry?", criteria=["Calm", "Angry", "Furious"])})
    assert seen == ["/v1/systemone"]
    assert response.choices["dept"].choice == "tech"
    assert response.nouls["urgent"].noul == pytest.approx(0.0067, abs=1e-3)
    assert response.answers["mood"].score == pytest.approx(2.0, abs=0.02)


@pytest.mark.parametrize("name", api.presets())
def test_bundled_calibration_matches_its_report(name):
    from pathlib import Path
    fixed = api.preset(name)
    assert set(fixed) == {"model", "revision", "calibration", "rotations", "think_budget"} and len(fixed["revision"]) == 40
    report = Path(__file__).resolve().parents[1] / "results" / json.loads(fixed["calibration"].read_text())["report"]
    if not report.exists():
        pytest.skip(f"needs {report.name}: unpack results.tar.gz from the GitHub release")
    bundled = api.calibration_from_report(fixed["calibration"], fixed["rotations"])
    full = api.calibration_from_report(report, fixed["rotations"])
    assert {k: v for k, v in bundled.items() if k != "source"} == {k: v for k, v in full.items() if k != "source"}


def test_presets_cover_the_release_models():
    assert api.presets() == ["qwen3.5-27b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.8-27b"]
    with pytest.raises(ValueError, match="bundled presets"):
        api.preset("qwen3.5-0.8b")
