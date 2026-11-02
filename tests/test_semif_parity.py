"""The vendored S1 readout must stay SemIf's frozen `direct-options-v1`.

A golden check pins the rendered messages. When SemIf itself is importable (for example in an
environment that still has the old pinned dependency), the vendored code is also compared with
it verbatim. The end-to-end check against real tokenizers is a smoke run that reproduces the
prompt_sha256 of SemIf's committed direct predictions.
"""

import inspect
import json

import pytest

from thinkjev import _semif

ROW = {
    "id": "golden",
    "state": {"invoice": "INV-7", "paid": "2026-03-02"},
    "question": "Was the invoice paid late?",
    "options": [{"id": "late", "description": "Paid after the due date."},
                {"id": "on_time", "description": "Paid on or before the due date."}],
}


def test_direct_messages_match_the_frozen_prompt():
    assert _semif.PROMPT_VERSION == "direct-options-v1"
    system, user = _semif.direct_messages(ROW)
    assert system == {"role": "system", "content": (
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning.")}
    assert user["role"] == "user"
    assert user["content"] == json.dumps({
        "evidence": {"invoice": "INV-7", "paid": "2026-03-02"},
        "criterion": "Was the invoice paid late?",
        "options": [{"letter": "A", "description": "Paid after the due date."},
                    {"letter": "B", "description": "Paid on or before the due date."}],
    }, ensure_ascii=False)


@pytest.mark.parametrize("name,module", [
    ("validate_row", "core"), ("direct_messages", "core"), ("softmax", "core"), ("digest", "core"),
    ("load_causal_model", "core"), ("_slot_ids", "direct"), ("encode_prompt", "direct"), ("score", "direct"),
])
def test_vendored_code_is_verbatim_semif(name, module):
    semif = pytest.importorskip(f"semif_phase1.{module}")
    assert inspect.getsource(getattr(_semif, name)) == inspect.getsource(getattr(semif, name))
