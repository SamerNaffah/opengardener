"""
Strategy coercion is the most subtle bit of agent behaviour: the LLM may return
a JSON-string, a dict, or a freeform string, and the wire format requires a dict.
These tests pin that contract so the wire never sees a non-dict.
"""
import pytest

# Force-skip if sentence-transformers/torch aren't available in the test env.
pytest.importorskip("sentence_transformers")

from base.agent import EmergentAgent  # noqa: E402


def test_coerce_dict_passthrough():
    out = EmergentAgent._coerce_strategy({"method": "x", "k": 1})
    assert out == {"method": "x", "k": 1}


def test_coerce_json_string_returns_dict():
    out = EmergentAgent._coerce_strategy('{"method": "y"}')
    assert out == {"method": "y"}


def test_coerce_non_json_string_wraps_as_raw():
    out = EmergentAgent._coerce_strategy("just some text")
    assert out == {"raw": "just some text"}


def test_coerce_json_array_wraps_as_raw():
    # JSON arrays are not valid strategies — wrap in raw rather than mis-typing.
    out = EmergentAgent._coerce_strategy('["a","b"]')
    assert out == {"raw": '["a","b"]'}


def test_coerce_other_types():
    assert EmergentAgent._coerce_strategy(42) == {"value": "42"}
    assert EmergentAgent._coerce_strategy(None) == {"value": "None"}
