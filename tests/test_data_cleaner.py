"""
Smoke tests for the DataCleaner specialist's strategy plumbing — does NOT spin
up grpc/Soil/Ollama. We patch SoilClient + LLMClient so we can drive the agent
synthetically.
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sentence_transformers")


def _stub_soil(*_a, **_kw):
    s = MagicMock()
    s.query_similar.return_value = []
    s.leave_trail.return_value = True
    s.mark_failure.return_value = True
    return s


def _stub_llm(*_a, **_kw):
    l = MagicMock()
    l.is_available.return_value = False
    l.suggest_approach.return_value = {}
    return l


@pytest.fixture
def tmp_csv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(textwrap.dedent("""\
        id,name,score
        1,a,1.0
        2,,2.0
        2,,2.0
        3,c,
    """))
    return str(p)


@patch("base.agent.SoilClient", _stub_soil)
@patch("base.agent.LLMClient", _stub_llm)
def test_data_cleaner_pandas_dropna(tmp_csv):
    from specialists.data_cleaner import DataCleanerAgent
    agent = DataCleanerAgent(agent_id="t-001")
    result = agent.clean(tmp_csv)
    assert result["success"] is True
    assert result["original_rows"] == 4
    # Drops the row missing `name` (id=2 dup) and the row missing `score` (id=3),
    # then de-dupes — leaves 1 row (id=1).
    assert result["cleaned_rows"] == 1
    assert os.path.exists(result["output"])
