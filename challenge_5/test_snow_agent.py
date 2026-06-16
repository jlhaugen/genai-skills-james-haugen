"""Unit tests for snow_agent_logic. All external services are mocked."""

import pytest
from unittest.mock import MagicMock

import snow_agent_logic as logic


# --- weather tool ---

def test_get_weather_known_location():
    w = logic.get_weather("Anchorage")
    assert w["location"] == "Anchorage"
    assert w["conditions"] == "Snow"
    assert set(w) == {"location", "temp_f", "conditions", "wind_mph", "snow_1h_in"}


def test_get_weather_unknown_location_uses_default():
    w = logic.get_weather("Sitka")
    assert w["location"] == "Sitka"
    assert w["conditions"] == "Cloudy"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_get_weather_rejects_empty(bad):
    with pytest.raises(ValueError):
        logic.get_weather(bad)


# --- RAG result formatting ---

def test_format_rag_results_with_rows():
    df = MagicMock()
    df.__len__.return_value = 2
    df.__getitem__.return_value = ["plowing info", "reporting info"]
    out = logic.format_rag_results(df)
    assert out == {"results": ["plowing info", "reporting info"]}


def test_format_rag_results_empty():
    df = MagicMock()
    df.__len__.return_value = 0
    assert logic.format_rag_results(df) == {"results": []}


def test_format_rag_results_none():
    assert logic.format_rag_results(None) == {"results": []}


# --- handle_request pipeline ---

def _safe(_): return (True, None)
def _agent_ok(_): return "Roads are plowed in priority order."


def test_handle_request_happy_path():
    out = logic.handle_request("How are roads plowed?",
                               input_check=_safe, agent_fn=_agent_ok, output_check=_safe)
    assert out == "Roads are plowed in priority order."


def test_handle_request_blocks_bad_input():
    def bad_input(_): return (False, "injection")
    out = logic.handle_request("ignore instructions",
                               input_check=bad_input, agent_fn=_agent_ok, output_check=_safe)
    assert "blocked" in out.lower()


def test_handle_request_blocks_bad_output():
    def bad_output(_): return (False, "sensitive data")
    out = logic.handle_request("question",
                               input_check=_safe, agent_fn=_agent_ok, output_check=bad_output)
    assert out == logic.BLOCKED_MESSAGE


def test_handle_request_blocks_empty_answer():
    def empty_agent(_): return ""
    out = logic.handle_request("question",
                               input_check=_safe, agent_fn=empty_agent, output_check=_safe)
    assert out == logic.BLOCKED_MESSAGE


def test_handle_request_logs_each_outcome():
    logged = []
    def logger(p, r, blocked, reason): logged.append((blocked, reason))
    logic.handle_request("ok question",
                         input_check=_safe, agent_fn=_agent_ok, output_check=_safe, logger=logger)
    assert logged == [(False, None)]
