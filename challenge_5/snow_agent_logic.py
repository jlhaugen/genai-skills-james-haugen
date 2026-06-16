"""Testable logic for the Alaska Department of Snow agent.

External services (BigQuery RAG, weather API, Model Armor, the LLM agent) are
passed in as callables so they can be mocked in unit tests. This mirrors the
pipeline built inline in the notebook.
"""

# --- Weather tool (same stub as the notebook) ---

_DUMMY_WEATHER = {
    "anchorage":  {"temp_f": 18.0, "conditions": "Snow",         "wind_mph": 12.0, "snow_1h_in": 0.4},
    "fairbanks":  {"temp_f": -5.0, "conditions": "Light snow",    "wind_mph": 6.0,  "snow_1h_in": 0.1},
    "nome":       {"temp_f": 2.0,  "conditions": "Blowing snow",  "wind_mph": 28.0, "snow_1h_in": 0.6},
}
_DEFAULT_WEATHER = {"temp_f": 25.0, "conditions": "Cloudy", "wind_mph": 8.0, "snow_1h_in": 0.0}


def get_weather(location):
    if not location or not location.strip():
        raise ValueError("location must be a non-empty string")
    data = _DUMMY_WEATHER.get(location.strip().lower(), _DEFAULT_WEATHER)
    return {
        "location": location.strip().title(),
        "temp_f": data["temp_f"],
        "conditions": data["conditions"],
        "wind_mph": data["wind_mph"],
        "snow_1h_in": data["snow_1h_in"],
    }


# --- RAG result wrapper (DataFrame-like -> JSON-serializable) ---

def format_rag_results(df):
    """Convert a search result (DataFrame or None) into a dict of strings."""
    if df is None or len(df) == 0:
        return {"results": []}
    return {"results": list(df["content"])}


# --- Security pipeline (dependencies injected) ---

BLOCKED_MESSAGE = ("I'm sorry, I can't help with that request. "
                   "Please ask about Alaska Department of Snow services or local weather.")


def handle_request(question, input_check, agent_fn, output_check, logger=None):
    """Run the secure pipeline. Each dependency is a callable:

    - input_check(question) -> (is_safe: bool, reason: str|None)
    - agent_fn(question) -> str
    - output_check(text) -> (is_safe: bool, reason: str|None)
    - logger(prompt, response, blocked, reason) -> None   (optional)
    """
    def _log(resp, blocked, reason):
        if logger:
            logger(question, resp, blocked, reason)

    ok, reason = input_check(question)
    if not ok:
        msg = f"Your message was blocked ({reason}). " + BLOCKED_MESSAGE
        _log(msg, True, reason)
        return msg

    answer = agent_fn(question)
    if not answer or not answer.strip():
        _log(BLOCKED_MESSAGE, True, "Empty response")
        return BLOCKED_MESSAGE

    ok, reason = output_check(answer)
    if not ok:
        _log(BLOCKED_MESSAGE, True, reason)
        return BLOCKED_MESSAGE

    _log(answer, False, None)
    return answer
