"""Alaska Department of Snow - Flask web app.

Serves the secure, logged RAG + weather agent over HTTP. Assumes the BigQuery
embedding model, snow_embeddings table, and interaction_log table already exist
(created by the notebook). Set PROJECT_ID via environment variable.

Run locally:   python app.py
Deploy:        container + Cloud Run (expose port 8080)
"""

import os
import time
from datetime import datetime, timezone

from flask import Flask, request, jsonify, Response

import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel, Tool, FunctionDeclaration,
    AutomaticFunctionCallingResponder, GenerationConfig,
    HarmCategory, HarmBlockThreshold, SafetySetting,
)
from google.cloud import bigquery
from google.cloud import modelarmor_v1
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted

# --- Config ---
PROJECT_ID = os.environ.get("PROJECT_ID", "qwiklabs-gcp-00-fc2622edeeb6")
BQ_LOCATION = "US"
DATASET = "alaska_snow"
DS = f"{PROJECT_ID}.{DATASET}"
EMBEDDING_MODEL = "embedding_model"
GEMINI_ENDPOINT = "gemini-2.5-flash"
LOG_TABLE = f"{DS}.interaction_log"

MA_LOCATION = "us"
PROMPT_TEMPLATE = f"projects/{PROJECT_ID}/locations/{MA_LOCATION}/templates/sanitize_prompt"
RESPONSE_TEMPLATE = f"projects/{PROJECT_ID}/locations/{MA_LOCATION}/templates/sanitize_reponse"
MA_ENDPOINT = f"modelarmor.{MA_LOCATION}.rep.googleapis.com"

# --- Clients ---
vertexai.init(project=PROJECT_ID, location="us-central1")
bq = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
ma = modelarmor_v1.ModelArmorClient(
    transport="rest", client_options=ClientOptions(api_endpoint=MA_ENDPOINT))

BLOCKED_MESSAGE = ("I'm sorry, I can't help with that request. "
                   "Please ask about Alaska Department of Snow services or local weather.")

# --- Tools ---
def rag_search(query):
    sql = f"""
    SELECT base.content AS content, distance
    FROM VECTOR_SEARCH(
      TABLE `{DS}.snow_embeddings`, 'ml_generate_embedding_result',
      (SELECT ml_generate_embedding_result
       FROM ML.GENERATE_EMBEDDING(
         MODEL `{DS}.{EMBEDDING_MODEL}`,
         (SELECT @q AS content),
         STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_QUERY' AS task_type))),
      top_k => 3, distance_type => 'COSINE')
    ORDER BY distance
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("q", "STRING", query)]))
    df = job.result().to_dataframe()
    return df

def search_snow_department_info(query: str) -> dict:
    """Search the Alaska Department of Snow knowledge base."""
    df = rag_search(query)
    return {"results": list(df["content"])} if len(df) else {"results": []}

_DUMMY = {"anchorage": (18.0, "Snow", 12.0, 0.4), "fairbanks": (-5.0, "Light snow", 6.0, 0.1),
          "nome": (2.0, "Blowing snow", 28.0, 0.6)}
def get_local_weather(location: str) -> dict:
    """Get current weather and recent snowfall for an Alaska location."""
    t, c, w, s = _DUMMY.get(location.strip().lower(), (25.0, "Cloudy", 8.0, 0.0))
    return {"location": location.strip().title(), "temp_f": t,
            "conditions": c, "wind_mph": w, "snow_1h_in": s}

# --- Agent ---
SYSTEM = ("You are the virtual assistant for the Alaska Department of Snow. "
          "Use search_snow_department_info for policy/service questions and "
          "get_local_weather for weather. Base answers on tool results; if they "
          "do not help, say you don't have that information. Be concise.")
SAFETY = [SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
          for c in (HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)]
tool = Tool(function_declarations=[
    FunctionDeclaration.from_func(search_snow_department_info),
    FunctionDeclaration.from_func(get_local_weather)])
agent_model = GenerativeModel(GEMINI_ENDPOINT, system_instruction=SYSTEM, tools=[tool],
                              safety_settings=SAFETY,
                              generation_config=GenerationConfig(temperature=0.2, max_output_tokens=2048))

def agent_answer(question, max_retries=4):
    delay = 5
    for attempt in range(max_retries):
        try:
            chat = agent_model.start_chat(responder=AutomaticFunctionCallingResponder(max_automatic_function_calls=5))
            resp = chat.send_message(question)
            try:
                return resp.text.strip()
            except (ValueError, AttributeError):
                return ""
        except ResourceExhausted:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay); delay *= 2

# --- Security + logging ---
def check_input(text):
    try:
        r = ma.sanitize_user_prompt(request=modelarmor_v1.SanitizeUserPromptRequest(
            name=PROMPT_TEMPLATE, user_prompt_data=modelarmor_v1.DataItem(text=text)))
        return r.sanitization_result.filter_match_state != modelarmor_v1.FilterMatchState.MATCH_FOUND
    except Exception:
        return False

def check_output(text):
    try:
        r = ma.sanitize_model_response(request=modelarmor_v1.SanitizeModelResponseRequest(
            name=RESPONSE_TEMPLATE, model_response_data=modelarmor_v1.DataItem(text=text)))
        return r.sanitization_result.filter_match_state != modelarmor_v1.FilterMatchState.MATCH_FOUND
    except Exception:
        return False

def log_interaction(prompt, response, blocked, reason=None):
    try:
        bq.insert_rows_json(LOG_TABLE, [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt, "response": response, "blocked": blocked, "reason": reason}])
    except Exception as e:
        print("log error:", e)

def handle_request(question):
    if not check_input(question):
        log_interaction(question, BLOCKED_MESSAGE, True, "input filtered")
        return BLOCKED_MESSAGE
    answer = agent_answer(question)
    if not answer or not answer.strip():
        log_interaction(question, BLOCKED_MESSAGE, True, "empty")
        return BLOCKED_MESSAGE
    if not check_output(answer):
        log_interaction(question, BLOCKED_MESSAGE, True, "output filtered")
        return BLOCKED_MESSAGE
    log_interaction(question, answer, False, None)
    return answer

# --- Flask ---
app = Flask(__name__)

PAGE = """<!doctype html><html><head><title>Alaska Department of Snow</title>
<style>body{font-family:sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
#log{border:1px solid #ccc;border-radius:8px;padding:12px;height:360px;overflow-y:auto;margin-bottom:12px}
.u{text-align:right;color:#0b5}.a{color:#235}input{width:78%;padding:8px}button{padding:8px 14px}</style></head>
<body><h2>Alaska Department of Snow Assistant</h2><div id="log"></div>
<input id="q" placeholder="Ask about snow removal or local weather..."/>
<button onclick="send()">Send</button>
<script>
async function send(){let q=document.getElementById('q').value;if(!q)return;
let log=document.getElementById('log');log.innerHTML+='<p class="u">'+q+'</p>';
document.getElementById('q').value='';
let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
let d=await r.json();log.innerHTML+='<p class="a"><b>Assistant:</b> '+d.response+'</p>';log.scrollTop=log.scrollHeight;}
</script></body></html>"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"response": "Please enter a question."})
    return jsonify({"response": handle_request(question)})

@app.route("/healthz")
def healthz():
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
