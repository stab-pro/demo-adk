"""Custom FastAPI web server that invokes the greeting agent directly.

Bypasses `adk web` / `adk api_server` — the Runner is wired up here and
invoked from a plain FastAPI route handler.
"""

import os
import uuid
import logging

from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, etc. from .env

# ---------------------------------------------------------------------------
# Langfuse — auto-instrumentation via OpenTelemetry + openinference
# Must be set up before any ADK code is imported/executed.
# ---------------------------------------------------------------------------
import base64 as _base64

from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as _OTLPSpanExporter
from langfuse import Langfuse as _Langfuse
from openinference.instrumentation.google_adk import GoogleADKInstrumentor as _GoogleADKInstrumentor

_lf_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
_lf_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_lf_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

# Langfuse OTel endpoint — Basic-auth header required
_lf_auth = _base64.b64encode(f"{_lf_public_key}:{_lf_secret_key}".encode()).decode()
_otlp_exporter = _OTLPSpanExporter(
    endpoint=f"{_lf_host}/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {_lf_auth}"},
)
_tracer_provider = _TracerProvider()
_tracer_provider.add_span_processor(_BatchSpanProcessor(_otlp_exporter))
_otel_trace.set_tracer_provider(_tracer_provider)

_langfuse = _Langfuse()
try:
    _langfuse.auth_check()
    logging.getLogger(__name__).info(
        "Langfuse connected — traces will be sent to %s",
        _lf_host,
    )
except Exception as _lf_err:
    logging.getLogger(__name__).warning(
        "Langfuse auth check failed (%s) — check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST",
        _lf_err,
    )

_GoogleADKInstrumentor().instrument(tracer_provider=_tracer_provider)

# ---------------------------------------------------------------------------
# Credential bootstrap — inject VS Code cached credentials before ADK loads
# ---------------------------------------------------------------------------
import json as _json
import google.auth as _google_auth
import google.oauth2.credentials as _oauth2_creds

_VSC_CREDS_PATH = os.path.expanduser(
    "~/.cache/google-vscode-extension/auth/credentials.json"
)

if os.path.exists(_VSC_CREDS_PATH):
    def _vsc_credentials(**kwargs):
        with open(_VSC_CREDS_PATH) as _f:
            _d = _json.load(_f)
        return _oauth2_creds.Credentials(
            token=_d["credentials"]["access_token"],
            refresh_token=_d.get("refreshToken"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=_d["credentials"].get("client_id"),
            client_secret=_d["credentials"].get("client_secret"),
            scopes=_d["credentials"]["scope"].split(),
        ), os.getenv("GOOGLE_CLOUD_PROJECT")

    _google_auth.default = _vsc_credentials

import uvicorn
from fastapi import FastAPI, HTTPException
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from greeting_agent.agent import root_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
    logger.warning(
        "GOOGLE_GENAI_USE_VERTEXAI is not set — defaulting to Gemini API. "
        "For Vertex AI set GOOGLE_GENAI_USE_VERTEXAI=1, GOOGLE_CLOUD_PROJECT, "
        "and GOOGLE_CLOUD_LOCATION in your .env file."
    )
else:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "<unset>")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    logger.info("Using Vertex AI — project=%s  location=%s", project, location)

# ---------------------------------------------------------------------------
# ADK wiring — one Runner for the lifetime of the process
# ---------------------------------------------------------------------------

APP_NAME = "greeting_app"

session_service = InMemorySessionService()

runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
    auto_create_session=True,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Greeting Agent Server")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None  # auto-generated when omitted


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the greeting agent and return its reply."""
    session_id = req.session_id or str(uuid.uuid4())

    new_message = types.Content(
        role="user",
        parts=[types.Part(text=req.message)],
    )

    reply_parts: list[str] = []

    try:
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            # Collect text from the final response event only
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        reply_parts.append(part.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not reply_parts:
        raise HTTPException(status_code=500, detail="Agent returned no response")

    return ChatResponse(session_id=session_id, reply="".join(reply_parts))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "agent": root_agent.name}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
