# Greeting Agent — Google ADK + Custom Web Server

A minimal Google ADK agent that greets users, served by a custom FastAPI server
(no `adk web` or `adk api_server` involved).

## Structure

```
demo-adk/
├── greeting_agent/
│   ├── __init__.py
│   └── agent.py        # ADK Agent definition (root_agent)
├── server.py           # Custom FastAPI server — wires up Runner directly
├── requirements.txt
└── .env                # (you create this) holds GOOGLE_API_KEY
```

## Setup

```bash
pip install -r requirements.txt

# Create a .env file with your Vertex AI config
cat > .env <<'EOF'
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
EOF
```

Authentication uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
Run `gcloud auth application-default login` if you haven't already.

## Run

```bash
python server.py
# or
uvicorn server:app --reload
```

## Usage

**Health check**
```bash
curl http://localhost:8000/health
```

**Greet the agent**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, I am Alice!", "user_id": "alice"}'
```

Response:
```json
{
  "session_id": "<uuid>",
  "reply": "Hello, Alice! ..."
}
```

Pass the same `session_id` back in subsequent requests to continue the same session.

## How it works

1. `greeting_agent/agent.py` defines a plain `root_agent` (an `Agent` instance).
2. `server.py` creates an ADK `Runner` backed by `InMemorySessionService` at
   startup — no ADK CLI server is used.
3. The `/chat` endpoint calls `runner.run_async(...)`, iterates the event stream,
   and returns the text from the final response event.
