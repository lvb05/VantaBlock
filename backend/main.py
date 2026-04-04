import json
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Separate directories for human and bot logs
HUMAN_LOG_DIR = LOG_DIR / "human_logs"
HUMAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
BOT_LOG_DIR = LOG_DIR / "bot_logs"
BOT_LOG_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/demo", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def next_log_index(log_dir: Path) -> int:
    """Get the next available log index for incremental naming (1.json, 2.json, etc.)"""
    max_idx = 0
    for path in log_dir.glob("*.json"):
        stem = path.stem
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))
    return max_idx + 1


def normalize_session_type(raw: str) -> str:
    return "agent" if str(raw).lower() == "agent" else "human"


def build_session_id(session_type: str) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = datetime.now().strftime("%f")[:6]
    return f"{session_type}_{now}_{unique}"


def validate_payload(payload: dict) -> None:
    required = [
        "session_id",
        "session_type",
        "start_time_ms",
        "end_time_ms",
        "user_agent",
        "viewport",
        "events",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")

    if not isinstance(payload["events"], list):
        raise HTTPException(status_code=400, detail="events must be an array")

    vp = payload.get("viewport", {})
    if not isinstance(vp, dict) or "width" not in vp or "height" not in vp:
        raise HTTPException(status_code=400, detail="viewport must contain width and height")


@app.post("/api/logs")
async def receive_session_log(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    validate_payload(payload)

    session_type = normalize_session_type(payload.get("session_type", "human"))
    session_id = str(payload.get("session_id", "")).strip()

    if not SESSION_ID_RE.match(session_id):
        session_id = build_session_id(session_type)

    if not session_id.startswith(f"{session_type}_"):
        session_id = f"{session_type}_{session_id}"

    payload["session_id"] = session_id
    payload["session_type"] = session_type

    # Ensure monotonic-style anchors are normalized as requested.
    payload["start_time_ms"] = 0
    payload["end_time_ms"] = int(payload.get("end_time_ms", 0))

    # Save to appropriate directory with incremental naming
    if session_type == "human":
        log_index = next_log_index(HUMAN_LOG_DIR)
        out_path = HUMAN_LOG_DIR / f"{log_index}.json"
    else:
        log_index = next_log_index(BOT_LOG_DIR)
        out_path = BOT_LOG_DIR / f"{log_index}.json"
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "status": "saved",
        "session_id": session_id,
        "file": out_path.name,
        "events": len(payload["events"]),
    }


@app.get("/sessions")
async def list_sessions(session_type: str = "human"):
    normalized = normalize_session_type(session_type)
    rows = []
    for path in sorted(LOG_DIR.glob(f"{normalized}_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "session_id": data.get("session_id", path.stem),
                "session_type": data.get("session_type", normalized),
                "events_count": len(data.get("events", [])),
                "end_time_ms": data.get("end_time_ms", 0),
                "file": path.name,
            }
        )
    return rows


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", session_id)
    path = LOG_DIR / f"{safe_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8001, reload=True)
