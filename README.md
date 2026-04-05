# VantaBlock 
A unified behavioural logging system that collects and analyzes user interactions (both human and bot) to detect automated traffic. Built for the HackByte 4.0 hackathon.

## What Is Implemented
- Unified event schema for both human (`frontend/logger.js`) and bot (`bot/bot_script.py`).
- Backend endpoint `POST /api/logs` that saves **one JSON file per session** in `logs/`.
- Session file naming convention:
  - `human_YYYYMMDD_HHMMSS_uniqueID.json`
  - `agent_YYYYMMDD_HHMMSS_uniqueID.json`
- Feature extraction pipeline: `extract_features.py` writes a CSV table for ML training.

## MockShop Behavioural Logging (Unified Schema)

This workspace collects comparable behavioural data for human and agent sessions.

## Run

### 1) Start backend (serves pages + log API)
```bash
cd backend
python main.py
```
Backend runs on `http://localhost:8001` and serves frontend at `http://localhost:8001/demo/index.html`.

### 2) Collect human session
Open in browser:
- `http://localhost:8001/demo/index.html`

Interact naturally across pages; logger sends full session snapshots every 30 seconds and on page unload.

### 3) Collect bot session
```bash
cd bot
python bot_script.py
```
The bot writes a local session file to `logs/` and also POSTs to `/api/logs`.

### 4) Inspect stored sessions
- `http://localhost:8001/sessions?session_type=human`
- `http://localhost:8001/sessions?session_type=agent`
- `http://localhost:8001/session/<session_id>`

### 5) Extract ML features
From project root:
```bash
python extract_features.py --logs-dir logs --output features.csv
```

## Session JSON Shape
```json
{
  "session_id": "human_20260404_153022_42ab9c",
  "session_type": "human",
  "start_time_ms": 0,
  "end_time_ms": 123456,
  "user_agent": "Mozilla/5.0 ...",
  "viewport": { "width": 1280, "height": 800 },
  "events": [
    {
      "event_id": 1,
      "timestamp_ms": 0,
      "type": "navigation",
      "data": { "url": "...", "referrer": "" }
    }
  ]
}
```
Built for HackByte 4.0.
Uses Playwright for browser automation and scikit‑learn for behavioural classification.
