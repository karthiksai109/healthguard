"""HealthGuard — Layer 6: FastAPI Gateway

API endpoints:
  /upload-photo    — Patient photo upload (wound, skin, medication)
  /voice-note      — Patient voice note upload
  /symptom         — Text symptom log
  /vital           — Record vital sign
  /logs            — View analysis logs
  /alerts          — View alerts
  /status          — Agent status + stats
  /audit           — Audit trail (verifiable receipts)
  /patients        — List patients
  /patient/{id}    — Patient detail + vitals
"""
import io
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.core.config import AppConfig, get_config
from app.core.database import Database
from app.core.clients import TelegramClient
from app.layers import ingestion, inference
from app.layers.agent import HealthGuardAgent
from app.layers.demo import load_demo_data, trigger_demo_events
from app.core.clients import get_venice_client, get_akashml_client

logger = structlog.get_logger()

app = FastAPI(title="HealthGuard", description="Decentralized Private AI Health Agent", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global references — initialized in startup
_agent: HealthGuardAgent = None
_config: AppConfig = None
_db: Database = None


@app.on_event("startup")
async def startup():
    global _agent, _config, _db
    _config = get_config()
    _db = Database(_config.data_dir, _config.encryption_salt)
    _agent = HealthGuardAgent(_config, _db)
    _agent.start()

    if _config.demo_mode:
        load_demo_data(_agent)
        trigger_demo_events(_agent)

    logger.info("gateway_started", port=_config.port, demo=_config.demo_mode)


# ── Upload Endpoints ──────────────────────────────────────────────────

@app.post("/upload-photo")
async def upload_photo(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
):
    """Upload patient health photo. EXIF stripped, processed by Venice Vision, raw deleted in 60s."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    item = ingestion.ingest_photo(_config.data_dir, raw, patient_id, ttl=_config.raw_file_ttl)
    _agent.event_queue.push(item)
    _db.audit({"type": "photo_uploaded", "patient_id": patient_id[:8] + "...", "size": len(raw), "session_id": item.session_id})
    return {"status": "queued", "session_id": item.session_id, "raw_ttl_seconds": _config.raw_file_ttl}


@app.post("/voice-note")
async def voice_note(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
):
    """Upload patient voice note. Transcribed by Venice STT, raw audio deleted in 60s."""
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 25MB)")

    item = ingestion.ingest_voice(_config.data_dir, raw, patient_id, ttl=_config.raw_file_ttl)
    _agent.event_queue.push(item)
    _db.audit({"type": "voice_uploaded", "patient_id": patient_id[:8] + "...", "size": len(raw), "session_id": item.session_id})
    return {"status": "queued", "session_id": item.session_id, "raw_ttl_seconds": _config.raw_file_ttl}


@app.post("/symptom")
async def submit_symptom(
    patient_id: str = Form(...),
    text: str = Form(...),
):
    """Submit text symptom log. Processed by AkashML SOAP note generation."""
    if not text.strip():
        raise HTTPException(400, "Text cannot be empty")
    item = ingestion.ingest_text(text, patient_id)
    _agent.event_queue.push(item)
    _db.audit({"type": "symptom_submitted", "patient_id": patient_id[:8] + "...", "chars": len(text), "session_id": item.session_id})
    return {"status": "queued", "session_id": item.session_id}


@app.post("/vital")
async def record_vital(
    patient_id: str = Form(...),
    metric_type: str = Form(...),
    value: float = Form(...),
    unit: str = Form(""),
):
    """Record a vital sign. Checked against rule engine immediately."""
    vid = _db.record_vital(patient_id, metric_type, value, unit=unit, source="api")

    # Also queue for agent processing (rule check + potential alert)
    item = ingestion.ingest_vital(patient_id, metric_type, value, unit)
    _agent.event_queue.push(item)

    _db.audit({"type": "vital_recorded", "patient_id": patient_id[:8] + "...", "metric": metric_type, "value": value})
    return {"status": "recorded", "vital_id": vid, "metric": metric_type, "value": value}


# ── Read Endpoints ────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    """Agent status, uptime, stats, Venice endpoints used."""
    if not _agent:
        return {"status": "initializing"}
    return JSONResponse(_agent.get_status())


@app.get("/patients")
def list_patients():
    """List all patients (IDs and creation dates only)."""
    return JSONResponse(_db.list_patients())


@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    """Patient detail with latest vitals."""
    patient = _db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    latest = _db.get_latest_vitals(patient_id)
    history = _db.get_vitals(patient_id, days=7)
    return JSONResponse({"patient": patient, "latest_vitals": latest, "vitals_history": history})


@app.get("/patient/{patient_id}/vitals")
def get_vitals(patient_id: str, days: int = 7):
    """Get vitals history for a patient."""
    return JSONResponse(_db.get_vitals(patient_id, days=days))


@app.get("/logs")
def get_logs(patient_id: str = None, limit: int = 50):
    """Analysis logs. Decrypted summaries returned."""
    if patient_id:
        return JSONResponse(_db.get_logs(patient_id, limit=limit))
    # All patients
    patients = _db.list_patients()
    all_logs = []
    for p in patients:
        all_logs.extend(_db.get_logs(p["id"], limit=limit))
    all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return JSONResponse(all_logs[:limit])


@app.get("/alerts")
def get_alerts(patient_id: str = None, limit: int = 50):
    """Alert history with delivery receipts."""
    return JSONResponse(_db.get_alerts(patient_id, limit=limit))


@app.get("/audit")
def get_audit(limit: int = 100):
    """Immutable audit trail — verifiable action receipts."""
    return JSONResponse(_db.get_audit_log(limit=limit))


@app.get("/health")
def health():
    """Health check for Akash deployment."""
    return {"status": "healthy", "uptime": round(time.time() - _agent.start_time, 1) if _agent else 0}


# ── Doctor Report — AI Clinical Analysis ──────────────────────────────

@app.get("/doctor-report/{patient_id}")
def doctor_report(patient_id: str):
    """Generate AI-powered doctor report: risk assessment, treatment plan, drug review, follow-up."""
    patient = _db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    context = _agent.memory.load_context(patient_id, days=14)
    context_text = _agent.memory.format_for_ai(context)
    report = inference.akashml_doctor_report(
        _agent.akashml, _config.akashml.primary_model, context_text
    )
    _agent.stats["akashml_calls"] += 1
    _db.audit({"type": "doctor_report_generated", "patient_id": patient_id[:8] + "..."})
    return JSONResponse({"patient": patient, "report": report})


# ── Patient Audio Briefing — Venice TTS ───────────────────────────────

@app.get("/patient-briefing/{patient_id}")
def patient_briefing(patient_id: str):
    """Generate patient-friendly spoken health briefing via AkashML + Venice TTS."""
    patient = _db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    context = _agent.memory.load_context(patient_id)
    context_text = _agent.memory.format_for_ai(context)
    briefing = inference.akashml_patient_briefing(
        _agent.akashml, _config.akashml.primary_model,
        patient["name"], context_text
    )
    _agent.stats["akashml_calls"] += 1
    # Generate TTS audio from the briefing text
    audio_b64 = None
    spoken = briefing.get("spoken_text", "")
    if spoken:
        audio_bytes = inference.venice_tts(_config, spoken)
        if audio_bytes:
            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            _agent.venice_endpoints_used.add("audio/speech")
            _agent.stats["venice_calls"] += 1
    _db.audit({"type": "patient_briefing_generated", "patient_id": patient_id[:8] + "...", "tts": audio_b64 is not None})
    return JSONResponse({"patient": patient, "briefing": briefing, "audio_b64": audio_b64})


# ── Wound Timeline — Structured Vision History ────────────────────────

@app.get("/wound-timeline/{patient_id}")
def wound_timeline(patient_id: str):
    """Get wound healing progression from stored vision analysis logs."""
    patient = _db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    logs = _db.get_logs(patient_id, limit=50)
    timeline = []
    for log in reversed(logs):
        if log.get("input_type") == "photo":
            summary = log.get("summary", "")
            timeline.append({
                "timestamp": log["timestamp"],
                "session_id": log["session_id"],
                "analysis": summary,
                "decision": log["decision"],
                "anomaly_score": log.get("anomaly_score", 0),
            })
    return JSONResponse({"patient": patient, "timeline": timeline, "total_photos": len(timeline)})


# ── Dashboard ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HealthGuard — Decentralized Private AI Health Agent</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:#0a0e17;color:#c8d6e5;min-height:100vh}

/* Header */
.header{display:flex;justify-content:space-between;align-items:center;padding:12px 24px;border-bottom:1px solid #1a2233;background:#0a0e17}
.logo h1{font-size:18px;font-weight:800;color:#e0f7fa;letter-spacing:-.5px}
.logo h1 em{color:#00e5ff;font-style:normal}
.nav{display:flex;gap:2px}
.nav-btn{padding:8px 18px;border:none;background:transparent;color:#546e7a;font-size:12px;font-weight:600;cursor:pointer;border-radius:6px 6px 0 0;transition:.2s;font-family:inherit;text-transform:uppercase;letter-spacing:1px}
.nav-btn.active{background:#111927;color:#00e5ff;border:1px solid #1a2d42;border-bottom:1px solid #111927}
.nav-btn:hover{color:#80cbc4}
.privacy-bar{display:flex;gap:6px;align-items:center;font-size:9px;color:#37474f}
.privacy-dot{width:6px;height:6px;border-radius:50%;background:#00e676;display:inline-block}
.live-tag{background:#00e676;color:#0a0e17;font-size:8px;font-weight:700;padding:1px 6px;border-radius:3px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

/* Status bar */
.status-bar{display:flex;gap:8px;padding:10px 24px;background:#0d1219;border-bottom:1px solid #1a2233;font-size:11px;color:#546e7a;align-items:center;flex-wrap:wrap}
.status-bar .tag{padding:2px 8px;border-radius:4px;font-weight:600;font-size:10px}
.status-bar .running{background:#00e67622;color:#00e676}

/* Tab content */
.tab-content{display:none;padding:20px 24px}
.tab-content.active{display:block}

/* Cards */
.card{background:#111927;border:1px solid #1a2d42;border-radius:10px;padding:18px;margin-bottom:14px}
.card-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#37474f;margin-bottom:12px}
.card h3{font-size:14px;font-weight:700;color:#e0f7fa;margin-bottom:8px}

/* Stats grid */
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.stat-card{background:#111927;border:1px solid #1a2d42;border-radius:8px;padding:14px;text-align:center}
.stat-card .label{font-size:9px;color:#546e7a;text-transform:uppercase;letter-spacing:1px}
.stat-card .value{font-size:28px;font-weight:800;color:#e0f7fa;margin:4px 0}
.stat-card .sub{font-size:10px;color:#37474f}

/* Vital cards */
.vitals-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.vital-card{background:#111927;border:1px solid #1a2d42;border-radius:8px;padding:14px}
.vital-card .metric{font-size:9px;color:#546e7a;text-transform:uppercase;letter-spacing:1px}
.vital-card .reading{font-size:22px;font-weight:800;color:#e0f7fa;margin:4px 0}
.vital-card .unit{font-size:11px;color:#455a64}
.vital-card .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:8px;font-weight:700;text-transform:uppercase}
.badge-normal{background:#00e67622;color:#00e676}
.badge-warning{background:#ffd74033;color:#ffd740}
.badge-critical{background:#ff525233;color:#ff5252}

/* Grids */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}

/* Alerts */
.alert-item{display:flex;gap:10px;padding:10px 14px;border-radius:6px;margin-bottom:6px;border:1px solid #1a2d42;background:#0d1219}
.alert-item.sev1{border-left:3px solid #ff5252}
.alert-item.sev2{border-left:3px solid #ffd740}
.alert-item.sev3{border-left:3px solid #69f0ae}
.alert-body{flex:1}
.alert-msg{font-size:12px;color:#b0bec5}
.alert-meta{font-size:10px;color:#546e7a;margin-top:3px}
.alert-tag{font-size:8px;padding:2px 6px;border-radius:3px;font-weight:600;margin-right:4px}
.tag-tg{background:#00e5ff22;color:#00e5ff}
.tag-tts{background:#76ff0322;color:#76ff03}
.tag-doc{background:#e040fb22;color:#e040fb}

/* Venice chips */
.ep-grid{display:flex;flex-wrap:wrap;gap:8px}
.ep-chip{display:flex;align-items:center;gap:6px;padding:8px 14px;border-radius:6px;font-size:11px;font-weight:600;background:#0d1219;border:1px solid #1a2d42;transition:.3s}
.ep-chip.active{border-color:var(--c);box-shadow:0 0 12px var(--c,#00e5ff)22}
.ep-chip .icon{font-size:14px}
.ep-chip .name{color:#546e7a;font-size:9px;font-family:monospace}
.ep-chip .label{color:#78909c}
.ep-chip.active .label{color:var(--c,#00e5ff)}

/* Audit */
.audit-entry{padding:5px 10px;border-bottom:1px solid #111927;font-family:monospace;font-size:10px;color:#546e7a}
.audit-entry .type{color:#00e5ff;font-weight:600}

/* Table */
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:8px 10px;font-size:9px;color:#37474f;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #1a2d42}
td{padding:8px 10px;font-size:11px;border-bottom:1px solid #111927}
tr:hover{background:#0d1219}
.sev-badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;text-transform:uppercase}
.s1{background:#b71c1c44;color:#ff5252}
.s2{background:#f9a82533;color:#ffd740}
.s3{background:#2e7d3233;color:#69f0ae}

/* Doctor Tab */
.report-section{margin-bottom:16px}
.report-section h4{font-size:11px;color:#00e5ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #1a2233}
.risk-bar{height:8px;border-radius:4px;background:#1a2d42;overflow:hidden;margin:8px 0}
.risk-fill{height:100%;border-radius:4px;transition:.5s}
.rec-item{padding:10px 14px;background:#0d1219;border-radius:6px;margin-bottom:6px;border-left:3px solid #00e5ff}
.rec-item .priority{font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#ffd740;margin-bottom:2px}
.rec-item .text{font-size:12px;color:#b0bec5}
.rec-item .rationale{font-size:10px;color:#546e7a;margin-top:2px}
.diag-tag{display:inline-block;padding:3px 10px;background:#1a2d4266;border:1px solid #1a2d42;border-radius:4px;font-size:11px;color:#80cbc4;margin:3px 4px 3px 0}

/* Audio player */
.audio-card{background:linear-gradient(135deg,#0d2137,#1b2838);border:1px solid #00e5ff33;border-radius:10px;padding:20px;text-align:center}
.audio-card h3{color:#00e5ff;margin-bottom:8px}
.audio-card p{font-size:12px;color:#80cbc4;margin-bottom:14px}
.audio-card audio{width:100%;margin-top:8px}

/* Upload */
.upload-zone{border:2px dashed #1a2d42;border-radius:10px;padding:40px;text-align:center;cursor:pointer;transition:.3s}
.upload-zone:hover{border-color:#00e5ff;background:#00e5ff08}
.upload-zone .icon{font-size:36px;margin-bottom:10px}
.upload-zone p{font-size:12px;color:#546e7a}
.form-group{margin-bottom:12px}
.form-group label{display:block;font-size:10px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.form-group input,.form-group textarea,.form-group select{width:100%;padding:10px 14px;background:#0d1219;border:1px solid #1a2d42;border-radius:6px;color:#e0f7fa;font-size:12px;font-family:inherit;outline:none;transition:.2s}
.form-group input:focus,.form-group textarea:focus{border-color:#00e5ff}
.btn{padding:10px 20px;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;transition:.2s}
.btn-primary{background:#00e5ff;color:#0a0e17}
.btn-primary:hover{background:#00b8d4}
.btn-secondary{background:#1a2d42;color:#80cbc4}
.btn-secondary:hover{background:#263d50}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* Wound timeline */
.timeline{position:relative;padding-left:24px}
.timeline::before{content:'';position:absolute;left:8px;top:0;bottom:0;width:2px;background:#1a2d42}
.timeline-item{position:relative;margin-bottom:16px}
.timeline-item::before{content:'';position:absolute;left:-20px;top:6px;width:10px;height:10px;border-radius:50%;background:#00e5ff;border:2px solid #0a0e17}
.timeline-item.worsening::before{background:#ff5252}
.timeline-item.improving::before{background:#00e676}

/* Patient selector */
.patient-select{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.patient-chip{padding:8px 16px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid #1a2d42;background:#111927;transition:.2s}
.patient-chip.active{border-color:#00e5ff;color:#00e5ff;background:#00e5ff11}
.patient-chip:hover{border-color:#00e5ff55}

.footer{text-align:center;padding:14px;font-size:10px;color:#1a2233;border-top:1px solid #111927;margin-top:16px}
.scrollbox{max-height:300px;overflow-y:auto}
.loading{color:#546e7a;font-size:12px;padding:20px;text-align:center}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="logo"><h1><em>HEALTH</em>GUARD</h1><span style="font-size:9px;color:#37474f;margin-left:8px">PRIVATE AI HEALTH AGENT</span></div>
  <div class="nav">
    <button class="nav-btn active" onclick="showTab('dashboard')">Dashboard</button>
    <button class="nav-btn" onclick="showTab('upload')">Upload</button>
    <button class="nav-btn" onclick="showTab('history')">History</button>
    <button class="nav-btn" onclick="showTab('doctor')">Doctor</button>
  </div>
  <div class="privacy-bar">
    <span class="privacy-dot"></span> PRIVACY ACTIVE
    <span style="margin:0 4px;color:#1a2233">|</span> AKASH <span style="margin:0 4px;color:#1a2233">|</span> VENICE <span style="margin:0 4px;color:#1a2233">|</span> E2E
    <span style="margin:0 4px;color:#1a2233">|</span> Patient: <strong id="hdrPatient" style="color:#80cbc4">—</strong>
  </div>
</div>

<!-- STATUS BAR -->
<div class="status-bar">
  <span class="live-tag">RUNNING</span>
  Loop: <strong id="sbLoop">0</strong>
  <span style="margin:0 6px;color:#1a2233">|</span>
  Uptime: <strong id="sbUptime">0s</strong>
  <span style="margin:0 6px;color:#1a2233">|</span>
  Akash Node: <strong id="sbNode" style="color:#80cbc4">—</strong>
  <span style="margin:0 6px;color:#1a2233">|</span>
  Storage: <strong style="color:#80cbc4">/data</strong>
</div>

<!-- ═══════════════════ DASHBOARD TAB ═══════════════════ -->
<div class="tab-content active" id="tab-dashboard">

  <!-- Vitals Row -->
  <div class="vitals-row" id="vitalsRow">
    <div class="vital-card"><div class="metric">Blood Pressure</div><div class="reading" id="vBP">—</div><div class="unit">mmHg</div></div>
    <div class="vital-card"><div class="metric">Glucose</div><div class="reading" id="vGlucose">—</div><div class="unit">mg/dL</div></div>
    <div class="vital-card"><div class="metric">Pain Level</div><div class="reading" id="vPain">—</div><div class="unit">/10</div></div>
    <div class="vital-card"><div class="metric">Medication Adherence</div><div class="reading" id="vMeds">N/A</div></div>
  </div>

  <div class="three-col">
    <!-- Left: Multimodal Input -->
    <div>
      <div class="card">
        <div class="card-title">Multimodal Input Panel</div>
        <div class="upload-zone" id="quickUpload" onclick="showTab('upload')">
          <div class="icon">📷</div>
          <p>Drop wound photo or click to upload</p>
        </div>
        <div style="margin-top:12px">
          <button class="btn btn-secondary" style="width:100%;margin-bottom:6px" onclick="showTab('upload')">Upload Photo / Voice</button>
          <button class="btn btn-primary" style="width:100%" onclick="generateBriefing()">🔊 Play Health Briefing</button>
        </div>
      </div>
      <div class="card" id="briefingCard" style="display:none">
        <div class="card-title">Patient Audio Briefing</div>
        <div class="audio-card">
          <h3>🎧 Your Health Update</h3>
          <p id="briefingText"></p>
          <audio id="briefingAudio" controls></audio>
        </div>
      </div>
    </div>

    <!-- Middle: Live Feed -->
    <div>
      <div class="card">
        <div class="card-title">Agent Live Feed</div>
        <div class="scrollbox" id="liveFeed" style="font-family:monospace;font-size:10px"></div>
      </div>
    </div>

    <!-- Right: Venice Pipeline -->
    <div>
      <div class="card">
        <div class="card-title">Venice AI Endpoints</div>
        <div class="ep-grid" id="epGrid"></div>
      </div>
      <div class="card">
        <div class="card-title">Stats</div>
        <div class="stats-grid" style="grid-template-columns:1fr 1fr">
          <div class="stat-card"><div class="label">Venice</div><div class="value" id="sVenice" style="color:#00e5ff;font-size:20px">0</div><div class="sub">zero retention</div></div>
          <div class="stat-card"><div class="label">AkashML</div><div class="value" id="sAkash" style="color:#76ff03;font-size:20px">0</div><div class="sub">structured only</div></div>
          <div class="stat-card"><div class="label">Alerts</div><div class="value" id="sAlerts" style="color:#ff5252;font-size:20px">0</div><div class="sub">verifiable</div></div>
          <div class="stat-card"><div class="label">Vitals</div><div class="value" id="sVitals" style="font-size:20px">0</div><div class="sub">recorded</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Alerts + Audit -->
  <div class="two-col">
    <div class="card">
      <div class="card-title">Autonomous Actions Log</div>
      <div class="scrollbox" id="alertsList"></div>
    </div>
    <div class="card">
      <div class="card-title">Inference Pipeline</div>
      <div style="font-size:12px;line-height:2.2">
        <div>🎤 <strong style="color:#00e5ff">Venice STT</strong> <span style="color:#546e7a">Whisper Large V3 — Audio deleted immediately</span></div>
        <div>👁 <strong style="color:#ff5252">Venice Vision</strong> <span style="color:#546e7a">Qwen2.5-VL 235B — Image deleted immediately</span></div>
        <div>🟡 <strong style="color:#ffd740">AkashML</strong> <span style="color:#546e7a">DeepSeek-V3.1 — Structured text only</span></div>
        <div>🔊 <strong style="color:#76ff03">Venice TTS</strong> <span style="color:#546e7a">Kokoro — Clean text to audio</span></div>
        <div>🖼 <strong style="color:#e040fb">Venice ImgGen</strong> <span style="color:#546e7a">Flux — Visual health reports</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════ UPLOAD TAB ═══════════════════ -->
<div class="tab-content" id="tab-upload">
  <div class="patient-select" id="patientChips"></div>
  <div class="two-col">
    <div class="card">
      <h3>📷 Upload Health Photo</h3>
      <p style="font-size:11px;color:#546e7a;margin-bottom:14px">EXIF stripped. Analyzed by Venice Vision. Raw deleted in 60s.</p>
      <form id="photoForm" onsubmit="return uploadPhoto(event)">
        <div class="form-group"><label>Photo</label><input type="file" id="photoFile" accept="image/*" required></div>
        <div class="form-group"><label>Description</label><input type="text" id="photoDesc" placeholder="e.g. surgical wound day 5, pain increasing"></div>
        <button type="submit" class="btn btn-primary" id="photoBtn">Analyze Photo</button>
      </form>
      <div id="photoResult" style="margin-top:14px"></div>
    </div>
    <div class="card">
      <h3>🎤 Voice Note</h3>
      <p style="font-size:11px;color:#546e7a;margin-bottom:14px">Transcribed by Venice STT. Audio deleted immediately.</p>
      <form id="voiceForm" onsubmit="return uploadVoice(event)">
        <div class="form-group"><label>Voice Recording</label><input type="file" id="voiceFile" accept="audio/*" required></div>
        <button type="submit" class="btn btn-primary" id="voiceBtn">Transcribe & Analyze</button>
      </form>
      <div id="voiceResult" style="margin-top:14px"></div>
    </div>
  </div>
  <div class="two-col" style="margin-top:14px">
    <div class="card">
      <h3>📝 Symptom Log</h3>
      <form id="symptomForm" onsubmit="return submitSymptom(event)">
        <div class="form-group"><label>Describe your symptoms</label><textarea id="symptomText" rows="3" placeholder="I woke up with a splitting headache and my vision is blurry..." required></textarea></div>
        <button type="submit" class="btn btn-primary" id="symptomBtn">Submit Symptoms</button>
      </form>
      <div id="symptomResult" style="margin-top:14px"></div>
    </div>
    <div class="card">
      <h3>💓 Record Vital Sign</h3>
      <form id="vitalForm" onsubmit="return submitVital(event)">
        <div class="form-group"><label>Metric</label>
          <select id="vitalMetric"><option value="bp_systolic">BP Systolic (mmHg)</option><option value="bp_diastolic">BP Diastolic (mmHg)</option><option value="glucose">Glucose (mg/dL)</option><option value="heart_rate">Heart Rate (bpm)</option><option value="oxygen_saturation">SpO2 (%)</option><option value="temperature">Temperature (F)</option><option value="pain_level">Pain Level (0-10)</option></select>
        </div>
        <div class="form-group"><label>Value</label><input type="number" id="vitalValue" step="0.1" required></div>
        <button type="submit" class="btn btn-primary" id="vitalBtn">Record Vital</button>
      </form>
      <div id="vitalResult" style="margin-top:14px"></div>
    </div>
  </div>
</div>

<!-- ═══════════════════ HISTORY TAB ═══════════════════ -->
<div class="tab-content" id="tab-history">
  <div class="patient-select" id="historyPatientChips"></div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Analysis Logs</div>
      <div class="scrollbox"><table><thead><tr><th>Time</th><th>Input</th><th>Decision</th><th>Reason</th><th>Anomaly</th></tr></thead><tbody id="histLogs"></tbody></table></div>
    </div>
    <div class="card">
      <div class="card-title">Alert History</div>
      <div class="scrollbox" id="histAlerts"></div>
    </div>
  </div>
  <div class="card" style="margin-top:14px">
    <div class="card-title">Wound Healing Timeline</div>
    <div id="woundTimeline" class="timeline"><p class="loading">Upload photos to build a wound healing timeline.</p></div>
  </div>
  <div class="card" style="margin-top:14px">
    <div class="card-title">Audit Trail — Verifiable Receipts</div>
    <div class="scrollbox" id="histAudit" style="max-height:200px"></div>
  </div>
</div>

<!-- ═══════════════════ DOCTOR TAB ═══════════════════ -->
<div class="tab-content" id="tab-doctor">
  <div class="patient-select" id="doctorPatientChips"></div>
  <div id="doctorContent">
    <div style="text-align:center;padding:40px">
      <button class="btn btn-primary" style="font-size:14px;padding:14px 28px" onclick="generateDoctorReport()">🩺 Generate AI Clinical Report</button>
      <p style="font-size:11px;color:#546e7a;margin-top:10px">AkashML analyzes all patient data and generates treatment recommendations</p>
    </div>
  </div>
  <div id="doctorReport" style="display:none">
    <div class="two-col">
      <div>
        <!-- Risk Assessment -->
        <div class="card">
          <div class="card-title">Risk Assessment</div>
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">
            <div id="drRiskLevel" style="font-size:22px;font-weight:800"></div>
            <div style="flex:1"><div class="risk-bar"><div class="risk-fill" id="drRiskBar"></div></div></div>
          </div>
          <div id="drRiskFactors"></div>
        </div>
        <!-- Treatment Recommendations -->
        <div class="card">
          <div class="card-title">Treatment Recommendations</div>
          <div id="drTreatment"></div>
        </div>
        <!-- Differential Diagnosis -->
        <div class="card">
          <div class="card-title">Differential Diagnosis</div>
          <div id="drDiagnosis"></div>
        </div>
      </div>
      <div>
        <!-- Clinical Summary -->
        <div class="card">
          <div class="card-title">Clinical Summary</div>
          <p id="drSummary" style="font-size:13px;color:#b0bec5;line-height:1.7"></p>
        </div>
        <!-- Medication Review -->
        <div class="card">
          <div class="card-title">Medication Review</div>
          <div id="drMeds"></div>
        </div>
        <!-- Follow-up Plan -->
        <div class="card">
          <div class="card-title">Follow-Up Plan</div>
          <div id="drFollowup"></div>
        </div>
        <!-- Patient Education -->
        <div class="card">
          <div class="card-title">Patient Education</div>
          <div id="drEducation"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="footer">HealthGuard v1.0 · Built on Akash · Venice AI · Zero data retention</div>

<script>
// ═══ State ═══
let currentPatient = null;
let patients = [];
const EP_META = {
  "audio/transcriptions":{icon:"🎤",label:"STT (Whisper)",color:"#00e5ff"},
  "vision":{icon:"👁",label:"Vision (Qwen2.5-VL)",color:"#ff5252"},
  "audio/speech":{icon:"🔊",label:"TTS (Kokoro)",color:"#76ff03"},
  "images/generations":{icon:"🖼",label:"ImgGen (Flux)",color:"#e040fb"},
  "chat/completions":{icon:"💬",label:"Chat",color:"#ffab40"},
};

// ═══ Tab Navigation ═══
function showTab(name){
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  event?.target?.classList?.add('active') || document.querySelectorAll('.nav-btn').forEach(b=>{if(b.textContent.toLowerCase()===name)b.classList.add('active')});
  if(name==='history') loadHistory();
}

// ═══ Patient Selection ═══
function selectPatient(pid){
  currentPatient = pid;
  document.querySelectorAll('.patient-chip').forEach(c=>{
    c.classList.toggle('active', c.dataset.pid===pid);
  });
  const p = patients.find(x=>x.id===pid);
  document.getElementById('hdrPatient').textContent = p?.name||pid.substring(0,16);
  loadPatientVitals(pid);
  loadHistory();
}

function renderPatientChips(){
  ['patientChips','historyPatientChips','doctorPatientChips'].forEach(id=>{
    const el=document.getElementById(id);
    if(!el)return;
    el.innerHTML='';
    patients.forEach(p=>{
      const c=document.createElement('div');
      c.className='patient-chip'+(p.id===currentPatient?' active':'');
      c.dataset.pid=p.id;
      c.textContent=p.name||p.id.substring(0,16);
      c.onclick=()=>selectPatient(p.id);
      el.appendChild(c);
    });
  });
}

// ═══ Load Patient Vitals ═══
async function loadPatientVitals(pid){
  try{
    const data = await fetch('/patient/'+pid).then(r=>r.json());
    const lv = data.latest_vitals||{};
    const bp_s = lv.bp_systolic?.value, bp_d = lv.bp_diastolic?.value;
    document.getElementById('vBP').textContent = bp_s&&bp_d ? `${bp_s}/${bp_d}` : (bp_s||'—');
    document.getElementById('vGlucose').textContent = lv.glucose?.value ?? '—';
    document.getElementById('vPain').textContent = lv.pain_level?.value ?? '—';
    // Badges
    const bpCard = document.getElementById('vBP').parentElement;
    const gCard = document.getElementById('vGlucose').parentElement;
    const pCard = document.getElementById('vPain').parentElement;
    bpCard.querySelectorAll('.badge').forEach(b=>b.remove());
    gCard.querySelectorAll('.badge').forEach(b=>b.remove());
    pCard.querySelectorAll('.badge').forEach(b=>b.remove());
    if(bp_s>=180) bpCard.innerHTML+='<span class="badge badge-critical">CRITICAL</span>';
    else if(bp_s>=140) bpCard.innerHTML+='<span class="badge badge-warning">WATCH</span>';
    else if(bp_s) bpCard.innerHTML+='<span class="badge badge-normal">NORMAL</span>';
    if(lv.glucose?.value<=70) gCard.innerHTML+='<span class="badge badge-critical">LOW</span>';
    else if(lv.glucose?.value) gCard.innerHTML+='<span class="badge badge-normal">NORMAL</span>';
    if(lv.pain_level?.value>=8) pCard.innerHTML+='<span class="badge badge-critical">ALERT</span>';
    else if(lv.pain_level?.value>=5) pCard.innerHTML+='<span class="badge badge-warning">WATCH</span>';
    else if(lv.pain_level?.value) pCard.innerHTML+='<span class="badge badge-normal">OK</span>';
    // Med adherence from logs
    const logs = data.vitals_history?.filter(v=>v.source==='text')||[];
    const medCount = logs.length;
    document.getElementById('vMeds').textContent = medCount>0?`${Math.min(medCount,7)}/7`:'N/A';
  }catch(e){console.error('vitals',e)}
}

// ═══ Dashboard Refresh ═══
async function refresh(){
  try{
    const [st,als,aud] = await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/alerts?limit=15').then(r=>r.json()),
      fetch('/audit?limit=20').then(r=>r.json()),
    ]);
    document.getElementById('sbLoop').textContent=st.loop_count||0;
    document.getElementById('sbUptime').textContent=st.uptime_seconds?Math.round(st.uptime_seconds)+'s':'0s';
    document.getElementById('sbNode').textContent=location.hostname.includes('akash')?location.hostname:'provider-local';
    document.getElementById('sVenice').textContent=st.venice_calls||0;
    document.getElementById('sAkash').textContent=st.akashml_calls||0;
    document.getElementById('sAlerts').textContent=st.db_stats?.total_alerts||0;
    document.getElementById('sVitals').textContent=st.db_stats?.vitals_recorded||0;

    // Endpoints
    const active=new Set(st.venice_endpoints_used||[]);
    const grid=document.getElementById('epGrid');
    grid.innerHTML='';
    for(const[ep,m]of Object.entries(EP_META)){
      const on=active.has(ep);
      const d=document.createElement('div');
      d.className='ep-chip'+(on?' active':'');
      d.style.setProperty('--c',m.color);
      d.innerHTML=`<span class="icon">${m.icon}</span><div><div class="label">${m.label}</div><div class="name">${ep}</div></div>`;
      grid.appendChild(d);
    }

    // Alerts
    const al=document.getElementById('alertsList');
    al.innerHTML='';
    for(const a of als){
      const sev=a.severity===1?'sev1':a.severity===2?'sev2':'sev3';
      const icon=a.severity===1?'🚨':a.severity===2?'⚠️':'ℹ️';
      const acts=(a.action_taken||'').split(',').map(s=>s.trim());
      const tags=acts.map(x=>{
        if(x.includes('telegram'))return'<span class="alert-tag tag-tg">📨 Telegram</span>';
        if(x.includes('tts'))return'<span class="alert-tag tag-tts">🔊 TTS</span>';
        if(x.includes('doctor'))return'<span class="alert-tag tag-doc">👨‍⚕️ Doctor</span>';
        return'';
      }).join('');
      al.innerHTML+=`<div class="alert-item ${sev}"><div>${icon}</div><div class="alert-body"><div class="alert-msg">${(a.message||'').substring(0,150)}</div><div class="alert-meta">${(a.timestamp||'').substring(11,19)} · ${(a.patient_id||'').substring(0,14)}</div><div>${tags}</div></div></div>`;
    }

    // Live feed from audit
    const feed=document.getElementById('liveFeed');
    feed.innerHTML='';
    for(const e of aud.reverse()){
      const ts=(e.timestamp||'').substring(11,19);
      const color=e.type?.includes('severity_1')?'#ff5252':e.type?.includes('severity_2')?'#ffd740':e.type?.includes('delivery')?'#e040fb':'#546e7a';
      feed.innerHTML+=`<div style="padding:3px 0;color:${color}">${ts} · ${e.type||''} ${e.reason?(': '+e.reason.substring(0,60)):''}</div>`;
    }

    // Load patients if not yet
    if(patients.length===0){
      const plist = await fetch('/patients').then(r=>r.json());
      // Get names
      for(const p of plist){
        const detail = await fetch('/patient/'+p.id).then(r=>r.json());
        patients.push({id:p.id, name:detail.patient?.name||p.id});
      }
      if(patients.length>0){
        currentPatient=patients[0].id;
        renderPatientChips();
        selectPatient(currentPatient);
      }
    }
  }catch(e){console.error('refresh',e)}
}

// ═══ History Tab ═══
async function loadHistory(){
  if(!currentPatient)return;
  try{
    const [logs,als,aud,wt] = await Promise.all([
      fetch('/logs?patient_id='+currentPatient+'&limit=20').then(r=>r.json()),
      fetch('/alerts?patient_id='+currentPatient+'&limit=15').then(r=>r.json()),
      fetch('/audit?limit=30').then(r=>r.json()),
      fetch('/wound-timeline/'+currentPatient).then(r=>r.json()),
    ]);
    // Logs table
    const lb=document.getElementById('histLogs');
    lb.innerHTML='';
    for(const l of logs){
      const sev=l.anomaly_score>=.7?'s1':l.anomaly_score>=.4?'s2':'s3';
      lb.innerHTML+=`<tr><td style="font-size:10px;color:#546e7a">${(l.timestamp||'').substring(11,19)}</td><td>${l.input_type||''}</td><td><span class="sev-badge ${sev}">${l.decision||''}</span></td><td style="max-width:250px;color:#b0bec5;font-size:11px">${(l.reason||'').substring(0,100)}</td><td style="color:${l.anomaly_score>=.7?'#ff5252':'#69f0ae'}">${(l.anomaly_score||0).toFixed(2)}</td></tr>`;
    }
    // Alert history
    const ha=document.getElementById('histAlerts');
    ha.innerHTML='';
    for(const a of als){
      const sev=a.severity===1?'sev1':a.severity===2?'sev2':'sev3';
      ha.innerHTML+=`<div class="alert-item ${sev}"><div class="alert-body"><div class="alert-msg">${(a.message||'').substring(0,120)}</div><div class="alert-meta">${(a.timestamp||'').substring(0,16)} · ${a.action_taken}</div></div></div>`;
    }
    // Audit
    const au=document.getElementById('histAudit');
    au.innerHTML='';
    for(const e of aud.reverse()){
      au.innerHTML+=`<div class="audit-entry"><span style="color:#546e7a">${(e.timestamp||'').substring(11,19)}</span> <span class="type">${e.type||''}</span> ${e.reason?e.reason.substring(0,70):''}</div>`;
    }
    // Wound timeline
    const wDiv=document.getElementById('woundTimeline');
    if(wt.timeline&&wt.timeline.length>0){
      wDiv.innerHTML='';
      for(const w of wt.timeline){
        const cls=w.analysis?.includes('worsening')?'worsening':w.analysis?.includes('heal')?'improving':'';
        wDiv.innerHTML+=`<div class="timeline-item ${cls}"><div style="font-size:10px;color:#546e7a">${(w.timestamp||'').substring(0,16)}</div><div style="font-size:12px;color:#b0bec5;margin-top:4px">${(w.analysis||'').substring(0,200)}</div><div style="margin-top:4px"><span class="sev-badge ${w.anomaly_score>=.5?'s1':'s3'}">${w.decision}</span></div></div>`;
      }
    }else{
      wDiv.innerHTML='<p class="loading">Upload wound photos over time to build a healing timeline. Structured analysis is stored — raw images are deleted.</p>';
    }
  }catch(e){console.error('history',e)}
}

// ═══ Upload Functions ═══
async function uploadPhoto(e){
  e.preventDefault();
  if(!currentPatient)return alert('Select a patient first');
  const btn=document.getElementById('photoBtn');
  btn.disabled=true; btn.textContent='Analyzing...';
  const fd=new FormData();
  fd.append('file',document.getElementById('photoFile').files[0]);
  fd.append('patient_id',currentPatient);
  try{
    const r=await fetch('/upload-photo',{method:'POST',body:fd}).then(r=>r.json());
    document.getElementById('photoResult').innerHTML=`<div class="card" style="border-color:#00e5ff33"><p style="color:#00e676;font-size:12px">✓ Photo queued. Session: ${r.session_id}. Raw deleted in ${r.raw_ttl_seconds}s.</p><p style="color:#546e7a;font-size:10px;margin-top:4px">🔒 Raw photo discarded · Only structured output retained</p></div>`;
  }catch(err){document.getElementById('photoResult').innerHTML=`<p style="color:#ff5252">${err}</p>`}
  btn.disabled=false; btn.textContent='Analyze Photo';
}

async function uploadVoice(e){
  e.preventDefault();
  if(!currentPatient)return alert('Select a patient first');
  const btn=document.getElementById('voiceBtn');
  btn.disabled=true; btn.textContent='Transcribing...';
  const fd=new FormData();
  fd.append('file',document.getElementById('voiceFile').files[0]);
  fd.append('patient_id',currentPatient);
  try{
    const r=await fetch('/voice-note',{method:'POST',body:fd}).then(r=>r.json());
    document.getElementById('voiceResult').innerHTML=`<div class="card" style="border-color:#00e5ff33"><p style="color:#00e676;font-size:12px">✓ Voice queued. Session: ${r.session_id}.</p><p style="color:#546e7a;font-size:10px;margin-top:4px">🔒 Raw audio discarded · Only structured output retained</p></div>`;
  }catch(err){document.getElementById('voiceResult').innerHTML=`<p style="color:#ff5252">${err}</p>`}
  btn.disabled=false; btn.textContent='Transcribe & Analyze';
}

async function submitSymptom(e){
  e.preventDefault();
  if(!currentPatient)return alert('Select a patient first');
  const btn=document.getElementById('symptomBtn');
  btn.disabled=true;
  const fd=new FormData();
  fd.append('patient_id',currentPatient);
  fd.append('text',document.getElementById('symptomText').value);
  try{
    const r=await fetch('/symptom',{method:'POST',body:fd}).then(r=>r.json());
    document.getElementById('symptomResult').innerHTML=`<div class="card" style="border-color:#00e5ff33"><p style="color:#00e676;font-size:12px">✓ Symptoms queued. Session: ${r.session_id}</p></div>`;
    document.getElementById('symptomText').value='';
  }catch(err){document.getElementById('symptomResult').innerHTML=`<p style="color:#ff5252">${err}</p>`}
  btn.disabled=false;
}

async function submitVital(e){
  e.preventDefault();
  if(!currentPatient)return alert('Select a patient first');
  const btn=document.getElementById('vitalBtn');
  btn.disabled=true;
  const units={bp_systolic:'mmHg',bp_diastolic:'mmHg',glucose:'mg/dL',heart_rate:'bpm',oxygen_saturation:'%',temperature:'F',pain_level:'/10'};
  const metric=document.getElementById('vitalMetric').value;
  const fd=new FormData();
  fd.append('patient_id',currentPatient);
  fd.append('metric_type',metric);
  fd.append('value',document.getElementById('vitalValue').value);
  fd.append('unit',units[metric]||'');
  try{
    const r=await fetch('/vital',{method:'POST',body:fd}).then(r=>r.json());
    document.getElementById('vitalResult').innerHTML=`<div class="card" style="border-color:#00e5ff33"><p style="color:#00e676;font-size:12px">✓ ${r.metric} = ${r.value} recorded</p></div>`;
    loadPatientVitals(currentPatient);
  }catch(err){document.getElementById('vitalResult').innerHTML=`<p style="color:#ff5252">${err}</p>`}
  btn.disabled=false;
}

// ═══ Doctor Report ═══
async function generateDoctorReport(){
  if(!currentPatient)return alert('Select a patient first');
  document.getElementById('doctorContent').innerHTML='<div class="loading">🩺 Generating AI clinical report... AkashML is analyzing all patient data...</div>';
  try{
    const data=await fetch('/doctor-report/'+currentPatient).then(r=>r.json());
    const rpt=data.report;
    document.getElementById('doctorReport').style.display='block';
    document.getElementById('doctorContent').style.display='none';

    // Risk
    const risk=rpt.risk_assessment||{};
    const riskColors={low:'#00e676',moderate:'#ffd740',high:'#ff9100',critical:'#ff5252'};
    const riskPct={low:20,moderate:50,high:75,critical:95};
    const rl=risk.overall_risk||'moderate';
    document.getElementById('drRiskLevel').textContent=rl.toUpperCase();
    document.getElementById('drRiskLevel').style.color=riskColors[rl]||'#ffd740';
    const bar=document.getElementById('drRiskBar');
    bar.style.width=(riskPct[rl]||50)+'%';
    bar.style.background=riskColors[rl]||'#ffd740';
    document.getElementById('drRiskFactors').innerHTML=(risk.risk_factors||[]).map(f=>`<div style="padding:4px 0;font-size:11px;color:#b0bec5">• ${f}</div>`).join('');

    // Treatment
    document.getElementById('drTreatment').innerHTML=(rpt.treatment_recommendations||[]).map((t,i)=>
      `<div class="rec-item"><div class="priority">Priority ${t.priority||i+1} · ${t.urgency||'routine'}</div><div class="text">${t.recommendation||''}</div><div class="rationale">${t.rationale||''}</div></div>`
    ).join('');

    // Diagnosis
    document.getElementById('drDiagnosis').innerHTML=(rpt.differential_diagnosis||[]).map(d=>`<span class="diag-tag">${d}</span>`).join('');

    // Summary
    document.getElementById('drSummary').textContent=rpt.clinical_summary||'No summary available.';

    // Meds
    const meds=rpt.medication_review||{};
    let medsHtml='';
    if(meds.current_concerns?.length) medsHtml+=`<div style="margin-bottom:8px"><strong style="color:#ff5252;font-size:10px">⚠ CONCERNS</strong>${meds.current_concerns.map(c=>`<div style="font-size:11px;color:#b0bec5;padding:2px 0">• ${c}</div>`).join('')}</div>`;
    if(meds.interactions_flagged?.length) medsHtml+=`<div style="margin-bottom:8px"><strong style="color:#ffd740;font-size:10px">💊 INTERACTIONS</strong>${meds.interactions_flagged.map(c=>`<div style="font-size:11px;color:#b0bec5;padding:2px 0">• ${c}</div>`).join('')}</div>`;
    if(meds.dosage_notes) medsHtml+=`<div style="font-size:11px;color:#546e7a">${meds.dosage_notes}</div>`;
    document.getElementById('drMeds').innerHTML=medsHtml||'<p style="color:#546e7a;font-size:11px">No medication concerns identified.</p>';

    // Follow-up
    const fu=rpt.follow_up_plan||{};
    let fuHtml='';
    if(fu.next_visit) fuHtml+=`<div style="padding:4px 0;font-size:12px;color:#e0f7fa">📅 Next visit: <strong>${fu.next_visit}</strong></div>`;
    if(fu.monitoring_frequency) fuHtml+=`<div style="padding:4px 0;font-size:12px;color:#b0bec5">📊 Monitoring: ${fu.monitoring_frequency}</div>`;
    if(fu.tests_recommended?.length) fuHtml+=`<div style="padding:4px 0;font-size:12px;color:#b0bec5">🔬 Tests: ${fu.tests_recommended.join(', ')}</div>`;
    if(fu.specialist_referral) fuHtml+=`<div style="padding:4px 0;font-size:12px;color:#ffd740">👨‍⚕️ Referral: ${fu.specialist_referral}</div>`;
    document.getElementById('drFollowup').innerHTML=fuHtml||'<p style="color:#546e7a;font-size:11px">No specific follow-up needed.</p>';

    // Education
    document.getElementById('drEducation').innerHTML=(rpt.patient_education||[]).map(e=>`<div style="padding:4px 0;font-size:12px;color:#80cbc4">📖 ${e}</div>`).join('')||'<p style="color:#546e7a;font-size:11px">No specific education points.</p>';

  }catch(err){
    document.getElementById('doctorContent').innerHTML=`<p style="color:#ff5252;text-align:center;padding:20px">${err}</p>`;
  }
}

// ═══ Patient Briefing (TTS) ═══
async function generateBriefing(){
  if(!currentPatient)return alert('Select a patient first');
  const card=document.getElementById('briefingCard');
  card.style.display='block';
  document.getElementById('briefingText').textContent='Generating your health briefing...';
  try{
    const data=await fetch('/patient-briefing/'+currentPatient).then(r=>r.json());
    document.getElementById('briefingText').textContent=data.briefing?.spoken_text||'Briefing generated.';
    if(data.audio_b64){
      const audio=document.getElementById('briefingAudio');
      audio.src='data:audio/mp3;base64,'+data.audio_b64;
      audio.style.display='block';
    }
  }catch(err){document.getElementById('briefingText').textContent='Failed to generate briefing: '+err}
}

// ═══ Init ═══
refresh();
setInterval(refresh,12000);
</script>
</body>
</html>"""
