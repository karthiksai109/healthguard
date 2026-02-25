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
import os
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "static"))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    patient_note: str = Form(""),
):
    """Upload patient health photo. EXIF stripped, processed by Venice Vision, raw deleted in 60s."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    item = ingestion.ingest_photo(_config.data_dir, raw, patient_id, ttl=_config.raw_file_ttl)
    _agent.event_queue.push(item)
    _db.audit({
        "type": "photo_uploaded",
        "patient_id": patient_id[:8] + "...",
        "size": len(raw),
        "session_id": item.session_id,
        "note_chars": len(patient_note or ""),
    })
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


# ── Instant Photo Analysis — Real-time Venice Vision + AkashML Triage ──

@app.post("/analyze-photo")
async def analyze_photo(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    note: str = Form(""),
):
    """INSTANT image analysis. Venice Vision analyzes the actual image, then AkashML
    provides full clinical triage. Image is NEVER stored — deleted from memory immediately.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    clean_bytes = ingestion.strip_exif(raw)

    # Step 1: Venice Vision — analyze the actual image (zero retention)
    vision_result = inference.venice_vision(_config, _agent.venice, clean_bytes)
    _agent.venice_endpoints_used.add("vision")
    _agent.stats["venice_calls"] += 1

    # Step 2: Load patient context for AkashML
    patient_context = ""
    patient = _db.get_patient(patient_id)
    if patient:
        context = _agent.memory.load_context(patient_id)
        patient_context = _agent.memory.format_for_ai(context)

    # Step 3: AkashML Clinical Triage — full treatment plan
    triage = inference.akashml_clinical_triage(
        _agent.akashml, _config.akashml.primary_model,
        vision_result, patient_context, note,
    )
    _agent.stats["akashml_calls"] += 1

    # Step 4: Log (structured text only — no image stored)
    sev_map = {"green_self_care": "normal", "yellow_see_doctor": "monitor", "orange_urgent_care": "alert", "red_emergency": "escalate"}
    decision_str = sev_map.get(triage.get("emergency_level", ""), "monitor")
    _db.record_log(
        patient_id=patient_id, session_id=f"instant_{int(time.time())}",
        input_type="photo", summary=f"Vision: {vision_result.get('observations', '')[:300]}",
        decision=decision_str,
        reason=triage.get("emergency_explanation", vision_result.get("patient_summary", ""))[:300],
        action_taken="instant_analysis",
        model_used=_config.akashml.primary_model,
        anomaly_score=triage.get("diagnosis_assessment", {}).get("confidence", 0.5),
    )

    # Step 5: If serious, trigger doctor notification
    doctor_notified = False
    if triage.get("emergency_level") in ("orange_urgent_care", "red_emergency"):
        notify = triage.get("doctor_notification", {})
        alert_msg = f"URGENT: {notify.get('reason', 'Serious condition detected')} — {notify.get('key_findings', '')}"
        sev = 1 if triage["emergency_level"] == "red_emergency" else 2
        _db.record_alert(patient_id, sev, alert_msg, action_taken="doctor_notified,instant_analysis")
        doctor_notified = True

    _db.audit({
        "type": "instant_photo_analysis",
        "patient_id": patient_id[:8] + "...",
        "emergency_level": triage.get("emergency_level"),
        "doctor_notified": doctor_notified,
    })

    # Image bytes are now garbage collected — never stored to disk
    return JSONResponse({
        "vision": vision_result,
        "triage": triage,
        "doctor_notified": doctor_notified,
        "privacy": {
            "image_stored": False,
            "image_sent_to": "Venice AI (zero retention)",
            "structured_output_only": True,
        },
    })


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
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


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

.header{display:flex;justify-content:space-between;align-items:center;padding:16px 28px;border-bottom:1px solid #1a2233;background:linear-gradient(90deg,#0a0e17,#0f1923)}
.logo{display:flex;align-items:center;gap:10px}
.logo h1{font-size:20px;font-weight:800;color:#e0f7fa;letter-spacing:-.5px}
.logo h1 span{color:#00e5ff}
.live{background:#00e676;color:#0a0e17;font-size:9px;font-weight:700;padding:2px 8px;border-radius:3px;text-transform:uppercase;letter-spacing:1px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.header-sub{font-size:11px;color:#546e7a}

.stats{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;padding:16px 28px}
.stat{background:#111927;border:1px solid #1a2d42;border-radius:8px;padding:14px}
.stat-label{font-size:9px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.stat-value{font-size:24px;font-weight:800;color:#e0f7fa}
.stat-sub{font-size:10px;color:#455a64;margin-top:2px}

.banner{margin:0 28px 14px;padding:12px 18px;border-radius:8px;background:linear-gradient(90deg,#1b2838,#0d2137);border:1px solid #00e5ff33;display:flex;align-items:center;gap:10px}
.banner-icon{font-size:18px}
.banner-text{font-size:11px;color:#80cbc4}
.banner-text strong{color:#00e5ff}

.section{padding:0 28px 16px}
.section-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#37474f;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #1a2233}

.ep-grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.ep-chip{display:flex;align-items:center;gap:6px;padding:8px 14px;border-radius:6px;font-size:11px;font-weight:600;background:#111927;border:1px solid #1a2d42;transition:.3s}
.ep-chip.active{border-color:var(--c);box-shadow:0 0 10px var(--c,#00e5ff)33}
.ep-chip .icon{font-size:14px}
.ep-chip .name{color:#78909c;font-size:9px;font-family:monospace}
.ep-chip .label{color:#e0f7fa}
.ep-chip.active .label{color:var(--c)}

.alerts-list{max-height:280px;overflow-y:auto}
.alert-item{display:flex;gap:10px;padding:10px 14px;border-radius:6px;margin-bottom:6px;border:1px solid #1a2d42;background:#111927}
.alert-item.sev1{border-left:3px solid #ff5252}
.alert-item.sev2{border-left:3px solid #ffd740}
.alert-item.sev3{border-left:3px solid #69f0ae}
.alert-sev{font-size:16px}
.alert-body{flex:1}
.alert-msg{font-size:12px;color:#b0bec5}
.alert-meta{font-size:10px;color:#546e7a;margin-top:3px}
.alert-actions{display:flex;gap:4px;margin-top:4px}
.alert-tag{font-size:8px;padding:2px 6px;border-radius:3px;font-weight:600}
.alert-tag.tg{background:#00e5ff22;color:#00e5ff}
.alert-tag.tts{background:#76ff0322;color:#76ff03}
.alert-tag.doc{background:#e040fb22;color:#e040fb}

.audit-list{max-height:240px;overflow-y:auto;font-family:monospace;font-size:10px}
.audit-entry{padding:6px 10px;border-bottom:1px solid #111927;color:#78909c}
.audit-entry .ts{color:#546e7a}
.audit-entry .type{color:#00e5ff;font-weight:600}

table{width:100%;border-collapse:collapse}
th{text-align:left;padding:8px 10px;font-size:9px;color:#37474f;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #1a2d42}
td{padding:10px;font-size:11px;border-bottom:1px solid #111927;vertical-align:top}
tr:hover{background:#111927}
.sev-badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;text-transform:uppercase}
.sev-badge.s1{background:#b71c1c44;color:#ff5252}
.sev-badge.s2{background:#f9a82533;color:#ffd740}
.sev-badge.s3{background:#2e7d3233;color:#69f0ae}

.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}

.footer{text-align:center;padding:16px;font-size:10px;color:#263238;border-top:1px solid #1a2233;margin-top:16px}
</style>
</head>
<body>

<div class="header">
  <div class="logo"><h1><span>Health</span>Guard <span class="live">LIVE</span></h1></div>
  <div class="header-sub">decentralized private ai health agent · venice ai · akash network · zero retention</div>
</div>

<div class="stats" id="statsRow">
  <div class="stat"><div class="stat-label">Status</div><div class="stat-value" id="sStatus">—</div><div class="stat-sub" id="sUptime"></div></div>
  <div class="stat"><div class="stat-label">Patients</div><div class="stat-value" id="sPatients">0</div><div class="stat-sub">monitored</div></div>
  <div class="stat"><div class="stat-label">Vitals</div><div class="stat-value" id="sVitals">0</div><div class="stat-sub">recorded</div></div>
  <div class="stat"><div class="stat-label">Venice Calls</div><div class="stat-value" id="sVenice" style="color:#00e5ff">0</div><div class="stat-sub">zero retention</div></div>
  <div class="stat"><div class="stat-label">AkashML Calls</div><div class="stat-value" id="sAkash" style="color:#76ff03">0</div><div class="stat-sub">structured only</div></div>
  <div class="stat"><div class="stat-label">Alerts Fired</div><div class="stat-value" id="sAlerts" style="color:#ff5252">0</div><div class="stat-sub">verifiable</div></div>
  <div class="stat"><div class="stat-label">Audit Entries</div><div class="stat-value" id="sAudit">0</div><div class="stat-sub">immutable log</div></div>
</div>

<div class="banner">
  <div class="banner-icon">🔐</div>
  <div class="banner-text"><strong>Zero-Knowledge Architecture</strong> — Raw photos &amp; audio deleted within 60s. Venice = zero retention. AkashML receives structured text only. SQLite encrypted AES-256-GCM. Akash provider sees encrypted bytes. Even a full server breach = unreadable data.</div>
</div>

<div class="section">
  <div class="section-title">Venice AI Endpoints — Multimodal Health Intelligence</div>
  <div class="ep-grid" id="epGrid"></div>
</div>

<div class="two-col" style="padding:0 28px 16px">
  <div>
    <div class="section-title">Recent Alerts</div>
    <div class="alerts-list" id="alertsList"></div>
  </div>
  <div>
    <div class="section-title">Audit Trail — Verifiable Receipts</div>
    <div class="audit-list" id="auditList"></div>
  </div>
</div>

<div class="section">
  <div class="section-title">Analysis Logs</div>
  <table><thead><tr><th>Time</th><th>Patient</th><th>Input</th><th>Decision</th><th>Reason</th><th>Model</th><th>Anomaly</th></tr></thead>
  <tbody id="logsBody"></tbody></table>
</div>

<div class="footer">healthguard · akash x venice ai open agents hackathon 2026 · decentralized private ai health agent · built by karthik</div>

<script>
const EP_META = {
  "audio/transcriptions":{icon:"🎤",label:"Speech to Text (Whisper)",color:"#00e5ff"},
  "vision":{icon:"👁️",label:"Medical Image Analysis (Qwen2.5-VL)",color:"#ff5252"},
  "audio/speech":{icon:"🔊",label:"Patient Audio Alerts (Kokoro TTS)",color:"#76ff03"},
  "images/generations":{icon:"🖼️",label:"Visual Health Reports (Flux)",color:"#e040fb"},
  "chat/completions":{icon:"💬",label:"Clinical Reasoning",color:"#ffab40"},
};

async function refresh(){
  try{
    const [st,alerts,audit,logs]=await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/alerts?limit=20').then(r=>r.json()),
      fetch('/audit?limit=30').then(r=>r.json()),
      fetch('/logs?limit=20').then(r=>r.json()),
    ]);

    document.getElementById('sStatus').textContent=st.running?'RUNNING':'IDLE';
    document.getElementById('sUptime').textContent=st.uptime_seconds?Math.round(st.uptime_seconds)+'s':'';
    document.getElementById('sVenice').textContent=st.venice_calls||0;
    document.getElementById('sAkash').textContent=st.akashml_calls||0;
    document.getElementById('sAlerts').textContent=st.db_stats?.total_alerts||0;
    document.getElementById('sAudit').textContent=st.db_stats?.audit_entries||0;
    document.getElementById('sPatients').textContent=st.db_stats?.patients||0;
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
    for(const a of alerts){
      const sevIcon=a.severity===1?'🚨':a.severity===2?'⚠️':'ℹ️';
      const actions=(a.action_taken||'').split(',').map(s=>s.trim()).filter(Boolean);
      const d=document.createElement('div');
      d.className=`alert-item sev${a.severity}`;
      d.innerHTML=`<div class="alert-sev">${sevIcon}</div><div class="alert-body"><div class="alert-msg">${a.message||''}</div><div class="alert-meta">${a.timestamp||''} · Patient: ${(a.patient_id||'').substring(0,12)}...</div><div class="alert-actions">${actions.map(a=>{
        if(a.includes('telegram'))return'<span class="alert-tag tg">📨 Telegram</span>';
        if(a.includes('tts'))return'<span class="alert-tag tts">🔊 TTS Alert</span>';
        if(a.includes('doctor'))return'<span class="alert-tag doc">👨‍⚕️ Doctor</span>';
        return`<span class="alert-tag">${a}</span>`;
      }).join('')}</div></div>`;
      al.appendChild(d);
    }

    // Audit
    const au=document.getElementById('auditList');
    au.innerHTML='';
    for(const e of audit.reverse()){
      const d=document.createElement('div');
      d.className='audit-entry';
      d.innerHTML=`<span class="ts">${(e.timestamp||'').substring(11,19)}</span> <span class="type">${e.type||''}</span> ${e.reason?e.reason.substring(0,80):''}${e.telegram_ok?' ✓TG':''}${e.tts_generated?' ✓TTS':''}`;
      au.appendChild(d);
    }

    // Logs
    const lb=document.getElementById('logsBody');
    lb.innerHTML='';
    for(const l of logs){
      const tr=document.createElement('tr');
      const sev=l.anomaly_score>=.7?'s1':l.anomaly_score>=.4?'s2':'s3';
      tr.innerHTML=`<td style="font-size:10px;color:#546e7a">${(l.timestamp||'').substring(11,19)}</td><td style="font-family:monospace;color:#80cbc4">${(l.patient_id||'').substring(0,12)}...</td><td>${l.input_type||''}</td><td><span class="sev-badge ${sev}">${l.decision||''}</span></td><td style="max-width:300px;color:#b0bec5">${(l.reason||'').substring(0,120)}</td><td style="font-size:10px;color:#546e7a">${l.model_used||'rules'}</td><td style="color:${l.anomaly_score>=.7?'#ff5252':l.anomaly_score>=.4?'#ffd740':'#69f0ae'}">${(l.anomaly_score||0).toFixed(2)}</td>`;
      lb.appendChild(tr);
    }
  }catch(e){console.error('refresh',e)}
}
refresh();
setInterval(refresh,10000);
</script>
</body>
</html>"""
