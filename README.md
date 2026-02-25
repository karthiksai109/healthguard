# HealthGuard — Decentralized Private AI Health Agent

> **Autonomous 24/7 health monitoring that keeps patient data private.**
> Built on Venice AI (zero data retention) + Akash Network (decentralized compute).

## The Problem

Patients need continuous health monitoring with AI analysis. But:
- **OpenAI retains data 30 days** — HIPAA liability for any health data
- **OpenAI content policies block** discussion of overdose thresholds, substance abuse scenarios
- **Enterprise BAAs cost $100K+/yr** — inaccessible to small clinics, rural hospitals
- **Centralized cloud = single point of data concentration** — one breach exposes everything

## The Solution

HealthGuard runs entirely on Akash decentralized compute. Venice AI processes patient data with **zero retention** — data is forgotten after inference. Raw photos and audio are **deleted within 60 seconds**. SQLite is **AES-256-GCM encrypted**. The Akash provider sees only encrypted noise.

## Architecture

```
Patient Input → Ingestion (EXIF strip, session ID, 60s TTL)
             → Venice STT (transcribe voice → delete audio)
             → Venice Vision (analyze image → delete photo)
             → AkashML (SOAP notes, anomaly detection — text only, no PHI)
             → Decision Engine (rules first, then AI — rules always win)
             → Delivery (Telegram alert + Venice TTS spoken alert)
             → Persistence (SQLite AES-256-GCM on Akash persistent volume)
             → Audit Log (immutable, append-only, verifiable receipts)
```

## Venice AI Endpoints — Each With a Purpose

| Endpoint | Use | Why Venice Not OpenAI |
|---|---|---|
| **STT (Whisper)** | Transcribe patient voice notes | Raw audio contains PHI — zero retention needed |
| **Vision (Qwen2.5-VL)** | Analyze wound photos, skin conditions | Images contain identifiable patient data |
| **TTS (Kokoro)** | Spoken alerts for patients | Diagnosis info in audio — zero retention |
| **Image Gen (Flux)** | Visual health report cards | Generated from private health summaries |
| **Chat** | Clinical reasoning when needed | Patient context is PHI |

## AkashML — Structured Output Only

AkashML **never receives raw photos, audio, or patient names**. It receives:
- Structured text from Venice STT/Vision output
- Anonymized vitals history
- Session IDs (not patient IDs)

Tasks routed to AkashML:
- SOAP note generation from transcripts
- Anomaly detection from combined signals
- Autonomous loop decisions (every 60 seconds)
- Weekly summary generation

## Dual-Layer Decision Engine

```
Layer 1 — RULES (deterministic, never misses emergencies)
  BP ≥ 180 systolic     → CRITICAL (severity 1)
  Glucose ≤ 50          → CRITICAL
  Pain ≥ 9/10           → CRITICAL
  SpO2 ≤ 90%            → CRITICAL
  Rules ALWAYS run first. AI cannot override safety rules.

Layer 2 — AI (nuanced pattern detection)
  AkashML analyzes combined signals
  Anomaly score > 0.7   → escalate
  Anomaly score > 0.4   → monitor
  AI adds context but never overrides rules
```

## Alert Delivery

| Severity | Actions |
|---|---|
| **1 — CRITICAL** | Telegram immediate + Doctor notify + Venice TTS spoken alert |
| **2 — WARNING** | Telegram notification + logged |
| **3 — INFO** | Logged only + periodic summaries |

Every alert produces a **verifiable receipt** in the immutable audit log:
```json
{
  "action_id": "b8e2f1",
  "timestamp": "2026-02-25T21:07:15Z",
  "type": "severity_1_delivery",
  "model_used": "deepseek-ai/DeepSeek-V3.1",
  "anomaly_score": 0.0,
  "actions_taken": ["telegram_alert", "tts_alert", "doctor_notify"],
  "telegram_ok": false,
  "tts_generated": true,
  "raw_data_retained": false
}
```

## Security

- **Transport**: HTTPS everywhere, TLS 1.3
- **Inference**: Venice = zero retention by architecture (not policy)
- **Storage**: AES-256-GCM, key derived via PBKDF2, never persisted
- **Identity**: Ephemeral session IDs during inference, never patient names
- **Raw Data**: Auto-deleted within 60 seconds after processing
- **Audit**: Append-only, immutable log of every action

## Quick Start

```bash
cp .env.example .env
# Fill in Venice + AkashML API keys
pip install -r requirements.txt
python main.py
# Dashboard: http://localhost:8080
# API docs: http://localhost:8080/docs
```

## Docker

```bash
docker build -t healthguard .
docker run -d -p 8080:8080 \
  -e VENICE_API_KEY=your_key \
  -e AKASHML_API_KEY=your_key \
  -e DEMO_MODE=true \
  healthguard
```

## Deploy to Akash

```bash
docker push karthiksai109/healthguard:latest
# Use deploy.sdl.yml in Akash Console
# Persistent volume at /data survives redeploys
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Live dashboard |
| `/upload-photo` | POST | Upload patient health photo |
| `/voice-note` | POST | Upload patient voice note |
| `/symptom` | POST | Submit text symptom log |
| `/vital` | POST | Record vital sign |
| `/status` | GET | Agent status + stats |
| `/patients` | GET | List patients |
| `/patient/{id}` | GET | Patient detail + vitals |
| `/logs` | GET | Analysis logs (decrypted) |
| `/alerts` | GET | Alert history |
| `/audit` | GET | Immutable audit trail |
| `/docs` | GET | Swagger API docs |

## Demo Cases

3 synthetic patients with escalating vitals:
- **Maria Santos** — BP trending 138→155→185 (hypertensive crisis, severity 1)
- **James Wilson** — Glucose dropping 110→85→62 (hypoglycemia, severity 2)
- **Aisha Patel** — Pain 4→6→9 + fever 99.8→102.1°F (emergency, severity 1)

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Database**: SQLite + cryptography (AES-256-GCM)
- **Venice AI**: STT, Vision, TTS, ImgGen, Chat (zero retention)
- **AkashML**: DeepSeek-V3.1 (structured output, no PHI)
- **Alerts**: Telegram Bot API
- **Deployment**: Docker + Akash SDL
- **No OpenAI, no AWS, no GCP** — fully decentralized

## Built For

**Akash x Venice AI Open Agents Hackathon 2026**

Built by Karthik Ramadugu
