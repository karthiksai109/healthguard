# HealthGuard — Hackathon Presentation Script
## Decentralized Private AI Health Agent

---

# SLIDE 1: THE PROBLEM (30 seconds)

**"What happens to your health data when AI analyzes it?"**

- OpenAI retains patient data for 30 days — HIPAA liability
- OpenAI content policies BLOCK discussions about overdose thresholds, substance abuse
- Enterprise healthcare BAAs cost $100K+/year — inaccessible to small clinics
- Centralized cloud = one breach exposes everything
- 725 healthcare data breaches in 2023 alone (HHS data)

**The healthcare industry needs AI that FORGETS.**

---

# SLIDE 2: THE SOLUTION (45 seconds)

**HealthGuard: An autonomous AI health agent that never remembers your data.**

- Runs 24/7 on Akash decentralized compute — no single point of failure
- Venice AI processes your vitals, photos, and voice notes with **zero data retention**
- Raw photos and audio are **deleted within 60 seconds** after analysis
- SQLite database is **AES-256-GCM encrypted** — even a full server breach = unreadable data
- The Akash provider literally cannot read your health records

**LIVE DEMO**: Our agent is running RIGHT NOW on Akash, autonomously monitoring 3 patients.

---

# SLIDE 3: HOW IT WORKS — LIVE WALKTHROUGH (2 minutes)

## Step 1: Patient submits data
- Upload a wound photo → EXIF metadata stripped instantly
- Record a voice note → "I've been having bad headaches, my vision is blurry"
- Enter vitals → BP: 185/115 mmHg

## Step 2: Venice AI processes with zero retention
- **Venice STT** (Whisper) transcribes voice → audio deleted immediately
- **Venice Vision** (Qwen2.5-VL) analyzes wound photo → image deleted immediately
- **Venice TTS** (Kokoro) generates spoken alert for the patient
- **Venice ImgGen** (Flux) creates visual health report card
- Venice processes and FORGETS. This is architectural, not a policy promise.

## Step 3: AkashML structures the output
- AkashML receives ONLY structured text — never raw images, audio, or patient names
- Generates SOAP notes (Subjective, Objective, Assessment, Plan)
- Detects anomaly patterns across vitals history
- Returns urgency level: routine / soon / urgent / emergency

## Step 4: Dual-layer decision engine
- **Layer 1 — Rules**: Hard-coded medical thresholds (BP ≥ 180 = CRITICAL)
- **Layer 2 — AI**: AkashML pattern detection (trending glucose drops)
- Rules ALWAYS win. AI cannot override safety thresholds.

## Step 5: Alert delivery with verifiable receipts
- Severity 1 (CRITICAL): Telegram alert + Doctor notify + TTS spoken warning
- Severity 2 (WARNING): Telegram notification
- Every alert creates an immutable audit log entry — verifiable proof of action

---

# SLIDE 4: DEMO SCENARIOS (show on dashboard)

## Patient 1 — Maria Santos (Hypertensive Crisis)
- BP trending: 138 → 155 → **185 mmHg** (3 days)
- Voice note: "Bad headaches, blurry vision, dizzy"
- **RESULT**: Rule engine fires severity 1 → Telegram alert + TTS audio + Doctor notified
- AkashML SOAP note: urgency = URGENT

## Patient 2 — James Wilson (Hypoglycemia)
- Glucose dropping: 110 → 85 → **62 mg/dL**
- Voice note: "Shaky, sweaty, forgot to eat, took insulin"
- **RESULT**: Rule engine fires severity 2 → Telegram warning
- AkashML detects insulin timing pattern

## Patient 3 — Aisha Patel (Acute Abdomen Emergency)
- Pain escalating: 4 → 6 → **9/10** + Fever 99.8 → **102.1°F**
- Voice note: "Lower right side pain, unbearable, nauseous"
- **RESULT**: Rule engine fires severity 1 (pain ≥ 9) → Full alert cascade
- AkashML SOAP: urgency = EMERGENCY

**All 3 alerts fired autonomously. No human intervention. The agent runs at 3am while the patient sleeps.**

---

# SLIDE 5: VENICE AI — WHY EVERY ENDPOINT MATTERS (45 seconds)

| Venice Endpoint | Clinical Purpose | Privacy Guarantee |
|---|---|---|
| STT (Whisper Large V3) | Transcribe patient voice notes into text | Raw audio contains PHI — zero retention |
| Vision (Qwen2.5-VL 235B) | Analyze wound photos, skin conditions, medications | Images are identifiable patient data |
| TTS (Kokoro) | Speak critical alerts to patients | Diagnosis info in audio |
| Image Gen (Flux) | Generate visual health report cards | Built from private health summaries |
| Chat (venice-uncensored) | Clinical reasoning without content filters | Patient context is PHI |

**Venice is not just "another LLM provider" — it's the ONLY provider that architecturally guarantees zero retention of health data.**

---

# SLIDE 6: AKASH NETWORK — WHY DECENTRALIZED MATTERS (30 seconds)

- **No single cloud provider** holds all patient data
- **Persistent encrypted volume** survives container restarts
- **$4.06/month** for 2 CPU, 4GB RAM, 10GB storage — vs $50+/month on AWS
- **Akash provider sees only encrypted bytes** — even with full server access
- The container is stateless — all state is in the encrypted SQLite on persistent volume
- If this provider disappears, redeploy to another in 2 minutes

---

# SLIDE 7: SECURITY ARCHITECTURE (30 seconds)

```
LAYER 1: Transport    — HTTPS everywhere, TLS 1.3
LAYER 2: Identity     — Ephemeral session IDs during inference (not patient names)
LAYER 3: Inference    — Venice zero retention (architectural, not policy)
LAYER 4: Structuring  — AkashML receives text only (no images, audio, or names)
LAYER 5: Storage      — AES-256-GCM encrypted SQLite, PBKDF2 key derivation
LAYER 6: Raw Data     — Auto-deleted within 60 seconds after processing
LAYER 7: Audit        — Append-only, immutable log of every action
LAYER 8: Compute      — Akash decentralized — no single point of data concentration
```

**8 layers of privacy. Zero trust at every boundary.**

---

# SLIDE 8: TECH STACK (15 seconds)

- **Backend**: Python 3.11 + FastAPI
- **AI Inference**: Venice AI (5 endpoints) + AkashML (DeepSeek-V3.1)
- **Database**: SQLite + AES-256-GCM encryption
- **Alerts**: Telegram Bot API
- **Frontend**: Live dark dashboard (embedded HTML)
- **Deployment**: Docker → DockerHub → Akash SDL
- **No OpenAI. No AWS. No GCP. Fully decentralized.**

---

# SLIDE 9: CLOSING (15 seconds)

**HealthGuard proves that privacy and AI intelligence are not trade-offs.**

- Venice AI gives us multimodal health analysis with zero retention
- Akash gives us decentralized compute at 1/10th the cost
- Together, they enable healthcare AI that patients can actually trust

**GitHub**: github.com/karthiksai109/healthguard
**Live on Akash**: Running right now, autonomously monitoring patients

Built by Karthik Ramadugu

---

# JUDGE Q&A PREP

**Q: How is this different from just using OpenAI with a BAA?**
A: Three things: (1) Venice is architecturally zero-retention, not policy-based — there's no 30-day retention window. (2) Venice is uncensored — it can discuss overdose thresholds, substance abuse scenarios that OpenAI blocks. (3) Cost — a healthcare BAA with OpenAI starts at $100K/year. Venice + Akash costs $4/month.

**Q: What happens if the Akash provider goes down?**
A: The container is stateless — all state is in the encrypted persistent volume. Redeploy to a new provider in 2 minutes. The data survives.

**Q: How do you handle HIPAA compliance?**
A: We go beyond HIPAA's minimum requirements. HIPAA allows 30-day data retention — we delete raw data in 60 seconds. HIPAA requires encryption at rest — we use AES-256-GCM. HIPAA requires audit trails — we have an immutable append-only log. The architecture is designed so that even a full infrastructure breach yields zero usable patient data.

**Q: Why not just use local on-device processing?**
A: On-device models are too small for clinical reasoning. Venice gives us 235B parameter vision models and full LLM reasoning — but with the privacy guarantees of local processing. Best of both worlds.

**Q: How many Venice AI endpoints do you use?**
A: All five — STT (Whisper), Vision (Qwen2.5-VL), TTS (Kokoro), Image Generation (Flux), and Chat. Each has a specific clinical purpose. This isn't token endpoint stuffing — each endpoint solves a real healthcare problem.

**Q: What's the autonomous agent actually doing?**
A: Every 60 seconds, the agent checks all patients. It loads their vitals history, recent logs, and alerts from encrypted storage. It runs the rule engine first (hard thresholds), then asks AkashML to analyze patterns. If anything is concerning, it fires alerts via Telegram with TTS audio. This runs 24/7 — it catches the 3am blood pressure spike that no human is watching.
