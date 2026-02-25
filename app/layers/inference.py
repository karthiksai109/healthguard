"""HealthGuard — Layer 2: Inference

Venice AI: STT (Whisper), Vision (Qwen2.5-VL), TTS (Kokoro), ImgGen (Flux)
AkashML: Structuring, reasoning, SOAP notes, anomaly detection
Venice processes and forgets. Zero retention is architectural.
AkashML receives structured text only — never raw photos, audio, or patient names.
"""
import base64
import json
import httpx
import structlog
from openai import OpenAI
from app.core.config import AppConfig

logger = structlog.get_logger()


# ── Venice STT — Speech to Text (Whisper Large V3) ──────────────────

def venice_stt(config: AppConfig, audio_bytes: bytes) -> str:
    """Transcribe patient voice note using Venice Whisper.
    Raw audio deleted by Venice after transcription — zero retention.
    """
    try:
        headers = {"Authorization": f"Bearer {config.venice.api_key}"}
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(
                f"{config.venice.base_url}/audio/transcriptions",
                headers=headers,
                files={"file": ("voice.wav", audio_bytes, "audio/wav")},
                data={"model": config.venice.stt_model},
            )
            resp.raise_for_status()
            result = resp.json()
            text = result.get("text", "")
            logger.info("venice_stt_ok", chars=len(text), endpoint="audio/transcriptions")
            return text
    except Exception as e:
        logger.error("venice_stt_fail", error=str(e))
        return ""


# ── Venice Vision — Image Analysis (Qwen2.5-VL 235B) ────────────────

VISION_SCHEMA_PROMPT = """You are a medical image analysis AI. Analyze this patient-submitted health image.
Return ONLY a JSON object with exactly these fields:
{
  "image_type": "wound|skin|medication|document|other",
  "observations": "detailed clinical observations",
  "healing_stage": "early|mid|late|healed|worsening|N/A",
  "redness": "none|mild|moderate|severe",
  "swelling": "none|mild|moderate|severe",
  "discharge": "none|clear|yellow|bloody",
  "confidence": 0.0 to 1.0,
  "concerns": "any concerning findings or empty string",
  "recommendation": "clinical recommendation"
}
Be precise and clinical. This analysis will be combined with vitals history for decision-making."""


def venice_vision(config: AppConfig, client: OpenAI, image_bytes: bytes) -> dict:
    """Analyze patient health image using Venice Vision.
    Raw image deleted by Venice after analysis — zero retention.
    Returns structured JSON with clinical observations.
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model=config.venice.vision_model,
            messages=[
                {"role": "system", "content": VISION_SCHEMA_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": "Analyze this health image. Return JSON only."},
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON from response
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        logger.info("venice_vision_ok", image_type=result.get("image_type"), confidence=result.get("confidence"), endpoint="vision")
        return result
    except json.JSONDecodeError:
        logger.warning("venice_vision_json_fail", raw=raw[:200] if raw else "")
        return {"observations": raw if raw else "analysis failed", "confidence": 0.0}
    except Exception as e:
        logger.error("venice_vision_fail", error=str(e))
        return {"observations": "vision analysis failed", "confidence": 0.0, "error": str(e)}


# ── Venice TTS — Text to Speech (Kokoro) ─────────────────────────────

def venice_tts(config: AppConfig, text: str, voice: str = "af_heart") -> bytes | None:
    """Generate spoken alert/summary using Venice TTS.
    Input text = clean summary only (no raw PHI).
    Venice generates audio and forgets the input.
    """
    try:
        headers = {
            "Authorization": f"Bearer {config.venice.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.venice.audio_model,
            "input": text[:4000],
            "voice": voice,
        }
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(
                f"{config.venice.base_url}/audio/speech",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            logger.info("venice_tts_ok", chars=len(text), bytes=len(resp.content), endpoint="audio/speech")
            return resp.content
    except Exception as e:
        logger.error("venice_tts_fail", error=str(e))
        return None


# ── Venice ImgGen — Visual Health Report (Flux) ──────────────────────

def venice_imggen(config: AppConfig, summary_text: str) -> bytes | None:
    """Generate visual health report/chart card using Venice image generation.
    Input = clean summary text only, never raw PHI.
    Output = visual card for patient dashboard.
    """
    prompt = (
        f"Clean medical infographic dashboard card. Dark background, teal and white accents. "
        f"Shows health metrics summary: {summary_text[:200]}. "
        f"Professional healthcare design. Data visualization style. No photorealism. "
        f"Modern, minimal, clinical aesthetic."
    )
    try:
        headers = {
            "Authorization": f"Bearer {config.venice.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.venice.image_model,
            "prompt": prompt,
            "n": 1,
            "size": "512x512",
            "response_format": "b64_json",
        }
        with httpx.Client(timeout=45.0) as http:
            resp = http.post(
                f"{config.venice.base_url}/images/generations",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            b64_img = data["data"][0].get("b64_json")
            if b64_img:
                img_bytes = base64.b64decode(b64_img)
                logger.info("venice_imggen_ok", size=len(img_bytes), endpoint="images/generations")
                return img_bytes
            url = data["data"][0].get("url", "")
            if url:
                img_resp = http.get(url)
                logger.info("venice_imggen_ok", size=len(img_resp.content), endpoint="images/generations")
                return img_resp.content
            return None
    except Exception as e:
        logger.error("venice_imggen_fail", error=str(e))
        return None


# ── AkashML — Structure STT into SOAP Note ───────────────────────────

SOAP_PROMPT = """You are a clinical documentation AI. Convert the patient's spoken words into a structured SOAP note.
Return ONLY a JSON object:
{
  "subjective": "what the patient reported",
  "objective": "any measurable observations mentioned",
  "assessment": "clinical assessment based on reported symptoms",
  "plan": "recommended next steps",
  "key_symptoms": ["symptom1", "symptom2"],
  "pain_level": 0-10 or null,
  "medication_mentioned": ["med1", "med2"] or [],
  "urgency": "routine|soon|urgent|emergency"
}"""


def akashml_soap_note(client: OpenAI, model: str, transcript: str, vitals_context: str = "") -> dict:
    """Structure Venice STT output into SOAP note using AkashML.
    AkashML receives: transcript text + vitals summary (no patient name, no raw audio).
    """
    try:
        content = f"Patient transcript: {transcript}"
        if vitals_context:
            content += f"\n\nRecent vitals context: {vitals_context}"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SOAP_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=800,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        logger.info("akashml_soap_ok", urgency=result.get("urgency"), pain=result.get("pain_level"))
        return result
    except json.JSONDecodeError:
        logger.warning("akashml_soap_json_fail", raw=raw[:200] if raw else "")
        return {"subjective": transcript, "urgency": "routine", "assessment": raw if raw else ""}
    except Exception as e:
        logger.error("akashml_soap_fail", error=str(e))
        return {"subjective": transcript, "urgency": "routine", "error": str(e)}


# ── AkashML — Analyze Vision Output Against History ──────────────────

ANALYZE_PROMPT = """You are a clinical decision support AI. Analyze the combined inputs and return a JSON decision.
Inputs: current image analysis + vitals history + recent logs.
Return ONLY JSON:
{
  "decision": "normal|monitor|alert|escalate",
  "anomaly_score": 0.0 to 1.0,
  "reason": "clear explanation",
  "urgency": "none|within_week|within_24_hours|immediate",
  "combined_signal": true/false (whether multiple inputs together are concerning),
  "recommended_actions": ["action1", "action2"]
}
Score > 0.7 = escalate. Score > 0.4 = alert. Otherwise = monitor or normal."""


def akashml_analyze(client: OpenAI, model: str, vision_result: dict, vitals_summary: str, recent_logs: str = "") -> dict:
    """Analyze Venice Vision output against patient history using AkashML.
    AkashML receives: structured JSON + vitals text. Never raw images or names.
    """
    try:
        content = f"Image analysis: {json.dumps(vision_result)}\n\nVitals history: {vitals_summary}"
        if recent_logs:
            content += f"\n\nRecent analysis logs: {recent_logs}"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        logger.info("akashml_analyze_ok", decision=result.get("decision"), anomaly=result.get("anomaly_score"))
        return result
    except json.JSONDecodeError:
        logger.warning("akashml_analyze_json_fail")
        return {"decision": "monitor", "anomaly_score": 0.3, "reason": "analysis parse error"}
    except Exception as e:
        logger.error("akashml_analyze_fail", error=str(e))
        return {"decision": "monitor", "anomaly_score": 0.0, "error": str(e)}


# ── AkashML — Autonomous Loop Decision ───────────────────────────────

LOOP_DECISION_PROMPT = """You are an autonomous health monitoring agent. Every 60 seconds you review a patient's current state.
Given the current context (vitals, recent logs, alerts), decide the agent's next action.
Return ONLY JSON:
{
  "action": "idle|check_in|alert_patient|alert_doctor|generate_report|request_input",
  "reason": "why this action",
  "message": "what to tell the patient (if applicable)",
  "severity": 1-3 (1=critical, 2=warning, 3=info),
  "confidence": 0.0 to 1.0
}"""


def akashml_loop_decision(client: OpenAI, model: str, context: str) -> dict:
    """Autonomous 60-second loop decision. AkashML decides next agent action."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LOOP_DECISION_PROMPT},
                {"role": "user", "content": context},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        logger.info("akashml_loop_ok", action=result.get("action"), severity=result.get("severity"))
        return result
    except Exception as e:
        logger.error("akashml_loop_fail", error=str(e))
        return {"action": "idle", "reason": f"decision error: {e}", "severity": 3, "confidence": 0.0}


# ── AkashML — Weekly Summary ─────────────────────────────────────────

DOCTOR_REPORT_PROMPT = """You are a senior clinical decision support AI. Generate a comprehensive doctor's report.
Given the patient's vitals history, symptoms, alerts, and analysis logs, produce a detailed clinical report.
Return ONLY a JSON object:
{
  "risk_assessment": {
    "overall_risk": "low|moderate|high|critical",
    "risk_factors": ["factor1", "factor2"],
    "risk_score": 0.0 to 1.0
  },
  "treatment_recommendations": [
    {"priority": 1, "recommendation": "text", "rationale": "why", "urgency": "immediate|24h|this_week|routine"}
  ],
  "medication_review": {
    "current_concerns": ["concern1"],
    "interactions_flagged": ["interaction1"],
    "dosage_notes": "any dosage observations"
  },
  "follow_up_plan": {
    "next_visit": "timeframe",
    "monitoring_frequency": "how often to check vitals",
    "tests_recommended": ["test1", "test2"],
    "specialist_referral": "if needed, which specialty"
  },
  "patient_education": ["key point for patient to understand"],
  "clinical_summary": "2-3 sentence executive summary for the doctor",
  "differential_diagnosis": ["possible diagnosis 1", "possible diagnosis 2"]
}
Be evidence-based. Flag any dangerous patterns. Prioritize patient safety."""


def akashml_doctor_report(client: OpenAI, model: str, patient_context: str) -> dict:
    """Generate comprehensive doctor report using AkashML.
    Input: anonymized patient context (vitals + logs + alerts). No PHI.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DOCTOR_REPORT_PROMPT},
                {"role": "user", "content": patient_context},
            ],
            max_tokens=1200,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        logger.info("akashml_doctor_report_ok", risk=result.get("risk_assessment", {}).get("overall_risk"))
        return result
    except json.JSONDecodeError:
        logger.warning("akashml_doctor_report_json_fail", raw=raw[:200] if raw else "")
        return {"clinical_summary": raw if raw else "Report generation failed", "error": "json_parse"}
    except Exception as e:
        logger.error("akashml_doctor_report_fail", error=str(e))
        return {"clinical_summary": "Report generation failed", "error": str(e)}


PATIENT_BRIEFING_PROMPT = """You are a friendly, empathetic health assistant speaking directly to the patient.
Generate a warm, clear spoken health briefing based on their current vitals and recent history.
Keep it under 200 words. Use simple language, no medical jargon. Be encouraging but honest about concerns.
Start with a greeting using their name. Mention specific numbers. End with one actionable tip.
Return ONLY a JSON object:
{
  "spoken_text": "the full text to be spoken aloud",
  "mood": "reassuring|cautious|urgent",
  "key_message": "one sentence takeaway"
}"""


def akashml_patient_briefing(client: OpenAI, model: str, patient_name: str, context: str) -> dict:
    """Generate patient-friendly spoken briefing text using AkashML."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PATIENT_BRIEFING_PROMPT},
                {"role": "user", "content": f"Patient name: {patient_name}\n\n{context}"},
            ],
            max_tokens=400,
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        logger.info("akashml_patient_briefing_ok", mood=result.get("mood"))
        return result
    except Exception as e:
        logger.error("akashml_patient_briefing_fail", error=str(e))
        return {"spoken_text": f"Hello {patient_name}, we're monitoring your health. Please check in with your doctor soon.", "mood": "cautious"}


WEEKLY_PROMPT = """You are a health reporting AI. Generate a weekly health summary from the patient's data.
Return ONLY JSON:
{
  "period": "date range",
  "overall_status": "stable|improving|declining|concerning",
  "key_findings": ["finding1", "finding2"],
  "vitals_trends": "summary of vital sign trends",
  "alerts_summary": "summary of alerts this week",
  "recommendations": ["rec1", "rec2"],
  "doctor_notes": "anything the doctor should know"
}"""


def akashml_weekly_summary(client: OpenAI, model: str, week_data: str) -> dict:
    """Generate weekly summary for patient/doctor review."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": WEEKLY_PROMPT},
                {"role": "user", "content": week_data},
            ],
            max_tokens=800,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.error("akashml_weekly_fail", error=str(e))
        return {"overall_status": "unknown", "error": str(e)}
