"""HealthGuard — Layer 3: Agent Core

The heart of the system. Orchestrator runs every 60 seconds.
- Memory Manager: loads patient context from SQLite (short/medium/long term)
- Decision Engine: dual-layer (rules first, then AI)
- Action Executor: Telegram, TTS, ImgGen, scheduling
- Event Queue: processes pending uploads and vitals

This loop never stops. It runs at 3am while the patient sleeps.
"""
import time
import threading
import traceback
import structlog
from openai import OpenAI
from app.core.config import AppConfig
from app.core.database import Database
from app.core.clients import TelegramClient, get_venice_client, get_akashml_client
from app.layers import ingestion, inference, decision
from app.layers.delivery import DeliveryEngine

logger = structlog.get_logger()


class EventQueue:
    """Thread-safe queue for incoming patient events."""

    def __init__(self):
        self._queue: list[ingestion.IngestedItem] = []
        self._lock = threading.Lock()

    def push(self, item: ingestion.IngestedItem):
        with self._lock:
            self._queue.append(item)

    def get_pending(self) -> list[ingestion.IngestedItem]:
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    def size(self) -> int:
        with self._lock:
            return len(self._queue)


class MemoryManager:
    """Loads patient context from SQLite for agent decisions."""

    def __init__(self, db: Database):
        self.db = db

    def load_context(self, patient_id: str, days: int = 7) -> dict:
        vitals = self.db.get_vitals(patient_id, days=days)
        latest = self.db.get_latest_vitals(patient_id)
        logs = self.db.get_logs(patient_id, limit=15)
        alerts = self.db.get_alerts(patient_id, limit=10)
        return {
            "vitals_history": vitals,
            "latest_vitals": latest,
            "recent_logs": logs,
            "recent_alerts": alerts,
        }

    def format_for_ai(self, context: dict) -> str:
        """Format context as rich text for AI. No patient names — only metrics and clinical data."""
        parts = []
        latest = context.get("latest_vitals", {})
        if latest:
            vital_lines = []
            for k, v in latest.items():
                val = v['value']
                unit = v.get('unit', '')
                status = ""
                if k == "bp_systolic":
                    if val >= 180: status = " [CRITICAL - hypertensive crisis]"
                    elif val >= 140: status = " [HIGH - stage 2 hypertension]"
                    elif val >= 130: status = " [ELEVATED]"
                    else: status = " [normal]"
                elif k == "bp_diastolic":
                    if val >= 120: status = " [CRITICAL]"
                    elif val >= 90: status = " [HIGH]"
                    else: status = " [normal]"
                elif k == "glucose":
                    if val >= 250: status = " [CRITICAL - dangerously high]"
                    elif val >= 180: status = " [HIGH]"
                    elif val < 70: status = " [LOW - hypoglycemia risk]"
                    else: status = " [normal]"
                elif k == "temperature":
                    if val >= 103: status = " [HIGH FEVER]"
                    elif val >= 100.4: status = " [FEVER]"
                    else: status = " [normal]"
                elif k == "heart_rate":
                    if val >= 100: status = " [TACHYCARDIA]"
                    elif val < 60: status = " [BRADYCARDIA]"
                    else: status = " [normal]"
                elif k == "oxygen_saturation":
                    if val < 90: status = " [CRITICAL - severe hypoxemia]"
                    elif val < 95: status = " [LOW]"
                    else: status = " [normal]"
                elif k == "pain_level":
                    if val >= 8: status = " [SEVERE]"
                    elif val >= 5: status = " [MODERATE]"
                    else: status = " [MILD]"
                vital_lines.append(f"  {k}: {val}{unit}{status}")
            parts.append("CURRENT VITALS:\n" + "\n".join(vital_lines))

        history = context.get("vitals_history", [])
        if history:
            parts.append(f"\nVITALS HISTORY ({len(history)} readings, last 7 days):")
            for v in history[:10]:
                parts.append(f"  {v['metric_type']}={v['value']}{v.get('unit','')} at {v['timestamp']}")

        logs = context.get("recent_logs", [])
        if logs:
            parts.append(f"\nAI ANALYSIS HISTORY ({len(logs)} entries):")
            for l in logs[:5]:
                parts.append(f"  [{l['decision'].upper()}] {l.get('summary', l.get('reason', ''))[:150]}")

        alerts = context.get("recent_alerts", [])
        if alerts:
            parts.append(f"\nALERTS ({len(alerts)} total):")
            for a in alerts[:5]:
                parts.append(f"  [SEVERITY {a['severity']}] {a['message'][:150]}")

        return "\n".join(parts) if parts else "No patient data available yet."

    def format_vitals_summary(self, vitals: list[dict]) -> str:
        if not vitals:
            return "No vitals recorded."
        return "; ".join(f"{v['metric_type']}={v['value']}{v.get('unit','')}" for v in vitals[:15])


class HealthGuardAgent:
    """Main autonomous agent. Processes events and runs the 60s loop."""

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self.venice = get_venice_client(config)
        self.akashml = get_akashml_client(config)
        self.telegram = TelegramClient(config)
        self.delivery = DeliveryEngine(config, db, self.telegram, venice_tracker=self._track_venice)
        self.memory = MemoryManager(db)
        self.event_queue = EventQueue()
        self.running = False
        self.loop_count = 0
        self.start_time = time.time()
        self.venice_endpoints_used: set[str] = set()
        self.stats = {
            "venice_calls": 0,
            "akashml_calls": 0,
            "events_processed": 0,
            "loop_iterations": 0,
        }
        logger.info("agent_initialized", demo_mode=config.demo_mode)

    def _track_venice(self, endpoint: str):
        """Callback for tracking Venice endpoint usage from delivery layer."""
        self.venice_endpoints_used.add(endpoint)
        self.stats["venice_calls"] += 1

    # ── Process a single event (photo, voice, text, vital) ───────────
    def process_event(self, item: ingestion.IngestedItem) -> dict:
        """Full pipeline for one ingested event."""
        patient_id = item.patient_id
        context = self.memory.load_context(patient_id)
        vitals_summary = self.memory.format_vitals_summary(context["vitals_history"])
        result = {"session_id": item.session_id, "input_type": item.input_type, "actions": []}

        try:
            if item.input_type == "photo" and item.raw_bytes:
                result = self._process_photo(item, context, vitals_summary, result)
            elif item.input_type == "voice" and item.raw_bytes:
                result = self._process_voice(item, context, vitals_summary, result)
            elif item.input_type == "text" and item.text:
                result = self._process_text(item, context, vitals_summary, result)
            elif item.input_type == "vital" and item.text:
                result = self._process_vital_event(item, context, result)

            # Delete raw file immediately after processing
            if item.file_path:
                ingestion.delete_immediately(item.file_path)
                result["raw_deleted"] = True

            self.stats["events_processed"] += 1

        except Exception as e:
            logger.error("event_processing_failed", session=item.session_id, error=str(e))
            result["error"] = str(e)
            # Still delete raw file on error
            if item.file_path:
                ingestion.delete_immediately(item.file_path)

        return result

    def _process_photo(self, item, context, vitals_summary, result):
        """Photo pipeline: Single Venice Vision call (analysis+triage) → Decision → Delivery."""
        # Single Venice Vision call — does both analysis AND triage
        vision_result = inference.venice_vision(self.config, self.venice, item.raw_bytes)
        self.venice_endpoints_used.add("vision")
        self.stats["venice_calls"] += 1
        result["vision"] = vision_result
        result["ai_analysis"] = vision_result

        # Decision engine — use vision result directly as ai_analysis
        sev_map = {"green_self_care": 0.2, "yellow_see_doctor": 0.5, "orange_urgent_care": 0.7, "red_emergency": 0.95}
        ai_analysis = {
            "anomaly_score": sev_map.get(vision_result.get("emergency_level", ""), 0.5),
            "severity": vision_result.get("severity", "moderate"),
            "recommendation": vision_result.get("patient_message", ""),
        }
        rule_results = decision.evaluate_rules(context["latest_vitals"])
        combined = decision.combine_decisions(rule_results, ai_analysis)
        result["decision"] = combined

        # Delivery
        if combined["final_decision"] in ("alert", "escalate"):
            receipt = self.delivery.deliver(item.patient_id, combined)
            result["delivery"] = receipt

        # Log
        self.db.record_log(
            patient_id=item.patient_id, session_id=item.session_id,
            input_type="photo", summary=f"Vision: {vision_result.get('observations', '')[:200]}",
            decision=combined["final_decision"], reason=combined["reason"][:300],
            action_taken=", ".join(result.get("delivery", {}).get("actions_taken", ["logged"])),
            model_used="venice-vision",
            anomaly_score=ai_analysis.get("anomaly_score", 0.0),
        )
        return result

    def _process_voice(self, item, context, vitals_summary, result):
        """Voice pipeline: Venice STT → AkashML SOAP → Decision → Delivery."""
        # Venice STT
        transcript = inference.venice_stt(self.config, item.raw_bytes)
        self.venice_endpoints_used.add("audio/transcriptions")
        self.stats["venice_calls"] += 1
        result["transcript"] = transcript

        if not transcript:
            result["error"] = "STT returned empty"
            return result

        # AkashML SOAP note
        soap = inference.akashml_soap_note(
            self.venice, "llama-3.3-70b",
            transcript, vitals_summary,
        )
        self.stats["venice_calls"] += 1
        result["soap"] = soap

        # If pain level mentioned, record as vital and run rules
        if soap.get("pain_level") is not None and soap["pain_level"] is not None:
            self.db.record_vital(item.patient_id, "pain_level", float(soap["pain_level"]), source="voice")
            context["latest_vitals"]["pain_level"] = {"value": float(soap["pain_level"]), "unit": "/10"}

        # Decision
        rule_results = decision.evaluate_rules(context["latest_vitals"])
        ai_decision = {
            "decision": soap.get("urgency", "routine"),
            "anomaly_score": 0.7 if soap.get("urgency") in ("urgent", "emergency") else 0.3,
            "reason": soap.get("assessment", ""),
        }
        combined = decision.combine_decisions(rule_results, ai_decision)
        result["decision"] = combined

        if combined["final_decision"] in ("alert", "escalate"):
            receipt = self.delivery.deliver(item.patient_id, combined)
            result["delivery"] = receipt

        self.db.record_log(
            patient_id=item.patient_id, session_id=item.session_id,
            input_type="voice", summary=f"SOAP: S={soap.get('subjective','')} A={soap.get('assessment','')}",
            decision=combined["final_decision"], reason=combined["reason"][:300],
            action_taken=", ".join(result.get("delivery", {}).get("actions_taken", ["logged"])),
            model_used="llama-3.3-70b",
            anomaly_score=ai_decision.get("anomaly_score", 0.0),
        )
        return result

    def _process_text(self, item, context, vitals_summary, result):
        """Text pipeline: AkashML SOAP → Decision → Delivery."""
        soap = inference.akashml_soap_note(
            self.venice, "llama-3.3-70b",
            item.text, vitals_summary,
        )
        self.stats["venice_calls"] += 1
        result["soap"] = soap

        if soap.get("pain_level") is not None and soap["pain_level"] is not None:
            self.db.record_vital(item.patient_id, "pain_level", float(soap["pain_level"]), source="text")
            context["latest_vitals"]["pain_level"] = {"value": float(soap["pain_level"]), "unit": "/10"}

        rule_results = decision.evaluate_rules(context["latest_vitals"])
        ai_decision = {
            "decision": soap.get("urgency", "routine"),
            "anomaly_score": 0.6 if soap.get("urgency") in ("urgent", "emergency") else 0.2,
            "reason": soap.get("assessment", ""),
        }
        combined = decision.combine_decisions(rule_results, ai_decision)
        result["decision"] = combined

        if combined["final_decision"] in ("alert", "escalate"):
            receipt = self.delivery.deliver(item.patient_id, combined)
            result["delivery"] = receipt

        self.db.record_log(
            patient_id=item.patient_id, session_id=item.session_id,
            input_type="text", summary=item.text[:300],
            decision=combined["final_decision"], reason=combined["reason"][:300],
            action_taken=", ".join(result.get("delivery", {}).get("actions_taken", ["logged"])),
            model_used=self.config.akashml.primary_model,
            anomaly_score=ai_decision.get("anomaly_score", 0.0),
        )
        return result

    def _process_vital_event(self, item, context, result):
        """Vital sign event: record → rules check → delivery if needed."""
        # Parse vital from text "metric_type: value unit"
        parts = item.text.split(":")
        if len(parts) == 2:
            metric = parts[0].strip()
            val_parts = parts[1].strip().split()
            value = float(val_parts[0])
            unit = val_parts[1] if len(val_parts) > 1 else ""
            self.db.record_vital(item.patient_id, metric, value, unit=unit, source="manual")
            context["latest_vitals"][metric] = {"value": value, "unit": unit}

        rule_results = decision.evaluate_rules(context["latest_vitals"])
        ai_decision = {"decision": "normal", "anomaly_score": 0.0, "reason": "vital recorded"}
        combined = decision.combine_decisions(rule_results, ai_decision)
        result["decision"] = combined

        if combined["final_decision"] in ("alert", "escalate"):
            receipt = self.delivery.deliver(item.patient_id, combined)
            result["delivery"] = receipt

        self.db.record_log(
            patient_id=item.patient_id, session_id=item.session_id,
            input_type="vital", summary=item.text,
            decision=combined["final_decision"], reason=combined["reason"][:300],
            action_taken=", ".join(result.get("delivery", {}).get("actions_taken", ["logged"])),
        )
        return result

    # ── Autonomous Loop ───────────────────────────────────────────────
    def autonomous_loop(self):
        """60-second loop. Processes event queue + checks all patients."""
        logger.info("autonomous_loop_started", interval=self.config.agent_interval)
        self.running = True

        while self.running:
            try:
                self.loop_count += 1
                self.stats["loop_iterations"] = self.loop_count

                # 1. Process pending events
                events = self.event_queue.get_pending()
                for event in events:
                    logger.info("processing_event", session=event.session_id, type=event.input_type)
                    self.process_event(event)

                # 2. Cleanup expired ephemeral files
                deleted = ingestion.cleanup_expired()
                if deleted:
                    logger.info("ephemeral_cleanup", deleted=deleted)

                # 3. Periodic AkashML loop decision for each patient
                if self.loop_count % 5 == 0:  # Every 5 minutes
                    patients = self.db.list_patients()
                    for p in patients:
                        context = self.memory.load_context(p["id"])
                        context_text = self.memory.format_for_ai(context)
                        if context_text != "No patient data available yet.":
                            loop_decision = inference.akashml_loop_decision(
                                self.venice, "llama-3.3-70b", context_text,
                            )
                            self.stats["venice_calls"] += 1
                            if loop_decision.get("action") in ("alert_patient", "alert_doctor"):
                                combined = {
                                    "final_decision": "alert",
                                    "final_severity": loop_decision.get("severity", 2),
                                    "reason": loop_decision.get("reason", "Autonomous loop alert"),
                                    "source": "ai_autonomous_loop",
                                    "ai_decision": loop_decision,
                                }
                                self.delivery.deliver(p["id"], combined)

                logger.debug("loop_tick", iteration=self.loop_count, events=len(events), queue_size=self.event_queue.size())

            except Exception as e:
                logger.error("loop_error", iteration=self.loop_count, error=str(e), traceback=traceback.format_exc())

            time.sleep(self.config.agent_interval)

    def start(self):
        """Start the autonomous loop in a daemon thread."""
        thread = threading.Thread(target=self.autonomous_loop, daemon=True)
        thread.start()
        logger.info("agent_loop_thread_started")
        return thread

    def stop(self):
        self.running = False

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "loop_count": self.loop_count,
            "events_processed": self.stats["events_processed"],
            "venice_calls": self.stats["venice_calls"],
            "akashml_calls": self.stats["akashml_calls"],
            "venice_endpoints_used": list(self.venice_endpoints_used),
            "queue_size": self.event_queue.size(),
            "ephemeral_files": ingestion.get_ephemeral_count(),
            "delivery_stats": self.delivery.stats,
            "db_stats": self.db.get_stats(),
        }
