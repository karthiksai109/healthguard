"""HealthGuard — Layer 5: Delivery

Alert engine with severity levels:
  Severity 1 — CRITICAL: Telegram immediate + doctor webhook + TTS spoken alert
  Severity 2 — WARNING: App notification + daily summary flag
  Severity 3 — INFO: Weekly summary, medication adherence

Every alert produces a verifiable action receipt in the audit log.
"""
import structlog
from app.core.config import AppConfig
from app.core.clients import TelegramClient
from app.core.database import Database
from app.layers import inference

logger = structlog.get_logger()


class DeliveryEngine:
    """Executes alert actions and logs verifiable receipts."""

    def __init__(self, config: AppConfig, db: Database, telegram: TelegramClient, venice_tracker=None):
        self.config = config
        self.db = db
        self.telegram = telegram
        self.venice_tracker = venice_tracker
        self.stats = {
            "telegram_sent": 0,
            "tts_generated": 0,
            "reports_generated": 0,
            "total_actions": 0,
        }

    def deliver(self, patient_id: str, decision: dict) -> dict:
        """Execute delivery based on combined decision result."""
        severity = decision.get("final_severity", 3)
        reason = decision.get("reason", "")
        source = decision.get("source", "unknown")

        receipt = {
            "type": "delivery",
            "patient_id_hash": patient_id[:8] + "...",
            "severity": severity,
            "source": source,
            "actions_taken": [],
            "raw_data_retained": False,
        }

        if severity == 1:
            receipt = self._deliver_critical(patient_id, reason, receipt)
        elif severity == 2:
            receipt = self._deliver_warning(patient_id, reason, receipt)
        else:
            receipt = self._deliver_info(patient_id, reason, receipt)

        # Record alert in database
        actions_str = ", ".join(receipt["actions_taken"])
        webhook_resp = receipt.get("telegram_response", "")
        tts_gen = "tts_alert" in receipt["actions_taken"]
        self.db.record_alert(
            patient_id=patient_id,
            severity=severity,
            message=reason,
            action_taken=actions_str,
            webhook_response=str(webhook_resp),
            tts_generated=tts_gen,
        )

        # Audit log — immutable receipt
        self.db.audit({
            "type": f"severity_{severity}_delivery",
            "patient_id": patient_id[:8] + "...",
            "severity": severity,
            "reason": reason[:200],
            "source": source,
            "model_used": decision.get("ai_decision", {}).get("model", "rule_engine"),
            "anomaly_score": decision.get("ai_decision", {}).get("anomaly_score", 0.0),
            "actions_taken": receipt["actions_taken"],
            "telegram_ok": receipt.get("telegram_ok", False),
            "tts_generated": tts_gen,
            "raw_data_retained": False,
        })

        self.stats["total_actions"] += 1
        logger.info("delivery_complete", severity=severity, actions=receipt["actions_taken"])
        return receipt

    def _deliver_critical(self, patient_id: str, reason: str, receipt: dict) -> dict:
        """Severity 1: Telegram + doctor notify + TTS spoken alert."""
        # Telegram immediate
        tg_msg = f"🚨 <b>CRITICAL ALERT</b>\n\n{reason}\n\nPatient: {patient_id[:8]}...\nAction required immediately."
        tg_result = self.telegram.send_message(tg_msg)
        receipt["telegram_ok"] = tg_result.get("ok", False)
        receipt["telegram_response"] = f"{tg_result.get('status_code', 0)} {tg_result.get('ok', False)}"
        receipt["actions_taken"].append("telegram_alert")
        self.stats["telegram_sent"] += 1

        # TTS spoken alert
        tts_text = f"Critical health alert. {reason}. Please seek immediate medical attention or contact your doctor."
        audio = inference.venice_tts(self.config, tts_text)
        if audio:
            receipt["actions_taken"].append("tts_alert")
            receipt["tts_audio"] = audio
            self.stats["tts_generated"] += 1
            if self.venice_tracker:
                self.venice_tracker("audio/speech")
            # Also send audio to Telegram
            self.telegram.send_audio(audio, caption=f"🚨 Critical Alert Audio — {reason[:100]}")

        # Doctor notification (same Telegram for demo, separate in production)
        doc_msg = f"👨‍⚕️ <b>DOCTOR NOTIFICATION</b>\n\nPatient {patient_id[:8]}... requires immediate review.\n\n{reason}"
        self.telegram.send_message(doc_msg)
        receipt["actions_taken"].append("doctor_notify")

        return receipt

    def _deliver_warning(self, patient_id: str, reason: str, receipt: dict) -> dict:
        """Severity 2: Telegram notification + logged."""
        tg_msg = f"⚠️ <b>WARNING</b>\n\n{reason}\n\nPatient: {patient_id[:8]}...\nMonitor closely."
        tg_result = self.telegram.send_message(tg_msg)
        receipt["telegram_ok"] = tg_result.get("ok", False)
        receipt["telegram_response"] = f"{tg_result.get('status_code', 0)} {tg_result.get('ok', False)}"
        receipt["actions_taken"].append("telegram_warning")
        self.stats["telegram_sent"] += 1
        return receipt

    def _deliver_info(self, patient_id: str, reason: str, receipt: dict) -> dict:
        """Severity 3: Log only, periodic summaries."""
        receipt["actions_taken"].append("logged_only")
        return receipt

    def generate_weekly_report(self, patient_id: str, akashml_client, model: str) -> dict:
        """Generate weekly visual + audio report using Venice + AkashML."""
        vitals = self.db.get_vitals(patient_id, days=7)
        logs = self.db.get_logs(patient_id, limit=20)
        alerts = self.db.get_alerts(patient_id, limit=10)

        week_data = (
            f"Vitals this week ({len(vitals)} readings): "
            + "; ".join(f"{v['metric_type']}={v['value']}{v.get('unit','')}" for v in vitals[:20])
            + f"\n\nAnalysis logs ({len(logs)} entries): "
            + "; ".join(f"{l['decision']}: {l['reason'][:50]}" for l in logs[:10])
            + f"\n\nAlerts ({len(alerts)} total): "
            + "; ".join(f"sev{a['severity']}: {a['message'][:50]}" for a in alerts[:5])
        )

        # AkashML weekly summary
        summary = inference.akashml_weekly_summary(akashml_client, model, week_data)

        # Venice ImgGen — visual health report card
        img_summary = summary.get("overall_status", "unknown") + " - " + ", ".join(summary.get("key_findings", [])[:3])
        img_bytes = inference.venice_imggen(self.config, img_summary)

        # Venice TTS — spoken weekly report
        tts_text = (
            f"Weekly health summary. Overall status: {summary.get('overall_status', 'unknown')}. "
            + ". ".join(summary.get("key_findings", [])[:3])
            + ". " + ". ".join(summary.get("recommendations", [])[:2])
        )
        audio = inference.venice_tts(self.config, tts_text)

        self.stats["reports_generated"] += 1

        report = {
            "summary": summary,
            "image_bytes": img_bytes,
            "audio_bytes": audio,
            "week_data_size": len(week_data),
        }

        # Audit
        self.db.audit({
            "type": "weekly_report",
            "patient_id": patient_id[:8] + "...",
            "overall_status": summary.get("overall_status"),
            "findings_count": len(summary.get("key_findings", [])),
            "image_generated": img_bytes is not None,
            "audio_generated": audio is not None,
        })

        return report
