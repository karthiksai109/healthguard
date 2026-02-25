"""HealthGuard — Layer 3: Decision Engine

Dual-layer decision system:
  Layer 1 — Rule-based: Hard-coded thresholds, deterministic, never misses emergencies
  Layer 2 — AI-based: AkashML pattern detection for nuanced decisions

Rule layer always runs first. AI never overrides safety rules.
"""
import structlog

logger = structlog.get_logger()


# ── Hard-Coded Clinical Thresholds ────────────────────────────────────

RULES = {
    "bp_systolic": {
        "critical": 180,   # Severity 1 — immediate
        "warning": 150,    # Severity 2 — within 24h
        "low_critical": 80,
    },
    "bp_diastolic": {
        "critical": 120,
        "warning": 100,
        "low_critical": 50,
    },
    "glucose": {
        "critical_high": 400,
        "warning_high": 250,
        "critical_low": 50,
        "warning_low": 70,
    },
    "heart_rate": {
        "critical_high": 150,
        "warning_high": 120,
        "critical_low": 40,
        "warning_low": 50,
    },
    "temperature": {
        "critical_high": 104.0,  # Fahrenheit
        "warning_high": 101.5,
        "critical_low": 95.0,
        "warning_low": 96.5,
    },
    "oxygen_saturation": {
        "critical_low": 90,
        "warning_low": 93,
    },
    "pain_level": {
        "critical": 9,    # 9-10 = emergency
        "warning": 7,     # 7-8 = needs attention
    },
}


class RuleResult:
    """Result from rule-based evaluation."""
    def __init__(self, triggered: bool, severity: int, metric: str, value: float,
                 threshold: float, message: str):
        self.triggered = triggered
        self.severity = severity   # 1=critical, 2=warning, 3=info
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.message = message

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "severity": self.severity,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
        }


def evaluate_rules(vitals: dict) -> list[RuleResult]:
    """Layer 1: Deterministic rule evaluation against vitals.
    Returns list of triggered rules sorted by severity.
    This runs FIRST. AI cannot override these.
    """
    results = []

    # Blood pressure systolic
    if "bp_systolic" in vitals:
        v = vitals["bp_systolic"]["value"] if isinstance(vitals["bp_systolic"], dict) else vitals["bp_systolic"]
        if v >= RULES["bp_systolic"]["critical"]:
            results.append(RuleResult(True, 1, "bp_systolic", v, 180, f"CRITICAL: Systolic BP {v} mmHg ≥ 180. Hypertensive crisis."))
        elif v >= RULES["bp_systolic"]["warning"]:
            results.append(RuleResult(True, 2, "bp_systolic", v, 150, f"WARNING: Systolic BP {v} mmHg ≥ 150. Elevated."))
        elif v <= RULES["bp_systolic"]["low_critical"]:
            results.append(RuleResult(True, 1, "bp_systolic", v, 80, f"CRITICAL: Systolic BP {v} mmHg ≤ 80. Hypotension."))

    # Blood pressure diastolic
    if "bp_diastolic" in vitals:
        v = vitals["bp_diastolic"]["value"] if isinstance(vitals["bp_diastolic"], dict) else vitals["bp_diastolic"]
        if v >= RULES["bp_diastolic"]["critical"]:
            results.append(RuleResult(True, 1, "bp_diastolic", v, 120, f"CRITICAL: Diastolic BP {v} mmHg ≥ 120."))
        elif v >= RULES["bp_diastolic"]["warning"]:
            results.append(RuleResult(True, 2, "bp_diastolic", v, 100, f"WARNING: Diastolic BP {v} mmHg ≥ 100."))

    # Glucose
    if "glucose" in vitals:
        v = vitals["glucose"]["value"] if isinstance(vitals["glucose"], dict) else vitals["glucose"]
        if v >= RULES["glucose"]["critical_high"]:
            results.append(RuleResult(True, 1, "glucose", v, 400, f"CRITICAL: Blood glucose {v} mg/dL ≥ 400. Diabetic emergency."))
        elif v >= RULES["glucose"]["warning_high"]:
            results.append(RuleResult(True, 2, "glucose", v, 250, f"WARNING: Blood glucose {v} mg/dL ≥ 250. Hyperglycemia."))
        elif v <= RULES["glucose"]["critical_low"]:
            results.append(RuleResult(True, 1, "glucose", v, 50, f"CRITICAL: Blood glucose {v} mg/dL ≤ 50. Severe hypoglycemia."))
        elif v <= RULES["glucose"]["warning_low"]:
            results.append(RuleResult(True, 2, "glucose", v, 70, f"WARNING: Blood glucose {v} mg/dL ≤ 70. Low glucose."))

    # Heart rate
    if "heart_rate" in vitals:
        v = vitals["heart_rate"]["value"] if isinstance(vitals["heart_rate"], dict) else vitals["heart_rate"]
        if v >= RULES["heart_rate"]["critical_high"]:
            results.append(RuleResult(True, 1, "heart_rate", v, 150, f"CRITICAL: Heart rate {v} bpm ≥ 150. Tachycardia."))
        elif v >= RULES["heart_rate"]["warning_high"]:
            results.append(RuleResult(True, 2, "heart_rate", v, 120, f"WARNING: Heart rate {v} bpm ≥ 120."))
        elif v <= RULES["heart_rate"]["critical_low"]:
            results.append(RuleResult(True, 1, "heart_rate", v, 40, f"CRITICAL: Heart rate {v} bpm ≤ 40. Bradycardia."))

    # Temperature
    if "temperature" in vitals:
        v = vitals["temperature"]["value"] if isinstance(vitals["temperature"], dict) else vitals["temperature"]
        if v >= RULES["temperature"]["critical_high"]:
            results.append(RuleResult(True, 1, "temperature", v, 104.0, f"CRITICAL: Temperature {v}°F ≥ 104. High fever."))
        elif v >= RULES["temperature"]["warning_high"]:
            results.append(RuleResult(True, 2, "temperature", v, 101.5, f"WARNING: Temperature {v}°F ≥ 101.5. Fever."))
        elif v <= RULES["temperature"]["critical_low"]:
            results.append(RuleResult(True, 1, "temperature", v, 95.0, f"CRITICAL: Temperature {v}°F ≤ 95. Hypothermia."))

    # Oxygen saturation
    if "oxygen_saturation" in vitals:
        v = vitals["oxygen_saturation"]["value"] if isinstance(vitals["oxygen_saturation"], dict) else vitals["oxygen_saturation"]
        if v <= RULES["oxygen_saturation"]["critical_low"]:
            results.append(RuleResult(True, 1, "oxygen_saturation", v, 90, f"CRITICAL: SpO2 {v}% ≤ 90. Severe hypoxia."))
        elif v <= RULES["oxygen_saturation"]["warning_low"]:
            results.append(RuleResult(True, 2, "oxygen_saturation", v, 93, f"WARNING: SpO2 {v}% ≤ 93. Low oxygen."))

    # Pain level
    if "pain_level" in vitals:
        v = vitals["pain_level"]["value"] if isinstance(vitals["pain_level"], dict) else vitals["pain_level"]
        if v >= RULES["pain_level"]["critical"]:
            results.append(RuleResult(True, 1, "pain_level", v, 9, f"CRITICAL: Pain level {v}/10. Severe pain."))
        elif v >= RULES["pain_level"]["warning"]:
            results.append(RuleResult(True, 2, "pain_level", v, 7, f"WARNING: Pain level {v}/10. Significant pain."))

    results.sort(key=lambda r: r.severity)
    if results:
        logger.info("rules_triggered", count=len(results), worst_severity=results[0].severity)
    return results


def combine_decisions(rule_results: list[RuleResult], ai_decision: dict) -> dict:
    """Combine rule and AI decisions. Rules always take priority for safety.

    - If any rule fires severity 1 → always alert, regardless of AI
    - If AI anomaly_score > 0.7 but no rules fired → still escalate
    - If AI says normal but rules say alert → rules win
    """
    worst_rule_severity = min((r.severity for r in rule_results), default=99)
    ai_anomaly = ai_decision.get("anomaly_score", 0.0)
    ai_action = ai_decision.get("decision", "normal")

    # Rules always win for critical
    if worst_rule_severity == 1:
        return {
            "final_decision": "alert",
            "final_severity": 1,
            "reason": rule_results[0].message,
            "source": "rule_engine",
            "ai_agreed": ai_action in ("alert", "escalate"),
            "rule_triggers": [r.to_dict() for r in rule_results],
            "ai_decision": ai_decision,
        }

    # Rules warning
    if worst_rule_severity == 2:
        severity = 1 if ai_anomaly > 0.7 else 2
        return {
            "final_decision": "alert",
            "final_severity": severity,
            "reason": rule_results[0].message + (f" (AI anomaly score: {ai_anomaly})" if ai_anomaly > 0.4 else ""),
            "source": "rule_engine+ai" if ai_anomaly > 0.4 else "rule_engine",
            "ai_agreed": ai_action in ("alert", "escalate"),
            "rule_triggers": [r.to_dict() for r in rule_results],
            "ai_decision": ai_decision,
        }

    # No rules but AI detects something
    if ai_anomaly > 0.7:
        return {
            "final_decision": "alert",
            "final_severity": 2,
            "reason": ai_decision.get("reason", "AI detected anomaly pattern"),
            "source": "ai_engine",
            "ai_agreed": True,
            "rule_triggers": [],
            "ai_decision": ai_decision,
        }

    if ai_anomaly > 0.4:
        return {
            "final_decision": "monitor",
            "final_severity": 3,
            "reason": ai_decision.get("reason", "Mild anomaly detected"),
            "source": "ai_engine",
            "ai_agreed": True,
            "rule_triggers": [],
            "ai_decision": ai_decision,
        }

    # Everything normal
    return {
        "final_decision": "normal",
        "final_severity": 3,
        "reason": "All vitals within normal range. No anomalies detected.",
        "source": "combined",
        "ai_agreed": True,
        "rule_triggers": [],
        "ai_decision": ai_decision,
    }
