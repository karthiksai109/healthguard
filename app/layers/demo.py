"""HealthGuard — Demo Data Loader

Pre-loads realistic patient data for live hackathon demo.
Triggers autonomous alerts on stage without human intervention.
All data is synthetic — no real PHI.
"""
import time
import structlog
from app.core.database import Database
from app.layers import ingestion
from app.layers.agent import HealthGuardAgent

logger = structlog.get_logger()


DEMO_PATIENTS = [
    {"id": "demo-patient-001", "name": "Maria Santos"},
    {"id": "demo-patient-002", "name": "James Wilson"},
    {"id": "demo-patient-003", "name": "Aisha Patel"},
]

DEMO_VITALS = {
    "demo-patient-001": [
        {"metric": "bp_systolic", "value": 138, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 88, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 78, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 145, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 98.6, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 97, "unit": "%", "source": "pulse_ox"},
        {"metric": "bp_systolic", "value": 155, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 95, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 85, "unit": "bpm", "source": "monitor"},
        {"metric": "bp_systolic", "value": 185, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 115, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 98, "unit": "bpm", "source": "monitor"},
    ],
    "demo-patient-002": [
        {"metric": "bp_systolic", "value": 125, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 80, "unit": "mmHg", "source": "monitor"},
        {"metric": "glucose", "value": 110, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "heart_rate", "value": 72, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 98.4, "unit": "F", "source": "thermometer"},
        {"metric": "glucose", "value": 85, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "glucose", "value": 62, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "heart_rate", "value": 95, "unit": "bpm", "source": "monitor"},
    ],
    "demo-patient-003": [
        {"metric": "bp_systolic", "value": 130, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 82, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 80, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 99.8, "unit": "F", "source": "thermometer"},
        {"metric": "pain_level", "value": 4, "unit": "/10", "source": "self_report"},
        {"metric": "pain_level", "value": 6, "unit": "/10", "source": "self_report"},
        {"metric": "temperature", "value": 100.8, "unit": "F", "source": "thermometer"},
        {"metric": "pain_level", "value": 9, "unit": "/10", "source": "self_report"},
        {"metric": "temperature", "value": 102.1, "unit": "F", "source": "thermometer"},
        {"metric": "heart_rate", "value": 110, "unit": "bpm", "source": "monitor"},
    ],
}

DEMO_TEXT_EVENTS = [
    {
        "patient_id": "demo-patient-001",
        "text": (
            "I've been having really bad headaches for the past two days. "
            "My vision gets blurry sometimes and I feel dizzy when I stand up. "
            "I took my blood pressure medication this morning but I'm not sure it's working."
        ),
    },
    {
        "patient_id": "demo-patient-002",
        "text": (
            "I'm feeling shaky and sweaty. I think my blood sugar is dropping again. "
            "I forgot to eat breakfast and I took my insulin at the usual time. "
            "My hands are trembling and I feel lightheaded."
        ),
    },
    {
        "patient_id": "demo-patient-003",
        "text": (
            "The pain in my lower right side is getting worse, now it's a 9 out of 10. "
            "I also have a fever and I feel nauseous. I can barely move without sharp pain. "
            "It started two days ago as a dull ache but now it's unbearable."
        ),
    },
]


def load_demo_data(agent: HealthGuardAgent):
    """Load all demo data into the system. Call once on startup in demo mode."""
    logger.info("demo_loading_started", patients=len(DEMO_PATIENTS))

    for p in DEMO_PATIENTS:
        pid, access_key = agent.db.create_patient(p["name"], patient_id=p["id"])
        logger.info("demo_patient_created", id=p["id"], name=p["name"], access_key=access_key)

    for patient_id, vitals in DEMO_VITALS.items():
        for v in vitals:
            agent.db.record_vital(
                patient_id=patient_id,
                metric_type=v["metric"],
                value=v["value"],
                unit=v["unit"],
                source=v["source"],
            )
        logger.info("demo_vitals_loaded", patient=patient_id, count=len(vitals))

    agent.db.audit({
        "type": "demo_data_loaded",
        "patients": len(DEMO_PATIENTS),
        "vitals": sum(len(v) for v in DEMO_VITALS.values()),
    })

    logger.info("demo_data_loaded", patients=len(DEMO_PATIENTS))


def trigger_demo_events(agent: HealthGuardAgent):
    """Queue demo text events to trigger the full pipeline.
    These will be processed by the autonomous loop and fire alerts."""
    logger.info("demo_events_triggering", count=len(DEMO_TEXT_EVENTS))

    for event in DEMO_TEXT_EVENTS:
        item = ingestion.ingest_text(event["text"], event["patient_id"])
        agent.event_queue.push(item)
        logger.info("demo_event_queued", patient=event["patient_id"], type="text")
        time.sleep(1)

    logger.info("demo_events_queued", total=len(DEMO_TEXT_EVENTS))
