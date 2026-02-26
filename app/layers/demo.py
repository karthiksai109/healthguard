"""HealthGuard — Demo Data Loader

Pre-loads realistic patient data for live hackathon demo.
20 patients, 10 doctors, chat histories, vitals, reports.
All data is synthetic — no real PHI.
"""
import time
import hashlib
import structlog
from app.core.database import Database
from app.layers import ingestion
from app.layers.agent import HealthGuardAgent

logger = structlog.get_logger()

# ── 20 Demo Patients ────────────────────────────────────────────────
DEMO_PATIENTS = [
    {"id": "demo-patient-001", "name": "Maria Santos"},
    {"id": "demo-patient-002", "name": "James Wilson"},
    {"id": "demo-patient-003", "name": "Aisha Patel"},
    {"id": "demo-patient-004", "name": "Robert Chen"},
    {"id": "demo-patient-005", "name": "Elena Rodriguez"},
    {"id": "demo-patient-006", "name": "David Kim"},
    {"id": "demo-patient-007", "name": "Sarah Johnson"},
    {"id": "demo-patient-008", "name": "Michael Brown"},
    {"id": "demo-patient-009", "name": "Fatima Al-Hassan"},
    {"id": "demo-patient-010", "name": "Thomas Anderson"},
    {"id": "demo-patient-011", "name": "Priya Sharma"},
    {"id": "demo-patient-012", "name": "John O'Brien"},
    {"id": "demo-patient-013", "name": "Lisa Chang"},
    {"id": "demo-patient-014", "name": "Ahmed Hassan"},
    {"id": "demo-patient-015", "name": "Jennifer Martinez"},
    {"id": "demo-patient-016", "name": "Wei Zhang"},
    {"id": "demo-patient-017", "name": "Rachel Green"},
    {"id": "demo-patient-018", "name": "Carlos Mendoza"},
    {"id": "demo-patient-019", "name": "Hannah Baker"},
    {"id": "demo-patient-020", "name": "Raj Krishnamurthy"},
]

# ── 10 Demo Doctors ─────────────────────────────────────────────────
DEMO_DOCTORS = [
    {"name": "Dr. Amanda Foster", "email": "amanda.foster@healthguard.ai", "spec": "Cardiology", "rate": "$150/hr", "bio": "15 years experience in interventional cardiology. Board certified. Specializes in hypertension management and heart failure."},
    {"name": "Dr. Benjamin Park", "email": "benjamin.park@healthguard.ai", "spec": "Endocrinology", "rate": "$140/hr", "bio": "Expert in diabetes management, thyroid disorders, and metabolic syndrome. Published researcher in insulin resistance."},
    {"name": "Dr. Catherine Wright", "email": "catherine.wright@healthguard.ai", "spec": "General Surgery", "rate": "$200/hr", "bio": "Fellowship-trained surgeon specializing in minimally invasive procedures. 12 years of surgical experience."},
    {"name": "Dr. Daniel Okafor", "email": "daniel.okafor@healthguard.ai", "spec": "Neurology", "rate": "$160/hr", "bio": "Neurologist specializing in migraines, epilepsy, and neurodegenerative diseases. Research focus on AI-assisted diagnostics."},
    {"name": "Dr. Emily Tanaka", "email": "emily.tanaka@healthguard.ai", "spec": "Pediatrics", "rate": "$120/hr", "bio": "Pediatrician with expertise in childhood asthma, allergies, and developmental disorders. 10 years practice."},
    {"name": "Dr. Farhan Malik", "email": "farhan.malik@healthguard.ai", "spec": "Orthopedics", "rate": "$175/hr", "bio": "Sports medicine and joint replacement specialist. Team physician for multiple professional sports organizations."},
    {"name": "Dr. Grace Liu", "email": "grace.liu@healthguard.ai", "spec": "Dermatology", "rate": "$145/hr", "bio": "Dermatologist specializing in skin cancer screening, wound healing, and cosmetic procedures. Teledermatology pioneer."},
    {"name": "Dr. Hassan Ibrahim", "email": "hassan.ibrahim@healthguard.ai", "spec": "Pulmonology", "rate": "$155/hr", "bio": "Pulmonologist focusing on COPD, asthma, and sleep apnea. Critical care certified. COVID long-haul specialist."},
    {"name": "Dr. Isabella Rossi", "email": "isabella.rossi@healthguard.ai", "spec": "Psychiatry", "rate": "$130/hr", "bio": "Psychiatrist with focus on anxiety, depression, and PTSD. Certified in cognitive behavioral therapy and psychopharmacology."},
    {"name": "Dr. Kevin Nguyen", "email": "kevin.nguyen@healthguard.ai", "spec": "Emergency Medicine", "rate": "$180/hr", "bio": "ER physician with 20 years experience in acute care, trauma, and triage. Disaster medicine certified."},
]

# ── Vitals for all 20 patients ──────────────────────────────────────
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
    "demo-patient-004": [
        {"metric": "bp_systolic", "value": 170, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 105, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 92, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 220, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "oxygen_saturation", "value": 94, "unit": "%", "source": "pulse_ox"},
        {"metric": "temperature", "value": 98.9, "unit": "F", "source": "thermometer"},
    ],
    "demo-patient-005": [
        {"metric": "bp_systolic", "value": 118, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 75, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 68, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 95, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 98.4, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 99, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-006": [
        {"metric": "bp_systolic", "value": 142, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 90, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 88, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 310, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 99.1, "unit": "F", "source": "thermometer"},
        {"metric": "glucose", "value": 285, "unit": "mg/dL", "source": "glucometer"},
    ],
    "demo-patient-007": [
        {"metric": "bp_systolic", "value": 105, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 65, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 110, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 101.5, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 93, "unit": "%", "source": "pulse_ox"},
        {"metric": "pain_level", "value": 7, "unit": "/10", "source": "self_report"},
    ],
    "demo-patient-008": [
        {"metric": "bp_systolic", "value": 135, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 85, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 76, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 130, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "oxygen_saturation", "value": 96, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-009": [
        {"metric": "bp_systolic", "value": 160, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 100, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 95, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 180, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 98.7, "unit": "F", "source": "thermometer"},
        {"metric": "pain_level", "value": 5, "unit": "/10", "source": "self_report"},
    ],
    "demo-patient-010": [
        {"metric": "bp_systolic", "value": 120, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 78, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 65, "unit": "bpm", "source": "monitor"},
        {"metric": "oxygen_saturation", "value": 88, "unit": "%", "source": "pulse_ox"},
        {"metric": "temperature", "value": 98.2, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 85, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-011": [
        {"metric": "bp_systolic", "value": 115, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 72, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 130, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 98.6, "unit": "F", "source": "thermometer"},
        {"metric": "pain_level", "value": 8, "unit": "/10", "source": "self_report"},
    ],
    "demo-patient-012": [
        {"metric": "bp_systolic", "value": 148, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 92, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 82, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 155, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "oxygen_saturation", "value": 95, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-013": [
        {"metric": "bp_systolic", "value": 110, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 70, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 72, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 103.2, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 91, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-014": [
        {"metric": "bp_systolic", "value": 190, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 120, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 105, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 200, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 98.8, "unit": "F", "source": "thermometer"},
    ],
    "demo-patient-015": [
        {"metric": "bp_systolic", "value": 122, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 78, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 70, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 100, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 98.5, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 98, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-016": [
        {"metric": "bp_systolic", "value": 155, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 98, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 90, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 250, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 99.0, "unit": "F", "source": "thermometer"},
    ],
    "demo-patient-017": [
        {"metric": "bp_systolic", "value": 108, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 68, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 62, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 98.3, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 99, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-018": [
        {"metric": "bp_systolic", "value": 145, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 94, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 88, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 175, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "pain_level", "value": 6, "unit": "/10", "source": "self_report"},
    ],
    "demo-patient-019": [
        {"metric": "bp_systolic", "value": 100, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 60, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 55, "unit": "bpm", "source": "monitor"},
        {"metric": "temperature", "value": 97.5, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 97, "unit": "%", "source": "pulse_ox"},
    ],
    "demo-patient-020": [
        {"metric": "bp_systolic", "value": 165, "unit": "mmHg", "source": "monitor"},
        {"metric": "bp_diastolic", "value": 102, "unit": "mmHg", "source": "monitor"},
        {"metric": "heart_rate", "value": 100, "unit": "bpm", "source": "monitor"},
        {"metric": "glucose", "value": 280, "unit": "mg/dL", "source": "glucometer"},
        {"metric": "temperature", "value": 99.5, "unit": "F", "source": "thermometer"},
        {"metric": "oxygen_saturation", "value": 92, "unit": "%", "source": "pulse_ox"},
        {"metric": "pain_level", "value": 7, "unit": "/10", "source": "self_report"},
    ],
}

# ── Chat histories for patients (pre-loaded conversations) ──────────
DEMO_CHATS = {
    "demo-patient-001": [
        ("user", "I have been having bad headaches for the past two days and my vision gets blurry sometimes."),
        ("assistant", "Headaches with blurry vision alongside your elevated BP of 185/115 is concerning. This could indicate a hypertensive crisis. Please monitor your blood pressure closely, take your prescribed medication, and if vision changes persist, seek emergency care immediately."),
        ("user", "I took my blood pressure medication this morning but I'm not sure it's working."),
        ("assistant", "If your BP remains above 180/110 even after medication, please go to the ER. Do not double your dose without doctor approval. Meanwhile, avoid caffeine, rest in a quiet room, and recheck BP in 30 minutes. I recommend consulting Dr. Foster (Cardiology) through our platform."),
    ],
    "demo-patient-002": [
        ("user", "I'm feeling shaky and sweaty. I think my blood sugar is dropping again."),
        ("assistant", "With your glucose at 62 mg/dL, you're experiencing hypoglycemia. Eat 15g of fast-acting carbs immediately — juice, glucose tablets, or candy. Recheck in 15 minutes. Your insulin timing may need adjustment. Please consult Dr. Park (Endocrinology)."),
        ("user", "I forgot to eat breakfast and took insulin at the usual time."),
        ("assistant", "That explains the drop. Always eat before or shortly after insulin. Keep glucose tablets handy. Your pattern shows glucose swings from 110 to 62 mg/dL. I recommend setting meal reminders and discussing dosage adjustment with your endocrinologist."),
    ],
    "demo-patient-003": [
        ("user", "The pain in my lower right side is getting worse, now it's a 9 out of 10."),
        ("assistant", "Lower right abdominal pain at 9/10 with fever (102.1F) and nausea strongly suggests appendicitis. This is urgent — please go to the emergency room immediately. Do not eat or drink anything. You may need surgical evaluation."),
        ("user", "It started two days ago as a dull ache but now it's unbearable."),
        ("assistant", "Progressive pain migrating to the lower right with fever escalation from 99.8F to 102.1F is a classic appendicitis presentation. Call 911 or have someone drive you to the ER now. Time is critical to prevent rupture. I've flagged this as an emergency alert."),
    ],
    "demo-patient-004": [
        ("user", "I've been having chest tightness and shortness of breath when climbing stairs."),
        ("assistant", "With your BP at 170/105 and glucose at 220 mg/dL, chest tightness during exertion is concerning for cardiovascular issues. Please rest, avoid physical exertion, and schedule an urgent appointment with a cardiologist. If chest pain becomes severe or radiates to arm/jaw, call 911."),
    ],
    "demo-patient-006": [
        ("user", "I'm very thirsty all the time and urinating frequently. My vision is getting blurry."),
        ("assistant", "Your glucose readings of 310 and 285 mg/dL are dangerously high, indicating poorly controlled diabetes. Excessive thirst, frequent urination, and blurry vision are classic hyperglycemia symptoms. Please seek medical attention today. You may need insulin adjustment or emergency intervention."),
        ("user", "I've been taking my metformin but it doesn't seem to help anymore."),
        ("assistant", "Metformin alone may be insufficient at these glucose levels. You likely need combination therapy or insulin. Do not stop metformin without doctor guidance. Stay hydrated, avoid sugary foods, and see Dr. Park (Endocrinology) urgently. Watch for signs of diabetic ketoacidosis: fruity breath, confusion, vomiting."),
    ],
    "demo-patient-007": [
        ("user", "I have a high fever and I'm coughing up greenish mucus. My chest hurts when I breathe."),
        ("assistant", "Fever of 101.5F with productive cough and pleuritic chest pain suggests pneumonia. Your oxygen saturation at 93% is below normal. Please see a doctor today — you may need antibiotics and a chest X-ray. If breathing becomes difficult, go to the ER. I recommend Dr. Ibrahim (Pulmonology)."),
    ],
    "demo-patient-009": [
        ("user", "I get terrible headaches almost every day now. The pain is behind my right eye."),
        ("assistant", "Daily retro-orbital headaches with your BP at 160/100 could be tension headaches, migraines, or hypertension-related. Your elevated blood pressure needs attention. Keep a headache diary noting triggers, duration, and intensity. I recommend consulting Dr. Okafor (Neurology) and Dr. Foster (Cardiology)."),
    ],
    "demo-patient-010": [
        ("user", "I get out of breath just walking to the kitchen. My lips sometimes turn bluish."),
        ("assistant", "Your oxygen saturation at 85% is critically low. Cyanosis (bluish lips) with dyspnea at rest indicates severe respiratory compromise. Please call 911 or go to the ER immediately. You may need supplemental oxygen and urgent pulmonary evaluation."),
    ],
    "demo-patient-011": [
        ("user", "My heart feels like it's racing all the time. Sometimes I feel like I might faint."),
        ("assistant", "Heart rate of 130 bpm at rest is tachycardia and needs evaluation. Combined with pain at 8/10 and near-syncope, this could indicate an arrhythmia, thyroid issue, or cardiac condition. Please go to urgent care or ER for an ECG. Avoid caffeine and stimulants."),
    ],
    "demo-patient-014": [
        ("user", "I went to the pharmacy and the BP machine showed 190/120. I have a terrible headache."),
        ("assistant", "BP of 190/120 with headache is a hypertensive emergency. Go to the nearest ER immediately. Do not drive yourself. This level of blood pressure can cause stroke, heart attack, or organ damage. Take any prescribed BP medications now if you haven't already. Call 911 if you develop chest pain, vision changes, or confusion."),
    ],
    "demo-patient-020": [
        ("user", "I'm diabetic and my feet are tingling and numb. I also found a small sore on my toe that won't heal."),
        ("assistant", "Peripheral neuropathy (tingling/numbness) and non-healing wounds are serious diabetic complications. Your glucose at 280 mg/dL is very high, impairing wound healing. Keep the sore clean and dry. See a doctor within 24 hours — non-healing foot wounds can lead to infection. I recommend Dr. Park (Endocrinology) and wound care assessment."),
        ("user", "The sore has been there for about a week and it's getting slightly red around the edges."),
        ("assistant", "Redness around the wound suggests early infection. Do NOT ignore this. Clean with saline, apply antibiotic ointment, and keep it covered. Elevate your foot. See a doctor TODAY. With your glucose levels, infection can progress rapidly. You may need oral antibiotics and aggressive glucose management."),
    ],
}

# ── Text events to trigger autonomous pipeline ──────────────────────
DEMO_TEXT_EVENTS = [
    {"patient_id": "demo-patient-001", "text": "I've been having really bad headaches for the past two days. My vision gets blurry sometimes and I feel dizzy when I stand up. I took my blood pressure medication this morning but I'm not sure it's working."},
    {"patient_id": "demo-patient-002", "text": "I'm feeling shaky and sweaty. I think my blood sugar is dropping again. I forgot to eat breakfast and I took my insulin at the usual time. My hands are trembling and I feel lightheaded."},
    {"patient_id": "demo-patient-003", "text": "The pain in my lower right side is getting worse, now it's a 9 out of 10. I also have a fever and I feel nauseous. It started two days ago as a dull ache but now it's unbearable."},
    {"patient_id": "demo-patient-004", "text": "Chest tightness and shortness of breath when climbing stairs. Been happening for 3 days."},
    {"patient_id": "demo-patient-010", "text": "I get out of breath just walking to the kitchen. My lips sometimes turn bluish. Very scared."},
    {"patient_id": "demo-patient-014", "text": "Pharmacy BP machine showed 190/120. I have a terrible headache and feel dizzy."},
]


def load_demo_data(agent: HealthGuardAgent):
    """Load all demo data: 20 patients, 10 doctors, chats, vitals."""
    logger.info("demo_loading_started", patients=len(DEMO_PATIENTS), doctors=len(DEMO_DOCTORS))

    # ── Create 20 patients ──
    patient_keys = {}
    for p in DEMO_PATIENTS:
        pid, access_key = agent.db.create_patient(p["name"], patient_id=p["id"])
        patient_keys[p["id"]] = access_key
        logger.info("demo_patient_created", id=p["id"], name=p["name"], access_key=access_key)

    # ── Load vitals ──
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

    # ── Load chat histories ──
    for patient_id, chats in DEMO_CHATS.items():
        for role, content in chats:
            agent.db.save_chat_message(patient_id, role, content)
        logger.info("demo_chats_loaded", patient=patient_id, messages=len(chats))

    # ── Create 10 doctors ──
    doctor_keys = {}
    cert_hash = hashlib.sha256(b"demo_certificate_healthguard").hexdigest()
    for doc in DEMO_DOCTORS:
        did, access_key = agent.db.create_doctor(
            doc["name"], doc["email"], doc["spec"],
            doc["rate"], cert_hash, "certificate.pdf", doc["bio"]
        )
        agent.db.verify_doctor(did)
        doctor_keys[doc["name"]] = {"id": did, "access_key": access_key, "spec": doc["spec"]}
        logger.info("demo_doctor_created", name=doc["name"], spec=doc["spec"], access_key=access_key)

    agent.db.audit({
        "type": "demo_data_loaded",
        "patients": len(DEMO_PATIENTS),
        "doctors": len(DEMO_DOCTORS),
        "vitals": sum(len(v) for v in DEMO_VITALS.values()),
        "chats": sum(len(c) for c in DEMO_CHATS.values()),
    })

    logger.info("demo_data_loaded", patients=len(DEMO_PATIENTS), doctors=len(DEMO_DOCTORS))


def trigger_demo_events(agent: HealthGuardAgent):
    """Queue demo text events to trigger the full pipeline."""
    logger.info("demo_events_triggering", count=len(DEMO_TEXT_EVENTS))
    for event in DEMO_TEXT_EVENTS:
        item = ingestion.ingest_text(event["text"], event["patient_id"])
        agent.event_queue.push(item)
        logger.info("demo_event_queued", patient=event["patient_id"], type="text")
        time.sleep(1)
    logger.info("demo_events_queued", total=len(DEMO_TEXT_EVENTS))
