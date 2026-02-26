"""HealthGuard — Persistence Layer

SQLite database with AES-256-GCM encryption for sensitive fields.
Tables: patients, vitals, logs, alerts
Audit log: append-only JSON lines file.
"""
import os
import json
import uuid
import time
import sqlite3
import hashlib
import base64
from datetime import datetime
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import structlog

logger = structlog.get_logger()


class EncryptionEngine:
    """AES-256-GCM encryption. Key derived from passphrase via PBKDF2."""

    def __init__(self, passphrase: str, salt: str):
        salt_bytes = salt.encode("utf-8")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=1_000,
        )
        self._key = kdf.derive(passphrase.encode("utf-8"))
        self._aesgcm = AESGCM(self._key)
        logger.info("encryption_initialized", key_hash=hashlib.sha256(self._key).hexdigest()[:12])

    def encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ct).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        raw = base64.b64decode(ciphertext)
        nonce, ct = raw[:12], raw[12:]
        return self._aesgcm.decrypt(nonce, ct, None).decode("utf-8")


class Database:
    """SQLite persistence with encrypted sensitive fields."""

    def __init__(self, data_dir: str, encryption_salt: str):
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "healthguard.db")
        self.audit_path = os.path.join(data_dir, "audit.jsonl")
        self.encryption = EncryptionEngine("healthguard_patient_key", encryption_salt)
        self._init_tables()
        logger.info("database_initialized", path=self.db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name_encrypted TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    access_key TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS vitals (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT DEFAULT '',
                    note_encrypted TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    summary_encrypted TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    model_used TEXT DEFAULT '',
                    anomaly_score REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    webhook_response TEXT DEFAULT '',
                    tts_generated INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                );
                CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vitals(patient_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_logs_patient ON logs(patient_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id, timestamp);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_encrypted TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                );
                CREATE INDEX IF NOT EXISTS idx_chat_patient ON chat_messages(patient_id, timestamp);
                CREATE TABLE IF NOT EXISTS doctors (
                    id TEXT PRIMARY KEY,
                    name_encrypted TEXT NOT NULL,
                    email TEXT NOT NULL,
                    specialization TEXT NOT NULL,
                    pay_rate TEXT NOT NULL,
                    certificate_hash TEXT NOT NULL,
                    certificate_filename TEXT NOT NULL,
                    bio_encrypted TEXT DEFAULT '',
                    verified INTEGER DEFAULT 0,
                    access_key TEXT UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS consultations (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'requested',
                    problem_description_encrypted TEXT DEFAULT '',
                    patient_approved INTEGER DEFAULT 0,
                    doctor_notes_encrypted TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id),
                    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
                );
                CREATE INDEX IF NOT EXISTS idx_consult_patient ON consultations(patient_id, status);
                CREATE INDEX IF NOT EXISTS idx_consult_doctor ON consultations(doctor_id, status);
            """)
        # Migrate: add access_key column if missing
        try:
            with self._conn() as conn:
                conn.execute("ALTER TABLE patients ADD COLUMN access_key TEXT UNIQUE")
        except Exception:
            pass

    # ── Patients ──────────────────────────────────────────────────────
    def _generate_access_key(self) -> str:
        """Generate a unique 6-char alphanumeric access key."""
        import random, string
        with self._conn() as conn:
            for _ in range(100):
                key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                existing = conn.execute("SELECT id FROM patients WHERE access_key = ?", (key,)).fetchone()
                if not existing:
                    return key
        return str(uuid.uuid4())[:8].upper()

    def create_patient(self, name: str, patient_id: str = None) -> tuple:
        """Create patient and return (patient_id, access_key)."""
        pid = patient_id or str(uuid.uuid4())
        name_enc = self.encryption.encrypt(name)
        key_hash = hashlib.sha256(pid.encode()).hexdigest()[:16]
        access_key = self._generate_access_key()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO patients (id, name_encrypted, created_at, key_hash, access_key) VALUES (?, ?, ?, ?, ?)",
                (pid, name_enc, datetime.utcnow().isoformat(), key_hash, access_key),
            )
        return pid, access_key

    def login_patient(self, access_key: str) -> dict | None:
        """Login with access key. Returns patient dict or None."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM patients WHERE access_key = ?", (access_key.strip().upper(),)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": self.encryption.decrypt(row["name_encrypted"]),
            "created_at": row["created_at"],
            "access_key": row["access_key"],
        }

    def get_patient(self, patient_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": self.encryption.decrypt(row["name_encrypted"]),
            "created_at": row["created_at"],
        }

    def list_patients(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, created_at FROM patients ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ── Vitals ────────────────────────────────────────────────────────
    def record_vital(self, patient_id: str, metric_type: str, value: float, unit: str = "", note: str = "", source: str = "manual") -> str:
        vid = str(uuid.uuid4())
        note_enc = self.encryption.encrypt(note) if note else ""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO vitals (id, patient_id, metric_type, value, unit, note_encrypted, timestamp, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (vid, patient_id, metric_type, value, unit, note_enc, datetime.utcnow().isoformat(), source),
            )
        return vid

    def get_vitals(self, patient_id: str, days: int = 7, metric_type: str = None) -> list[dict]:
        query = "SELECT * FROM vitals WHERE patient_id = ? AND timestamp >= datetime('now', ?) ORDER BY timestamp DESC"
        params = [patient_id, f"-{days} days"]
        if metric_type:
            query = "SELECT * FROM vitals WHERE patient_id = ? AND metric_type = ? AND timestamp >= datetime('now', ?) ORDER BY timestamp DESC"
            params = [patient_id, metric_type, f"-{days} days"]
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("note_encrypted"):
                try:
                    d["note"] = self.encryption.decrypt(d["note_encrypted"])
                except Exception:
                    d["note"] = ""
            del d["note_encrypted"]
            results.append(d)
        return results

    def get_latest_vitals(self, patient_id: str) -> dict:
        """Get latest value for each metric type."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT v.metric_type, v.value, v.unit, v.timestamp
                FROM vitals v
                INNER JOIN (
                    SELECT metric_type, MAX(timestamp) as max_ts
                    FROM vitals WHERE patient_id = ?
                    GROUP BY metric_type
                ) latest ON v.metric_type = latest.metric_type AND v.timestamp = latest.max_ts
                WHERE v.patient_id = ?
                ORDER BY v.timestamp DESC
            """, (patient_id, patient_id)).fetchall()
        return {r["metric_type"]: {"value": r["value"], "unit": r["unit"], "timestamp": r["timestamp"]} for r in rows}

    # ── Logs ──────────────────────────────────────────────────────────
    def record_log(self, patient_id: str, session_id: str, input_type: str, summary: str,
                   decision: str, reason: str, action_taken: str, model_used: str = "", anomaly_score: float = 0.0) -> str:
        lid = str(uuid.uuid4())
        summary_enc = self.encryption.encrypt(summary)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO logs (id, patient_id, session_id, input_type, summary_encrypted, decision, reason, action_taken, model_used, anomaly_score, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (lid, patient_id, session_id, input_type, summary_enc, decision, reason, action_taken, model_used, anomaly_score, datetime.utcnow().isoformat()),
            )
        return lid

    def get_logs(self, patient_id: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM logs WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["summary"] = self.encryption.decrypt(d["summary_encrypted"])
            except Exception:
                d["summary"] = "[decryption failed]"
            del d["summary_encrypted"]
            results.append(d)
        return results

    # ── Alerts ────────────────────────────────────────────────────────
    def record_alert(self, patient_id: str, severity: int, message: str,
                     action_taken: str, webhook_response: str = "", tts_generated: bool = False) -> str:
        aid = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO alerts (id, patient_id, severity, message, action_taken, webhook_response, tts_generated, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (aid, patient_id, severity, message, action_taken, webhook_response, int(tts_generated), datetime.utcnow().isoformat()),
            )
        return aid

    def get_alerts(self, patient_id: str = None, limit: int = 50) -> list[dict]:
        if patient_id:
            query = "SELECT * FROM alerts WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?"
            params = (patient_id, limit)
        else:
            query = "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?"
            params = (limit,)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Audit Log (append-only, immutable) ────────────────────────────
    def audit(self, entry: dict):
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        entry["action_id"] = str(uuid.uuid4())[:8]
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        if not os.path.exists(self.audit_path):
            return []
        lines = open(self.audit_path).readlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                pass
        return entries

    # ── Stats ─────────────────────────────────────────────────────────
    # ── Chat Messages ─────────────────────────────────────────────────
    def save_chat_message(self, patient_id: str, role: str, content: str) -> str:
        mid = str(uuid.uuid4())
        content_enc = self.encryption.encrypt(content)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_messages (id, patient_id, role, content_encrypted, timestamp) VALUES (?, ?, ?, ?, ?)",
                (mid, patient_id, role, content_enc, datetime.utcnow().isoformat()),
            )
        return mid

    def get_chat_history(self, patient_id: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["content"] = self.encryption.decrypt(d["content_encrypted"])
            except Exception:
                d["content"] = "[decryption failed]"
            del d["content_encrypted"]
            results.append(d)
        results.reverse()
        return results

    def clear_chat(self, patient_id: str) -> int:
        with self._conn() as conn:
            count = conn.execute("DELETE FROM chat_messages WHERE patient_id = ?", (patient_id,)).rowcount
        return count

    # ── Doctors ─────────────────────────────────────────────────────────
    def _generate_doctor_access_key(self) -> str:
        import random, string
        with self._conn() as conn:
            for _ in range(100):
                key = 'DR' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
                existing = conn.execute("SELECT id FROM doctors WHERE access_key = ?", (key,)).fetchone()
                if not existing:
                    return key
        return 'DR' + str(uuid.uuid4())[:6].upper()

    def create_doctor(self, name: str, email: str, specialization: str,
                      pay_rate: str, certificate_hash: str, certificate_filename: str,
                      bio: str = "") -> tuple:
        did = str(uuid.uuid4())
        name_enc = self.encryption.encrypt(name)
        bio_enc = self.encryption.encrypt(bio) if bio else ""
        access_key = self._generate_doctor_access_key()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO doctors (id, name_encrypted, email, specialization, pay_rate,
                   certificate_hash, certificate_filename, bio_encrypted, verified, access_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (did, name_enc, email.lower().strip(), specialization, pay_rate,
                 certificate_hash, certificate_filename, bio_enc, access_key,
                 datetime.utcnow().isoformat()),
            )
        return did, access_key

    def login_doctor(self, access_key: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM doctors WHERE access_key = ?", (access_key.strip().upper(),)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": self.encryption.decrypt(row["name_encrypted"]),
            "email": row["email"],
            "specialization": row["specialization"],
            "pay_rate": row["pay_rate"],
            "verified": bool(row["verified"]),
            "access_key": row["access_key"],
            "created_at": row["created_at"],
        }

    def verify_doctor(self, doctor_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("UPDATE doctors SET verified = 1 WHERE id = ?", (doctor_id,))
        return True

    def list_doctors(self, specialization: str = None, verified_only: bool = True) -> list[dict]:
        query = "SELECT * FROM doctors"
        params = []
        conditions = []
        if verified_only:
            conditions.append("verified = 1")
        if specialization:
            conditions.append("specialization = ?")
            params.append(specialization)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = {
                "id": r["id"],
                "name": self.encryption.decrypt(r["name_encrypted"]),
                "email": r["email"],
                "specialization": r["specialization"],
                "pay_rate": r["pay_rate"],
                "verified": bool(r["verified"]),
                "created_at": r["created_at"],
            }
            if r["bio_encrypted"]:
                try:
                    d["bio"] = self.encryption.decrypt(r["bio_encrypted"])
                except Exception:
                    d["bio"] = ""
            else:
                d["bio"] = ""
            results.append(d)
        return results

    def get_doctor(self, doctor_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM doctors WHERE id = ?", (doctor_id,)).fetchone()
        if not row:
            return None
        d = {
            "id": row["id"],
            "name": self.encryption.decrypt(row["name_encrypted"]),
            "email": row["email"],
            "specialization": row["specialization"],
            "pay_rate": row["pay_rate"],
            "verified": bool(row["verified"]),
            "access_key": row["access_key"],
            "created_at": row["created_at"],
        }
        if row["bio_encrypted"]:
            try:
                d["bio"] = self.encryption.decrypt(row["bio_encrypted"])
            except Exception:
                d["bio"] = ""
        else:
            d["bio"] = ""
        return d

    # ── Consultations ──────────────────────────────────────────────────
    def create_consultation(self, patient_id: str, doctor_id: str, problem: str) -> str:
        cid = str(uuid.uuid4())
        problem_enc = self.encryption.encrypt(problem) if problem else ""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO consultations (id, patient_id, doctor_id, status,
                   problem_description_encrypted, patient_approved, created_at, updated_at)
                   VALUES (?, ?, ?, 'requested', ?, 0, ?, ?)""",
                (cid, patient_id, doctor_id, problem_enc, now, now),
            )
        return cid

    def get_consultations_for_patient(self, patient_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT c.*, d.name_encrypted as doc_name_enc, d.specialization, d.email as doc_email, d.pay_rate "
                "FROM consultations c JOIN doctors d ON c.doctor_id = d.id "
                "WHERE c.patient_id = ? ORDER BY c.updated_at DESC",
                (patient_id,),
            ).fetchall()
        return self._format_consultations(rows)

    def get_consultations_for_doctor(self, doctor_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT c.*, d.name_encrypted as doc_name_enc, d.specialization, d.email as doc_email, d.pay_rate "
                "FROM consultations c JOIN doctors d ON c.doctor_id = d.id "
                "WHERE c.doctor_id = ? ORDER BY c.updated_at DESC",
                (doctor_id,),
            ).fetchall()
        return self._format_consultations(rows)

    def _format_consultations(self, rows) -> list[dict]:
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["doctor_name"] = self.encryption.decrypt(d["doc_name_enc"])
            except Exception:
                d["doctor_name"] = "Doctor"
            if d.get("problem_description_encrypted"):
                try:
                    d["problem"] = self.encryption.decrypt(d["problem_description_encrypted"])
                except Exception:
                    d["problem"] = "[encrypted]"
            else:
                d["problem"] = ""
            if d.get("doctor_notes_encrypted"):
                try:
                    d["doctor_notes"] = self.encryption.decrypt(d["doctor_notes_encrypted"])
                except Exception:
                    d["doctor_notes"] = ""
            else:
                d["doctor_notes"] = ""
            # Clean up internal fields
            for key in ["doc_name_enc", "problem_description_encrypted", "doctor_notes_encrypted"]:
                d.pop(key, None)
            results.append(d)
        return results

    def update_consultation_status(self, consultation_id: str, status: str, patient_approved: bool = None, doctor_notes: str = None) -> bool:
        updates = ["status = ?", "updated_at = ?"]
        params = [status, datetime.utcnow().isoformat()]
        if patient_approved is not None:
            updates.append("patient_approved = ?")
            params.append(int(patient_approved))
        if doctor_notes is not None:
            updates.append("doctor_notes_encrypted = ?")
            params.append(self.encryption.encrypt(doctor_notes))
        params.append(consultation_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE consultations SET {', '.join(updates)} WHERE id = ?", params)
        return True

    def get_consultation(self, consultation_id: str) -> dict | None:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT c.*, d.name_encrypted as doc_name_enc, d.specialization, d.email as doc_email, d.pay_rate "
                "FROM consultations c JOIN doctors d ON c.doctor_id = d.id "
                "WHERE c.id = ?",
                (consultation_id,),
            ).fetchall()
        formatted = self._format_consultations(rows)
        return formatted[0] if formatted else None

    def get_stats(self) -> dict:
        with self._conn() as conn:
            patients = conn.execute("SELECT COUNT(*) as c FROM patients").fetchone()["c"]
            vitals = conn.execute("SELECT COUNT(*) as c FROM vitals").fetchone()["c"]
            logs = conn.execute("SELECT COUNT(*) as c FROM logs").fetchone()["c"]
            alerts = conn.execute("SELECT COUNT(*) as c FROM alerts").fetchone()["c"]
            sev1 = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE severity = 1").fetchone()["c"]
            sev2 = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE severity = 2").fetchone()["c"]
            sev3 = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE severity = 3").fetchone()["c"]
        audit_count = 0
        if os.path.exists(self.audit_path):
            with open(self.audit_path) as f:
                audit_count = sum(1 for _ in f)
        return {
            "patients": patients,
            "vitals_recorded": vitals,
            "analysis_logs": logs,
            "total_alerts": alerts,
            "critical_alerts": sev1,
            "warning_alerts": sev2,
            "info_alerts": sev3,
            "audit_entries": audit_count,
        }
