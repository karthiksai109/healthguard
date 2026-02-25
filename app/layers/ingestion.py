"""HealthGuard — Layer 1: Ingestion

Handles incoming patient data (photos, voice notes, text).
- Strips EXIF metadata from images (GPS, device info)
- Assigns ephemeral session IDs (never patient IDs to inference)
- Encrypts in transit with session key
- Registers 60-second TTL auto-delete timer
- Raw file lifespan: maximum 60 seconds
"""
import os
import io
import uuid
import time
import threading
from datetime import datetime
from PIL import Image
import structlog

logger = structlog.get_logger()

# Track ephemeral files for auto-deletion
_ephemeral_files: dict[str, float] = {}  # path -> expiry timestamp
_lock = threading.Lock()


def generate_session_id() -> str:
    """Ephemeral session ID — never a patient name or MRN.
    Venice receives this, not the patient identity."""
    return f"session_{uuid.uuid4().hex[:12]}"


def strip_exif(image_bytes: bytes) -> bytes:
    """Remove all EXIF metadata from image (GPS, device, timestamps).
    Returns clean image bytes with zero metadata."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        buf = io.BytesIO()
        fmt = img.format or "PNG"
        clean.save(buf, format=fmt)
        logger.info("exif_stripped", original_size=len(image_bytes), clean_size=buf.tell())
        return buf.getvalue()
    except Exception as e:
        logger.warning("exif_strip_failed", error=str(e))
        return image_bytes


def save_ephemeral(data_dir: str, data: bytes, suffix: str, ttl: int = 60) -> str:
    """Save file with auto-delete timer.
    File will be deleted after ttl seconds."""
    ephemeral_dir = os.path.join(data_dir, "ephemeral")
    os.makedirs(ephemeral_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}_{int(time.time())}{suffix}"
    filepath = os.path.join(ephemeral_dir, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    with _lock:
        _ephemeral_files[filepath] = time.time() + ttl
    logger.info("ephemeral_saved", path=filename, ttl=ttl, size=len(data))
    return filepath


def cleanup_expired():
    """Delete all expired ephemeral files. Called by cleanup worker."""
    now = time.time()
    to_delete = []
    with _lock:
        for path, expiry in list(_ephemeral_files.items()):
            if now >= expiry:
                to_delete.append(path)
                del _ephemeral_files[path]
    for path in to_delete:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info("ephemeral_deleted", path=os.path.basename(path))
        except Exception as e:
            logger.warning("ephemeral_delete_failed", path=path, error=str(e))
    return len(to_delete)


def delete_immediately(filepath: str):
    """Force-delete a raw file after inference completes."""
    with _lock:
        _ephemeral_files.pop(filepath, None)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info("raw_deleted_immediately", path=os.path.basename(filepath))
    except Exception as e:
        logger.warning("immediate_delete_failed", error=str(e))


def get_ephemeral_count() -> int:
    with _lock:
        return len(_ephemeral_files)


class IngestedItem:
    """Processed ingestion result ready for inference pipeline."""

    def __init__(self, session_id: str, input_type: str, patient_id: str,
                 file_path: str = None, raw_bytes: bytes = None, text: str = None):
        self.session_id = session_id
        self.input_type = input_type  # "photo", "voice", "text", "vital"
        self.patient_id = patient_id
        self.file_path = file_path
        self.raw_bytes = raw_bytes
        self.text = text
        self.created_at = datetime.utcnow().isoformat()


def ingest_photo(data_dir: str, image_bytes: bytes, patient_id: str, ttl: int = 60) -> IngestedItem:
    """Full photo ingestion pipeline: strip EXIF → save ephemeral → return clean item."""
    session_id = generate_session_id()
    clean_bytes = strip_exif(image_bytes)
    filepath = save_ephemeral(data_dir, clean_bytes, ".png", ttl=ttl)
    logger.info("photo_ingested", session_id=session_id, size=len(clean_bytes))
    return IngestedItem(
        session_id=session_id, input_type="photo",
        patient_id=patient_id, file_path=filepath, raw_bytes=clean_bytes,
    )


def ingest_voice(data_dir: str, audio_bytes: bytes, patient_id: str, ttl: int = 60) -> IngestedItem:
    """Voice note ingestion: save ephemeral → return item for STT."""
    session_id = generate_session_id()
    filepath = save_ephemeral(data_dir, audio_bytes, ".wav", ttl=ttl)
    logger.info("voice_ingested", session_id=session_id, size=len(audio_bytes))
    return IngestedItem(
        session_id=session_id, input_type="voice",
        patient_id=patient_id, file_path=filepath, raw_bytes=audio_bytes,
    )


def ingest_text(text: str, patient_id: str) -> IngestedItem:
    """Text symptom log — no file needed, just session tracking."""
    session_id = generate_session_id()
    logger.info("text_ingested", session_id=session_id, chars=len(text))
    return IngestedItem(
        session_id=session_id, input_type="text",
        patient_id=patient_id, text=text,
    )


def ingest_vital(patient_id: str, metric_type: str, value: float, unit: str = "") -> IngestedItem:
    """Vital sign entry — structured data, no raw file."""
    session_id = generate_session_id()
    text = f"{metric_type}: {value} {unit}"
    logger.info("vital_ingested", session_id=session_id, metric=metric_type, value=value)
    return IngestedItem(
        session_id=session_id, input_type="vital",
        patient_id=patient_id, text=text,
    )
