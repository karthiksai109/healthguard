# HealthGuard — Test Commands

Replace `BASE_URL` with your Akash deployment URL or `http://localhost:8080`

## Health Check
```bash
curl http://localhost:8080/health
curl http://localhost:8080/status
```

## List Patients
```bash
curl http://localhost:8080/patients
```

## Get Patient Detail + Vitals
```bash
curl http://localhost:8080/patient/demo-patient-001
curl http://localhost:8080/patient/demo-patient-002
curl http://localhost:8080/patient/demo-patient-003
```

## Submit Vital Signs (triggers rule engine)

### Normal BP — should log only (severity 3)
```bash
curl -X POST http://localhost:8080/vital \
  -d "patient_id=demo-patient-001" \
  -d "metric_type=bp_systolic" \
  -d "value=120" \
  -d "unit=mmHg"
```

### WARNING glucose — severity 2
```bash
curl -X POST http://localhost:8080/vital \
  -d "patient_id=demo-patient-002" \
  -d "metric_type=glucose" \
  -d "value=65" \
  -d "unit=mg/dL"
```

### CRITICAL BP — severity 1 (triggers Telegram + TTS)
```bash
curl -X POST http://localhost:8080/vital \
  -d "patient_id=demo-patient-001" \
  -d "metric_type=bp_systolic" \
  -d "value=190" \
  -d "unit=mmHg"
```

### CRITICAL heart rate — severity 1
```bash
curl -X POST http://localhost:8080/vital \
  -d "patient_id=demo-patient-003" \
  -d "metric_type=heart_rate" \
  -d "value=160" \
  -d "unit=bpm"
```

### CRITICAL oxygen — severity 1
```bash
curl -X POST http://localhost:8080/vital \
  -d "patient_id=demo-patient-003" \
  -d "metric_type=oxygen_saturation" \
  -d "value=88" \
  -d "unit=%"
```

## Submit Symptom Text (triggers AkashML SOAP note)

### Headache + blurry vision
```bash
curl -X POST http://localhost:8080/symptom \
  -d "patient_id=demo-patient-001" \
  -d "text=I woke up with a splitting headache. My vision is blurry and I feel pressure behind my eyes. I took my medication but it hasnt helped."
```

### Diabetic emergency
```bash
curl -X POST http://localhost:8080/symptom \
  -d "patient_id=demo-patient-002" \
  -d "text=My hands wont stop shaking and I am sweating a lot. I feel like I might pass out. I accidentally took double insulin this morning."
```

### Acute pain emergency
```bash
curl -X POST http://localhost:8080/symptom \
  -d "patient_id=demo-patient-003" \
  -d "text=The pain is now a 10 out of 10. I cannot move. I think something is seriously wrong. I am also vomiting and have a very high fever."
```

## View Alerts
```bash
curl http://localhost:8080/alerts?limit=10
```

## View Analysis Logs
```bash
curl http://localhost:8080/logs?limit=10
```

## View Audit Trail (immutable receipts)
```bash
curl http://localhost:8080/audit?limit=10
```

## View Swagger API Docs
Open in browser: http://localhost:8080/docs

## PowerShell Equivalents (Windows)

```powershell
# Status
Invoke-RestMethod -Uri "http://localhost:8080/status" | ConvertTo-Json -Depth 4

# Submit vital
Invoke-RestMethod -Uri "http://localhost:8080/vital" -Method Post -Body @{patient_id="demo-patient-001"; metric_type="bp_systolic"; value=190; unit="mmHg"} | ConvertTo-Json

# Submit symptom
Invoke-RestMethod -Uri "http://localhost:8080/symptom" -Method Post -Body @{patient_id="demo-patient-001"; text="Severe headache and blurry vision"} | ConvertTo-Json

# View alerts
Invoke-RestMethod -Uri "http://localhost:8080/alerts?limit=5" | ConvertTo-Json -Depth 3
```
