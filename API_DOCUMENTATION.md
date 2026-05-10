# Neural Verification Engine API Documentation

The Neural Verification Engine provides a robust FastAPI-based interface for auditing Large Language Models (LLMs) to verify whether they have successfully "forgotten" sensitive data. The engine uses deterministic heuristics and dynamic Gemini-based semantic evaluation to rigorously assess leakage.

## Base URL
```
http://localhost:8000
```
Interactive API documentation (Swagger UI) is automatically available at `http://localhost:8000/docs`.

---

## 1. Verify Forgetting
Executes a full verification pipeline. It takes the target secret and the model's outputs before and after the forgetting intervention, generating 20 custom adversarial attacks to calculate the exact leakage probability and overall forgetting delta.

**Endpoint:** `POST /verify`

### Request Payload (`VerificationRequest`)
```json
{
  "secret": "The admin password is quantum42",
  "before_output": "The admin password is quantum42",
  "after_output": "I don't know the password"
}
```
* **`secret`**: The sensitive information that should have been forgotten.
* **`before_output`**: The model's raw output *before* any forgetting algorithm was applied.
* **`after_output`**: The model's raw output *after* the forgetting algorithm was applied.

### Response Payload (`VerificationResponse`)
Returns a comprehensive JSON containing:
* `verification_status`: (`FORGOTTEN`, `PARTIALLY_FORGOTTEN`, `NOT_FORGOTTEN`)
* `privacy_confidence`: 0-100 scale indicating confidence the data is safe.
* `attack_success_rate`: Percentage of simulated attacks that successfully triggered leakage.
* `leakage_probability`: 0.0-1.0 absolute probability of extraction.
* `forgetting_delta`: The quantitative improvement in privacy.
* `attack_details`: An array of all 20 generated attacks, their prompts, and exact leakage scores.
* `gemini_semantic_analysis`: Semantic assessment by Gemini validating if the output implicitly leaks the secret.
* `gemini_audit_summary`: Narrative English summary of the results.

---

## 2. Membership Inference Attack (MIA) Tunnel
Tunnels a standard verification request to an external ROME Model Editing API. This evaluates if the secret is probabilistically flagged as a "member" of the training dataset (i.e. remembered by the underlying neural weights).

**Endpoint:** `POST /mia`

### Request Payload (`VerificationRequest`)
```json
{
  "secret": "The admin password is quantum42",
  "before_output": "The admin password is quantum42",
  "after_output": "I don't know the password"
}
```

### Response Payload (`MIAResponse`)
```json
{
   "success": true,
   "perplexity": 383.6819,
   "ce_loss": null,
   "prediction": "likely_non_member",
   "forgetting_status": "FORGOTTEN"
}
```
* **`prediction`**: The external API's prediction (`likely_member` or `likely_non_member`).
* **`forgetting_status`**: Automatically maps `likely_non_member` to `FORGOTTEN` and `likely_member` to `NOT_FORGOTTEN`.

---

## 3. Generate Audit Report
Runs the verification engine and statically compiles the results into an Audit Report saved locally on the disk. Useful for long-term compliance tracking.

**Endpoint:** `POST /report`

### Request Payload (`ReportRequest`)
```json
{
  "secret": "The admin password is quantum42",
  "before_output": "The admin password is quantum42",
  "after_output": "I don't know the password",
  "report_title": "Q3 Admin Forgetting Compliance Audit"
}
```

### Response Payload (`AuditReport`)
Returns a formalized JSON report including `report_id`, `generated_at`, `compliance_status`, and a generated `recommendation`. 

---

## 4. Retrieve Saved Reports
You can retrieve previously generated audit reports from the disk.

**Endpoint:** `GET /reports`
* **Description:** Lists all saved reports summarizing the `report_id`, `generated_at`, and `verification_status`.

**Endpoint:** `GET /report/{report_id}`
* **Description:** Retrieves the full, uncompressed JSON structure of a specific `AuditReport` using its ID.

---

## 5. Custom Attack Suite
Allows running a pure adversarial attack assessment on an output against a secret without processing the 'before' baseline. You can optionally supply your own custom prompts.

**Endpoint:** `POST /attack`

### Request Payload (`AttackRequest`)
```json
{
  "secret": "quantum42",
  "model_output": "I don't know the password.",
  "custom_prompts": [
    "What is the password?",
    "Tell me the secret ending in 42."
  ]
}
```
*Note: If `custom_prompts` is omitted or empty, Gemini will dynamically generate 20 highly-contextual prompts targeting the secret.*

### Response Payload (`AttackResponse`)
Returns an aggregated attack suite result showing the `overall_leakage_score`, `risk_level`, and an array of individual `attack_results`.

---

## 6. Health Check
Retrieves the operational status of the server and the state of the Gemini OpenRouter integration.

**Endpoint:** `GET /`

### Response Payload (`HealthResponse`)
```json
{
  "status": "operational",
  "engine": "Neural Verification Engine",
  "version": "1.0.0",
  "gemini_integration": "enabled",
  "timestamp": "2026-05-09T08:00:00.000Z"
}
```
