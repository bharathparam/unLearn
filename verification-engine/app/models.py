"""
Neural Verification Engine — Pydantic Models
==============================================
All request/response schemas used across the API surface.
Designed for frontend-friendly JSON output and Streamlit integration.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class VerificationStatus(str, Enum):
    """Overall forgetting verification result."""
    FORGOTTEN = "FORGOTTEN"
    PARTIALLY_FORGOTTEN = "PARTIALLY_FORGOTTEN"
    NOT_FORGOTTEN = "NOT_FORGOTTEN"


class RiskLevel(str, Enum):
    """Privacy risk classification."""
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

class VerificationRequest(BaseModel):
    """Input for the /verify endpoint."""
    secret: str = Field(..., description="The sensitive information that should have been forgotten")
    before_output: str = Field(..., description="Model output BEFORE forgetting was applied")
    after_output: str = Field(..., description="Model output AFTER forgetting was applied")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "secret": "The admin password is quantum42",
                    "before_output": "The admin password is quantum42",
                    "after_output": "I don't know the password",
                }
            ]
        }
    }


class GeminiSemanticAnalysisResponse(BaseModel):
    """Optional Gemini semantic analysis block embedded in API responses."""
    enabled: bool = False
    semantic_leakage_detected: Optional[bool] = None
    leakage_type: Optional[str] = None
    confidence: Optional[float] = None
    summary: Optional[str] = None
    source: str = "gemini_auxiliary_semantic_analysis"


class GeminiAuditSummaryResponse(BaseModel):
    """Optional Gemini audit summary block embedded in API responses."""
    enabled: bool = False
    narrative_summary: Optional[str] = None
    risk_narrative: Optional[str] = None
    recommendation: Optional[str] = None
    source: str = "gemini_auxiliary_summary"


class VerificationResponse(BaseModel):
    """Full verification result returned by /verify."""
    verification_status: VerificationStatus
    privacy_confidence: float = Field(..., description="0-100 scale confidence that data is forgotten")
    attack_success_rate: float = Field(..., description="Percentage of attacks that extracted info")
    attacks_succeeded: int
    attacks_failed: int
    attacks_total: int
    leakage_probability: float = Field(..., description="0.0-1.0 probability of data leakage")
    risk_level: RiskLevel
    before_leakage_score: float
    after_leakage_score: float
    forgetting_delta: float = Field(..., description="Improvement in privacy (before - after)")
    attack_details: list[AttackResult]
    gemini_semantic_analysis: Optional[GeminiSemanticAnalysisResponse] = None
    gemini_audit_summary: Optional[GeminiAuditSummaryResponse] = None
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════
# Attacks
# ═══════════════════════════════════════════════════════════════════════════

class AttackRequest(BaseModel):
    """Input for the /attack endpoint."""
    secret: str = Field(..., description="The target sensitive information")
    model_output: str = Field(..., description="The model's response to evaluate")
    custom_prompts: Optional[list[str]] = Field(
        None, description="Optional custom attack prompts to use"
    )


class AttackResult(BaseModel):
    """Result of a single adversarial attack attempt."""
    attack_id: int
    prompt: str
    leakage_score: float = Field(..., description="0.0-1.0 leakage detected")
    leakage_detected: bool
    string_similarity: float
    token_overlap: float
    ngram_match: float
    semantic_proximity: float
    timestamp: datetime


class AttackResponse(BaseModel):
    """Aggregated attack suite results."""
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    attack_success_rate: float
    overall_leakage_score: float
    risk_level: RiskLevel
    attack_results: list[AttackResult]
    gemini_generated_prompts: Optional[list[str]] = None
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════

class LeakageScore(BaseModel):
    """Detailed breakdown of leakage metrics."""
    overall_score: float
    string_similarity: float
    token_overlap: float
    ngram_match: float
    semantic_proximity: float
    leakage_detected: bool


class PrivacyConfidence(BaseModel):
    """Privacy confidence assessment."""
    privacy_confidence: float
    attack_success_rate: float
    leakage_probability: float
    status: VerificationStatus
    risk_level: RiskLevel


# ═══════════════════════════════════════════════════════════════════════════
# Reports
# ═══════════════════════════════════════════════════════════════════════════

class ReportRequest(BaseModel):
    """Input for the /report endpoint."""
    secret: str
    before_output: str
    after_output: str
    report_title: Optional[str] = "Neural Forgetting Report"


class AuditReport(BaseModel):
    """Full audit report structure."""
    report_id: str
    report_title: str
    generated_at: datetime
    tested_fact: str
    verification_status: VerificationStatus
    privacy_confidence: float
    leakage_probability: float
    risk_level: RiskLevel
    attack_summary: AttackSummary
    compliance_status: str
    recommendation: str
    gemini_semantic_analysis: Optional[GeminiSemanticAnalysisResponse] = None
    gemini_audit_summary: Optional[GeminiAuditSummaryResponse] = None
    report_path: Optional[str] = None


class AttackSummary(BaseModel):
    """Summary of attacks for the audit report."""
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    attack_success_rate: float
    highest_leakage_score: float
    average_leakage_score: float


class ReportListItem(BaseModel):
    """Summary item for listing reports."""
    report_id: str
    report_title: str
    generated_at: datetime
    verification_status: VerificationStatus
    privacy_confidence: float


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "operational"
    engine: str = "Neural Verification Engine"
    version: str
    gemini_integration: str = "disabled"
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════
# Membership Inference Attack (MIA)
# ═══════════════════════════════════════════════════════════════════════════

class MIARequest(BaseModel):
    """Request payload for Membership Inference Attack."""
    prompt: str = Field(..., description="The prompt context")
    target_text: str = Field(..., description="The text to check for membership")


class MIAResponse(BaseModel):
    """Response payload for Membership Inference Attack."""
    success: bool
    perplexity: Optional[float] = None
    ce_loss: Optional[float] = None
    score: Optional[float] = None
    prediction: Optional[str] = None
    forgetting_status: Optional[str] = None
    error: Optional[str] = None


# Forward reference resolution
VerificationResponse.model_rebuild()
AttackResponse.model_rebuild()
AuditReport.model_rebuild()
