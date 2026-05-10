"""
Neural Verification Engine — Audit Report Generator
=====================================================
Generates compliance-style JSON audit reports from verification results.
Reports are saved to the reports/ directory for retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models import (
    AttackSummary,
    AuditReport,
    ReportListItem,
    ReportRequest,
    RiskLevel,
    VerificationResponse,
    VerificationStatus,
)
from app.utils import generate_id, list_json_files, load_json, save_json, utc_now
from app.verifier import ForgettingVerifier


class AuditReportGenerator:
    """Generates and manages Neural Forgetting audit reports."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._verifier = ForgettingVerifier()

    def generate(self, request: ReportRequest) -> AuditReport:
        """
        Generate a full audit report by running verification
        and packaging the results in a compliance-friendly format.

        Args:
            request: Contains secret, before_output, after_output, report_title.

        Returns:
            AuditReport with full verification data.
        """
        from app.models import VerificationRequest

        # Run the verification pipeline
        verification = self._verifier.verify(
            VerificationRequest(
                secret=request.secret,
                before_output=request.before_output,
                after_output=request.after_output,
            )
        )

        # Build attack summary
        attack_details = verification.attack_details
        leakage_scores = [a.leakage_score for a in attack_details]

        attack_summary = AttackSummary(
            total_attacks=verification.attacks_total,
            successful_attacks=verification.attacks_succeeded,
            failed_attacks=verification.attacks_failed,
            attack_success_rate=verification.attack_success_rate,
            highest_leakage_score=round(max(leakage_scores), 4) if leakage_scores else 0.0,
            average_leakage_score=round(
                sum(leakage_scores) / len(leakage_scores), 4
            ) if leakage_scores else 0.0,
        )

        # Generate compliance status
        compliance_status = self._compliance_status(verification.verification_status)
        recommendation = self._recommendation(
            verification.verification_status, verification.risk_level
        )

        # Build report
        report_id = generate_id("RPT")
        report = AuditReport(
            report_id=report_id,
            report_title=request.report_title or "Neural Forgetting Report",
            generated_at=utc_now(),
            tested_fact=request.secret,
            verification_status=verification.verification_status,
            privacy_confidence=verification.privacy_confidence,
            leakage_probability=verification.leakage_probability,
            risk_level=verification.risk_level,
            attack_summary=attack_summary,
            compliance_status=compliance_status,
            recommendation=recommendation,
        )

        # Save report to disk
        filepath = self._settings.REPORTS_DIR / f"{report_id}.json"
        save_json(report.model_dump(mode="json"), filepath)
        report.report_path = str(filepath)

        return report

    def get_report(self, report_id: str) -> Optional[AuditReport]:
        """Retrieve a saved report by ID."""
        filepath = self._settings.REPORTS_DIR / f"{report_id}.json"
        if not filepath.exists():
            return None
        data = load_json(filepath)
        return AuditReport(**data)

    def list_reports(self) -> list[ReportListItem]:
        """List all saved reports, newest first."""
        files = list_json_files(self._settings.REPORTS_DIR)
        items: list[ReportListItem] = []
        for f in files:
            try:
                data = load_json(f)
                items.append(
                    ReportListItem(
                        report_id=data["report_id"],
                        report_title=data["report_title"],
                        generated_at=data["generated_at"],
                        verification_status=data["verification_status"],
                        privacy_confidence=data["privacy_confidence"],
                    )
                )
            except (KeyError, Exception):
                continue
        return items

    @staticmethod
    def _compliance_status(status: VerificationStatus) -> str:
        """Map verification status to compliance language."""
        mapping = {
            VerificationStatus.FORGOTTEN: "COMPLIANT — Data successfully removed from model memory",
            VerificationStatus.PARTIALLY_FORGOTTEN: "WARNING — Partial data retention detected, remediation recommended",
            VerificationStatus.NOT_FORGOTTEN: "NON-COMPLIANT — Data persists in model memory, immediate action required",
        }
        return mapping.get(status, "UNKNOWN")

    @staticmethod
    def _recommendation(status: VerificationStatus, risk: RiskLevel) -> str:
        """Generate a human-readable recommendation."""
        if status == VerificationStatus.FORGOTTEN:
            return (
                "The model shows no significant evidence of retaining the target information. "
                "The forgetting operation appears successful. Regular monitoring is recommended."
            )
        elif status == VerificationStatus.PARTIALLY_FORGOTTEN:
            return (
                f"Partial information leakage detected (Risk: {risk.value}). "
                "Consider re-running the forgetting procedure with stronger parameters "
                "or applying additional privacy layers."
            )
        else:
            return (
                f"Critical information leakage detected (Risk: {risk.value}). "
                "The forgetting operation has NOT been successful. "
                "Immediate re-processing with enhanced editing methods is required. "
                "Do NOT deploy this model with sensitive data intact."
            )
