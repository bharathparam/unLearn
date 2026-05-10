"""
Neural Verification Engine — Forgetting Verifier
==================================================
Orchestrates the full verification pipeline:
  1. Score BEFORE output against the secret (baseline leakage)
  2. Run adversarial attack suite against AFTER output
  3. Compute forgetting delta and privacy confidence
  4. Classify verification status and risk level

This is the core brain of the verification engine.
"""

from __future__ import annotations

from app.attacks import AdversarialAttackEngine
from app.config import get_settings
from app.models import (
    AttackRequest,
    RiskLevel,
    VerificationRequest,
    VerificationResponse,
    VerificationStatus,
)
from app.scoring import LeakageScorer
from app.utils import utc_now


class ForgettingVerifier:
    """
    Main verification orchestrator.

    Compares before/after model outputs to determine whether
    sensitive information has been successfully forgotten.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._scorer = LeakageScorer()
        self._attack_engine = AdversarialAttackEngine()

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        """
        Run the complete verification pipeline.

        Steps:
          1. Score the BEFORE output to establish baseline leakage
          2. Run adversarial attacks against the AFTER output
          3. Compute the forgetting delta
          4. Classify the result

        Args:
            request: Contains secret, before_output, after_output.

        Returns:
            VerificationResponse with full audit data.
        """
        # ── Step 1: Baseline leakage (before forgetting) ────────────────
        before_score = self._scorer.compute(request.secret, request.before_output)
        before_leakage = before_score.overall_score

        # ── Optional Gemini Attack Generation ───────────────────────────
        custom_prompts = None
        try:
            from app.gemini.attack_generator import AdversarialPromptGenerator
            attack_gen = AdversarialPromptGenerator()
            gemini_attacks = attack_gen.generate(secret=request.secret, count=20)
            if gemini_attacks.enabled and gemini_attacks.prompts:
                custom_prompts = gemini_attacks.prompts
        except Exception:
            pass

        # ── Step 2: Post-forgetting attack suite ────────────────────────
        attack_request = AttackRequest(
            secret=request.secret,
            model_output=request.after_output,
            custom_prompts=custom_prompts
        )
        attack_response = self._attack_engine.run_attack_suite(attack_request)

        # ── Step 3: Compute after-forgetting leakage ────────────────────
        after_score = self._scorer.compute(request.secret, request.after_output)
        after_leakage = after_score.overall_score

        # ── Step 4: Compute forgetting delta ────────────────────────────
        forgetting_delta = round(before_leakage - after_leakage, 4)

        # ── Step 5: Compute leakage probability ─────────────────────────
        # Use the attack suite's overall leakage score as the basis
        leakage_probability = attack_response.overall_leakage_score

        # ── Step 6: Compute privacy confidence ──────────────────────────
        privacy_confidence = round(100.0 - (leakage_probability * 100.0), 2)
        privacy_confidence = max(0.0, min(100.0, privacy_confidence))

        # ── Step 7: Classify verification status ────────────────────────
        verification_status = self._classify_status(leakage_probability)

        # ── Step 8: Classify risk level ─────────────────────────────────
        risk_level = self._classify_risk(leakage_probability)

        return VerificationResponse(
            verification_status=verification_status,
            privacy_confidence=privacy_confidence,
            attack_success_rate=attack_response.attack_success_rate,
            attacks_succeeded=attack_response.successful_attacks,
            attacks_failed=attack_response.failed_attacks,
            attacks_total=attack_response.total_attacks,
            leakage_probability=round(leakage_probability, 4),
            risk_level=risk_level,
            before_leakage_score=round(before_leakage, 4),
            after_leakage_score=round(after_leakage, 4),
            forgetting_delta=forgetting_delta,
            attack_details=attack_response.attack_results,
            timestamp=utc_now(),
        )

    def _classify_status(self, leakage: float) -> VerificationStatus:
        """Classify leakage probability into verification status."""
        s = self._settings
        if leakage < s.LEAKAGE_THRESHOLD_LOW:
            return VerificationStatus.FORGOTTEN
        elif leakage < s.LEAKAGE_THRESHOLD_MEDIUM:
            return VerificationStatus.PARTIALLY_FORGOTTEN
        else:
            return VerificationStatus.NOT_FORGOTTEN

    def _classify_risk(self, leakage: float) -> RiskLevel:
        """Classify leakage probability into risk level."""
        s = self._settings
        if leakage < s.RISK_NEGLIGIBLE:
            return RiskLevel.NEGLIGIBLE
        elif leakage < s.RISK_LOW:
            return RiskLevel.LOW
        elif leakage < s.RISK_MEDIUM:
            return RiskLevel.MEDIUM
        elif leakage < s.RISK_HIGH:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
