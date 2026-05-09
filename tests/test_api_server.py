"""
tests/test_api_server.py
========================
Comprehensive pytest test suite for api_server.py.

Covers all 5 endpoints across 9 test classes:

  TestHealth           — /health
  TestQuery            — /query  (happy path + edge cases)
  TestQueryValidation  — /query  (422 validation)
  TestEdit             — /edit   (happy path)
  TestEditValidation   — /edit   (422 validation + edge cases)
  TestRestore          — /restore
  TestMIA              — /mia    (happy path + predictions)
  TestMIAValidation    — /mia    (422 validation)
  TestConcurrency      — parallel read/write safety

Run (from ROME-2/):
  rome_env\\Scripts\\pytest tests/test_api_server.py -v --tb=short -p no:warnings

NOTE: Model loads once per module — takes ~15s.  Each ROME edit test takes
~90s because it runs the full ROME pipeline (30 grad steps).
"""

import sys
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Single model load for the entire module."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(autouse=True)
def restore_after(client):
    """Always restore weights after every test to reset model state."""
    yield
    client.post("/restore")


# ─────────────────────────────────────────────────────────────────────────────
# Shared payloads
# ─────────────────────────────────────────────────────────────────────────────

EDIT_GOOGLE = {
    "subject": "Google",
    "prompt": "Google was founded by",
    "target": "Elon Musk",
    "layer": 15,
    "v_num_grad_steps": 3,   # minimal steps for fast tests
    "v_lr": 0.1,
}

QUERY_GOOGLE = {
    "prompt": "Google was founded by",
    "max_new_tokens": 10,
    "temperature": 0.1,
    "do_sample": False,
}

MIA_KNOWN = {
    "prompt": "The capital of France is",
    "target_text": "Paris",
}

MIA_UNKNOWN = {
    "prompt": "The capital of France is",
    "target_text": "xyzzywobblethorp",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1 — /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_status_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_device_field_present(self, client):
        body = client.get("/health").json()
        assert "device" in body
        assert body["device"] in ("cuda", "cpu")

    def test_active_edits_starts_zero(self, client):
        client.post("/restore")
        assert client.get("/health").json()["active_edits"] == 0

    def test_active_edits_increments(self, client):
        client.post("/edit", json=EDIT_GOOGLE)
        assert client.get("/health").json()["active_edits"] >= 1

    def test_active_edits_resets_after_restore(self, client):
        client.post("/edit", json=EDIT_GOOGLE)
        client.post("/restore")
        assert client.get("/health").json()["active_edits"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2 — /query (happy path)
# ─────────────────────────────────────────────────────────────────────────────

class TestQuery:

    def test_returns_200(self, client):
        assert client.post("/query", json=QUERY_GOOGLE).status_code == 200

    def test_success_true(self, client):
        assert client.post("/query", json=QUERY_GOOGLE).json()["success"] is True

    def test_response_not_empty(self, client):
        body = client.post("/query", json=QUERY_GOOGLE).json()
        assert len(body["response"]) > 0

    def test_elapsed_ms_positive(self, client):
        body = client.post("/query", json=QUERY_GOOGLE).json()
        assert body["elapsed_ms"] > 0

    def test_response_contains_prompt(self, client):
        """Generated text should echo the prompt (greedy, no truncation)."""
        r = client.post("/query", json={**QUERY_GOOGLE, "max_new_tokens": 5})
        assert QUERY_GOOGLE["prompt"] in r.json()["response"]

    def test_greedy_deterministic(self, client):
        """Two identical greedy requests should produce identical output."""
        p = {**QUERY_GOOGLE, "do_sample": False, "max_new_tokens": 5}
        r1 = client.post("/query", json=p).json()["response"]
        r2 = client.post("/query", json=p).json()["response"]
        assert r1 == r2

    def test_max_tokens_1(self, client):
        r = client.post("/query", json={**QUERY_GOOGLE, "max_new_tokens": 1})
        assert r.status_code == 200
        assert len(r.json()["response"]) > 0

    def test_max_tokens_200(self, client):
        r = client.post("/query", json={**QUERY_GOOGLE, "max_new_tokens": 200,
                                        "do_sample": True})
        assert r.status_code == 200

    def test_temperature_high(self, client):
        r = client.post("/query", json={**QUERY_GOOGLE, "temperature": 2.0,
                                        "do_sample": True})
        assert r.status_code == 200

    def test_temperature_low(self, client):
        r = client.post("/query", json={**QUERY_GOOGLE, "temperature": 0.01,
                                        "do_sample": True})
        assert r.status_code == 200

    def test_unicode_prompt(self, client):
        r = client.post("/query", json={**QUERY_GOOGLE,
                                        "prompt": "Napoléon était un général de"})
        assert r.status_code == 200

    def test_long_prompt(self, client):
        long_p = "Tell me about the history of Google. " * 5
        r = client.post("/query", json={**QUERY_GOOGLE, "prompt": long_p,
                                        "max_new_tokens": 5})
        assert r.status_code == 200

    def test_query_after_edit(self, client):
        """Query should still work after a ROME edit."""
        client.post("/edit", json=EDIT_GOOGLE)
        r = client.post("/query", json=QUERY_GOOGLE)
        assert r.status_code == 200
        assert len(r.json()["response"]) > 0

    def test_query_after_restore(self, client):
        client.post("/edit", json=EDIT_GOOGLE)
        client.post("/restore")
        r = client.post("/query", json=QUERY_GOOGLE)
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 3 — /query validation (422)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:

    def test_missing_prompt(self, client):
        assert client.post("/query", json={"max_new_tokens": 10}).status_code == 422

    def test_empty_prompt(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "prompt": ""}).status_code == 422

    def test_blank_prompt(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "prompt": "   "}).status_code == 422

    def test_max_tokens_zero(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "max_new_tokens": 0}).status_code == 422

    def test_max_tokens_over_limit(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "max_new_tokens": 1025}).status_code == 422

    def test_temperature_zero(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "temperature": 0.0}).status_code == 422

    def test_temperature_over_limit(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "temperature": 2.1}).status_code == 422

    def test_top_p_zero(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "top_p": 0.0}).status_code == 422

    def test_top_p_over_limit(self, client):
        assert client.post("/query", json={**QUERY_GOOGLE, "top_p": 1.01}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 4 — /edit (happy path)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdit:

    def test_returns_200(self, client):
        assert client.post("/edit", json=EDIT_GOOGLE).status_code == 200

    def test_success_true(self, client):
        assert client.post("/edit", json=EDIT_GOOGLE).json()["success"] is True

    def test_message_field(self, client):
        body = client.post("/edit", json=EDIT_GOOGLE).json()
        assert "ROME edit applied" in body["message"]

    def test_edit_id_uuid4(self, client):
        body = client.post("/edit", json=EDIT_GOOGLE).json()
        assert len(body["edit_id"]) == 36
        assert body["edit_id"].count("-") == 4

    def test_update_norm_positive(self, client):
        body = client.post("/edit", json=EDIT_GOOGLE).json()
        assert body["update_norm"] > 0.0

    def test_fields_echo(self, client):
        body = client.post("/edit", json=EDIT_GOOGLE).json()
        assert body["subject"] == EDIT_GOOGLE["subject"]
        assert body["target"] == EDIT_GOOGLE["target"]
        assert body["edit_layer"] == EDIT_GOOGLE["layer"]

    def test_applied_at_iso_format(self, client):
        body = client.post("/edit", json=EDIT_GOOGLE).json()
        from datetime import datetime
        # Should not raise
        datetime.fromisoformat(body["applied_at"].replace("Z", "+00:00"))

    def test_edit_layer_override(self, client):
        r = client.post("/edit", json={**EDIT_GOOGLE, "layer": 10, "v_num_grad_steps": 2})
        assert r.json()["edit_layer"] == 10

    def test_edit_layer_boundary_low(self, client):
        r = client.post("/edit", json={**EDIT_GOOGLE, "layer": 0, "v_num_grad_steps": 2})
        assert r.status_code == 200

    def test_edit_layer_boundary_high(self, client):
        r = client.post("/edit", json={**EDIT_GOOGLE, "layer": 27, "v_num_grad_steps": 2})
        assert r.status_code == 200

    def test_subject_not_in_prompt_succeeds(self, client):
        """ROME falls back gracefully when subject is not found in prompt."""
        r = client.post("/edit", json={
            **EDIT_GOOGLE,
            "subject": "Antarctica",   # not in prompt
            "v_num_grad_steps": 2,
        })
        assert r.status_code == 200

    def test_unicode_target(self, client):
        r = client.post("/edit", json={
            **EDIT_GOOGLE,
            "target": "Elon Musk 🚀",
            "v_num_grad_steps": 2,
        })
        assert r.status_code == 200

    def test_multiple_edits_accumulate(self, client):
        edit2 = {**EDIT_GOOGLE, "subject": "France",
                 "prompt": "The capital of France is", "target": "Lyon",
                 "v_num_grad_steps": 2}
        client.post("/edit", json=EDIT_GOOGLE)
        client.post("/edit", json=edit2)
        assert client.get("/health").json()["active_edits"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5 — /edit validation (422)
# ─────────────────────────────────────────────────────────────────────────────

class TestEditValidation:

    def test_missing_subject(self, client):
        d = {k: v for k, v in EDIT_GOOGLE.items() if k != "subject"}
        assert client.post("/edit", json=d).status_code == 422

    def test_missing_prompt(self, client):
        d = {k: v for k, v in EDIT_GOOGLE.items() if k != "prompt"}
        assert client.post("/edit", json=d).status_code == 422

    def test_missing_target(self, client):
        d = {k: v for k, v in EDIT_GOOGLE.items() if k != "target"}
        assert client.post("/edit", json=d).status_code == 422

    def test_blank_subject(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "subject": "  "}).status_code == 422

    def test_blank_prompt(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "prompt": "  "}).status_code == 422

    def test_blank_target(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "target": "  "}).status_code == 422

    def test_layer_negative(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "layer": -1}).status_code == 422

    def test_layer_too_high(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "layer": 28}).status_code == 422

    def test_v_lr_zero(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "v_lr": 0.0}).status_code == 422

    def test_v_lr_negative(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "v_lr": -0.1}).status_code == 422

    def test_v_steps_zero(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "v_num_grad_steps": 0}).status_code == 422

    def test_v_steps_over_limit(self, client):
        assert client.post("/edit", json={**EDIT_GOOGLE, "v_num_grad_steps": 201}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 6 — /restore
# ─────────────────────────────────────────────────────────────────────────────

class TestRestore:

    def test_restore_returns_200(self, client):
        assert client.post("/restore").status_code == 200

    def test_restore_success_true(self, client):
        assert client.post("/restore").json()["success"] is True

    def test_restore_when_no_edits(self, client):
        client.post("/restore")
        r = client.post("/restore")
        assert r.status_code == 200
        assert r.json()["edits_cleared"] == 0

    def test_restore_clears_history(self, client):
        client.post("/edit", json=EDIT_GOOGLE)
        r = client.post("/restore")
        assert r.json()["edits_cleared"] >= 1
        assert client.get("/health").json()["active_edits"] == 0

    def test_restore_multiple_safe(self, client):
        for _ in range(3):
            assert client.post("/restore").status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 7 — /mia (happy path)
# ─────────────────────────────────────────────────────────────────────────────

class TestMIA:

    def test_returns_200(self, client):
        assert client.post("/mia", json=MIA_KNOWN).status_code == 200

    def test_success_true(self, client):
        assert client.post("/mia", json=MIA_KNOWN).json()["success"] is True

    def test_perplexity_positive(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert body["perplexity"] > 0.0

    def test_avg_token_loss_positive(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert body["avg_token_loss"] > 0.0

    def test_membership_score_in_range(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert 0.0 <= body["membership_score"] <= 1.0

    def test_prediction_valid_values(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert body["prediction"] in ("likely_member", "likely_non_member")

    def test_num_target_tokens_positive(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert body["num_target_tokens"] > 0

    def test_elapsed_ms_positive(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert body["elapsed_ms"] > 0

    def test_known_fact_lower_perplexity_than_unknown(self, client):
        """
        'Paris' as continuation of 'The capital of France is' should have
        lower perplexity than a random nonsense string.
        """
        known_ppl   = client.post("/mia", json=MIA_KNOWN).json()["perplexity"]
        unknown_ppl = client.post("/mia", json=MIA_UNKNOWN).json()["perplexity"]
        assert known_ppl < unknown_ppl, (
            f"Expected known fact perplexity ({known_ppl}) < unknown ({unknown_ppl})"
        )

    def test_known_fact_higher_score_than_unknown(self, client):
        known_score   = client.post("/mia", json=MIA_KNOWN).json()["membership_score"]
        unknown_score = client.post("/mia", json=MIA_UNKNOWN).json()["membership_score"]
        assert known_score > unknown_score

    def test_known_fact_prediction_member(self, client):
        body = client.post("/mia", json=MIA_KNOWN).json()
        assert body["prediction"] == "likely_member"

    def test_unknown_text_prediction_non_member(self, client):
        body = client.post("/mia", json=MIA_UNKNOWN).json()
        assert body["prediction"] == "likely_non_member"

    def test_mia_with_long_target(self, client):
        r = client.post("/mia", json={
            "prompt": "The capital of France is",
            "target_text": "Paris, a major European city with a rich cultural heritage "
                           "and home to the Eiffel Tower.",
        })
        assert r.status_code == 200

    def test_mia_unicode_target(self, client):
        r = client.post("/mia", json={
            "prompt": "Napoléon était",
            "target_text": "un général français très célèbre",
        })
        assert r.status_code == 200

    def test_mia_after_edit(self, client):
        """MIA should still work after a ROME edit."""
        client.post("/edit", json=EDIT_GOOGLE)
        r = client.post("/mia", json=MIA_KNOWN)
        assert r.status_code == 200
        assert r.json()["perplexity"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8 — /mia validation (422)
# ─────────────────────────────────────────────────────────────────────────────

class TestMIAValidation:

    def test_missing_prompt(self, client):
        assert client.post("/mia", json={"target_text": "Paris"}).status_code == 422

    def test_missing_target_text(self, client):
        assert client.post("/mia", json={"prompt": "test"}).status_code == 422

    def test_empty_prompt(self, client):
        assert client.post("/mia", json={**MIA_KNOWN, "prompt": ""}).status_code == 422

    def test_empty_target(self, client):
        assert client.post("/mia", json={**MIA_KNOWN, "target_text": ""}).status_code == 422

    def test_blank_prompt(self, client):
        assert client.post("/mia", json={**MIA_KNOWN, "prompt": "   "}).status_code == 422

    def test_blank_target(self, client):
        assert client.post("/mia", json={**MIA_KNOWN, "target_text": "   "}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 9 — Concurrency (RWLock correctness)
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_parallel_query_requests_all_succeed(self, client):
        """Multiple simultaneous /query requests must all return 200."""
        results = {}

        def do_query(idx):
            r = client.post("/query", json={**QUERY_GOOGLE, "max_new_tokens": 3})
            results[idx] = r.status_code

        threads = [threading.Thread(target=do_query, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=60)

        assert all(v == 200 for v in results.values()), f"Failures: {results}"

    def test_parallel_mia_requests_all_succeed(self, client):
        results = {}

        def do_mia(idx):
            r = client.post("/mia", json=MIA_KNOWN)
            results[idx] = r.status_code

        threads = [threading.Thread(target=do_mia, args=(i,)) for i in range(3)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=60)

        assert all(v == 200 for v in results.values()), f"Failures: {results}"

    def test_edit_then_query_sequential(self, client):
        """Edit followed by query — both should succeed."""
        edit_r = client.post("/edit", json=EDIT_GOOGLE)
        assert edit_r.status_code == 200

        query_r = client.post("/query", json=QUERY_GOOGLE)
        assert query_r.status_code == 200

    def test_health_always_available(self, client):
        """Health check should return 200 at any time."""
        results = {}

        def check_health(idx):
            r = client.get("/health")
            results[idx] = r.status_code

        threads = [threading.Thread(target=check_health, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10)

        assert all(v == 200 for v in results.values())
