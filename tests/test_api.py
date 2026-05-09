"""
tests/test_api.py — comprehensive pytest test suite for the ROME FastAPI.

Covers
──────
  Happy-path tests
    - health / status endpoints
    - basic edit + generate cycle
    - restore clears history
    - list edits endpoint
    - per-request hparam overrides (layer, steps, lr)

  Validation / edge-case tests
    - blank / whitespace-only fields
    - subject not appearing in prompt (graceful fallback)
    - prompt too long (> max_length)
    - unicode subject / prompt / target
    - multiple sequential edits
    - generate before any edit
    - generate after restore
    - delete /edits/{id} → 501 Not Implemented
    - delete non-existent edit id → 404
    - out-of-range edit layer
    - edge layer values (layer 0, layer 27)
    - very short prompt (single word)
    - very long target (multi-word phrase)
    - max_new_tokens boundary values (1, 500)
    - temperature boundary values (just above 0, 2.0)
    - do_sample=False (greedy)
    - restore when no edits applied
    - concurrent requests (serialisation check)

Run
───
  # from ROME-2/
  rome_env\\Scripts\\pytest tests/test_api.py -v
  # or with output:
  rome_env\\Scripts\\pytest tests/test_api.py -v -s

NOTE: These tests use TestClient (sync WSGI mode) so the model loads once
per module via the module-scoped fixture.  Full model load takes ~10-15s.
"""

import sys
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

# ── Make sure the parent dir is on sys.path ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient.  The FastAPI lifespan (model load) runs once
    for the whole module, not per test — this is intentional to avoid the
    expensive model-load overhead for every single test.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(autouse=True)
def restore_after_test(client):
    """Restore model weights after every test that may have applied an edit."""
    yield
    # POST /restore is idempotent — safe to call even if nothing was edited
    client.post("/restore")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_EDIT = {
    "subject": "Google",
    "prompt": "Google was founded by",
    "target_new": "Elon Musk",
}

VALID_GENERATE = {
    "prompt": "Google was founded by",
    "max_new_tokens": 10,
    "temperature": 0.1,
    "do_sample": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1 — Utility endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilityEndpoints:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_ready_true(self, client):
        r = client.get("/health")
        assert r.json()["model_ready"] is True

    def test_status_returns_200(self, client):
        r = client.get("/status")
        assert r.status_code == 200

    def test_status_fields_present(self, client):
        r = client.get("/status")
        body = r.json()
        for field in ("model_ready", "model_name", "device", "edit_layer",
                      "num_edits_applied", "startup_time", "edits"):
            assert field in body, f"Missing field: {field}"

    def test_status_model_ready(self, client):
        r = client.get("/status")
        assert r.json()["model_ready"] is True

    def test_status_model_name(self, client):
        r = client.get("/status")
        assert "Qwen" in r.json()["model_name"]


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Happy-path edit → generate → restore cycle
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_apply_edit_200(self, client):
        r = client.post("/edit", json=VALID_EDIT)
        assert r.status_code == 200

    def test_apply_edit_returns_edit_id(self, client):
        r = client.post("/edit", json=VALID_EDIT)
        assert "edit_id" in r.json()
        assert len(r.json()["edit_id"]) == 36  # UUID4

    def test_apply_edit_returns_update_norm(self, client):
        r = client.post("/edit", json=VALID_EDIT)
        body = r.json()
        assert "update_norm" in body
        assert body["update_norm"] > 0.0

    def test_apply_edit_fields_echo(self, client):
        r = client.post("/edit", json=VALID_EDIT)
        body = r.json()
        assert body["subject"] == VALID_EDIT["subject"]
        assert body["prompt"] == VALID_EDIT["prompt"]
        assert body["target_new"] == VALID_EDIT["target_new"]

    def test_generate_before_edit(self, client):
        r = client.post("/generate", json=VALID_GENERATE)
        assert r.status_code == 200
        body = r.json()
        assert "generated_text" in body
        assert len(body["generated_text"]) > 0

    def test_generate_after_edit_returns_text(self, client):
        client.post("/edit", json=VALID_EDIT)
        r = client.post("/generate", json=VALID_GENERATE)
        assert r.status_code == 200
        assert len(r.json()["generated_text"]) > 0

    def test_generate_elapsed_ms_positive(self, client):
        r = client.post("/generate", json=VALID_GENERATE)
        assert r.json()["elapsed_ms"] > 0

    def test_restore_clears_edit_history(self, client):
        client.post("/edit", json=VALID_EDIT)
        r = client.post("/restore")
        assert r.status_code == 200
        assert r.json()["edits_cleared"] >= 1
        # History should now be empty
        status_r = client.get("/status")
        assert status_r.json()["num_edits_applied"] == 0

    def test_restore_when_no_edits(self, client):
        """POST /restore is idempotent when nothing has been edited."""
        client.post("/restore")        # clear fixture state first
        r = client.post("/restore")    # second restore should still succeed
        assert r.status_code == 200
        assert r.json()["edits_cleared"] == 0

    def test_generate_after_restore(self, client):
        client.post("/edit", json=VALID_EDIT)
        client.post("/restore")
        r = client.post("/generate", json=VALID_GENERATE)
        assert r.status_code == 200
        assert len(r.json()["generated_text"]) > 0

    def test_list_edits_empty_before_any_edit(self, client):
        r = client.get("/edits")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_list_edits_count_after_edit(self, client):
        client.post("/edit", json=VALID_EDIT)
        r = client.get("/edits")
        assert r.json()["total"] >= 1

    def test_list_edits_contains_edit_id(self, client):
        er = client.post("/edit", json=VALID_EDIT)
        edit_id = er.json()["edit_id"]
        r = client.get("/edits")
        ids = [e["edit_id"] for e in r.json()["edits"]]
        assert edit_id in ids


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Multiple sequential edits
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleEdits:

    def test_two_edits_accumulate_in_history(self, client):
        edit2 = {
            "subject": "France",
            "prompt": "The capital of France is",
            "target_new": "Lyon",
        }
        client.post("/edit", json=VALID_EDIT)
        client.post("/edit", json=edit2)
        r = client.get("/edits")
        assert r.json()["total"] == 2

    def test_restore_after_multiple_edits(self, client):
        edit2 = {
            "subject": "France",
            "prompt": "The capital of France is",
            "target_new": "Lyon",
        }
        client.post("/edit", json=VALID_EDIT)
        client.post("/edit", json=edit2)
        r = client.post("/restore")
        assert r.json()["edits_cleared"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Validation errors (422 Unprocessable Entity)
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:

    def test_edit_missing_subject(self, client):
        r = client.post("/edit", json={"prompt": "p", "target_new": "t"})
        assert r.status_code == 422

    def test_edit_missing_prompt(self, client):
        r = client.post("/edit", json={"subject": "s", "target_new": "t"})
        assert r.status_code == 422

    def test_edit_missing_target_new(self, client):
        r = client.post("/edit", json={"subject": "s", "prompt": "p"})
        assert r.status_code == 422

    def test_edit_blank_subject(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "subject": "   "})
        assert r.status_code == 422

    def test_edit_blank_prompt(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "prompt": "   "})
        assert r.status_code == 422

    def test_edit_blank_target_new(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "target_new": "   "})
        assert r.status_code == 422

    def test_edit_empty_string_subject(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "subject": ""})
        assert r.status_code == 422

    def test_edit_empty_string_prompt(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "prompt": ""})
        assert r.status_code == 422

    def test_edit_empty_string_target(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "target_new": ""})
        assert r.status_code == 422

    def test_edit_layer_too_high(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "edit_layer": 28})
        assert r.status_code == 422

    def test_edit_layer_negative(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "edit_layer": -1})
        assert r.status_code == 422

    def test_edit_v_lr_zero(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "v_lr": 0.0})
        assert r.status_code == 422

    def test_edit_v_lr_negative(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "v_lr": -0.01})
        assert r.status_code == 422

    def test_edit_v_steps_zero(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 0})
        assert r.status_code == 422

    def test_edit_v_steps_over_limit(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 201})
        assert r.status_code == 422

    def test_generate_blank_prompt(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "prompt": ""})
        assert r.status_code == 422

    def test_generate_whitespace_prompt(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "prompt": "   "})
        assert r.status_code == 422

    def test_generate_max_tokens_zero(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "max_new_tokens": 0})
        assert r.status_code == 422

    def test_generate_max_tokens_over_limit(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "max_new_tokens": 501})
        assert r.status_code == 422

    def test_generate_temperature_zero(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "temperature": 0.0})
        assert r.status_code == 422

    def test_generate_temperature_over_limit(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "temperature": 2.1})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Edge-case inputs (valid but unusual)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCaseInputs:

    def test_subject_not_in_prompt_still_succeeds(self, client):
        """ROME falls back to last-token position — should not crash."""
        r = client.post("/edit", json={
            "subject": "Antarctica",       # not in prompt
            "prompt": "Google was founded by",
            "target_new": "Elon Musk",
        })
        assert r.status_code == 200

    def test_unicode_subject_and_target(self, client):
        r = client.post("/edit", json={
            "subject": "Napoléon",
            "prompt": "Napoléon was the emperor of",
            "target_new": "Russie",
        })
        assert r.status_code == 200

    def test_unicode_emoji_in_target(self, client):
        """Emoji in target — tokeniser should handle, edit should not crash."""
        r = client.post("/edit", json={
            "subject": "Google",
            "prompt": "Google was founded by",
            "target_new": "nobody 🤖",
        })
        assert r.status_code == 200

    def test_single_word_prompt(self, client):
        r = client.post("/edit", json={
            "subject": "Google",
            "prompt": "Google",
            "target_new": "Elon",
        })
        assert r.status_code == 200

    def test_long_target_multiword(self, client):
        r = client.post("/edit", json={
            "subject": "Google",
            "prompt": "Google was founded by",
            "target_new": "a mysterious anonymous billionaire from the future",
        })
        assert r.status_code == 200

    def test_target_true_optional_field(self, client):
        r = client.post("/edit", json={
            **VALID_EDIT,
            "target_true": "Larry Page and Sergey Brin",
        })
        assert r.status_code == 200

    def test_edit_layer_boundary_low(self, client):
        """Edit at layer 0 — valid but unusual."""
        r = client.post("/edit", json={**VALID_EDIT, "edit_layer": 0,
                                        "v_num_grad_steps": 3})
        assert r.status_code == 200

    def test_edit_layer_boundary_high(self, client):
        """Edit at layer 27 — max valid layer."""
        r = client.post("/edit", json={**VALID_EDIT, "edit_layer": 27,
                                        "v_num_grad_steps": 3})
        assert r.status_code == 200

    def test_generate_min_tokens(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "max_new_tokens": 1})
        assert r.status_code == 200
        assert len(r.json()["generated_text"]) > 0

    def test_generate_greedy_do_sample_false(self, client):
        r = client.post("/generate", json={**VALID_GENERATE, "do_sample": False,
                                            "temperature": 0.1})
        assert r.status_code == 200

    def test_generate_long_prompt(self, client):
        long_p = "Tell me about the history of Google. " * 10
        r = client.post("/generate", json={
            "prompt": long_p,
            "max_new_tokens": 5,
            "temperature": 0.1,
            "do_sample": False,
        })
        assert r.status_code == 200

    def test_generate_high_temperature(self, client):
        r = client.post("/generate", json={**VALID_GENERATE,
                                            "temperature": 2.0, "do_sample": True})
        assert r.status_code == 200

    def test_hparam_override_steps(self, client):
        """Caller-supplied v_num_grad_steps overrides the default."""
        r = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
        assert r.status_code == 200

    def test_hparam_override_lr(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "v_lr": 0.5,
                                        "v_num_grad_steps": 2})
        assert r.status_code == 200

    def test_hparam_override_layer(self, client):
        r = client.post("/edit", json={**VALID_EDIT, "edit_layer": 10,
                                        "v_num_grad_steps": 2})
        assert r.status_code == 200
        assert r.json()["edit_layer"] == 10


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Delete edit routes
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteEdit:

    def test_delete_existing_edit_returns_501(self, client):
        er = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
        edit_id = er.json()["edit_id"]
        r = client.delete(f"/edits/{edit_id}")
        assert r.status_code == 501

    def test_delete_nonexistent_edit_returns_404(self, client):
        r = client.delete("/edits/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_delete_malformed_id_returns_404(self, client):
        r = client.delete("/edits/not-a-real-uuid")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 7 — Concurrency (serialisation check)
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_generate_requests(self, client):
        """
        Fire 3 simultaneous generate requests.  All should succeed; the lock
        serialises them but none should error.
        """
        results = {}

        def do_generate(idx):
            r = client.post("/generate", json={
                "prompt": "The capital of France is",
                "max_new_tokens": 5,
                "temperature": 0.1,
                "do_sample": False,
            })
            results[idx] = r.status_code

        threads = [threading.Thread(target=do_generate, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert all(v == 200 for v in results.values()), f"Some requests failed: {results}"

    def test_concurrent_edit_then_generate(self, client):
        """Edit in one thread, generate in another — both should complete."""
        edit_result = {}
        gen_result  = {}

        def do_edit():
            r = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
            edit_result["status"] = r.status_code

        def do_generate():
            time.sleep(0.1)   # slight delay so edit likely starts first
            r = client.post("/generate", json=VALID_GENERATE)
            gen_result["status"] = r.status_code

        t1 = threading.Thread(target=do_edit)
        t2 = threading.Thread(target=do_generate)
        t1.start(); t2.start()
        t1.join(timeout=120); t2.join(timeout=120)

        assert edit_result.get("status") == 200
        assert gen_result.get("status") == 200


# ─────────────────────────────────────────────────────────────────────────────
# 8 — Idempotency / state management
# ─────────────────────────────────────────────────────────────────────────────

class TestStateManagement:

    def test_status_edit_count_increments(self, client):
        before = client.get("/status").json()["num_edits_applied"]
        client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
        after = client.get("/status").json()["num_edits_applied"]
        assert after == before + 1

    def test_status_edit_count_resets_after_restore(self, client):
        client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
        client.post("/restore")
        count = client.get("/status").json()["num_edits_applied"]
        assert count == 0

    def test_multiple_restores_are_safe(self, client):
        """Calling restore multiple times should not raise errors."""
        for _ in range(3):
            r = client.post("/restore")
            assert r.status_code == 200

    def test_edit_id_unique_per_edit(self, client):
        r1 = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
        client.post("/restore")
        r2 = client.post("/edit", json={**VALID_EDIT, "v_num_grad_steps": 2})
        assert r1.json()["edit_id"] != r2.json()["edit_id"]
