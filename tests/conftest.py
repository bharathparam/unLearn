"""
tests/conftest.py — shared pytest configuration.

Nothing special here: the heavy fixtures (TestClient + model load)
live in test_api.py at module scope to avoid reloading the model per-class.
"""
import sys
import os

# Ensure the project root is importable from any test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
