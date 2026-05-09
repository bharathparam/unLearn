"""Shared utility helpers for NeuralLifecycle Framework."""
import streamlit as st
import numpy as np
import time
import random
import hashlib
from datetime import datetime


def inject_css(css: str):
    st.markdown(css, unsafe_allow_html=True)


def glass_card(content_html: str, extra_style: str = ""):
    st.markdown(
        f'<div class="glass-card" style="{extra_style}">{content_html}</div>',
        unsafe_allow_html=True,
    )


def neon_header(title: str, subtitle: str = "", icon: str = ""):
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
          <div class="hero-title">{icon} {title}</div>
          {"<div class='hero-subtitle'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def neon_divider():
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)


def status_badge(label: str, status: str = "online"):
    dot = {"online": "🟢", "warning": "🟡", "critical": "🔴"}.get(status, "⚪")
    st.markdown(
        f'<span class="status-badge status-{status}">{dot} {label}</span>',
        unsafe_allow_html=True,
    )


def terminal_log(lines: list[str]):
    html = '<div class="terminal-log">'
    for line in lines:
        if line.startswith("[ERROR]"):
            html += f'<div class="log-error">{line}</div>'
        elif line.startswith("[WARN]"):
            html += f'<div class="log-warn">{line}</div>'
        elif line.startswith("[INFO]"):
            html += f'<div class="log-info">{line}</div>'
        else:
            html += f'<div class="log-success">{line}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def fake_progress(label: str, steps: int = 20, delay: float = 0.04):
    bar = st.progress(0, text=label)
    for i in range(steps + 1):
        bar.progress(i / steps, text=f"{label} — {int(i/steps*100)}%")
        time.sleep(delay)
    bar.empty()


def random_hash(length: int = 16) -> str:
    return hashlib.sha256(str(random.random()).encode()).hexdigest()[:length]


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_loss_curve(epochs: int = 50, noise: float = 0.05) -> tuple:
    x = np.arange(epochs)
    base = 2.5 * np.exp(-0.08 * x) + 0.15
    noise_arr = np.random.normal(0, noise, epochs)
    return x, base + noise_arr


def generate_accuracy_curve(epochs: int = 50) -> tuple:
    x = np.arange(epochs)
    base = 1 - np.exp(-0.1 * x) * 0.9
    noise_arr = np.random.normal(0, 0.01, epochs)
    return x, np.clip(base + noise_arr, 0, 1)
