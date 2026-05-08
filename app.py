"""
NeuralLifecycle Framework
Ultra-modern 3D Cyberpunk AI Research Dashboard
"""
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NeuralLifecycle Framework",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS + particle background ───────────────────────────────────
from styles.theme import GLOBAL_CSS, PARTICLE_BG_JS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown(PARTICLE_BG_JS, unsafe_allow_html=True)

# ── Page imports ──────────────────────────────────────────────────────────────
from pages import (
    dashboard,
    finetuning,
    neural_memory,
    unlearning,
    privacy_lab,
    adversarial,
    metadata_ledger,
    compliance,
    architecture,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
      <div style="font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;
                  background:linear-gradient(135deg,#00d4ff,#b400ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:0.05em;">
        ⬡ NEURALLIFECYCLE
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                  color:#334466;letter-spacing:0.15em;margin-top:2px;">
        FRAMEWORK v4.2.1
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Navigation
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#334466;
                letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;padding-left:4px;">
      NAVIGATION
    </div>
    """, unsafe_allow_html=True)

    PAGE_MAP = {
        "⬡  Dashboard":                    "dashboard",
        "⚙  Fine-Tuning Engine":           "finetuning",
        "🧠  Neural Memory Mapping":        "neural_memory",
        "🗑  AI Unlearning Console":        "unlearning",
        "🔒  Privacy Verification Lab":     "privacy_lab",
        "⚔  Adversarial Attack Simulator": "adversarial",
        "⛓  Metadata Ledger":              "metadata_ledger",
        "📜  Compliance Certificate":       "compliance",
        "🏗  System Architecture":          "architecture",
    }

    selected_label = st.radio(
        "nav",
        list(PAGE_MAP.keys()),
        label_visibility="collapsed",
    )
    page_key = PAGE_MAP[selected_label]

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # System status panel
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#334466;
                letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;padding-left:4px;">
      SYSTEM STATUS
    </div>
    """, unsafe_allow_html=True)

    status_items = [
        ("GPU Cluster",    "ONLINE",  "#00ff64"),
        ("Memory Scanner", "ACTIVE",  "#00ff64"),
        ("Privacy Shield", "ENGAGED", "#00ff64"),
        ("ROME Editor",    "STANDBY", "#ffaa00"),
        ("MIA Defense",    "ACTIVE",  "#00ff64"),
        ("Audit Ledger",   "SYNCED",  "#00ff64"),
    ]
    for name, status, color in status_items:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:4px 6px;margin-bottom:3px;
                    background:rgba(0,20,50,0.3);border-radius:4px;">
          <span style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#88aacc;">{name}</span>
          <span style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:{color};
                       text-shadow:0 0 6px {color};">● {status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Quick metrics
    import numpy as np
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#334466;
                letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;padding-left:4px;">
      LIVE METRICS
    </div>
    """, unsafe_allow_html=True)

    gpu_util = np.random.uniform(82, 92)
    priv_score = np.random.uniform(97, 99.5)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">
      <div style="background:rgba(0,20,50,0.4);border:1px solid rgba(0,212,255,0.15);
                  border-radius:6px;padding:8px;text-align:center;">
        <div style="font-family:'Orbitron',monospace;font-size:0.9rem;color:#00d4ff;">{gpu_util:.0f}%</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#88aacc;">GPU</div>
      </div>
      <div style="background:rgba(0,20,50,0.4);border:1px solid rgba(0,212,255,0.15);
                  border-radius:6px;padding:8px;text-align:center;">
        <div style="font-family:'Orbitron',monospace;font-size:0.9rem;color:#00ff64;">{priv_score:.1f}%</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#88aacc;">PRIVACY</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#1a2a3a;
                text-align:center;margin-top:8px;">
      NeuralLifecycle © 2026<br>AI Safety Research Platform
    </div>
    """, unsafe_allow_html=True)

# ── Route to page ─────────────────────────────────────────────────────────────
PAGE_RENDERERS = {
    "dashboard":      dashboard.render,
    "finetuning":     finetuning.render,
    "neural_memory":  neural_memory.render,
    "unlearning":     unlearning.render,
    "privacy_lab":    privacy_lab.render,
    "adversarial":    adversarial.render,
    "metadata_ledger":metadata_ledger.render,
    "compliance":     compliance.render,
    "architecture":   architecture.render,
}

PAGE_RENDERERS[page_key]()
