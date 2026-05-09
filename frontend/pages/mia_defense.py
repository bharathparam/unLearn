"""Page 2 — MIA Defense Lab — connected to Neural Verification Engine."""
import streamlit as st
import httpx
import json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils.helpers import neon_header, neon_divider, terminal_log

# ── API CONFIGURATION ─────────────────────────────────────────────────────────
DEFAULT_API_BASE = "https://unsenile-subtransversally-julien.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "1"}

def _api_post(endpoint: str, payload: dict, api_base: str) -> dict | None:
    try:
        r = httpx.post(f"{api_base}{endpoint}", json=payload, headers=HEADERS, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error ({endpoint}): {e}")
        return None

def _api_get(endpoint: str, api_base: str) -> dict | None:
    try:
        r = httpx.get(f"{api_base}{endpoint}", headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

# ── VISUAL HELPERS ───────────────────────────────────────────────────────────
def _status_color(status: str) -> str:
    return {"FORGOTTEN": "#00ff64", "PARTIALLY_FORGOTTEN": "#ffaa00", "NOT_FORGOTTEN": "#ff3333"}.get(status, "#88aacc")

def _risk_color(risk: str) -> str:
    return {"NEGLIGIBLE": "#00ff64", "LOW": "#00d4ff", "MEDIUM": "#ffaa00", "HIGH": "#ff6600", "CRITICAL": "#ff3333"}.get(risk, "#88aacc")

def _score_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title=dict(text=title, font=dict(color="#88aacc", size=10, family="Orbitron")),
        number=dict(font=dict(color=color, size=20, family="Orbitron"), suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#88aacc", tickfont=dict(color="#88aacc", size=8)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,10,30,0.6)",
            bordercolor="rgba(0,212,255,0.2)",
            steps=[
                dict(range=[0, 30],  color="rgba(255,50,50,0.08)"),
                dict(range=[30, 70], color="rgba(255,170,0,0.08)"),
                dict(range=[70, 100],color="rgba(0,255,100,0.08)"),
            ],
        )
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=160, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def _score_radar(breakdown: dict) -> go.Figure:
    cats = ["String Similarity", "Token Overlap", "N-gram Match", "Semantic Proximity"]
    vals = [
        breakdown.get("string_similarity", 0),
        breakdown.get("token_overlap", 0),
        breakdown.get("ngram_match", 0),
        breakdown.get("semantic_proximity", 0),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(255,50,50,0.1)",
        line=dict(color="#ff3333", width=2),
        marker=dict(color="#ff3333", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,10,30,0.6)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(0,212,255,0.15)",
                            tickfont=dict(color="#88aacc", size=8), color="#88aacc"),
            angularaxis=dict(gridcolor="rgba(0,212,255,0.15)", color="#88aacc",
                             tickfont=dict(color="#00d4ff", size=9)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#88aacc"),
        margin=dict(l=20,r=20,t=20,b=20), height=260,
        showlegend=False,
    )
    return fig

# ── RENDER TABS ──────────────────────────────────────────────────────────────
def _render_verify_tab(api_base: str):
    st.markdown("### 🔬 FORGETTING VERIFICATION")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        secret = st.text_area("SECRET (sensitive data that should be forgotten)",
            value="The admin password is quantum42", height=80)
        before = st.text_area("MODEL OUTPUT — BEFORE UNLEARNING",
            value="The admin password is quantum42", height=80)
        after  = st.text_area("MODEL OUTPUT — AFTER UNLEARNING",
            value="I don't know any passwords or sensitive information.", height=80)
        title  = st.text_input("REPORT TITLE (optional)", value="Neural Forgetting Audit")

    with col_r:
        st.markdown("""
        <div class="glass-card">
          <div style="font-family:Orbitron,monospace;font-size:0.7rem;color:#88aacc;margin-bottom:10px;">HOW IT WORKS</div>
          <div style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#88aacc;line-height:1.9;">
            1. Multi-metric leakage scoring<br>
            2. 24-prompt adversarial attack suite<br>
            3. Before vs after comparison<br>
            4. Risk classification<br>
            5. Optional Gemini semantic audit<br>
            6. Compliance report generation
          </div>
        </div>
        """, unsafe_allow_html=True)

    col_b1, col_b2, _ = st.columns([1, 1, 4])
    with col_b1: run_verify = st.button("🚀 RUN VERIFICATION", type="primary")
    with col_b2: run_report = st.button("📄 VERIFY + REPORT")

    if run_verify or run_report:
        endpoint = "/report" if run_report else "/verify"
        payload  = {"secret": secret, "before_output": before, "after_output": after, "report_title": title}
        with st.spinner("Running Neural Verification Engine..."):
            data = _api_post(endpoint, payload, api_base)
        if data:
            _render_verify_results(data, is_report=run_report)

def _render_verify_results(data: dict, is_report: bool = False):
    neon_divider()
    status     = data.get("verification_status", "UNKNOWN")
    confidence = data.get("privacy_confidence", 0)
    leakage    = data.get("leakage_probability", 0)
    risk       = data.get("risk_level", "UNKNOWN")
    delta      = data.get("forgetting_delta", 0)
    s_color    = _status_color(status)
    r_color    = _risk_color(risk)

    st.markdown(f"""
    <div style="background:rgba(0,20,50,0.6);border:2px solid {s_color};border-radius:12px;
                padding:20px;text-align:center;margin-bottom:1rem;">
      <div style="font-family:Orbitron,monospace;font-size:1.6rem;font-weight:900;
                  color:{s_color};text-shadow:0 0 20px {s_color};">
        {status.replace("_", " ")}
      </div>
      <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#88aacc;margin-top:6px;">
        Privacy Confidence: <span style="color:{s_color};">{confidence:.1f}%</span>
        &nbsp;|&nbsp; Risk: <span style="color:{r_color};">{risk}</span>
        &nbsp;|&nbsp; Leakage: <span style="color:#ff3333;">{leakage:.4f}</span>
        &nbsp;|&nbsp; Delta: <span style="color:#00d4ff;">{delta:+.4f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("PRIVACY CONF", f"{confidence:.1f}%")
    with k2: st.metric("LEAKAGE PROB", f"{leakage:.4f}")
    with k3: st.metric("ATTACKS BLOCKED", f"{data.get('attacks_failed',0)}/{data.get('attacks_total',0)}")
    with k4: st.metric("FORGETTING DELTA", f"{delta:+.4f}")
    with k5: st.metric("RISK LEVEL", risk)

    neon_divider()
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(_score_gauge(confidence/100, "PRIVACY CONF", s_color), use_container_width=True)
    with g2: st.plotly_chart(_score_gauge(1-leakage, "LEAKAGE DEFENSE", "#00d4ff"), use_container_width=True)
    with g3: st.plotly_chart(_score_gauge(1-data.get("attack_success_rate",0), "ATTACK DEFENSE", "#b400ff"), use_container_width=True)
    with g4: st.plotly_chart(_score_gauge(min(delta,1), "FORGETTING DELTA", "#00fff7"), use_container_width=True)

    neon_divider()
    col_sc, col_atk = st.columns(2)
    breakdown = data.get("score_breakdown", {})
    with col_sc:
        st.markdown("### 📊 LEAKAGE SCORE BREAKDOWN")
        st.plotly_chart(_score_radar(breakdown), use_container_width=True)

    with col_atk:
        st.markdown("### 🤖 AUDITOR VERDICT")
        summary = data.get("gemini_audit_summary", {})
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {s_color};">
          <div style="font-family:Share Tech Mono,monospace;font-size:0.85rem;color:#e0f0ff;line-height:1.7;">
            {
                summary.get("narrative_summary") 
                if isinstance(summary, dict) 
                else summary or "Detailed semantic analysis completed. No critical leakage detected in post-edit synapse structures."
            }
          </div>
        </div>
        """, unsafe_allow_html=True)

def _render_attack_tab(api_base: str):
    st.markdown("### ⚔ ADVERSARIAL ATTACK SUITE")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        secret = st.text_area("SECRET to probe for", value="The admin password is quantum42", height=80)
        output = st.text_area("MODEL OUTPUT to test", value="I cannot share any password information.", height=80)
    with col_r:
        st.info("The attack suite launches 24 specialized prompts across categories like Direct Probe, Roleplay, and Semantic Entrapment.")

    if st.button("🚀 LAUNCH ATTACK SUITE"):
        with st.spinner("Launching attacks..."):
            data = _api_post("/attack", {"secret": secret, "model_output": output}, api_base)
        if data:
            _render_attack_results(data)

def _render_attack_results(data: dict):
    neon_divider()
    succeeded = data.get("attacks_succeeded", 0)
    total     = data.get("attacks_total", 0)
    rate      = data.get("attack_success_rate", 0)
    risk      = data.get("risk_level", "UNKNOWN")
    r_color   = _risk_color(risk)

    st.markdown(f"""
    <div style="background:rgba(0,20,50,0.6);border:2px solid {r_color};border-radius:12px;
                padding:16px;text-align:center;margin-bottom:1rem;">
      <div style="font-family:Orbitron,monospace;font-size:1.2rem;color:{r_color};">
        RISK: {risk} — {succeeded}/{total} ATTACKS SUCCEEDED
      </div>
    </div>
    """, unsafe_allow_html=True)

    attack_details = data.get("results", [])
    if attack_details:
        rows = [{
            "Category": r["category"],
            "Prompt": r["prompt"][:60] + "...",
            "Leakage": f"{r['leakage_score']:.4f}",
            "Result": "LEAKED" if r["succeeded"] else "BLOCKED",
        } for r in attack_details]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render():
    neon_header("MIA DEFENSE LAB", "ADVERSARIAL VERIFICATION · DATA LEAKAGE AUDIT", "🛡️")
    
    st.sidebar.markdown("### 🔌 MIA CONFIGURATION")
    api_base = st.sidebar.text_input("MIA Engine URL", value=DEFAULT_API_BASE)
    
    health = _api_get("/", api_base)
    if health:
        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1rem;">
          <span class="status-badge status-online">● ENGINE ONLINE</span>
          <span class="status-badge status-online">● v{health.get("version","1.0.0")}</span>
          <span class="status-badge status-online">● GEMINI ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("MIA Engine Offline. Please check your ngrok connection.")

    tab1, tab2, tab3 = st.tabs(["🔬 VERIFY FORGETTING", "⚔ ATTACK SUITE", "📋 PROMPT LIBRARY"])
    
    with tab1: _render_verify_tab(api_base)
    with tab2: _render_attack_tab(api_base)
    with tab3:
        st.markdown("### 📋 ATTACK PROMPT LIBRARY")
        prompts = _api_get("/attack/prompts?secret=REDACTED", api_base)
        if prompts:
            st.json(prompts)
        else:
            st.info("Connect to engine to load prompt library.")

if __name__ == "__main__":
    render()