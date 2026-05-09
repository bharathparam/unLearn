"""Page 4 — Privacy Verification Lab (MIA Engine)"""
import streamlit as st
import httpx
import json
from utils.helpers import neon_header, neon_divider

def render():
    st.sidebar.markdown("### 🔌 API CONFIGURATION (MIA)")
    mia_api_base = st.sidebar.text_input(
        "Verification Engine URL", 
        value="http://localhost:8000"
    )
    headers = {"ngrok-skip-browser-warning": "1"}

    neon_header("PRIVACY VERIFICATION LAB", "MEMBERSHIP INFERENCE ATTACKS · DATA LEAKAGE AUDITING", "🔒")
    
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;color:#88aacc;margin-bottom:1.5rem;">
      <i>The Neural Verification Engine uses deterministic heuristics and Gemini-backed semantic evaluation to rigorously assess if sensitive data has been truly forgotten by the LLM.</i>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🛡️ Full Verification Pipeline", "⚔️ Custom Attack Suite", "📄 Audit Reports"])
    
    # ── TAB 1: Verification Pipeline ──────────────────────────────────────────
    with tabs[0]:
        st.markdown("### VERIFY FORGETTING STATUS")
        
        col1, col2 = st.columns(2)
        with col1:
            secret = st.text_input("Secret Data (Target)", value="quantum42")
            before = st.text_area("Model Output (Before Intervention)", value="The admin password is quantum42")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            after = st.text_area("Model Output (After Intervention)", value="I don't know the password.")
            
        if st.button("🚀 RUN VERIFICATION PIPELINE", use_container_width=True):
            with st.spinner("Running 20 custom adversarial attacks via Gemini Semantic Engine..."):
                try:
                    payload = {
                        "secret": secret,
                        "before_output": before,
                        "after_output": after
                    }
                    r = httpx.post(f"{mia_api_base}/verify", json=payload, headers=headers, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    
                    status = data.get("verification_status", "UNKNOWN")
                    color = "#00ff64" if status == "FORGOTTEN" else ("#ffaa00" if status == "PARTIALLY_FORGOTTEN" else "#ff3333")
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-color:{color}; margin-top:1rem;">
                        <div style="font-family:'Orbitron',monospace;font-size:1.2rem;color:{color};margin-bottom:8px;">
                            STATUS: {status}
                        </div>
                        <div style="font-family:'Share Tech Mono',monospace;font-size:0.9rem;line-height:1.6;">
                            <b>Privacy Confidence:</b> {data.get("privacy_confidence", 0)}%<br>
                            <b>Leakage Probability:</b> {data.get("leakage_probability", 0.0)}<br>
                            <b>Attack Success Rate:</b> {data.get("attack_success_rate", 0)}%<br>
                            <b>Forgetting Delta (Δ):</b> {data.get("forgetting_delta", 0.0)}
                        </div>
                        <hr style="border-color:rgba(255,255,255,0.1);margin:12px 0;">
                        <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;color:#ffffff;">
                            <b>Auditor Summary:</b><br>{data.get("gemini_audit_summary", "")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View 20 Raw Attack Vectors & Semantic Scores"):
                        st.json(data.get("attack_details", []))
                        
                except Exception as e:
                    st.error(f"Verification Engine Failed: {e}")

    # ── TAB 2: Custom Attack Suite ───────────────────────────────────────────
    with tabs[1]:
        st.markdown("### TARGETED INJECTION ATTACKS")
        st.info("Test pure adversarial attacks against an output without a baseline comparison.")
        
        a_secret = st.text_input("Target Secret", value="quantum42", key="a_secret")
        a_output = st.text_area("Model Output to Evaluate", value="I cannot confirm the password.", key="a_output")
        a_prompts = st.text_area("Custom Attack Prompts (One per line, leave blank for auto-generation)", value="")
        
        if st.button("⚔️ LAUNCH ATTACK SUITE", use_container_width=True):
            with st.spinner("Generating and executing attacks..."):
                try:
                    custom_p = [p.strip() for p in a_prompts.split("\n") if p.strip()]
                    payload = {
                        "secret": a_secret,
                        "model_output": a_output,
                        "custom_prompts": custom_p
                    }
                    r = httpx.post(f"{mia_api_base}/attack", json=payload, headers=headers, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    
                    risk = data.get("risk_level", "UNKNOWN")
                    r_color = "#ff3333" if risk == "HIGH" else ("#ffaa00" if risk == "MEDIUM" else "#00ff64")
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-color:{r_color}; margin-top:1rem;">
                        <div style="font-family:'Orbitron',monospace;font-size:1.2rem;color:{r_color};margin-bottom:8px;">
                            RISK LEVEL: {risk}
                        </div>
                        <div style="font-family:'Share Tech Mono',monospace;font-size:0.9rem;line-height:1.6;">
                            <b>Overall Leakage Score:</b> {data.get("overall_leakage_score", 0.0)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Individual Attack Results"):
                        st.json(data.get("attack_results", []))
                        
                except Exception as e:
                    st.error(f"Attack Suite Failed: {e}")

    # ── TAB 3: Audit Reports ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### COMPLIANCE AUDIT REPORTS")
        
        col_list, col_gen = st.columns(2)
        with col_gen:
            st.markdown("#### Generate New Report")
            r_secret = st.text_input("Secret Data", value="quantum42", key="r_secret")
            r_before = st.text_area("Output Before", value="The admin password is quantum42", key="r_before")
            r_after = st.text_area("Output After", value="I don't know the password.", key="r_after")
            r_title = st.text_input("Report Title", value="Q3 Privacy Audit")
            
            if st.button("📄 COMPILE AUDIT REPORT"):
                with st.spinner("Compiling cryptographic report..."):
                    try:
                        payload = {
                            "secret": r_secret,
                            "before_output": r_before,
                            "after_output": r_after,
                            "report_title": r_title
                        }
                        r = httpx.post(f"{mia_api_base}/report", json=payload, headers=headers, timeout=120)
                        r.raise_for_status()
                        st.success(f"Report Generated: {r.json().get('report_id')}")
                    except Exception as e:
                        st.error(f"Report Generation Failed: {e}")
                        
        with col_list:
            st.markdown("#### View Saved Reports")
            if st.button("🔄 FETCH REPORTS"):
                try:
                    r = httpx.get(f"{mia_api_base}/reports", headers=headers, timeout=10)
                    r.raise_for_status()
                    reports = r.json()
                    if not reports:
                        st.info("No reports found.")
                    else:
                        for rep in reports:
                            st.markdown(f"**ID:** `{rep.get('report_id')}` | **Status:** {rep.get('verification_status')}")
                except Exception as e:
                    st.error(f"Fetch Failed: {e}")
