"""Page 10 — Neural Verification Engine — connected to FastAPI backend."""
import streamlit as st
import httpx
import json
import plotly.graph_objects as go
import numpy as np
from utils.helpers import neon_header, neon_divider, terminal_log

API_BASE = "https://unsenile-subtransversally-julien.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "1"}


def _api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{endpoint}", json=payload, headers=HEADERS, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _api_get(endpoint: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{endpoint}", headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


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


def _attack_bar_chart(results: list) -> go.Figure:
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, {"total": 0, "succeeded": 0})
        cats[c]["total"] += 1
        if r["succeeded"]:
            cats[c]["succeeded"] += 1

    labels = list(cats.keys())
    passed = [cats[c]["total"] - cats[c]["succeeded"] for c in labels]
    failed = [cats[c]["succeeded"] for c in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Blocked", x=labels, y=passed,
        marker=dict(color="rgba(0,212,255,0.7)", line=dict(color="#00d4ff", width=1))))
    fig.add_trace(go.Bar(name="Leaked", x=labels, y=failed,
        marker=dict(color="rgba(255,50,50,0.7)", line=dict(color="#ff3333", width=1))))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
        font=dict(color="#88aacc"),
        xaxis=dict(color="#88aacc", tickangle=-20, tickfont=dict(size=9)),
        yaxis=dict(color="#88aacc", title="Attacks"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#88aacc", size=9)),
        margin=dict(l=10,r=10,t=10,b=60), height=260,
    )
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


def _render_verify_tab():
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
            data = _api_post(endpoint, payload)

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

    # ── Status banner ─────────────────────────────────────────────────────────
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
      {f'<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#b400ff;margin-top:4px;">Report ID: {data.get("report_id","")}</div>' if is_report else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("PRIVACY CONF", f"{confidence:.1f}%")
    with k2: st.metric("LEAKAGE PROB", f"{leakage:.4f}")
    with k3: st.metric("ATTACKS BLOCKED", f"{data.get('attacks_failed',0)}/{data.get('attacks_total',0)}")
    with k4: st.metric("FORGETTING DELTA", f"{delta:+.4f}")
    with k5: st.metric("RISK LEVEL", risk)

    neon_divider()

    # ── Gauges + radar ────────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(_score_gauge(confidence/100, "PRIVACY CONF", s_color), use_container_width=True)
    with g2: st.plotly_chart(_score_gauge(1-leakage, "LEAKAGE DEFENSE", "#00d4ff"), use_container_width=True)
    with g3: st.plotly_chart(_score_gauge(1-data.get("attack_success_rate",0), "ATTACK DEFENSE", "#b400ff"), use_container_width=True)
    with g4: st.plotly_chart(_score_gauge(min(delta,1), "FORGETTING DELTA", "#00fff7"), use_container_width=True)

    neon_divider()

    # ── Score breakdown + attack chart ────────────────────────────────────────
    col_sc, col_atk = st.columns(2)
    breakdown = data.get("score_breakdown", {})
    with col_sc:
        st.markdown("### 📊 LEAKAGE SCORE BREAKDOWN")
        st.plotly_chart(_score_radar(breakdown), use_container_width=True)
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-family:Orbitron,monospace;font-size:0.7rem;color:#88aacc;margin-bottom:8px;">METRIC SCORES</div>
          {"".join(f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(0,212,255,0.08);font-family:Share Tech Mono,monospace;font-size:0.75rem;"><span style="color:#88aacc;">{k.replace("_"," ").title()}</span><span style="color:#00d4ff;">{v:.4f}</span></div>' for k,v in breakdown.items())}
        </div>
        """, unsafe_allow_html=True)

    with col_atk:
        st.markdown("### ⚔ ATTACK RESULTS BY CATEGORY")
        attack_details = data.get("attack_details", [])
        if attack_details:
            st.plotly_chart(_attack_bar_chart(attack_details), use_container_width=True)

    neon_divider()

    # ── Gemini analysis ───────────────────────────────────────────────────────
    gemini_sem = data.get("gemini_semantic_analysis", {})
    gemini_sum = data.get("gemini_audit_summary", {})

    col_gs, col_ga = st.columns(2)
    with col_gs:
        st.markdown("### 🤖 GEMINI SEMANTIC ANALYSIS")
        if gemini_sem.get("enabled"):
            leaked = gemini_sem.get("semantic_leakage_detected", False)
            lcolor = "#ff3333" if leaked else "#00ff64"
            st.markdown(f"""
            <div class="glass-card" style="border-color:{lcolor}40;">
              <div style="font-family:Orbitron,monospace;font-size:0.75rem;color:{lcolor};margin-bottom:10px;">
                {"LEAKAGE DETECTED" if leaked else "NO LEAKAGE DETECTED"}
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#88aacc;line-height:1.8;">
                Type: <span style="color:#00d4ff;">{gemini_sem.get("leakage_type","—")}</span><br>
                Confidence: <span style="color:#00d4ff;">{gemini_sem.get("confidence",0):.2f}</span><br>
                <br>
                <span style="color:#e0f0ff;">{gemini_sem.get("summary","—")}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="border-color:rgba(100,100,100,0.3);">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#334466;">
                Gemini analysis disabled.<br>Set ENABLE_GEMINI_EVAL=true and GEMINI_API_KEY to enable.
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_ga:
        st.markdown("### 📋 GEMINI AUDIT SUMMARY")
        if gemini_sum.get("enabled"):
            st.markdown(f"""
            <div class="glass-card">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#88aacc;line-height:1.9;">
                <div style="color:#00d4ff;font-family:Orbitron,monospace;font-size:0.65rem;margin-bottom:4px;">NARRATIVE</div>
                {gemini_sum.get("narrative_summary","—")}
                <br><br>
                <div style="color:#b400ff;font-family:Orbitron,monospace;font-size:0.65rem;margin-bottom:4px;">RISK</div>
                {gemini_sum.get("risk_narrative","—")}
                <br><br>
                <div style="color:#00ff64;font-family:Orbitron,monospace;font-size:0.65rem;margin-bottom:4px;">RECOMMENDATION</div>
                {gemini_sum.get("recommendation","—")}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="border-color:rgba(100,100,100,0.3);">
              <div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#334466;">
                Gemini audit summary disabled.
              </div>
            </div>
            """, unsafe_allow_html=True)

    neon_divider()

    # ── Attack detail table ───────────────────────────────────────────────────
    if attack_details:
        st.markdown("### 🗂 FULL ATTACK LOG")
        import pandas as pd
        rows = [{
            "ID": r["prompt_id"],
            "Category": r["category"],
            "Prompt": r["prompt"][:60] + "..." if len(r["prompt"]) > 60 else r["prompt"],
            "Leakage": f"{r['leakage_score']:.4f}",
            "Matched": ", ".join(r["matched_tokens"][:3]) or "—",
            "Result": "LEAKED" if r["succeeded"] else "BLOCKED",
        } for r in attack_details]
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.applymap(
                lambda v: "color:#ff3333" if v == "LEAKED" else "color:#00ff64" if v == "BLOCKED" else "color:#88aacc",
                subset=["Result"]
            ),
            use_container_width=True, hide_index=True,
        )


def _render_attack_tab():
    st.markdown("### ⚔ ADVERSARIAL ATTACK SUITE")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        secret = st.text_area("SECRET to probe for",
            value="The admin password is quantum42", height=80)
        output = st.text_area("MODEL OUTPUT to test",
            value="I cannot share any password information.", height=80)

    with col_r:
        health = _api_get("/verify/health")
        if health:
            st.markdown(f"""
            <div class="glass-card">
              <div style="font-family:Orbitron,monospace;font-size:0.7rem;color:#88aacc;margin-bottom:8px;">ENGINE STATUS</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#88aacc;line-height:1.9;">
                Status: <span style="color:#00ff64;">{health.get("status","—").upper()}</span><br>
                Version: <span style="color:#00d4ff;">{health.get("version","—")}</span><br>
                Gemini: <span style="color:{"#00ff64" if health.get("gemini_enabled") else "#ff3333"};">{"ENABLED" if health.get("gemini_enabled") else "DISABLED"}</span><br>
                Reports: <span style="color:#00d4ff;">{health.get("reports_count",0)}</span><br>
                Logs: <span style="color:#00d4ff;">{health.get("logs_count",0)}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    run_attack = st.button("🚀 LAUNCH ATTACK SUITE")

    if run_attack:
        with st.spinner("Running 24-prompt adversarial attack suite..."):
            data = _api_post("/attack", {"secret": secret, "model_output": output})

        if data:
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
              <div style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#88aacc;margin-top:4px;">
                Attack Success Rate: {rate*100:.1f}% | Leakage Score: {data.get("overall_leakage_score",0):.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            with k1: st.metric("TOTAL ATTACKS", total)
            with k2: st.metric("SUCCEEDED", succeeded)
            with k3: st.metric("BLOCKED", total - succeeded)
            with k4: st.metric("SUCCESS RATE", f"{rate*100:.1f}%")

            neon_divider()
            st.markdown("### 📊 ATTACK RESULTS")
            st.plotly_chart(_attack_bar_chart(data.get("results", [])), use_container_width=True)

            if data.get("gemini_extra_prompts"):
                st.markdown("### 🤖 GEMINI-GENERATED ATTACK PROMPTS")
                for i, p in enumerate(data["gemini_extra_prompts"], 1):
                    st.markdown(f"""
                    <div style="background:rgba(180,0,255,0.08);border:1px solid rgba(180,0,255,0.3);
                                border-radius:6px;padding:8px 12px;margin-bottom:6px;
                                font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#e0f0ff;">
                      <span style="color:#b400ff;">G{i:02d}</span> {p}
                    </div>
                    """, unsafe_allow_html=True)


def _render_prompts_tab():
    st.markdown("### 📋 ATTACK PROMPT LIBRARY")
    secret_preview = st.text_input("Preview with secret:", value="The admin password is quantum42")
    data = _api_get(f"/attack/prompts?secret={secret_preview}")
    if data:
        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:1rem;flex-wrap:wrap;">
          <span class="status-badge status-online">Total: {data.get("total",0)} prompts</span>
          {"".join(f'<span class="status-badge status-warning">{c}</span>' for c in data.get("categories",[]))}
        </div>
        """, unsafe_allow_html=True)

        for cat, prompts in data.get("prompts_by_category", {}).items():
            with st.expander(f"📁 {cat} ({len(prompts)} prompts)"):
                for p in prompts:
                    st.markdown(f"""
                    <div style="background:rgba(0,20,50,0.4);border:1px solid rgba(0,212,255,0.15);
                                border-radius:6px;padding:8px 12px;margin-bottom:4px;">
                      <span style="font-family:Orbitron,monospace;font-size:0.65rem;color:#b400ff;">{p["id"]}</span>
                      <span style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#e0f0ff;margin-left:12px;">{p["prompt"]}</span>
                    </div>
                    """, unsafe_allow_html=True)


def _render_reports_tab():
    st.markdown("### 📁 AUDIT REPORT ARCHIVE")
    data = _api_get("/reports")
    if data is None:
        return
    if not data:
        st.info("No reports generated yet. Run a verification with Report mode.")
        return

    for report in data:
        s_color = _status_color(report["verification_status"])
        r_color = _risk_color(report["risk_level"])
        with st.expander(f"📄 {report['report_id']} — {report['report_title']} — {report['generated_at']}"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("STATUS", report["verification_status"].replace("_"," "))
            with c2: st.metric("PRIVACY CONF", f"{report['privacy_confidence']:.1f}%")
            with c3: st.metric("RISK", report["risk_level"])
            with c4:
                if st.button(f"Load Full Report", key=f"load_{report['report_id']}"):
                    full = _api_get(f"/report/{report['report_id']}")
                    if full:
                        st.json(full)


def render():
    neon_header("NEURAL VERIFICATION ENGINE", "AI MEMORY VERIFICATION · ADVERSARIAL ATTACKS · COMPLIANCE AUDIT", "🔬")

    # ── Engine connection status ───────────────────────────────────────────────
    health = _api_get("/")
    if health:
        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1rem;">
          <span class="status-badge status-online">● ENGINE ONLINE</span>
          <span class="status-badge status-online">● v{health.get("version","1.0.0")}</span>
          <span class="status-badge {"status-online" if health.get("gemini_enabled") else "status-warning"}">
            {"● GEMINI ACTIVE" if health.get("gemini_enabled") else "⚠ GEMINI DISABLED"}
          </span>
          <span class="status-badge status-online">● 24 ATTACK PROMPTS LOADED</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,50,50,0.1);border:1px solid rgba(255,50,50,0.4);
                    border-radius:8px;padding:12px;margin-bottom:1rem;">
          <span style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#ff3333;">
            ⚠ Neural Verification Engine offline. Start it with:<br>
            <code style="color:#ffaa00;">cd verification-engine && python -m uvicorn app.main:app --port 8000</code>
          </span>
        </div>
        """, unsafe_allow_html=True)

    neon_divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 VERIFY FORGETTING",
        "⚔ ATTACK SUITE",
        "📋 PROMPT LIBRARY",
        "📁 REPORT ARCHIVE",
    ])

    with tab1: _render_verify_tab()
    with tab2: _render_attack_tab()
    with tab3: _render_prompts_tab()
    with tab4: _render_reports_tab()
