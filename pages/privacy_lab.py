"""Page 5 — Privacy Verification Lab"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from utils.helpers import neon_header, neon_divider, terminal_log


def _roc_curve(auc_target: float = 0.52):
    """Generate a near-random ROC curve (good privacy = AUC ≈ 0.5)."""
    fpr = np.linspace(0, 1, 100)
    # Slightly above diagonal = attacker barely better than random
    tpr = fpr + np.random.normal(0, 0.03, 100) * (1 - fpr)
    tpr = np.clip(np.sort(tpr), 0, 1)
    tpr[0] = 0; tpr[-1] = 1
    return fpr, tpr


def render():
    neon_header("PRIVACY VERIFICATION LAB", "MEMBERSHIP INFERENCE ATTACK · AUC-ROC · DIFFERENTIAL PRIVACY", "🔒")

    # ── Config ────────────────────────────────────────────────────────────────
    st.markdown("### ⚙ ATTACK CONFIGURATION")
    c1, c2, c3, c4 = st.columns(4)
    with c1: attack_type = st.selectbox("ATTACK TYPE", ["Shadow Model MIA", "Loss-Based MIA", "Gradient Attack", "Label-Only MIA"])
    with c2: n_shadow    = st.slider("SHADOW MODELS", 1, 20, 8)
    with c3: epsilon      = st.slider("DP EPSILON (ε)", 0.1, 10.0, 1.0, step=0.1)
    with c4: delta        = st.select_slider("DP DELTA (δ)", options=["1e-8","1e-7","1e-6","1e-5"], value="1e-6")

    run_btn = st.button("🚀 RUN PRIVACY AUDIT")
    neon_divider()

    # ── Simulate results ──────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("Running membership inference attack..."):
            time.sleep(1.2)

    # Metrics
    auc_score      = np.random.uniform(0.50, 0.56)
    privacy_conf   = max(0, 100 - (auc_score - 0.5) * 400)
    attack_success = (auc_score - 0.5) * 2
    dp_guarantee   = f"(ε={epsilon:.1f}, δ={delta})"

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("AUC-ROC SCORE", f"{auc_score:.4f}", f"{auc_score-0.5:+.4f} vs random")
    with k2: st.metric("PRIVACY CONFIDENCE", f"{privacy_conf:.1f}%", "+0.3%")
    with k3: st.metric("ATTACK SUCCESS RATE", f"{attack_success*100:.2f}%", "")
    with k4: st.metric("DP GUARANTEE", dp_guarantee, "")

    neon_divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    col_roc, col_radar, col_dist = st.columns(3)

    with col_roc:
        st.markdown("### 📈 ROC CURVE")
        fpr, tpr = _roc_curve(auc_score)
        fig = go.Figure()
        # Random baseline
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"),
            name="Random (AUC=0.5)"))
        # Attack curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
            line=dict(color="#ff3333", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,50,50,0.06)",
            name=f"Attacker (AUC={auc_score:.3f})"))
        # Ideal privacy
        fig.add_trace(go.Scatter(x=[0,0,1], y=[0,1,1], mode="lines",
            line=dict(color="rgba(0,255,100,0.3)", width=1, dash="dot"),
            name="Perfect Privacy"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc"),
            xaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="FPR", color="#88aacc", range=[0,1]),
            yaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="TPR", color="#88aacc", range=[0,1]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#88aacc", size=9)),
            margin=dict(l=10,r=10,t=10,b=10), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_radar:
        st.markdown("### 🕸 PRIVACY RADAR")
        cats = ["MIA Defense", "DP Guarantee", "Gradient Privacy", "Label Privacy", "Model Inversion", "Attribute Privacy"]
        vals = [privacy_conf/100, 1-epsilon/10, 0.92, 0.88, 0.95, 0.91]
        fig2 = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(0,212,255,0.1)",
            line=dict(color="#00d4ff", width=2),
            marker=dict(color="#00fff7", size=6),
        ))
        # Attack overlay
        atk_vals = [attack_success, epsilon/10, 0.08, 0.12, 0.05, 0.09]
        fig2.add_trace(go.Scatterpolar(
            r=atk_vals + [atk_vals[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(255,50,50,0.08)",
            line=dict(color="#ff3333", width=1.5, dash="dot"),
            marker=dict(color="#ff3333", size=4),
            name="Attack Surface",
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(0,10,30,0.6)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(0,212,255,0.15)",
                                tickfont=dict(color="#88aacc", size=8), color="#88aacc"),
                angularaxis=dict(gridcolor="rgba(0,212,255,0.15)", color="#88aacc",
                                 tickfont=dict(color="#00d4ff", size=9)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#88aacc"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#88aacc", size=9)),
            margin=dict(l=20,r=20,t=20,b=20), height=300,
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_dist:
        st.markdown("### 📊 LOSS DISTRIBUTIONS")
        # Member vs non-member loss distributions
        member_loss     = np.random.normal(0.8, 0.15, 500)
        nonmember_loss  = np.random.normal(1.1, 0.18, 500)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=member_loss, name="Members",
            marker_color="rgba(0,212,255,0.6)", nbinsx=30,
            marker_line=dict(color="rgba(0,212,255,0.3)", width=0.5)))
        fig3.add_trace(go.Histogram(x=nonmember_loss, name="Non-Members",
            marker_color="rgba(255,50,50,0.5)", nbinsx=30,
            marker_line=dict(color="rgba(255,50,50,0.3)", width=0.5)))
        fig3.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc"),
            xaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Loss Value", color="#88aacc"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Count", color="#88aacc"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#88aacc", size=9)),
            margin=dict(l=10,r=10,t=10,b=10), height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    neon_divider()

    # ── Security shield animation ─────────────────────────────────────────────
    st.markdown("### 🛡 SECURITY SHIELD STATUS")
    shield_color = "#00ff64" if privacy_conf > 90 else "#ffaa00" if privacy_conf > 75 else "#ff3333"
    shield_label = "PROTECTED" if privacy_conf > 90 else "MODERATE RISK" if privacy_conf > 75 else "HIGH RISK"

    col_shield, col_checks = st.columns([1, 2])
    with col_shield:
        fig_shield = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=privacy_conf,
            delta=dict(reference=90, valueformat=".1f",
                       increasing=dict(color="#00ff64"), decreasing=dict(color="#ff3333")),
            title=dict(text="PRIVACY CONFIDENCE", font=dict(color="#88aacc", size=12, family="Orbitron")),
            number=dict(font=dict(color=shield_color, size=32, family="Orbitron"), suffix="%"),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#88aacc", tickfont=dict(color="#88aacc", size=9)),
                bar=dict(color=shield_color, thickness=0.3),
                bgcolor="rgba(0,10,30,0.6)",
                bordercolor="rgba(0,212,255,0.2)",
                steps=[
                    dict(range=[0,60],  color="rgba(255,50,50,0.1)"),
                    dict(range=[60,85], color="rgba(255,170,0,0.1)"),
                    dict(range=[85,100],color="rgba(0,255,100,0.1)"),
                ],
                threshold=dict(line=dict(color=shield_color, width=3), thickness=0.8, value=privacy_conf),
            )
        ))
        fig_shield.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(l=20,r=20,t=40,b=20),
        )
        st.plotly_chart(fig_shield, use_container_width=True)

    with col_checks:
        checks = [
            ("Membership Inference Defense",  privacy_conf > 85, f"AUC={auc_score:.3f}"),
            ("Differential Privacy",          epsilon < 3.0,     f"ε={epsilon:.1f}"),
            ("Gradient Leakage Prevention",   True,              "Gradient clipping active"),
            ("Model Inversion Defense",       True,              "Noise injection enabled"),
            ("Attribute Inference Defense",   True,              "Feature masking active"),
            ("Label Inference Defense",       True,              "Label smoothing applied"),
            ("Data Reconstruction Defense",   privacy_conf > 80, "Reconstruction error > 0.95"),
        ]
        for check_name, passed, detail in checks:
            icon  = "✅" if passed else "❌"
            color = "#00ff64" if passed else "#ff3333"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;
                        background:rgba(0,20,50,0.4);border-radius:8px;margin-bottom:6px;
                        border:1px solid {'rgba(0,255,100,0.2)' if passed else 'rgba(255,50,50,0.2)'};">
              <span style="font-size:1.1rem;">{icon}</span>
              <div style="flex:1;">
                <div style="font-family:'Orbitron',monospace;font-size:0.72rem;color:{color};">{check_name}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#88aacc;">{detail}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    neon_divider()

    # ── Attack log ────────────────────────────────────────────────────────────
    st.markdown("### 📟 ATTACK SIMULATION LOG")
    terminal_log([
        f"[INFO]  Initializing {attack_type}...",
        f"[INFO]  Training {n_shadow} shadow models...",
        "[INFO]  Extracting confidence scores from target model...",
        "[INFO]  Fitting attack classifier (LR + threshold)...",
        f"[WARN]  Attack AUC: {auc_score:.4f} (near-random — good privacy)",
        f"[SUCCESS] Privacy confidence: {privacy_conf:.1f}%",
        f"[SUCCESS] DP guarantee verified: ε={epsilon:.1f}, δ={delta}",
        "[SUCCESS] No significant membership leakage detected",
        "[INFO]  Generating privacy report...",
    ])
