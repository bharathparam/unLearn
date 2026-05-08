"""Page 1 — Dashboard"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import random
from utils.helpers import neon_header, neon_divider, glass_card, terminal_log, generate_loss_curve


def render():
    neon_header("NEURALLIFECYCLE FRAMEWORK", "AI MEMORY CONTROL SYSTEM · v4.2.1 · ONLINE", "⬡")

    # ── Live status row ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1.5rem;">
      <span class="status-badge status-online">● SYSTEM ONLINE</span>
      <span class="status-badge status-online">● GPU CLUSTER ACTIVE</span>
      <span class="status-badge status-warning">⚠ MEMORY SCAN RUNNING</span>
      <span class="status-badge status-online">● PRIVACY SHIELD ENGAGED</span>
      <span class="status-badge status-online">● COMPLIANCE MONITOR ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    neon_divider()

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("GPU UTIL", "87.4%", "+2.1%")
    with c2: st.metric("TRAIN LOSS", "0.0312", "-0.0018")
    with c3: st.metric("PRIVACY SCORE", "98.7%", "+0.3%")
    with c4: st.metric("ACTIVE LAYERS", "32 / 32", "0")
    with c5: st.metric("MEMORY NODES", "14,892", "-47")
    with c6: st.metric("COMPLIANCE", "PASS ✓", "")

    neon_divider()

    # ── Charts row ───────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📈 REAL-TIME TRAINING METRICS")
        epochs, loss = generate_loss_curve(60)
        _, acc  = (epochs, np.clip(1 - np.exp(-0.1*epochs)*0.9 + np.random.normal(0,0.008,60), 0, 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=loss, name="Training Loss",
            line=dict(color="#00d4ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=acc, name="Accuracy", yaxis="y2",
            line=dict(color="#b400ff", width=2.5, dash="dot"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc", family="Rajdhani"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#88aacc")),
            xaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Epoch", color="#88aacc"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Loss", color="#00d4ff"),
            yaxis2=dict(overlaying="y", side="right", title="Accuracy", color="#b400ff",
                        gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🧠 NEURAL ACTIVITY RADAR")
        categories = ["Attention", "Memory", "Reasoning", "Privacy", "Safety", "Compliance"]
        vals = [0.87, 0.74, 0.91, 0.98, 0.95, 0.99]
        fig2 = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,212,255,0.12)",
            line=dict(color="#00d4ff", width=2),
            marker=dict(color="#00fff7", size=6),
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(0,10,30,0.6)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(0,212,255,0.15)",
                                tickfont=dict(color="#88aacc", size=9), color="#88aacc"),
                angularaxis=dict(gridcolor="rgba(0,212,255,0.15)", color="#88aacc",
                                 tickfont=dict(color="#00d4ff", size=10)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#88aacc"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    neon_divider()

    # ── GPU + Memory gauges ──────────────────────────────────────────────────
    st.markdown("### ⚡ SYSTEM RESOURCE MONITOR")
    g1, g2, g3, g4 = st.columns(4)

    def gauge(val, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title=dict(text=title, font=dict(color="#88aacc", size=11, family="Orbitron")),
            number=dict(font=dict(color=color, size=22, family="Orbitron"), suffix="%"),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#88aacc", tickfont=dict(color="#88aacc", size=8)),
                bar=dict(color=color, thickness=0.25),
                bgcolor="rgba(0,10,30,0.6)",
                bordercolor="rgba(0,212,255,0.2)",
                steps=[
                    dict(range=[0,60],  color="rgba(0,212,255,0.05)"),
                    dict(range=[60,85], color="rgba(180,0,255,0.08)"),
                    dict(range=[85,100],color="rgba(255,0,100,0.1)"),
                ],
                threshold=dict(line=dict(color=color, width=2), thickness=0.75, value=val),
            )
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=180,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    with g1: st.plotly_chart(gauge(87.4, "GPU UTIL", "#00d4ff"), use_container_width=True)
    with g2: st.plotly_chart(gauge(63.2, "VRAM", "#b400ff"), use_container_width=True)
    with g3: st.plotly_chart(gauge(45.8, "CPU", "#00fff7"), use_container_width=True)
    with g4: st.plotly_chart(gauge(71.3, "RAM", "#ff00aa"), use_container_width=True)

    neon_divider()

    # ── Bottom row: heatmap + logs ───────────────────────────────────────────
    col_heat, col_log = st.columns([3, 2])

    with col_heat:
        st.markdown("### 🔥 LAYER ACTIVATION HEATMAP")
        layers = [f"L{i:02d}" for i in range(1, 33)]
        heads  = [f"H{i}" for i in range(1, 17)]
        z = np.random.rand(32, 16) * np.linspace(0.3, 1.0, 32).reshape(-1,1)
        fig3 = go.Figure(go.Heatmap(
            z=z, x=heads, y=layers,
            colorscale=[[0,"rgba(0,10,30,0.8)"],[0.3,"#003366"],
                        [0.6,"#0066cc"],[0.8,"#b400ff"],[1,"#00fff7"]],
            showscale=True,
            colorbar=dict(tickfont=dict(color="#88aacc"), title=dict(text="Activation", font=dict(color="#88aacc"))),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc"),
            xaxis=dict(title="Attention Head", color="#88aacc"),
            yaxis=dict(title="Layer", color="#88aacc"),
            margin=dict(l=10, r=10, t=10, b=10), height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_log:
        st.markdown("### 📟 SYSTEM LOG STREAM")
        logs = [
            "[INFO]  2026-05-08T09:14:01Z  Model loaded: GPT-NL-7B",
            "[INFO]  2026-05-08T09:14:03Z  LoRA adapters injected: 128 layers",
            "[SUCCESS] Privacy shield initialized",
            "[INFO]  2026-05-08T09:14:07Z  Memory scan started",
            "[WARN]  Sensitive node detected: layer_18.attn.head_4",
            "[INFO]  Causal tracing: 3 facts located",
            "[SUCCESS] Gradient projection ready",
            "[INFO]  ROME editor: standby",
            "[SUCCESS] MIA defense: ACTIVE",
            "[INFO]  Compliance monitor: watching 14,892 nodes",
            "[WARN]  Anomaly score: 0.023 (threshold 0.05)",
            "[SUCCESS] All systems nominal",
        ]
        terminal_log(logs)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌐 ACTIVE OPERATIONS")
        ops = {
            "Fine-Tuning": 78,
            "Memory Scan": 45,
            "Privacy Check": 92,
            "Compliance": 100,
        }
        for op, pct in ops.items():
            color = "#00d4ff" if pct < 80 else "#00ff64"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#88aacc;margin-bottom:4px;">
                <span>{op}</span><span style="color:{color};">{pct}%</span>
              </div>
              <div style="background:rgba(0,20,50,0.6);border-radius:4px;height:6px;overflow:hidden;border:1px solid rgba(0,212,255,0.15);">
                <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#00d4ff,#b400ff);border-radius:4px;box-shadow:0 0 8px rgba(0,212,255,0.5);"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
