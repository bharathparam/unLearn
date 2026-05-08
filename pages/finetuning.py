"""Page 2 — Fine-Tuning Engine"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from utils.helpers import neon_header, neon_divider, terminal_log, fake_progress


def render():
    neon_header("FINE-TUNING ENGINE", "LoRA ADAPTER INJECTION · GRADIENT FLOW · WEIGHT OPTIMIZATION", "⚙")

    # ── Config panel ─────────────────────────────────────────────────────────
    st.markdown("### 🔧 TRAINING CONFIGURATION")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model = st.selectbox("BASE MODEL", ["GPT-NL-7B", "LLaMA-3-8B", "Mistral-7B", "Falcon-7B"])
    with c2:
        rank = st.slider("LoRA RANK", 4, 128, 64, step=4)
    with c3:
        lr = st.select_slider("LEARNING RATE", options=["1e-5","3e-5","5e-5","1e-4","3e-4"], value="3e-5")
    with c4:
        epochs = st.slider("EPOCHS", 1, 20, 5)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        batch = st.slider("BATCH SIZE", 4, 64, 16, step=4)
    with c6:
        alpha = st.slider("LoRA ALPHA", 8, 256, 128, step=8)
    with c7:
        dropout = st.slider("DROPOUT", 0.0, 0.5, 0.1, step=0.05)
    with c8:
        target = st.multiselect("TARGET MODULES", ["q_proj","v_proj","k_proj","o_proj","gate_proj"], default=["q_proj","v_proj"])

    neon_divider()

    # ── LoRA Architecture Diagram ─────────────────────────────────────────────
    st.markdown("### 🧬 LoRA ADAPTER ARCHITECTURE")
    col_arch, col_ctrl = st.columns([3, 1])

    with col_arch:
        # Sankey-style LoRA flow
        fig = go.Figure()

        # Base weight matrix W
        fig.add_shape(type="rect", x0=0.1, y0=0.3, x1=0.35, y1=0.7,
                      fillcolor="rgba(0,212,255,0.15)", line=dict(color="#00d4ff", width=2))
        fig.add_annotation(x=0.225, y=0.5, text="W<br><sub>Base Weight</sub>",
                           font=dict(color="#00d4ff", size=12, family="Orbitron"), showarrow=False)

        # A matrix
        fig.add_shape(type="rect", x0=0.45, y0=0.55, x1=0.6, y1=0.75,
                      fillcolor="rgba(180,0,255,0.15)", line=dict(color="#b400ff", width=2))
        fig.add_annotation(x=0.525, y=0.65, text="A<br><sub>Down</sub>",
                           font=dict(color="#b400ff", size=11, family="Orbitron"), showarrow=False)

        # B matrix
        fig.add_shape(type="rect", x0=0.45, y0=0.25, x1=0.6, y1=0.45,
                      fillcolor="rgba(0,255,247,0.15)", line=dict(color="#00fff7", width=2))
        fig.add_annotation(x=0.525, y=0.35, text="B<br><sub>Up</sub>",
                           font=dict(color="#00fff7", size=11, family="Orbitron"), showarrow=False)

        # Output
        fig.add_shape(type="rect", x0=0.72, y0=0.3, x1=0.92, y1=0.7,
                      fillcolor="rgba(255,0,170,0.15)", line=dict(color="#ff00aa", width=2))
        fig.add_annotation(x=0.82, y=0.5, text="W + BA<br><sub>Adapted</sub>",
                           font=dict(color="#ff00aa", size=11, family="Orbitron"), showarrow=False)

        # Arrows
        for (x0,y0,x1,y1,col) in [
            (0.35,0.5,0.72,0.5,"#00d4ff"),
            (0.35,0.5,0.45,0.65,"#b400ff"),
            (0.6,0.65,0.72,0.55,"#b400ff"),
            (0.35,0.5,0.45,0.35,"#00fff7"),
            (0.6,0.35,0.72,0.45,"#00fff7"),
        ]:
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y",
                               arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=col,
                               showarrow=True, text="")

        # Rank label
        fig.add_annotation(x=0.525, y=0.82, text=f"rank = {rank}  |  α = {alpha}",
                           font=dict(color="#ffaa00", size=11, family="Share Tech Mono"), showarrow=False)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False, range=[0,1]),
            height=300, margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_ctrl:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#88aacc;margin-bottom:8px;">ADAPTER PARAMS</div>
          <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#00d4ff;">
            {rank * alpha // 8 :,}
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#88aacc;">trainable params</div>
          <br>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#88aacc;margin-bottom:4px;">COMPRESSION</div>
          <div style="font-family:'Orbitron',monospace;font-size:1.2rem;color:#b400ff;">
            {100 - rank*100//512:.1f}%
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#88aacc;">param reduction</div>
        </div>
        """, unsafe_allow_html=True)

    neon_divider()

    # ── Training simulation ───────────────────────────────────────────────────
    st.markdown("### 🚀 TRAINING SIMULATION")
    col_btn1, col_btn2, col_btn3, _ = st.columns([1,1,1,3])
    with col_btn1: start = st.button("▶ START TRAINING")
    with col_btn2: pause = st.button("⏸ PAUSE")
    with col_btn3: reset = st.button("↺ RESET")

    if start:
        # Animated training run
        chart_placeholder = st.empty()
        log_placeholder   = st.empty()
        metric_cols       = st.columns(4)

        loss_hist, acc_hist, lr_hist, grad_hist = [], [], [], []
        ep_x = []

        for ep in range(1, epochs + 1):
            for step in range(1, 11):
                frac = (ep - 1 + step/10) / epochs
                loss_val = 2.5 * np.exp(-4 * frac) + 0.12 + np.random.normal(0, 0.02)
                acc_val  = 1 - np.exp(-5 * frac) * 0.9 + np.random.normal(0, 0.005)
                lr_val   = float(lr.replace("e-","e-")) * (0.95 ** ep)
                grad_val = abs(np.random.normal(0.8 - frac*0.5, 0.1))

                loss_hist.append(loss_val)
                acc_hist.append(np.clip(acc_val, 0, 1))
                lr_hist.append(lr_val)
                grad_hist.append(grad_val)
                ep_x.append(frac * epochs)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ep_x, y=loss_hist, name="Loss",
                    line=dict(color="#00d4ff", width=2), fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.06)"))
                fig.add_trace(go.Scatter(x=ep_x, y=acc_hist, name="Accuracy", yaxis="y2",
                    line=dict(color="#b400ff", width=2, dash="dot")))
                fig.add_trace(go.Scatter(x=ep_x, y=grad_hist, name="Grad Norm", yaxis="y3",
                    line=dict(color="#ff00aa", width=1.5, dash="dash")))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
                    font=dict(color="#88aacc", family="Rajdhani"),
                    xaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Epoch", color="#88aacc"),
                    yaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Loss", color="#00d4ff"),
                    yaxis2=dict(overlaying="y", side="right", title="Acc", color="#b400ff",
                                gridcolor="rgba(0,0,0,0)", range=[0,1]),
                    yaxis3=dict(overlaying="y", side="right", position=0.95, title="Grad",
                                color="#ff00aa", gridcolor="rgba(0,0,0,0)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10,r=60,t=10,b=10), height=300,
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                with log_placeholder.container():
                    terminal_log([
                        f"[INFO]  Epoch {ep:02d}/step {step:02d}  loss={loss_val:.4f}  acc={acc_val:.4f}",
                        f"[INFO]  lr={lr_val:.2e}  grad_norm={grad_val:.4f}",
                        f"[SUCCESS] Tokens/sec: {random.randint(1800,2400)}",
                    ])
                time.sleep(0.05)

        st.success("✅ Training complete! LoRA adapters saved.")

    else:
        # Static preview charts
        ep_x = np.linspace(0, epochs, 60)
        loss_y = 2.5 * np.exp(-4 * ep_x / epochs) + 0.12 + np.random.normal(0, 0.02, 60)
        acc_y  = np.clip(1 - np.exp(-5 * ep_x / epochs) * 0.9, 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_x, y=loss_y, name="Loss",
            line=dict(color="#00d4ff", width=2), fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)"))
        fig.add_trace(go.Scatter(x=ep_x, y=acc_y, name="Accuracy", yaxis="y2",
            line=dict(color="#b400ff", width=2, dash="dot")))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc", family="Rajdhani"),
            xaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Epoch", color="#88aacc"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.08)", title="Loss", color="#00d4ff"),
            yaxis2=dict(overlaying="y", side="right", title="Accuracy", color="#b400ff",
                        gridcolor="rgba(0,0,0,0)", range=[0,1]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=40,t=10,b=10), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    neon_divider()

    # ── Weight update 3D surface ──────────────────────────────────────────────
    st.markdown("### 🌊 LOSS LANDSCAPE (3D)")
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X) * np.cos(Y) * np.exp(-0.2*(X**2+Y**2)) +
         0.3 * np.exp(-((X-1)**2+(Y-1)**2)) +
         0.5 * np.exp(-((X+1)**2+(Y+1)**2)*0.5))

    fig4 = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0,"#020818"],[0.2,"#003366"],[0.5,"#0066cc"],
                    [0.7,"#b400ff"],[0.9,"#00fff7"],[1,"#ffffff"]],
        opacity=0.85,
        contours=dict(z=dict(show=True, color="rgba(0,212,255,0.3)", width=1)),
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
    ))
    # Gradient descent path
    path_x = np.linspace(2.5, 0.1, 30) + np.random.normal(0, 0.05, 30)
    path_y = np.linspace(2.5, 0.1, 30) + np.random.normal(0, 0.05, 30)
    path_z = (np.sin(path_x)*np.cos(path_y)*np.exp(-0.2*(path_x**2+path_y**2)) + 0.05)
    fig4.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode="lines+markers",
        line=dict(color="#ff00aa", width=4),
        marker=dict(size=3, color="#ff00aa"),
        name="Gradient Descent",
    ))
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,10,30,0.8)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#88aacc"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#88aacc"),
            zaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#88aacc"),
        ),
        font=dict(color="#88aacc"),
        margin=dict(l=0,r=0,t=0,b=0), height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig4, use_container_width=True)

import random
