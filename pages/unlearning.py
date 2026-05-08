"""Page 4 — AI Unlearning Console"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from utils.helpers import neon_header, neon_divider, terminal_log, random_hash


def _unlearning_3d(phase: str = "before"):
    """3D visualization of memory deletion process."""
    fig = go.Figure()
    np.random.seed(99)

    n_layers = 8
    for layer in range(n_layers):
        y = layer * 2.0
        theta = np.linspace(0, 2*np.pi, 50)
        r = 2.2
        if phase == "during" and layer in [3, 4]:
            # Cracking ring — broken segments
            for seg in range(5):
                t_seg = theta[seg*10:(seg+1)*10]
                opacity = np.random.uniform(0.2, 0.8)
                fig.add_trace(go.Scatter3d(
                    x=r*np.cos(t_seg), y=[y]*len(t_seg), z=r*np.sin(t_seg),
                    mode="lines",
                    line=dict(color="#ff3333", width=3),
                    opacity=opacity, showlegend=False,
                ))
        elif phase == "after" and layer in [3, 4]:
            # Dimmed ring
            fig.add_trace(go.Scatter3d(
                x=r*np.cos(theta), y=[y]*50, z=r*np.sin(theta),
                mode="lines",
                line=dict(color="rgba(100,100,100,0.3)", width=1),
                opacity=0.3, showlegend=False,
            ))
        else:
            color = "#00d4ff" if layer not in [3,4] else "#ff6600"
            fig.add_trace(go.Scatter3d(
                x=r*np.cos(theta), y=[y]*50, z=r*np.sin(theta),
                mode="lines",
                line=dict(color=color, width=2),
                opacity=0.7, showlegend=False,
            ))

        # Attention heads
        for h in range(6):
            angle = h * np.pi / 3
            hx = 1.4 * np.cos(angle)
            hz = 1.4 * np.sin(angle)
            if phase == "during" and layer in [3, 4]:
                hcolor = f"rgba(255,{np.random.randint(0,100)},0,{np.random.uniform(0.3,1.0):.2f})"
                hsize  = np.random.randint(4, 14)
            elif phase == "after" and layer in [3, 4]:
                hcolor = "rgba(80,80,80,0.4)"
                hsize  = 3
            else:
                hcolor = "#00d4ff"
                hsize  = 5
            fig.add_trace(go.Scatter3d(
                x=[hx], y=[y], z=[hz],
                mode="markers",
                marker=dict(size=hsize, color=hcolor),
                showlegend=False,
            ))

    # Memory nodes
    mem_positions = [
        (1.8, 3*2.0, 0.5, True),
        (-1.5, 4*2.0, 1.0, True),
        (0.5, 3.5*2.0, -1.2, True),
        (2.5, 1*2.0, 0.8, False),
        (-2.0, 6*2.0, -0.5, False),
        (1.0, 2*2.0, 2.0, False),
    ]
    for mx, my, mz, is_sensitive in mem_positions:
        if is_sensitive:
            if phase == "before":
                mcolor, msize, mopacity, msymbol = "#ff3333", 12, 0.9, "circle"
            elif phase == "during":
                mcolor = f"rgba(255,{np.random.randint(0,80)},0,{np.random.uniform(0.2,0.7):.2f})"
                msize, mopacity, msymbol = np.random.randint(4,16), np.random.uniform(0.2,0.8), "circle"
                # Scatter dissolving particles
                for _ in range(4):
                    px = mx + np.random.normal(0, 0.5)
                    py = my + np.random.normal(0, 0.5)
                    pz = mz + np.random.normal(0, 0.5)
                    fig.add_trace(go.Scatter3d(
                        x=[px], y=[py], z=[pz],
                        mode="markers",
                        marker=dict(size=np.random.randint(2,6),
                                    color=f"rgba(255,100,0,{np.random.uniform(0.2,0.6):.2f})"),
                        showlegend=False,
                    ))
            else:  # after
                mcolor, msize, mopacity, msymbol = "rgba(80,80,80,0.2)", 4, 0.2, "circle"
        else:
            mcolor, msize, mopacity, msymbol = "#00fff7", 7, 0.8, "circle"

        fig.add_trace(go.Scatter3d(
            x=[mx], y=[my], z=[mz],
            mode="markers",
            marker=dict(size=msize, color=mcolor, opacity=mopacity, symbol=msymbol,
                        line=dict(color="rgba(255,255,255,0.3)", width=1)),
            showlegend=False,
        ))

    # Gradient projection vectors (during/after)
    if phase in ["during", "after"]:
        for _ in range(8):
            gx0 = np.random.uniform(-2, 2)
            gy0 = np.random.uniform(4, 10)
            gz0 = np.random.uniform(-2, 2)
            gx1 = gx0 + np.random.uniform(-1, 1)
            gy1 = gy0 + np.random.uniform(-0.5, 0.5)
            gz1 = gz0 + np.random.uniform(-1, 1)
            fig.add_trace(go.Scatter3d(
                x=[gx0, gx1], y=[gy0, gy1], z=[gz0, gz1],
                mode="lines",
                line=dict(color="#b400ff", width=2),
                opacity=0.6, showlegend=False,
            ))

    title_map = {
        "before": "PRE-UNLEARNING STATE",
        "during": "UNLEARNING IN PROGRESS — MEMORY DISSOLVING",
        "after":  "POST-UNLEARNING — MEMORY ERASED",
    }
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,5,20,0.95)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.05)", color="#334466", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,212,255,0.05)", color="#334466",
                       title=dict(text="Layer Depth", font=dict(color="#88aacc", size=9))),
            zaxis=dict(gridcolor="rgba(0,212,255,0.05)", color="#334466", showticklabels=False),
            camera=dict(eye=dict(x=1.8, y=0.7, z=1.8)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.2, z=1),
        ),
        title=dict(text=title_map[phase], font=dict(color="#ff00aa" if phase=="during" else "#00d4ff",
                                                     size=12, family="Orbitron"), x=0.5),
        font=dict(color="#88aacc"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    return fig


def render():
    neon_header("AI UNLEARNING CONSOLE", "ROME SURGICAL EDITING · GRADIENT ORTHOGONALIZATION · MEMORY DELETION", "🗑")

    # ── Input panel ───────────────────────────────────────────────────────────
    st.markdown("### ⚠ SENSITIVE DATA DELETION REQUEST")
    col_inp, col_cfg = st.columns([3, 2])

    with col_inp:
        sensitive_data = st.text_area(
            "Enter sensitive data to unlearn:",
            value="Delete John's password abc123\nRemove SSN: 123-45-6789\nForget credit card: 4111-1111-1111-1111",
            height=120,
        )
        method = st.selectbox("UNLEARNING METHOD", [
            "ROME (Rank-One Model Editing)",
            "Gradient Ascent",
            "Selective Forgetting",
            "Gradient Projection",
            "Neuron Masking",
        ])

    with col_cfg:
        st.markdown("""
        <div class="glass-card">
          <div style="font-family:'Orbitron',monospace;font-size:0.75rem;color:#88aacc;margin-bottom:12px;">DELETION PARAMETERS</div>
        """, unsafe_allow_html=True)
        strength   = st.slider("DELETION STRENGTH", 0.1, 1.0, 0.85, step=0.05)
        iterations = st.slider("ITERATIONS", 10, 500, 100, step=10)
        threshold  = st.slider("CONFIDENCE THRESHOLD", 0.5, 0.99, 0.95, step=0.01)
        st.markdown("</div>", unsafe_allow_html=True)

    neon_divider()

    # ── Phase selector ────────────────────────────────────────────────────────
    st.markdown("### 🎬 UNLEARNING VISUALIZATION")
    phase_col1, phase_col2, phase_col3, _ = st.columns([1,1,1,3])
    with phase_col1: show_before = st.button("👁 BEFORE")
    with phase_col2: execute_btn = st.button("💥 EXECUTE UNLEARNING")
    with phase_col3: show_after  = st.button("✅ AFTER")

    if "unlearn_phase" not in st.session_state:
        st.session_state.unlearn_phase = "before"

    if show_before:  st.session_state.unlearn_phase = "before"
    if show_after:   st.session_state.unlearn_phase = "after"

    if execute_btn and sensitive_data.strip():
        st.session_state.unlearn_phase = "during"

        # Animated log
        log_ph = st.empty()
        prog_ph = st.empty()

        steps = [
            ("[INFO]  Parsing sensitive data tokens...", 5),
            ("[INFO]  Locating memory clusters via causal tracing...", 15),
            ("[WARN]  Sensitive nodes found: layer_3.attn, layer_4.ffn", 25),
            ("[INFO]  Initializing ROME surgical editor...", 35),
            ("[INFO]  Computing rank-one update matrices...", 50),
            ("[INFO]  Applying gradient orthogonalization...", 65),
            ("[WARN]  Synapse rewiring in progress...", 75),
            ("[INFO]  Weight matrix update: Δ||W|| = 0.0023", 85),
            ("[INFO]  Verifying deletion completeness...", 92),
            ("[SUCCESS] Memory nodes dissolved: 3/3", 97),
            ("[SUCCESS] UNLEARNING COMPLETE — Privacy verified", 100),
        ]

        for msg, pct in steps:
            log_ph.markdown(f"""
            <div class="terminal-log">
              <div class="{'log-success' if 'SUCCESS' in msg else 'log-warn' if 'WARN' in msg else 'log-info'}">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            prog_ph.progress(pct / 100, text=f"Unlearning... {pct}%")
            time.sleep(0.18)

        prog_ph.empty()
        st.session_state.unlearn_phase = "after"
        st.success("✅ Unlearning complete. Sensitive memories have been surgically removed.")

    # ── 3D visualization ──────────────────────────────────────────────────────
    fig = _unlearning_3d(st.session_state.unlearn_phase)
    st.plotly_chart(fig, use_container_width=True)

    neon_divider()

    # ── Weight matrix diff ────────────────────────────────────────────────────
    st.markdown("### 📊 WEIGHT MATRIX DELTA (ΔW)")
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown("#### Before Unlearning")
        W_before = np.random.randn(20, 20)
        fig_w1 = go.Figure(go.Heatmap(
            z=W_before,
            colorscale=[[0,"#020818"],[0.3,"#003366"],[0.6,"#0066cc"],[1,"#00fff7"]],
            showscale=False,
        ))
        fig_w1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            margin=dict(l=5,r=5,t=5,b=5), height=220,
            xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_w1, use_container_width=True)

    with col_w2:
        st.markdown("#### After Unlearning (ΔW applied)")
        W_after = W_before.copy()
        W_after[8:12, 8:12] *= 0.05  # zeroed-out region
        fig_w2 = go.Figure(go.Heatmap(
            z=W_after,
            colorscale=[[0,"#020818"],[0.3,"#003366"],[0.6,"#0066cc"],[0.85,"#00fff7"],[1,"#ff3333"]],
            showscale=False,
        ))
        # Highlight deleted region
        fig_w2.add_shape(type="rect", x0=7.5, y0=7.5, x1=12.5, y1=12.5,
                         line=dict(color="#ff3333", width=2, dash="dot"),
                         fillcolor="rgba(255,50,50,0.05)")
        fig_w2.add_annotation(x=10, y=10, text="DELETED", font=dict(color="#ff3333", size=9, family="Orbitron"),
                               showarrow=False)
        fig_w2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            margin=dict(l=5,r=5,t=5,b=5), height=220,
            xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_w2, use_container_width=True)

    neon_divider()

    # ── Audit trail ───────────────────────────────────────────────────────────
    st.markdown("### 📋 DELETION AUDIT TRAIL")
    import pandas as pd
    audit_data = {
        "Timestamp":    ["2026-05-08T09:14:01Z", "2026-05-08T09:14:03Z", "2026-05-08T09:14:07Z"],
        "Data Token":   ["password_abc123", "SSN_123456789", "CC_4111xxxx"],
        "Layer":        ["layer_3.attn.head_2", "layer_4.ffn.neuron_847", "layer_3.attn.head_5"],
        "Method":       [method.split("(")[0].strip()] * 3,
        "Status":       ["✅ DELETED", "✅ DELETED", "✅ DELETED"],
        "Checksum":     [random_hash(12), random_hash(12), random_hash(12)],
    }
    df = pd.DataFrame(audit_data)
    st.dataframe(
        df.style.applymap(lambda v: "color: #00ff64" if "DELETED" in str(v) else "color: #88aacc"),
        use_container_width=True, hide_index=True,
    )
