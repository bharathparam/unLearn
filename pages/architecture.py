"""Page 9 — System Architecture Visualization"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from utils.helpers import neon_header, neon_divider


PIPELINE_NODES = [
    {"id": 0, "label": "USER DATA",          "sublabel": "Input Layer",          "color": "#00d4ff", "icon": "👤"},
    {"id": 1, "label": "FINE-TUNING",         "sublabel": "LoRA Adapter Injection","color": "#b400ff", "icon": "⚙"},
    {"id": 2, "label": "LAYER TRACKING",      "sublabel": "Causal Tracing",       "color": "#00fff7", "icon": "🔍"},
    {"id": 3, "label": "ROME EDITING",        "sublabel": "Surgical Memory Edit",  "color": "#ff00aa", "icon": "✂"},
    {"id": 4, "label": "GRAD PROJECTION",     "sublabel": "Orthogonal Forgetting", "color": "#ffaa00", "icon": "📐"},
    {"id": 5, "label": "MIA VERIFICATION",    "sublabel": "Privacy Audit",         "color": "#00ff64", "icon": "🔒"},
    {"id": 6, "label": "COMPLIANCE CERT",     "sublabel": "Certificate Generation","color": "#00d4ff", "icon": "📜"},
]

EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]


def _hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Convert #rrggbb to rgba(r,g,b,a) for Plotly Sankey compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _pipeline_sankey():
    """Animated pipeline as a Sankey diagram."""
    labels = [f"{n['icon']} {n['label']}" for n in PIPELINE_NODES]
    colors = [n["color"] for n in PIPELINE_NODES]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30, thickness=25,
            line=dict(color="rgba(0,212,255,0.3)", width=1),
            label=labels,
            color=[_hex_to_rgba(c, 0.6) for c in colors],
            customdata=[n["sublabel"] for n in PIPELINE_NODES],
            hovertemplate="%{label}<br>%{customdata}<extra></extra>",
        ),
        link=dict(
            source=[e[0] for e in EDGES],
            target=[e[1] for e in EDGES],
            value=[100, 95, 90, 85, 88, 92],
            color=["rgba(0,212,255,0.15)","rgba(180,0,255,0.15)","rgba(0,255,247,0.15)",
                   "rgba(255,0,170,0.15)","rgba(255,170,0,0.15)","rgba(0,255,100,0.15)"],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#88aacc", family="Orbitron", size=9),
        margin=dict(l=10,r=10,t=10,b=10), height=350,
    )
    return fig


def _pipeline_3d():
    """3D pipeline visualization with animated flow."""
    fig = go.Figure()
    n = len(PIPELINE_NODES)

    # Node positions on a helix
    t = np.linspace(0, 2*np.pi, n)
    xs = 2 * np.cos(t)
    ys = np.linspace(0, n*1.5, n)
    zs = 2 * np.sin(t)

    # Edges / flow lines
    for i, (src, tgt) in enumerate(EDGES):
        x_line = np.linspace(xs[src], xs[tgt], 20)
        y_line = np.linspace(ys[src], ys[tgt], 20)
        z_line = np.linspace(zs[src], zs[tgt], 20)
        fig.add_trace(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode="lines",
            line=dict(color=PIPELINE_NODES[src]["color"], width=3),
            opacity=0.6, showlegend=False,
        ))
        # Animated data packet (midpoint marker)
        mx, my, mz = np.mean([xs[src],xs[tgt]]), np.mean([ys[src],ys[tgt]]), np.mean([zs[src],zs[tgt]])
        fig.add_trace(go.Scatter3d(
            x=[mx], y=[my], z=[mz],
            mode="markers",
            marker=dict(size=5, color=PIPELINE_NODES[src]["color"], opacity=0.8,
                        symbol="diamond"),
            showlegend=False,
        ))

    # Nodes
    for i, node in enumerate(PIPELINE_NODES):
        fig.add_trace(go.Scatter3d(
            x=[xs[i]], y=[ys[i]], z=[zs[i]],
            mode="markers+text",
            marker=dict(size=18, color=_hex_to_rgba(node["color"], 0.2),
                        line=dict(color=node["color"], width=2),
                        symbol="circle"),
            text=[node["label"]],
            textposition="middle right",
            textfont=dict(color=node["color"], size=9, family="Orbitron"),
            name=node["label"],
            showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,5,20,0.95)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.05)", color="#334466", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,212,255,0.05)", color="#334466",
                       title=dict(text="Pipeline Stage", font=dict(color="#88aacc", size=9))),
            zaxis=dict(gridcolor="rgba(0,212,255,0.05)", color="#334466", showticklabels=False),
            camera=dict(eye=dict(x=2.0, y=0.5, z=1.5)),
        ),
        font=dict(color="#88aacc"),
        legend=dict(bgcolor="rgba(0,10,30,0.8)", font=dict(color="#88aacc", size=8)),
        margin=dict(l=0,r=0,t=0,b=0), height=500,
    )
    return fig


def render():
    neon_header("SYSTEM ARCHITECTURE", "FULL PIPELINE VISUALIZATION · DATA FLOW · SECURITY CHECKPOINTS", "🏗")

    # ── Architecture overview cards ───────────────────────────────────────────
    st.markdown("### 🗺 PIPELINE OVERVIEW")
    cols = st.columns(len(PIPELINE_NODES))
    for col, node in zip(cols, PIPELINE_NODES):
        with col:
            st.markdown(f"""
            <div style="background:rgba(0,20,50,0.5);border:1px solid {node['color']}40;
                        border-radius:10px;padding:12px;text-align:center;
                        transition:all 0.3s ease;">
              <div style="font-size:1.5rem;">{node['icon']}</div>
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:{node['color']};
                          margin-top:6px;line-height:1.3;">{node['label']}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#88aacc;
                          margin-top:4px;">{node['sublabel']}</div>
            </div>
            """, unsafe_allow_html=True)

    neon_divider()

    # ── Tabs for different views ──────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🌊 SANKEY FLOW", "🌐 3D PIPELINE", "📊 COMPONENT METRICS"])

    with tab1:
        st.markdown("### 🌊 DATA FLOW SANKEY DIAGRAM")
        st.plotly_chart(_pipeline_sankey(), use_container_width=True)

    with tab2:
        st.markdown("### 🌐 3D PIPELINE ARCHITECTURE")
        st.plotly_chart(_pipeline_3d(), use_container_width=True)

    with tab3:
        st.markdown("### 📊 COMPONENT PERFORMANCE METRICS")
        components = ["Fine-Tuning", "Layer Tracking", "ROME Editing", "Grad Projection", "MIA Verify", "Cert Gen"]
        metrics = {
            "Latency (ms)":   [245, 89, 312, 178, 523, 45],
            "Throughput":     [0.92, 0.99, 0.87, 0.94, 0.96, 1.0],
            "Memory (GB)":    [4.2, 0.8, 2.1, 1.5, 0.6, 0.1],
            "Accuracy":       [0.97, 0.99, 0.95, 0.98, 0.96, 1.0],
        }

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_lat = go.Figure(go.Bar(
                x=components, y=metrics["Latency (ms)"],
                marker=dict(
                    color=metrics["Latency (ms)"],
                    colorscale=[[0,"#00d4ff"],[0.5,"#b400ff"],[1,"#ff3333"]],
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                ),
                text=[f"{v}ms" for v in metrics["Latency (ms)"]],
                textposition="outside",
                textfont=dict(color="#88aacc", size=9),
            ))
            fig_lat.update_layout(
                title=dict(text="Component Latency", font=dict(color="#88aacc", size=11, family="Orbitron")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
                font=dict(color="#88aacc"),
                xaxis=dict(color="#88aacc", tickangle=-20, tickfont=dict(size=9)),
                yaxis=dict(color="#88aacc", title="ms"),
                margin=dict(l=10,r=10,t=40,b=60), height=280, showlegend=False,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_m2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatterpolar(
                r=metrics["Accuracy"] + [metrics["Accuracy"][0]],
                theta=components + [components[0]],
                fill="toself", fillcolor="rgba(0,212,255,0.1)",
                line=dict(color="#00d4ff", width=2),
                marker=dict(color="#00fff7", size=6),
                name="Accuracy",
            ))
            fig_acc.add_trace(go.Scatterpolar(
                r=metrics["Throughput"] + [metrics["Throughput"][0]],
                theta=components + [components[0]],
                fill="toself", fillcolor="rgba(180,0,255,0.08)",
                line=dict(color="#b400ff", width=2, dash="dot"),
                marker=dict(color="#b400ff", size=4),
                name="Throughput",
            ))
            fig_acc.update_layout(
                title=dict(text="Accuracy & Throughput", font=dict(color="#88aacc", size=11, family="Orbitron")),
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
                margin=dict(l=20,r=20,t=40,b=20), height=280,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    neon_divider()

    # ── Live pipeline simulation ──────────────────────────────────────────────
    st.markdown("### ▶ LIVE PIPELINE SIMULATION")
    col_sim1, col_sim2, _ = st.columns([1, 1, 4])
    with col_sim1: run_sim = st.button("▶ RUN PIPELINE")
    with col_sim2: st.button("⏹ STOP")

    if run_sim:
        progress_ph = st.empty()
        status_ph   = st.empty()

        for i, node in enumerate(PIPELINE_NODES):
            pct = int((i + 1) / len(PIPELINE_NODES) * 100)
            progress_ph.progress(pct / 100, text=f"Processing: {node['label']}...")
            status_ph.markdown(f"""
            <div style="background:rgba(0,20,50,0.5);border:1px solid {node['color']}60;
                        border-radius:8px;padding:12px;font-family:'Share Tech Mono',monospace;
                        font-size:0.8rem;color:{node['color']};">
              ⚡ ACTIVE: {node['icon']} {node['label']} — {node['sublabel']}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)

        progress_ph.empty()
        status_ph.empty()
        st.success("✅ Pipeline execution complete. All stages passed.")

    neon_divider()

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown("### 🛠 TECHNOLOGY STACK")
    tech_cols = st.columns(4)
    tech_stack = [
        ("FRONTEND",  ["Streamlit", "Plotly 3D", "Custom CSS", "HTML5 Canvas"], "#00d4ff"),
        ("ML CORE",   ["PyTorch", "Transformers", "LoRA/PEFT", "ROME Editor"], "#b400ff"),
        ("PRIVACY",   ["Differential Privacy", "MIA Defense", "Gradient Clip", "Noise Injection"], "#00fff7"),
        ("INFRA",     ["FastAPI", "WebSocket", "Audit Ledger", "PDF Generator"], "#ff00aa"),
    ]
    for col, (category, items, color) in zip(tech_cols, tech_stack):
        with col:
            items_html = "".join(f'<div style="padding:4px 0;border-bottom:1px solid rgba(0,212,255,0.08);font-size:0.75rem;color:#e0f0ff;">{item}</div>' for item in items)
            st.markdown(f"""
            <div class="glass-card" style="border-color:{color}40;">
              <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:{color};
                          margin-bottom:10px;letter-spacing:0.1em;">{category}</div>
              <div style="font-family:'Share Tech Mono',monospace;">{items_html}</div>
            </div>
            """, unsafe_allow_html=True)
