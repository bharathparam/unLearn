"""Page 3 — Neural Memory Mapping (most visually impressive page)"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.helpers import neon_header, neon_divider, terminal_log


def _build_3d_transformer(highlight_layer: int = -1, fact_text: str = ""):
    """Full 3D transformer architecture visualization."""
    fig = go.Figure()

    n_layers = 12
    n_heads  = 8
    colors   = []

    # ── Transformer rings (one per layer) ────────────────────────────────────
    for layer in range(n_layers):
        y_pos  = layer * 1.8
        radius = 2.5
        theta  = np.linspace(0, 2*np.pi, 60)
        x_ring = radius * np.cos(theta)
        z_ring = radius * np.sin(theta)
        y_ring = np.full_like(theta, y_pos)

        is_highlight = (layer == highlight_layer)
        ring_color   = "#ff00aa" if is_highlight else "#00d4ff"
        ring_width   = 4 if is_highlight else 1.5
        ring_opacity = 1.0 if is_highlight else 0.5

        fig.add_trace(go.Scatter3d(
            x=x_ring, y=y_ring, z=z_ring,
            mode="lines",
            line=dict(color=ring_color, width=ring_width),
            opacity=ring_opacity,
            name=f"Layer {layer+1}",
            showlegend=False,
        ))

        # Layer label
        fig.add_trace(go.Scatter3d(
            x=[0], y=[y_pos], z=[radius + 0.6],
            mode="text",
            text=[f"L{layer+1:02d}"],
            textfont=dict(color=ring_color, size=9, family="Orbitron"),
            showlegend=False,
        ))

        # ── Attention heads (orbiting particles) ─────────────────────────────
        head_angles = np.linspace(0, 2*np.pi, n_heads, endpoint=False)
        for h, angle in enumerate(head_angles):
            hx = (radius * 0.7) * np.cos(angle)
            hz = (radius * 0.7) * np.sin(angle)
            intensity = np.random.rand()
            if is_highlight:
                hcolor = f"rgba(255,{int(intensity*200)},0,0.9)"
                hsize  = 8
            else:
                hcolor = f"rgba(0,{int(100+intensity*155)},255,0.7)"
                hsize  = 5

            fig.add_trace(go.Scatter3d(
                x=[hx], y=[y_pos], z=[hz],
                mode="markers",
                marker=dict(size=hsize, color=hcolor,
                            symbol="circle",
                            line=dict(color="rgba(255,255,255,0.3)", width=1)),
                showlegend=False,
            ))

    # ── Knowledge embedding spheres ──────────────────────────────────────────
    np.random.seed(42)
    for i in range(25):
        ex = np.random.uniform(-4, 4)
        ey = np.random.uniform(0, n_layers * 1.8)
        ez = np.random.uniform(-4, 4)
        dist = np.sqrt(ex**2 + ez**2)
        if dist < 2.2:
            continue
        is_sensitive = (i % 7 == 0)
        ecolor = "#ff3333" if is_sensitive else "#00fff7"
        esize  = 10 if is_sensitive else 6
        fig.add_trace(go.Scatter3d(
            x=[ex], y=[ey], z=[ez],
            mode="markers",
            marker=dict(size=esize, color=ecolor, opacity=0.85,
                        symbol="circle",
                        line=dict(color="rgba(255,255,255,0.4)", width=1)),
            name="Sensitive" if is_sensitive else "Knowledge",
            showlegend=False,
        ))

    # ── Vertical data flow lines ──────────────────────────────────────────────
    for _ in range(6):
        fx = np.random.uniform(-1.5, 1.5)
        fz = np.random.uniform(-1.5, 1.5)
        fy = np.linspace(0, n_layers * 1.8, 40)
        alpha_vals = np.linspace(0.1, 0.8, 40)
        fig.add_trace(go.Scatter3d(
            x=[fx]*40, y=fy, z=[fz]*40,
            mode="lines",
            line=dict(color="#b400ff", width=1.5),
            opacity=0.4,
            showlegend=False,
        ))

    # ── Highlighted fact annotation ───────────────────────────────────────────
    if fact_text and highlight_layer >= 0:
        hl_y = highlight_layer * 1.8
        fig.add_trace(go.Scatter3d(
            x=[3.5], y=[hl_y], z=[0],
            mode="markers+text",
            marker=dict(size=14, color="#ff00aa", symbol="circle",
                        line=dict(color="#ffffff", width=2)),
            text=[f"<- {fact_text[:20]}"],
            textfont=dict(color="#ff00aa", size=10, family="Share Tech Mono"),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,5,20,0.95)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466",
                       title="", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466",
                       title=dict(text="Layer Depth", font=dict(color="#88aacc", size=10))),
            zaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466",
                       title="", showticklabels=False),
            camera=dict(eye=dict(x=1.6, y=0.8, z=1.6)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.5, z=1),
        ),
        font=dict(color="#88aacc"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
    )
    return fig


def render():
    neon_header("NEURAL MEMORY MAPPING", "TRANSFORMER ARCHITECTURE · CAUSAL TRACING · ATTENTION ROUTING", "🧠")

    # ── Fact tracer input ─────────────────────────────────────────────────────
    st.markdown("### 🔍 CAUSAL FACT TRACER")
    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        fact = st.text_input(
            "Enter a fact to trace inside the model:",
            value="The CEO of Tesla is Elon Musk",
            placeholder="e.g. The capital of France is Paris",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        trace = st.button("🔬 TRACE MEMORY")

    highlight_layer = -1
    trace_results   = {}

    if trace and fact:
        # Simulate causal tracing
        highlight_layer = np.random.randint(6, 10)
        trace_results = {
            "Primary Layer":    f"Layer {highlight_layer + 1}",
            "Secondary Layer":  f"Layer {highlight_layer - 1}",
            "Attention Heads":  f"H{np.random.randint(1,4)}, H{np.random.randint(4,8)}",
            "Activation Score": f"{np.random.uniform(0.72, 0.98):.4f}",
            "Memory Cluster":   f"Cluster-{np.random.randint(100,999)}",
            "Causal Influence": f"{np.random.uniform(0.65, 0.95):.4f}",
        }

        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(255,0,170,0.5);margin-bottom:1rem;">
          <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:#ff00aa;margin-bottom:12px;">
            ⚡ CAUSAL TRACE RESULTS — "{fact}"
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            {"".join(f'<div><div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#88aacc;">{k}</div><div style="font-family:Orbitron,monospace;font-size:1rem;color:#00d4ff;">{v}</div></div>' for k,v in trace_results.items())}
          </div>
        </div>
        """, unsafe_allow_html=True)

    neon_divider()

    # ── 3D Transformer visualization ──────────────────────────────────────────
    st.markdown("### 🌐 3D TRANSFORMER ARCHITECTURE — TRAVELING INSIDE THE AI BRAIN")

    col_viz, col_info = st.columns([3, 1])

    with col_viz:
        fig = _build_3d_transformer(highlight_layer, fact if trace else "")
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:12px;">
          <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:#88aacc;margin-bottom:10px;">LEGEND</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;line-height:2;">
            <span style="color:#00d4ff;">━━</span> Transformer Ring<br>
            <span style="color:#00d4ff;">●</span> Attention Head<br>
            <span style="color:#00fff7;">●</span> Knowledge Node<br>
            <span style="color:#ff3333;">●</span> Sensitive Memory<br>
            <span style="color:#b400ff;">│</span> Data Flow<br>
            <span style="color:#ff00aa;">◆</span> Traced Fact<br>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
          <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:#88aacc;margin-bottom:10px;">ARCHITECTURE</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#88aacc;line-height:1.8;">
            Layers: 12<br>
            Heads/Layer: 8<br>
            Hidden Dim: 4096<br>
            FFN Dim: 16384<br>
            Vocab: 32,000<br>
            Params: 7.2B<br>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if trace_results:
            st.markdown(f"""
            <div class="glass-card" style="margin-top:12px;border-color:rgba(255,0,170,0.4);">
              <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:#ff00aa;margin-bottom:8px;">TRACE ACTIVE</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#88aacc;line-height:1.8;">
                Layer: <span style="color:#ff00aa;">{trace_results.get('Primary Layer','—')}</span><br>
                Score: <span style="color:#00d4ff;">{trace_results.get('Activation Score','—')}</span><br>
                Cluster: <span style="color:#00fff7;">{trace_results.get('Memory Cluster','—')}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    neon_divider()

    # ── Attention flow heatmap ────────────────────────────────────────────────
    st.markdown("### 🔥 ATTENTION FLOW HEATMAP")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("#### Layer-wise Attention Intensity")
        tokens = ["The", "CEO", "of", "Tesla", "is", "Elon", "Musk"]
        n_tok  = len(tokens)
        attn   = np.random.rand(n_tok, n_tok)
        attn   = attn / attn.sum(axis=1, keepdims=True)
        # Boost Tesla→Elon, CEO→Elon
        attn[1, 5] += 0.3; attn[3, 5] += 0.4; attn[5, 6] += 0.5
        attn = np.clip(attn, 0, 1)

        fig2 = go.Figure(go.Heatmap(
            z=attn, x=tokens, y=tokens,
            colorscale=[[0,"rgba(0,10,30,0.9)"],[0.3,"#003366"],
                        [0.6,"#0066cc"],[0.8,"#b400ff"],[1,"#ff00aa"]],
            showscale=True,
            colorbar=dict(tickfont=dict(color="#88aacc"),
                          title=dict(text="Attention", font=dict(color="#88aacc"))),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc"),
            xaxis=dict(title="Key Token", color="#88aacc"),
            yaxis=dict(title="Query Token", color="#88aacc"),
            margin=dict(l=10,r=10,t=10,b=10), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_h2:
        st.markdown("#### Layer Influence Scores")
        layers = [f"L{i:02d}" for i in range(1, 13)]
        influence = np.array([0.12,0.18,0.22,0.31,0.45,0.52,0.61,0.78,0.91,0.85,0.67,0.43])
        if highlight_layer >= 0:
            influence[highlight_layer] = 0.98

        fig3 = go.Figure()
        bar_colors = ["#ff00aa" if i == highlight_layer else "#00d4ff" for i in range(12)]
        fig3.add_trace(go.Bar(
            x=layers, y=influence,
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(0,212,255,0.3)", width=1),
            ),
            name="Influence",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
            font=dict(color="#88aacc"),
            xaxis=dict(gridcolor="rgba(0,212,255,0.08)", color="#88aacc"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.08)", color="#88aacc", title="Influence Score"),
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    neon_divider()

    # ── Memory cluster 3D scatter ─────────────────────────────────────────────
    st.markdown("### 💎 KNOWLEDGE EMBEDDING SPACE (3D)")
    np.random.seed(7)
    n_pts = 200
    # Cluster centers
    centers = [(0,0,0), (3,2,1), (-2,3,2), (1,-3,2), (-3,-2,3)]
    labels  = ["General", "Science", "People", "Places", "Events"]
    clr_map = ["#00d4ff", "#b400ff", "#00fff7", "#ff00aa", "#ffaa00"]

    fig4 = go.Figure()
    for (cx,cy,cz), lbl, clr in zip(centers, labels, clr_map):
        n = n_pts // len(centers)
        xs = np.random.normal(cx, 0.8, n)
        ys = np.random.normal(cy, 0.8, n)
        zs = np.random.normal(cz, 0.8, n)
        fig4.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=3, color=clr, opacity=0.7,
                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
            name=lbl,
        ))

    # Sensitive nodes
    sx = np.random.normal(3, 0.3, 5)
    sy = np.random.normal(2, 0.3, 5)
    sz = np.random.normal(1, 0.3, 5)
    fig4.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="markers",
        marker=dict(size=10, color="#ff3333", opacity=0.9, symbol="diamond",
                    line=dict(color="#ffffff", width=1)),
        name="⚠ Sensitive",
    ))

    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,5,20,0.95)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466"),
            zaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        font=dict(color="#88aacc"),
        legend=dict(bgcolor="rgba(0,10,30,0.8)", font=dict(color="#88aacc")),
        margin=dict(l=0,r=0,t=0,b=0), height=450,
    )
    st.plotly_chart(fig4, use_container_width=True)
