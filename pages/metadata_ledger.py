"""Page 7 — Metadata Ledger (blockchain-inspired)"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import hashlib, time, random
from utils.helpers import neon_header, neon_divider, random_hash, timestamp_now


def _make_block(index: int, prev_hash: str, data: dict) -> dict:
    content = f"{index}{prev_hash}{data}"
    block_hash = hashlib.sha256(content.encode()).hexdigest()
    return {
        "index":     index,
        "timestamp": timestamp_now(),
        "data":      data,
        "prev_hash": prev_hash[:16] + "...",
        "hash":      block_hash[:16] + "...",
        "full_hash": block_hash,
        "nonce":     random.randint(10000, 99999),
    }


def _generate_chain(n: int = 8) -> list:
    chain = []
    prev  = "0" * 64
    operations = [
        {"op": "FINE_TUNE",   "user": "researcher_01", "adapter": "lora_v1", "layers": "all"},
        {"op": "MEMORY_SCAN", "user": "researcher_01", "nodes_scanned": 14892, "sensitive": 3},
        {"op": "UNLEARN",     "user": "admin_02",      "tokens": ["pwd_abc", "ssn_123"], "method": "ROME"},
        {"op": "PRIVACY_AUDIT","user": "auditor_01",   "auc": 0.523, "result": "PASS"},
        {"op": "FINE_TUNE",   "user": "researcher_03", "adapter": "lora_v2", "layers": "q,v"},
        {"op": "UNLEARN",     "user": "admin_02",      "tokens": ["cc_4111"], "method": "GradProj"},
        {"op": "COMPLIANCE",  "user": "compliance_01", "score": 98.7, "cert": "CERT-2026-001"},
        {"op": "CHECKPOINT",  "user": "system",        "checksum": random_hash(32), "version": "4.2.1"},
    ]
    for i in range(min(n, len(operations))):
        block = _make_block(i, prev, operations[i])
        chain.append(block)
        prev = block["full_hash"]
    return chain


def render():
    neon_header("METADATA LEDGER", "IMMUTABLE AUDIT CHAIN · OPERATION HISTORY · COMPLIANCE TRAIL", "⛓")

    chain = _generate_chain(8)

    # ── Chain stats ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CHAIN LENGTH", f"{len(chain)} blocks")
    with c2: st.metric("TOTAL OPERATIONS", len(chain))
    with c3: st.metric("CHAIN INTEGRITY", "✅ VALID")
    with c4: st.metric("LAST BLOCK", f"#{len(chain)-1}")

    neon_divider()

    # ── Blockchain visualization ──────────────────────────────────────────────
    st.markdown("### ⛓ BLOCKCHAIN VISUALIZATION")

    fig = go.Figure()
    n = len(chain)
    op_colors = {
        "FINE_TUNE":     "#00d4ff",
        "MEMORY_SCAN":   "#b400ff",
        "UNLEARN":       "#ff3333",
        "PRIVACY_AUDIT": "#00fff7",
        "COMPLIANCE":    "#00ff64",
        "CHECKPOINT":    "#ffaa00",
    }

    for i, block in enumerate(chain):
        x = i * 2.5
        op = block["data"]["op"]
        color = op_colors.get(op, "#88aacc")

        # Block rectangle
        fig.add_shape(type="rect", x0=x-0.9, y0=-0.6, x1=x+0.9, y1=0.6,
                      fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
                      line=dict(color=color, width=2))

        # Block label
        fig.add_annotation(x=x, y=0.25, text=f"#{i}", showarrow=False,
                           font=dict(color=color, size=14, family="Orbitron"))
        fig.add_annotation(x=x, y=-0.05, text=op, showarrow=False,
                           font=dict(color=color, size=8, family="Share Tech Mono"))
        fig.add_annotation(x=x, y=-0.35, text=block["hash"], showarrow=False,
                           font=dict(color="#334466", size=7, family="Share Tech Mono"))

        # Chain link arrow
        if i < n - 1:
            fig.add_annotation(
                x=(i+1)*2.5 - 0.9, y=0, ax=x+0.9, ay=0,
                axref="x", ayref="y", xref="x", yref="y",
                arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#334466",
                showarrow=True, text="",
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,10,30,0.6)",
        xaxis=dict(visible=False, range=[-1, n*2.5]),
        yaxis=dict(visible=False, range=[-1, 1]),
        height=180, margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    neon_divider()

    # ── Block explorer ────────────────────────────────────────────────────────
    st.markdown("### 🔍 BLOCK EXPLORER")
    selected_block = st.slider("SELECT BLOCK", 0, len(chain)-1, 0)
    block = chain[selected_block]
    op    = block["data"]["op"]
    color = op_colors.get(op, "#88aacc")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown(f"""
        <div class="glass-card" style="border-color:{color}40;">
          <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:{color};margin-bottom:16px;">
            BLOCK #{block['index']} — {op}
          </div>
          <table style="width:100%;font-family:'Share Tech Mono',monospace;font-size:0.75rem;border-collapse:collapse;">
            <tr><td style="color:#88aacc;padding:4px 0;">Index</td><td style="color:#e0f0ff;">{block['index']}</td></tr>
            <tr><td style="color:#88aacc;padding:4px 0;">Timestamp</td><td style="color:#e0f0ff;">{block['timestamp']}</td></tr>
            <tr><td style="color:#88aacc;padding:4px 0;">Nonce</td><td style="color:#e0f0ff;">{block['nonce']}</td></tr>
            <tr><td style="color:#88aacc;padding:4px 0;">Prev Hash</td><td style="color:#334466;">{block['prev_hash']}</td></tr>
            <tr><td style="color:#88aacc;padding:4px 0;">Block Hash</td><td style="color:{color};">{block['hash']}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown(f"""
        <div class="glass-card" style="border-color:{color}40;">
          <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:{color};margin-bottom:16px;">
            OPERATION DATA
          </div>
          <table style="width:100%;font-family:'Share Tech Mono',monospace;font-size:0.75rem;border-collapse:collapse;">
            {"".join(f'<tr><td style="color:#88aacc;padding:4px 0;">{k}</td><td style="color:#e0f0ff;">{v}</td></tr>' for k,v in block["data"].items())}
          </table>
        </div>
        """, unsafe_allow_html=True)

    neon_divider()

    # ── Full ledger table ─────────────────────────────────────────────────────
    st.markdown("### 📋 FULL AUDIT LEDGER")
    rows = []
    for b in chain:
        rows.append({
            "Block #":    b["index"],
            "Timestamp":  b["timestamp"],
            "Operation":  b["data"]["op"],
            "User":       b["data"].get("user", "—"),
            "Block Hash": b["hash"],
            "Prev Hash":  b["prev_hash"],
            "Nonce":      b["nonce"],
            "Status":     "✅ VERIFIED",
        })
    df = pd.DataFrame(rows)

    def color_op(val):
        c = op_colors.get(val, "#88aacc")
        return f"color: {c}"

    st.dataframe(
        df.style.applymap(color_op, subset=["Operation"])
               .applymap(lambda v: "color: #00ff64", subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

    neon_divider()

    # ── Operation distribution ────────────────────────────────────────────────
    st.markdown("### 📊 OPERATION DISTRIBUTION")
    op_counts = {}
    for b in chain:
        op = b["data"]["op"]
        op_counts[op] = op_counts.get(op, 0) + 1

    fig2 = go.Figure(go.Pie(
        labels=list(op_counts.keys()),
        values=list(op_counts.values()),
        hole=0.55,
        marker=dict(
            colors=[op_colors.get(op, "#88aacc") for op in op_counts.keys()],
            line=dict(color="rgba(0,0,0,0.5)", width=2),
        ),
        textfont=dict(color="#e0f0ff", family="Share Tech Mono", size=10),
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#88aacc"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#88aacc")),
        margin=dict(l=10,r=10,t=10,b=10), height=300,
        annotations=[dict(text="OPS", x=0.5, y=0.5, font=dict(color="#00d4ff", size=16, family="Orbitron"),
                          showarrow=False)],
    )
    st.plotly_chart(fig2, use_container_width=True)
