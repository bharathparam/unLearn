"""Page 3 — Neural Memory Mapping (Animated 3D Visual & Tracing)"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import httpx
import time
import threading
from utils.helpers import neon_header, neon_divider

def threaded_api_call(payload, api_base, headers, result_dict, endpoint):
    """Run an API call in a separate thread so Streamlit can animate."""
    try:
        r = httpx.post(f"{api_base}/{endpoint}", json=payload, headers=headers, timeout=300)
        r.raise_for_status()
        result_dict['data'] = r.json()
        result_dict['status'] = 'success'
    except Exception as e:
        result_dict['status'] = 'error'
        result_dict['error'] = str(e)

def _build_3d_transformer(highlight_layer: int = -1, fact_text: str = "", animation_step: int = -1, edit_state: str = "none"):
    """Full 3D transformer architecture visualization with animation support."""
    fig = go.Figure()
    visual_layers = 14
    n_heads  = 8

    # ── Transformer rings (one per layer) ────────────────────────────────────
    for v_layer in range(visual_layers):
        actual_layer = v_layer * 2
        y_pos  = v_layer * 1.5
        radius = 2.5
        theta  = np.linspace(0, 2*np.pi, 60)
        x_ring = radius * np.cos(theta)
        z_ring = radius * np.sin(theta)
        y_ring = np.full_like(theta, y_pos)

        visual_anim_step = animation_step // 2 if animation_step >= 0 else -1
        visual_hl_layer = highlight_layer // 2 if highlight_layer >= 0 else -1
        
        is_traced    = (v_layer <= visual_anim_step) if visual_anim_step >= 0 else False
        is_highlight = (v_layer == visual_hl_layer)
        
        if is_highlight:
            ring_color = "#ff00aa" if edit_state != "editing" else "#00ff64"
        elif is_traced:
            ring_color = "#b400ff"
        else:
            ring_color = "#00d4ff"

        ring_width   = 4 if is_highlight else (3 if is_traced else 1.5)
        ring_opacity = 1.0 if (is_highlight or is_traced) else 0.5

        fig.add_trace(go.Scatter3d(
            x=x_ring, y=y_ring, z=z_ring,
            mode="lines",
            line=dict(color=ring_color, width=ring_width),
            opacity=ring_opacity,
            name=f"Layer {actual_layer}",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[y_pos], z=[radius + 0.6],
            mode="text",
            text=[f"L{actual_layer:02d}"],
            textfont=dict(color=ring_color, size=9, family="Orbitron"),
            showlegend=False,
        ))

        # ── Attention heads (orbiting particles) ─────────────────────────────
        head_angles = np.linspace(0, 2*np.pi, n_heads, endpoint=False)
        for h, angle in enumerate(head_angles):
            # Dynamic orbiting based on animation step if editing
            if edit_state in ["editing", "forward_pass"] and animation_step >= 0:
                angle += (animation_step * 0.1) * (h % 2 == 0 and 1 or -1)

            hx = (radius * 0.7) * np.cos(angle)
            hz = (radius * 0.7) * np.sin(angle)
            intensity = np.random.rand()
            
            if is_highlight:
                if edit_state == "editing":
                    # Laser pulsing
                    pulse = abs(np.sin(animation_step * 0.5)) if animation_step >= 0 else 1.0
                    hcolor = f"rgba(255,0,100,{0.5 + 0.5 * pulse})"
                    hsize  = 8 + (4 * pulse)
                elif edit_state == "queried_post":
                    hcolor = f"rgba(0,255,100,0.9)"
                    hsize  = 8
                else:
                    hcolor = f"rgba(255,{int(intensity*200)},0,0.9)"
                    hsize  = 8
            elif is_traced:
                hcolor = f"rgba(180,0,255,0.8)"
                hsize  = 6
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

    # ── Vertical data flow lines ───────────────────────
    for i in range(6):
        fx = np.cos(i * np.pi/3) * 1.5
        fz = np.sin(i * np.pi/3) * 1.5
        fy = np.linspace(0, visual_layers * 1.5, 40)
        
        flow_color = "#ff00aa" if (animation_step >= 0 and i % 2 == 0 and edit_state == "editing") else "#b400ff"
        
        fig.add_trace(go.Scatter3d(
            x=[fx]*40, y=fy, z=[fz]*40,
            mode="lines",
            line=dict(color=flow_color, width=1.5 if animation_step < 0 else 3),
            opacity=0.4 if animation_step < 0 else 0.8,
            showlegend=False,
        ))

    # ── Highlighted fact annotation ───────────────────────────────────────────
    if fact_text and highlight_layer >= 0 and (animation_step >= highlight_layer or edit_state == "queried_post"):
        visual_hl_layer = highlight_layer // 2
        hl_y = visual_hl_layer * 1.5
        
        node_color = "#ff00aa"
        text_color = "#ff00aa"
        if edit_state == "editing":
            pulse = abs(np.sin(animation_step * 0.5))
            node_color = f"rgba(255,0,100,{0.5 + 0.5*pulse})"
            text_color = "#ffaa00"
        elif edit_state == "queried_post":
            node_color = "#00ff64"
            text_color = "#00ff64"
            
        fig.add_trace(go.Scatter3d(
            x=[3.5], y=[hl_y], z=[0],
            mode="markers+text",
            marker=dict(size=16 if edit_state != "editing" else 20, 
                        color=node_color, symbol="circle",
                        line=dict(color="#ffffff", width=2)),
            text=[f"<- {fact_text[:30]}"],
            textfont=dict(color=text_color, size=11, family="Share Tech Mono"),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,5,20,0.95)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466", title="", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466", title=dict(text="Layer Depth", font=dict(color="#88aacc", size=10))),
            zaxis=dict(gridcolor="rgba(0,212,255,0.06)", color="#334466", title="", showticklabels=False),
            camera=dict(eye=dict(x=1.6, y=0.8, z=1.6)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.5, z=1),
        ),
        font=dict(color="#88aacc"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )
    return fig


def render():
    st.sidebar.markdown("### 🔌 API CONFIGURATION")
    api_base = st.sidebar.text_input("ROME Backend URL", value="https://postlabially-overinstructive-aurore.ngrok-free.dev")
    mia_api_base = st.sidebar.text_input("MIA Engine URL", value="https://unsenile-subtransversally-julien.ngrok-free.dev")
    headers = {"ngrok-skip-browser-warning": "1"}

    if st.sidebar.button("🧹 RESTORE ORIGINAL WEIGHTS"):
        with st.spinner("Restoring weights..."):
            try:
                r = httpx.post(f"{api_base}/restore", headers=headers, timeout=60)
                st.sidebar.success(f"Restored: {r.json().get('edits_cleared')} edits.")
                st.session_state.nm_state = "init"
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Restore failed: {e}")

    neon_header("NEURAL MEMORY MAPPING", "LIVE ROME EDITING · QWEN 1.5B · CAUSAL TRACING", "🧠")

    # State management
    if "nm_state" not in st.session_state: st.session_state.nm_state = "init"
    if "nm_prompt" not in st.session_state: st.session_state.nm_prompt = ""
    if "nm_subject" not in st.session_state: st.session_state.nm_subject = ""
    if "nm_target" not in st.session_state: st.session_state.nm_target = ""
    if "nm_layer" not in st.session_state: st.session_state.nm_layer = 15
    if "nm_response_pre" not in st.session_state: st.session_state.nm_response_pre = ""
    if "nm_response_post" not in st.session_state: st.session_state.nm_response_post = ""

    # ── Phase 1: Query Input ──────────────────────────────────────────────────
    st.markdown("### 🔍 1. QUERY THE MODEL")
    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        prompt_val = st.text_input("Enter a factual prompt (without chat template):", value=st.session_state.nm_prompt)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_query = st.button("🤖 QUERY MODEL")

    query_out_ph = st.empty()
    
    neon_divider()

    # ── Phase 2: 3D Visualization ─────────────────────────────────────────────
    st.markdown("### 🌐 2. CAUSAL TRACE VISUALIZATION")
    col_viz, col_info = st.columns([3, 1])
    with col_viz:
        graph_ph = st.empty()
    with col_info:
        info_ph = st.empty()
        
    edit_ph = st.empty()
    verify_ph = st.empty()

    # ── Logic ─────────────────────────────────────────────────────────────────
    
    # Render static legend in info panel
    info_ph.markdown("""
    <div class="glass-card" style="margin-bottom:12px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:#88aacc;margin-bottom:10px;">LEGEND</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;line-height:2;">
        <span style="color:#00d4ff;">━━</span> Transformer Ring<br>
        <span style="color:#b400ff;">━━</span> Traced Path<br>
        <span style="color:#ff00aa;">◆</span> Target Memory<br>
        <span style="color:#00ff64;">◆</span> Edited Memory<br>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. HANDLE NEW QUERY
    if (btn_query or (prompt_val and prompt_val != st.session_state.nm_prompt)):
        st.session_state.nm_prompt = prompt_val
        st.session_state.nm_state = "init"
        
        # Start Thread
        query_res = {'status': 'running'}
        payload = {"prompt": prompt_val, "max_new_tokens": 15, "do_sample": False, "use_chat_template": True}
        t = threading.Thread(target=threaded_api_call, args=(payload, api_base, headers, query_res, "query"))
        t.start()
        
        # Animate Forward Pass
        step = 0
        while query_res['status'] == 'running':
            try:
                fig = _build_3d_transformer(animation_step=step, edit_state="forward_pass")
                graph_ph.plotly_chart(fig, use_container_width=True)
                step += 2
                time.sleep(0.15)
            except Exception:
                break
            
        if query_res['status'] == 'success':
            st.session_state.nm_response_pre = query_res['data'].get("response", "").strip()
            st.session_state.nm_state = "queried_pre"
            st.rerun()
        else:
            st.error(f"Query Failed: {query_res.get('error')}")

    # Display Query Output if available
    if st.session_state.nm_state in ["queried_pre", "editing", "queried_post"]:
        query_out_ph.markdown(f"""
        <div class="glass-card" style="border-color:rgba(0,212,255,0.5);margin-bottom:1rem;">
          <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:#00d4ff;margin-bottom:8px;">
            OUTPUT (PRE-EDIT)
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#ffffff;">
            {st.session_state.nm_response_pre}
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. HANDLE EDIT UI & BRAIN SURGERY
    if st.session_state.nm_state == "queried_pre":
        with edit_ph.container():
            neon_divider()
            st.markdown("### ✍️ 3. EDIT MODEL MEMORY")
            e_col1, e_col2, e_col3 = st.columns([2, 2, 1])
            with e_col1:
                subj_val = st.text_input("Subject to edit (must be exactly in prompt):", value=st.session_state.nm_subject, placeholder="e.g. india")
            with e_col2:
                target_val = st.text_input("New Target (the new answer):", value=st.session_state.nm_target, placeholder="e.g. Mumbai")
            with e_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                apply_edit = st.button("🚀 EXECUTE BRAIN SURGERY")
                
            if apply_edit and subj_val and target_val:
                st.session_state.nm_subject = subj_val
                st.session_state.nm_target = target_val
                st.session_state.nm_state = "editing"
                st.rerun()

    if st.session_state.nm_state == "editing":
        # Render info block
        with info_ph.container():
            st.markdown(f"""
            <div class="glass-card" style="margin-top:12px;border-color:rgba(255,0,170,0.4);">
              <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:#ff00aa;margin-bottom:8px;">MEMORY DISCOVERED</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#88aacc;line-height:1.8;">
                Subject: <span style="color:#ff00aa;">{st.session_state.nm_subject}</span><br>
                Layer: <span style="color:#ff00aa;">L{st.session_state.nm_layer}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        edit_ph.warning("⚡ **BRAIN SURGERY IN PROGRESS**: ROME algorithm is performing Rank-One Modification on the MLP weights. This takes ~60-120 seconds. Do not refresh...")
        
        # Start Threaded Edit
        edit_res = {'status': 'running'}
        payload = {
            "prompt": st.session_state.nm_prompt,
            "target": st.session_state.nm_target,
            "subject": st.session_state.nm_subject,
            "layer": st.session_state.nm_layer,
            "v_num_grad_steps": 30,
            "v_lr": 0.1
        }
        t_edit = threading.Thread(target=threaded_api_call, args=(payload, api_base, headers, edit_res, "edit"))
        t_edit.start()
        
        # Animate Brain Surgery continuously
        step = 0
        while edit_res['status'] == 'running':
            try:
                # Cap pulse step so tracing stops at the layer but animation continues
                fig = _build_3d_transformer(st.session_state.nm_layer, st.session_state.nm_subject, step, "editing")
                graph_ph.plotly_chart(fig, use_container_width=True)
                step += 1
                time.sleep(0.3)
            except Exception:
                break
            
        if edit_res['status'] == 'success':
            # Auto query again
            query_res = {'status': 'running'}
            payload_q = {"prompt": st.session_state.nm_prompt, "max_new_tokens": 15, "do_sample": False, "use_chat_template": True}
            t_q = threading.Thread(target=threaded_api_call, args=(payload_q, api_base, headers, query_res, "query"))
            t_q.start()
            
            # Spin until query finishes
            while query_res['status'] == 'running':
                time.sleep(0.1)
                
            if query_res['status'] == 'success':
                st.session_state.nm_response_post = query_res['data'].get("response", "").strip()
                st.session_state.nm_state = "queried_post"
                st.rerun()
            else:
                st.error("Edit succeeded but verification query failed.")
        else:
            st.error(f"Brain Surgery Failed: {edit_res.get('error')}")
            st.session_state.nm_state = "queried_pre"
            if st.button("Retry"):
                st.rerun()

    # 3. STATIC RENDER FOR END STATE
    if st.session_state.nm_state == "queried_post":
        fig = _build_3d_transformer(st.session_state.nm_layer, st.session_state.nm_subject, st.session_state.nm_layer, "queried_post")
        graph_ph.plotly_chart(fig, use_container_width=True)
        
        with info_ph.container():
            st.markdown(f"""
            <div class="glass-card" style="margin-top:12px;border-color:rgba(0,255,100,0.4);">
              <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:#00ff64;margin-bottom:8px;">MEMORY REWRITTEN</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#88aacc;line-height:1.8;">
                Subject: <span style="color:#00ff64;">{st.session_state.nm_subject}</span><br>
                Target: <span style="color:#00ff64;">{st.session_state.nm_target}</span><br>
                Layer: <span style="color:#00ff64;">L{st.session_state.nm_layer}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with edit_ph.container():
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(0,255,100,0.6);margin-top:1rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:#00ff64;margin-bottom:8px;">
                OUTPUT (POST-EDIT) — Verification Successful
              </div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#ffffff;">
                {st.session_state.nm_response_post}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ ROME edit successfully applied. The MLP weight matrix has been modified.")
            
        with verify_ph.container():
            neon_divider()
            st.markdown("### 🔒 4. PRIVACY VERIFICATION (MIA)")
            st.info("Run a Membership Inference Attack to verify data leakage probabilistically.")
            
            if st.button("🛡️ RUN PRIVACY AUDIT"):
                with st.spinner("Calling Neural Verification Engine..."):
                    try:
                        payload = {
                            "secret": st.session_state.nm_prompt,
                            "before_output": st.session_state.nm_response_pre,
                            "after_output": st.session_state.nm_response_post
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
                                <b>Attack Success Rate:</b> {data.get("attack_success_rate", 0)}%
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
                        st.error(f"MIA Engine Failed: {e}")
    # Render idle graph if not animated
    elif st.session_state.nm_state == "init":
        fig = _build_3d_transformer(-1, "", -1, "none")
        graph_ph.plotly_chart(fig, use_container_width=True)
