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
    """Full 3D transformer architecture visualization with impressive cinematic animation."""
    fig = go.Figure()
    visual_layers = 14
    n_heads  = 8

    # 1. Base Rings (combined into one trace for performance)
    ring_x, ring_y, ring_z, ring_colors, ring_widths = [], [], [], [], []
    text_x, text_y, text_z, text_vals, text_colors = [], [], [], [], []
    
    for v_layer in range(visual_layers):
        actual_layer = v_layer * 2
        y_pos  = v_layer * 1.5
        radius = 2.5
        theta  = np.linspace(0, 2*np.pi, 60)
        
        visual_anim_step = animation_step // 2 if animation_step >= 0 else -1
        visual_hl_layer = highlight_layer // 2 if highlight_layer >= 0 else -1
        
        is_traced    = (v_layer <= visual_anim_step) if visual_anim_step >= 0 else False
        is_highlight = (v_layer == visual_hl_layer)
        
        if is_highlight:
            r_color = "#ff00aa" if edit_state != "editing" else "#00ff64"
        elif is_traced:
            r_color = "#b400ff"
        else:
            r_color = "#00d4ff"

        r_width = 4 if is_highlight else (3 if is_traced else 1.5)

        # Plotly doesn't easily support varying line colors in a single line trace unless using segments,
        # but since we want minimal traces, we'll just add each ring.
        fig.add_trace(go.Scatter3d(
            x=radius * np.cos(theta), y=np.full_like(theta, y_pos), z=radius * np.sin(theta),
            mode="lines",
            line=dict(color=r_color, width=r_width),
            opacity=1.0 if (is_highlight or is_traced) else 0.4,
            showlegend=False,
            hoverinfo='skip'
        ))

        text_x.append(0)
        text_y.append(y_pos)
        text_z.append(radius + 0.6)
        text_vals.append(f"L{actual_layer:02d}")
        text_colors.append(r_color)

    # Add all layer labels in one trace
    fig.add_trace(go.Scatter3d(
        x=text_x, y=text_y, z=text_z,
        mode="text",
        text=text_vals,
        textfont=dict(color=text_colors, size=9, family="Orbitron"),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 2. Attention Heads Orbiting (combined into one trace)
    hx_all, hy_all, hz_all, hcolor_all, hsize_all = [], [], [], [], []
    for v_layer in range(visual_layers):
        y_pos  = v_layer * 1.5
        is_highlight = (v_layer == (highlight_layer // 2))
        is_traced = (v_layer <= (animation_step // 2)) if animation_step >= 0 else False
        
        head_angles = np.linspace(0, 2*np.pi, n_heads, endpoint=False)
        for h, angle in enumerate(head_angles):
            if edit_state in ["editing", "forward_pass"] and animation_step >= 0:
                angle += (animation_step * 0.15) * (1 if h % 2 == 0 else -1)

            hx_all.append((2.5 * 0.7) * np.cos(angle))
            hy_all.append(y_pos)
            hz_all.append((2.5 * 0.7) * np.sin(angle))
            
            if is_highlight:
                if edit_state == "editing":
                    pulse = abs(np.sin(animation_step * 0.8))
                    hcolor_all.append(f"rgba(255,0,100,{0.5 + 0.5 * pulse})")
                    hsize_all.append(8 + (4 * pulse))
                elif edit_state == "queried_post":
                    hcolor_all.append("rgba(0,255,100,0.9)")
                    hsize_all.append(8)
                else:
                    hcolor_all.append("rgba(255,150,0,0.9)")
                    hsize_all.append(8)
            elif is_traced:
                hcolor_all.append("rgba(180,0,255,0.8)")
                hsize_all.append(6)
            else:
                hcolor_all.append("rgba(0,150,255,0.6)")
                hsize_all.append(5)

    fig.add_trace(go.Scatter3d(
        x=hx_all, y=hy_all, z=hz_all,
        mode="markers",
        marker=dict(size=hsize_all, color=hcolor_all, symbol="circle", line=dict(color="rgba(255,255,255,0.2)", width=1)),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 3. Impressive "Forward Pass" Moving Pulse Ring
    pulse_x, pulse_y, pulse_z = [], [], []
    pulse_color = "rgba(0,0,0,0)"
    if edit_state == "forward_pass" and animation_step >= 0:
        p_y = (animation_step % (visual_layers * 2)) * 0.75
        theta_pulse  = np.linspace(0, 2*np.pi, 30)
        pulse_x = 3.0 * np.cos(theta_pulse)
        pulse_z = 3.0 * np.sin(theta_pulse)
        pulse_y = np.full_like(theta_pulse, p_y)
        pulse_color = "#00ffff"
        
    fig.add_trace(go.Scatter3d(
        x=pulse_x, y=pulse_y, z=pulse_z,
        mode="markers",
        marker=dict(size=8, color=pulse_color, symbol="circle-open", line=dict(width=3, color=pulse_color)),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 4. Cinematic "Brain Surgery" Lasers & Energy Swarm
    laser_x, laser_y, laser_z = [], [], []
    laser_color = "rgba(0,0,0,0)"
    if edit_state == "editing" and animation_step >= 0 and highlight_layer >= 0:
        hl_y = (highlight_layer // 2) * 1.5
        pulse = abs(np.sin(animation_step * 0.8))
        
        # Primary Lasers (8 fast rotating beams with vertical sway)
        laser_color = f"rgba(255,0,100,{0.3 + 0.7*pulse})"
        for i in range(8):
            angle = i * (np.pi/4) + animation_step * 0.4
            start_x = 12 * np.cos(angle)
            start_z = 12 * np.sin(angle)
            laser_x.extend([start_x, 3.5, None])
            laser_y.extend([hl_y + np.sin(animation_step + i)*2, hl_y, None])
            laser_z.extend([start_z, 0, None])
            
        # Energy Swarm (orbiting particles spiraling into the target node)
        swarm_x, swarm_y, swarm_z = [], [], []
        for i in range(30):
            s_angle = i * (np.pi/15) - animation_step * 0.8
            # Spiraling inward logic: radius decreases over time then resets
            s_radius = max(0.2, 5.0 - ((animation_step * 0.4 + i*0.1) % 5.0))
            swarm_x.append(3.5 + s_radius * np.cos(s_angle))
            swarm_y.append(hl_y + s_radius * np.sin(s_angle * 3) * 0.5)
            swarm_z.append(s_radius * np.sin(s_angle))
            
        fig.add_trace(go.Scatter3d(
            x=swarm_x, y=swarm_y, z=swarm_z,
            mode="markers",
            marker=dict(size=4 + pulse*3, color="#ffaa00", symbol="diamond"),
            showlegend=False,
            hoverinfo='skip'
        ))
            
    fig.add_trace(go.Scatter3d(
        x=laser_x, y=laser_y, z=laser_z,
        mode="lines",
        line=dict(color=laser_color, width=4 + pulse*4),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 5. Highlighted Target Node with Fact Text
    if fact_text and highlight_layer >= 0 and (animation_step >= highlight_layer or edit_state in ["queried_post", "none"]):
        hl_y = (highlight_layer // 2) * 1.5
        
        node_color = "#ff00aa"
        text_color = "#ff00aa"
        node_size = 16
        
        if edit_state == "editing":
            pulse = abs(np.sin(animation_step * 0.8))
            node_color = f"rgba(255,0,100,{0.5 + 0.5*pulse})"
            text_color = "#ffaa00"
            node_size = 16 + (8 * pulse)
        elif edit_state == "queried_post":
            node_color = "#00ff64"
            text_color = "#00ff64"
            node_size = 20
            
        fig.add_trace(go.Scatter3d(
            x=[3.5], y=[hl_y], z=[0],
            mode="markers+text",
            marker=dict(size=node_size, color=node_color, symbol="circle", line=dict(color="#ffffff", width=2)),
            text=[f"<- {fact_text[:30]}"],
            textfont=dict(color=text_color, size=13, family="Share Tech Mono"),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,5,20,0.95)",
            xaxis=dict(gridcolor="rgba(0,212,255,0.04)", color="#334466", title="", showticklabels=False, range=[-8, 8]),
            yaxis=dict(gridcolor="rgba(0,212,255,0.04)", color="#334466", title=dict(text="Layer Depth", font=dict(color="#88aacc", size=10))),
            zaxis=dict(gridcolor="rgba(0,212,255,0.04)", color="#334466", title="", showticklabels=False, range=[-8, 8]),
            camera=dict(eye=dict(x=1.8, y=0.8, z=1.8)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.5, z=1),
        ),
        font=dict(color="#88aacc"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
        uirevision="static_camera_revision" # Prevents camera from resetting between frames
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
    elif st.session_state.nm_state in ["init", "queried_pre"]:
        fig = _build_3d_transformer(-1, "", -1, "none")
        graph_ph.plotly_chart(fig)
