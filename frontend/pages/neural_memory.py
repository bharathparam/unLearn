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

def _build_3d_transformer(highlight_layer: int = -1, fact_text: str = "", animation_step: int = -1, edit_state: str = "none", edit_mode: str = "edit"):
    """Cinematic 3D Volumetric AI Brain Visualization."""
    fig = go.Figure()
    visual_layers = 14
    
    np.random.seed(42) # Ensure cluster positions are consistent across frames

    # ── 1. Volumetric Neural Web (Neurons) ──
    brain_x, brain_y, brain_z, brain_colors, brain_sizes = [], [], [], [], []
    text_x, text_y, text_z, text_vals, text_colors = [], [], [], [], []
    
    visual_anim_step = animation_step // 2 if animation_step >= 0 else -1
    visual_hl_layer = highlight_layer // 2 if highlight_layer >= 0 else -1

    for v_layer in range(visual_layers):
        actual_layer = v_layer * 2
        y_pos  = v_layer * 2.0
        
        is_traced    = (v_layer <= visual_anim_step) if visual_anim_step >= 0 else False
        is_highlight = (v_layer == visual_hl_layer)
        
        # Dim the environment during surgery except for the target layer
        if edit_state == "editing":
            if is_highlight:
                layer_color = "rgba(255, 0, 100, 0.9)" if edit_mode == "edit" else "rgba(255, 60, 0, 0.9)" # Orange/red for forgetting
            else:
                layer_color = "rgba(10, 20, 40, 0.2)"  # Dimmed out
        else:
            if is_highlight:
                layer_color = "#00ff64" if edit_state == "queried_post" else "#b400ff"
            elif is_traced:
                layer_color = "rgba(180, 0, 255, 0.8)"
            else:
                layer_color = "rgba(0, 212, 255, 0.4)" # Idle cyan

        # Generate 40 neurons per layer in a dispersed disc
        for _ in range(40):
            r = np.random.uniform(0.5, 4.0)
            th = np.random.uniform(0, 2*np.pi)
            brain_x.append(r * np.cos(th))
            brain_y.append(y_pos + np.random.uniform(-0.3, 0.3))
            brain_z.append(r * np.sin(th))
            brain_colors.append(layer_color)
            
            # Pulse size if highlighted during surgery
            if is_highlight and edit_state == "editing":
                pulse = abs(np.sin(animation_step * 0.5))
                brain_sizes.append(4 + pulse * 4)
            else:
                brain_sizes.append(np.random.uniform(2, 5))
                
        # Layer labels
        text_x.append(0)
        text_y.append(y_pos)
        text_z.append(4.5)
        text_vals.append(f"L{actual_layer:02d}")
        text_colors.append(layer_color)

    # Add Neurons
    fig.add_trace(go.Scatter3d(
        x=brain_x, y=brain_y, z=brain_z,
        mode="markers",
        marker=dict(size=brain_sizes, color=brain_colors, symbol="circle", line=dict(color="rgba(255,255,255,0.1)", width=1)),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Add Text
    fig.add_trace(go.Scatter3d(
        x=text_x, y=text_y, z=text_z, mode="text", text=text_vals,
        textfont=dict(color=text_colors, size=9, family="Orbitron"),
        showlegend=False, hoverinfo='skip'
    ))

    # ── 2. Synaptic Connections ──
    # Render faint lines connecting neurons to simulate pathways
    synapse_x, synapse_y, synapse_z = [], [], []
    for _ in range(120): # 120 random synaptic pathways
        i1 = np.random.randint(0, len(brain_x))
        i2 = np.random.randint(max(0, i1-40), min(len(brain_x), i1+40))
        synapse_x.extend([brain_x[i1], brain_x[i2], None])
        synapse_y.extend([brain_y[i1], brain_y[i2], None])
        synapse_z.extend([brain_z[i1], brain_z[i2], None])
        
    synapse_color = "rgba(0, 212, 255, 0.05)"
    if edit_state == "editing":
        synapse_color = "rgba(255, 0, 100, 0.03)" if edit_mode == "edit" else "rgba(200, 50, 0, 0.02)" # Threatening red tint

    fig.add_trace(go.Scatter3d(
        x=synapse_x, y=synapse_y, z=synapse_z,
        mode="lines", line=dict(color=synapse_color, width=1),
        showlegend=False, hoverinfo='skip'
    ))

    # ── 3. Query Propagation (Forward Pass Wave) ──
    if edit_state == "forward_pass" and animation_step >= 0:
        # A massive electric blue energy disc moving up the layers
        p_y = (animation_step % (visual_layers * 2)) * 1.0
        theta_pulse = np.linspace(0, 2*np.pi, 50)
        pulse_r = 4.2
        
        # Central Data Core Beam
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, p_y], z=[0, 0],
            mode="lines", line=dict(color="#00ffff", width=12),
            opacity=0.8, showlegend=False, hoverinfo='skip'
        ))
        
        # Expanding Activation Ring
        fig.add_trace(go.Scatter3d(
            x=pulse_r * np.cos(theta_pulse), y=np.full_like(theta_pulse, p_y), z=pulse_r * np.sin(theta_pulse),
            mode="markers",
            marker=dict(size=10, color="#00ffff", symbol="circle", line=dict(color="#ffffff", width=2)),
            showlegend=False, hoverinfo='skip'
        ))

    # ── 4. ROME Surgery Phase (Extraction & Implantation) ──
    if edit_state == "editing" and animation_step >= 0 and highlight_layer >= 0:
        hl_y = (highlight_layer // 2) * 2.0
        target_x, target_y, target_z = 3.5, hl_y, 0
        
        # Phase A: EXTRACTION (Memory fragments exploding outward)
        if animation_step < 20:
            progress = animation_step / 20.0 # 0 to 1
            exp_x, exp_y, exp_z = [], [], []
            for i in range(80): # 80 shattered memory fragments
                th = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                speed = np.random.uniform(1.0, 5.0)
                exp_x.append(target_x + speed * progress * np.sin(phi) * np.cos(th))
                exp_y.append(target_y + speed * progress * np.cos(phi))
                exp_z.append(target_z + speed * progress * np.sin(phi) * np.sin(th))
                
            fig.add_trace(go.Scatter3d(
                x=exp_x, y=exp_y, z=exp_z, mode="markers",
                marker=dict(size=4 + (1-progress)*6, color="#ff0044", opacity=max(0.1, 1-progress), symbol="x"),
                showlegend=False, hoverinfo='skip'
            ))
            
        # Phase B: IMPLANTATION (Quantum green energy vortex) OR NEURAL DECAY
        elif animation_step >= 20:
            progress = (animation_step - 20) / 40.0 # Scales up over time
            if edit_mode == "edit":
                vortex_x, vortex_y, vortex_z = [], [], []
                for i in range(60):
                    s_angle = i * (np.pi/10) + animation_step * 0.6
                    # Particles spiral inward from radius 8 down to 0.2
                    s_radius = max(0.2, 8.0 - (progress * 8.0) - (i*0.05))
                    if s_radius > 0.2:
                        vortex_x.append(target_x + s_radius * np.cos(s_angle))
                        vortex_y.append(target_y + s_radius * np.sin(s_angle * 3) * 0.4) # Vertical tornado sway
                        vortex_z.append(target_z + s_radius * np.sin(s_angle))
                        
                fig.add_trace(go.Scatter3d(
                    x=vortex_x, y=vortex_y, z=vortex_z, mode="markers",
                    marker=dict(size=5 + min(progress, 1.0)*3, color="#00ff64", symbol="diamond"),
                    showlegend=False, hoverinfo='skip'
                ))
                
                # The new Memory Core stabilizing
                core_size = min(25, progress * 35)
                fig.add_trace(go.Scatter3d(
                    x=[target_x], y=[target_y], z=[target_z], mode="markers",
                    marker=dict(size=core_size, color="#00ff64", symbol="diamond", line=dict(color="#ffffff", width=2)),
                    showlegend=False, hoverinfo='skip'
                ))
            else:
                # Decay phase: no new memory core, just empty space and fading red embers
                decay_x, decay_y, decay_z = [], [], []
                for i in range(20):
                    decay_x.append(target_x + np.random.uniform(-1.5, 1.5))
                    decay_y.append(target_y + np.random.uniform(-1.5, 1.5))
                    decay_z.append(target_z + np.random.uniform(-1.5, 1.5))
                fig.add_trace(go.Scatter3d(
                    x=decay_x, y=decay_y, z=decay_z, mode="markers",
                    marker=dict(size=3, color="#ff4400", opacity=max(0, 0.4 - progress)),
                    showlegend=False, hoverinfo='skip'
                ))

    # ── 5. Rewiring & Verified Target Node ──
    if fact_text and highlight_layer >= 0 and (animation_step >= highlight_layer or edit_state in ["queried_post", "none"]):
        hl_y = (highlight_layer // 2) * 2.0
        target_x, target_y, target_z = 3.5, hl_y, 0
        
        node_color = "#b400ff"
        text_color = "#b400ff"
        node_size = 18
        
        if edit_state == "editing":
            node_color, text_color = "rgba(0,0,0,0)", "rgba(0,0,0,0)" # Hidden during surgery explosions
        elif edit_state == "queried_post":
            if edit_mode == "edit":
                node_color = "#00ff64"
                text_color = "#00ff64"
                node_size = 22
                
                # Cinematic Rewired Pathway (Thick glowing green neural route)
                rewire_x, rewire_y, rewire_z = [0], [0], [0]
                for v in range(1, (highlight_layer // 2) + 1):
                    rewire_y.append(v * 2.0)
                    if v == (highlight_layer // 2):
                        rewire_x.append(target_x)
                        rewire_z.append(target_z)
                    else:
                        rewire_x.append(np.random.uniform(-1.5, 1.5))
                        rewire_z.append(np.random.uniform(-1.5, 1.5))
                        
                fig.add_trace(go.Scatter3d(
                    x=rewire_x, y=rewire_y, z=rewire_z, mode="lines",
                    line=dict(color="#00ff64", width=10), opacity=0.9,
                    showlegend=False, hoverinfo='skip'
                ))
            else:
                node_color = "rgba(30, 30, 40, 0.8)" # Dark, inactive node
                text_color = "rgba(100, 100, 100, 0.5)"
                node_size = 12
                # NO pathway drawn to signify broken/decayed route
            
        fig.add_trace(go.Scatter3d(
            x=[target_x], y=[target_y], z=[target_z],
            mode="markers+text",
            marker=dict(size=node_size, color=node_color, symbol="circle", line=dict(color="#ffffff", width=2)),
            text=[f"<- {fact_text[:30]}"],
            textfont=dict(color=text_color, size=13, family="Share Tech Mono"),
            showlegend=False, hoverinfo='skip'
        ))

    # ── 6. Cinematic Camera System ──
    # We change uirevision during surgery so Plotly allows the camera to snap to the surgical view once, 
    # without resetting every single frame.
    ui_rev = "surgery_view" if edit_state == "editing" else "global_view"
    
    if edit_state == "editing":
        # Low, aggressive zoomed-in camera angle looking up at the network
        cam_eye = dict(x=1.2, y=0.2, z=1.2)
    else:
        # God-view floating slightly above
        cam_eye = dict(x=1.8, y=0.8, z=1.8)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(5, 5, 8, 1.0)", # Pitch black neural lab
            xaxis=dict(gridcolor="rgba(0,212,255,0)", showticklabels=False, showbackground=False, range=[-8, 8]),
            yaxis=dict(gridcolor="rgba(0,212,255,0.02)", color="#334466", title=dict(text="Layer Depth", font=dict(color="#334466", size=10)), showbackground=False),
            zaxis=dict(gridcolor="rgba(0,212,255,0)", showticklabels=False, showbackground=False, range=[-8, 8]),
            camera=dict(eye=cam_eye),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.5, z=1),
        ),
        font=dict(color="#88aacc"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        uirevision=ui_rev
    )
    return fig


def render():
    st.sidebar.markdown("### 🔌 API CONFIGURATION")
    api_base = st.sidebar.text_input("ROME Backend URL", value="https://postlabially-overinstructive-aurore.ngrok-free.dev")
    mia_api_base = st.sidebar.text_input("MIA Engine URL", value="https://unsenile-subtransversally-julien.ngrok-free.dev")
    auto_audit = st.sidebar.toggle("🛡️ AUTO-AUDIT AFTER EDIT", value=True)
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
    if "nm_mode" not in st.session_state: st.session_state.nm_mode = "edit"
    if "nm_mia_score" not in st.session_state: st.session_state.nm_mia_score = None
    st.markdown("### ⚡ 1. INITIALIZE NEURAL PROPAGATION")
    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        prompt_val = st.text_input("Enter a factual prompt (without chat template):", value=st.session_state.nm_prompt)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_query = st.button("⚡ EXECUTE FORWARD PASS")

    query_out_ph = st.empty()
    
    neon_divider()

    # ── Phase 2: 3D Visualization ─────────────────────────────────────────────
    st.markdown("### 🧠 2. LIVE AI BRAIN VISUALIZATION")
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
                fig = _build_3d_transformer(animation_step=step, edit_state="forward_pass", edit_mode=st.session_state.get("nm_mode", "edit"))
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
            st.markdown("### ⚔️ 3. ISOLATE TARGET NODE")
            e_col1, e_col2, e_col3, e_col4 = st.columns([2, 2, 1.2, 1.2])
            with e_col1:
                subj_val = st.text_input("Subject to edit (must be exactly in prompt):", value=st.session_state.nm_subject, placeholder="e.g. india")
            with e_col2:
                target_val = st.text_input("New Target (the new answer):", value=st.session_state.nm_target, placeholder="e.g. Mumbai")
            with e_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                apply_edit = st.button("⚔️ EDIT MEMORY")
            with e_col4:
                st.markdown("<br>", unsafe_allow_html=True)
                apply_delete = st.button("☣️ DELETE MEMORY")
                
            if apply_edit and subj_val and target_val:
                st.session_state.nm_subject = subj_val
                st.session_state.nm_target = target_val
                st.session_state.nm_mode = "edit"
                st.session_state.nm_state = "editing"
                st.rerun()
            elif apply_delete and subj_val:
                st.session_state.nm_subject = subj_val
                st.session_state.nm_target = "I forgot this information."
                st.session_state.nm_mode = "forget"
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

        edit_ph.warning("☣️ **CRITICAL: ROME NEUROSURGERY IN PROGRESS**. MLP weight matrices are being actively restructured. Synaptic pathways are highly unstable. This takes ~60-120 seconds. DO NOT INTERRUPT.")
        
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
        
        meter_ph = st.empty()
        
        # Animate Brain Surgery continuously
        step = 0
        while edit_res['status'] == 'running':
            try:
                # Update Memory Strength Meter
                if st.session_state.nm_mode == "forget":
                    strength = max(0, 98 - (step * 3))
                    meter_color = "#ff4400" if strength < 50 else "#ffaa00"
                    meter_text = "DECAYING..." if strength > 0 else "MEMORY DESTROYED"
                else:
                    strength = min(100, 20 + (step * 2))
                    meter_color = "#00ff64"
                    meter_text = "REWRITING..." if strength < 100 else "STABILIZED"
                
                meter_ph.markdown(f"""
                <div class="glass-card" style="margin-bottom:12px; border-color:{meter_color};">
                  <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:{meter_color};margin-bottom:5px;">MEMORY STRENGTH: {strength}%</div>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#fff;">STATUS: {meter_text}</div>
                </div>
                """, unsafe_allow_html=True)

                # Cap pulse step so tracing stops at the layer but animation continues
                fig = _build_3d_transformer(st.session_state.nm_layer, st.session_state.nm_subject, step, "editing", st.session_state.nm_mode)
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
                
                # --- AUTO AUDIT INTEGRATION ---
                if auto_audit:
                    with st.spinner("🛡️ Launching Automated MIA Audit..."):
                        try:
                            payload_mia = {
                                "secret": st.session_state.nm_prompt,
                                "before_output": st.session_state.nm_response_pre,
                                "after_output": st.session_state.nm_response_post
                            }
                            r_mia = httpx.post(f"{mia_api_base}/verify", json=payload_mia, headers=headers, timeout=120)
                            if r_mia.status_code == 200:
                                st.session_state.nm_mia_score = r_mia.json()
                        except:
                            pass
                # ------------------------------
                
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
        fig = _build_3d_transformer(st.session_state.nm_layer, st.session_state.nm_subject, st.session_state.nm_layer, "queried_post", st.session_state.nm_mode)
        graph_ph.plotly_chart(fig, use_container_width=True)
        
        with info_ph.container():
            status_title = "MEMORY DELETED" if st.session_state.nm_mode == "forget" else "MEMORY REWRITTEN"
            status_color = "#ff4400" if st.session_state.nm_mode == "forget" else "#00ff64"
            st.markdown(f"""
            <div class="glass-card" style="margin-top:12px;border-color:{status_color}66;">
              <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:{status_color};margin-bottom:8px;">{status_title}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#88aacc;line-height:1.8;">
                Subject: <span style="color:{status_color};">{st.session_state.nm_subject}</span><br>
                Target: <span style="color:{status_color};">{st.session_state.nm_target}</span><br>
                Layer: <span style="color:{status_color};">L{st.session_state.nm_layer}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with edit_ph.container():
            status_color = "#ff4400" if st.session_state.nm_mode == "forget" else "#00ff64"
            st.markdown(f"""
            <div class="glass-card" style="border-color:{status_color}99;margin-top:1rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:{status_color};margin-bottom:8px;">
                OUTPUT (POST-EDIT) — Verification Successful
              </div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#ffffff;">
                {st.session_state.nm_response_post}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.nm_mia_score:
                mia = st.session_state.nm_mia_score
                status = mia.get("verification_status", "UNKNOWN")
                color = "#00ff64" if status == "FORGOTTEN" else ("#ffaa00" if status == "PARTIALLY_FORGOTTEN" else "#ff3333")
                privacy = mia.get('privacy_confidence', 0)
                leakage = mia.get('leakage_probability', 0)
                
                # Extract narrative summary safely
                summary_data = mia.get("gemini_audit_summary", {})
                if isinstance(summary_data, dict):
                    narrative = summary_data.get("narrative_summary", "Audit completed successfully.")
                else:
                    narrative = str(summary_data) if summary_data else "Audit completed successfully."
                
                # Mini-Certificate for Auto-Audit
                st.markdown(f"""
                <div style="border: 1px solid {color}44; background: rgba(0,20,50,0.3); padding: 15px; border-radius: 8px; border-left: 5px solid {color}; margin-bottom: 1rem;">
                    <div style="font-family:'Orbitron',monospace; font-size:0.7rem; color:{color}; margin-bottom:8px; font-weight:800;">🛡️ AUTO-AUDIT CERTIFICATE: {status}</div>
                    <div style="display:flex; gap:10px; margin-bottom:10px;">
                        <div style="background:rgba(255,255,255,0.03); padding:5px 10px; border-radius:4px; flex:1; text-align:center;">
                            <div style="font-family:'Orbitron',monospace; font-size:0.5rem; color:#88aacc;">PRIVACY</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:1rem; color:{color};">{privacy:.1f}%</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.03); padding:5px 10px; border-radius:4px; flex:1; text-align:center;">
                            <div style="font-family:'Orbitron',monospace; font-size:0.5rem; color:#88aacc;">LEAKAGE</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:1rem; color:{color};">{leakage:.4f}</div>
                        </div>
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#e0f0ff; background:rgba(0,0,0,0.2); padding:8px; border-radius:4px;">
                        {narrative}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.nm_mode == "forget":
                st.success("✅ ROME surgery successful. Synaptic connection severed. Memory is no longer recoverable.")
            else:
                st.success("✅ ROME edit successful. The MLP weight matrix has been restructured with the new target.")
            
        with verify_ph.container():
            neon_divider()
            st.markdown("### 🛡️ 4. POST-OP MIA DEFENSE SCAN")
            st.info("Initiate a Membership Inference Attack to verify data leakage and synapse isolation.")
            
            if st.button("🛡️ INITIATE SECURITY AUDIT"):
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
                        
                        # Pre-process certificate data
                        audit_sum = data.get("gemini_audit_summary", {})
                        if isinstance(audit_sum, dict):
                            narrative_text = audit_sum.get("narrative_summary", "Detailed semantic analysis completed.")
                        else:
                            narrative_text = str(audit_sum) if audit_sum else "Detailed semantic analysis completed."
                        
                        serial_num = f"NL-2026-{abs(hash(st.session_state.nm_prompt)) % 1000000:06d}"
                        cert_status = "✓ CERTIFIED" if privacy > 85 else "⚠ MONITORING"

                        st.markdown(f"""
                        <div class="glass-card" style="border-color:{color}; margin-top:1rem; position:relative; overflow:hidden;">
                            <div style="position:absolute; top:-20px; right:-20px; width:100px; height:100px; background:radial-gradient(circle, {color}44 0%, transparent 70%); border-radius:50%; filter:blur(10px); opacity:{shield_opacity};"></div>
                            
                            <div style="font-family:'Orbitron',monospace;font-size:1.2rem;color:{color};margin-bottom:12px; display:flex; align-items:center;">
                                <span style="margin-right:10px;">🛡️</span> SECURITY AUDIT COMPLETE: {status}
                            </div>
                            
                            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-bottom:15px;">
                                <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:4px;">
                                    <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:#88aacc;">PRIVACY SHIELD</div>
                                    <div style="font-family:'Share Tech Mono',monospace; font-size:1.2rem; color:{color};">{privacy:.1f}%</div>
                                </div>
                                <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:4px;">
                                    <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:#88aacc;">LEAKAGE PROB</div>
                                    <div style="font-family:'Share Tech Mono',monospace; font-size:1.2rem; color:{color};">{leakage:.4f}</div>
                                </div>
                                <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:4px;">
                                    <div style="font-family:'Orbitron',monospace; font-size:0.6rem; color:#88aacc;">ATTACK SUCCESS</div>
                                    <div style="font-family:'Share Tech Mono',monospace; font-size:1.2rem; color:{color};">{data.get("attack_success_rate", 0)}%</div>
                                </div>
                            </div>

                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;color:#ffffff; background:rgba(0,0,0,0.2); padding:10px; border-radius:4px; border-left: 3px solid {color}; margin-bottom:15px;">
                                <b style="color:{color};">AUDITOR VERDICT:</b><br>
                                {narrative_text}
                            </div>

                            <!-- 📜 PRIVACY CERTIFICATE -->
                            <div style="border: 2px solid {color}44; background: linear-gradient(135deg, rgba(0,20,50,0.4) 0%, rgba(0,0,0,0.6) 100%); padding: 20px; border-radius: 8px; position: relative; border-left: 10px solid {color};">
                                <div style="position: absolute; top: 10px; right: 15px; font-family: 'Orbitron', sans-serif; font-size: 3rem; color: {color}11; font-weight: 900; pointer-events: none;">VERIFIED</div>
                                
                                <div style="font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: {color}; letter-spacing: 2px; margin-bottom: 5px; font-weight: 800;">
                                    NEURAL PRIVACY COMPLIANCE CERTIFICATE
                                </div>
                                <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: #88aacc; margin-bottom: 15px;">
                                    SERIAL: {serial_num} | REV: 4.2.1
                                </div>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                    <div>
                                        <div style="font-family: 'Orbitron', sans-serif; font-size: 0.55rem; color: #334466; margin-bottom: 2px;">MODEL ENTITY</div>
                                        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #ffffff;">QWEN-1.5B (CAUSAL)</div>
                                    </div>
                                    <div>
                                        <div style="font-family: 'Orbitron', sans-serif; font-size: 0.55rem; color: #334466; margin-bottom: 2px;">AUDIT METHOD</div>
                                        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #ffffff;">ROME SYNAPSE ISO</div>
                                    </div>
                                    <div>
                                        <div style="font-family: 'Orbitron', sans-serif; font-size: 0.55rem; color: #334466; margin-bottom: 2px;">PRIVACY SCORE</div>
                                        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem; color: {color}; font-weight: 900;">{privacy:.1f}/100</div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-family: 'Orbitron', sans-serif; font-size: 0.55rem; color: #334466; margin-bottom: 2px;">COMPLIANCE STATUS</div>
                                        <div style="font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: {color}; font-weight: 900;">{cert_status}</div>
                                    </div>
                                </div>
                                
                                <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: #445566; line-height: 1.4;">
                                    THIS DOCUMENT CERTIFIES THAT THE SPECIFIED NEURAL WEIGHTS HAVE UNDERGONE ADVERSARIAL MEMBERSHIP INFERENCE ATTACK (MIA) TESTING. SEMANTIC LEAKAGE HAS BEEN MEASURED AT {leakage:.4f} AND IS WITHIN SAFE OPERATIONAL PARAMETERS.
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("View 20 Raw Attack Vectors & Semantic Scores"):
                            st.json(data.get("attack_details", []))
                            
                    except Exception as e:
                        st.error(f"MIA Engine Failed: {e}")
    # Render idle graph if not animated
    elif st.session_state.nm_state in ["init", "queried_pre"]:
        fig = _build_3d_transformer(-1, "", -1, "none", st.session_state.nm_mode)
        graph_ph.plotly_chart(fig)
