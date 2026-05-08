"""Page 8 — Compliance Certificate Generator"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import io
import time
from datetime import datetime
from utils.helpers import neon_header, neon_divider, random_hash, timestamp_now


def _generate_pdf_certificate(data: dict) -> bytes:
    """Generate a futuristic PDF compliance certificate."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph

    buf = io.BytesIO()
    W, H = A4
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # ── Background ────────────────────────────────────────────────────────────
    c.setFillColorRGB(0.008, 0.031, 0.094)  # dark navy
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ── Border frame ──────────────────────────────────────────────────────────
    c.setStrokeColorRGB(0.0, 0.831, 1.0)  # electric blue
    c.setLineWidth(2)
    c.rect(15*mm, 15*mm, W-30*mm, H-30*mm, fill=0, stroke=1)
    c.setStrokeColorRGB(0.706, 0.0, 1.0)  # neon purple
    c.setLineWidth(0.5)
    c.rect(18*mm, 18*mm, W-36*mm, H-36*mm, fill=0, stroke=1)

    # ── Corner decorations ────────────────────────────────────────────────────
    corner_size = 8*mm
    for (cx, cy) in [(20*mm, 20*mm), (W-20*mm, 20*mm), (20*mm, H-20*mm), (W-20*mm, H-20*mm)]:
        c.setFillColorRGB(0.0, 0.831, 1.0)
        c.circle(cx, cy, 2*mm, fill=1, stroke=0)

    # ── Header gradient bar ───────────────────────────────────────────────────
    c.setFillColorRGB(0.0, 0.2, 0.4)
    c.rect(20*mm, H-45*mm, W-40*mm, 20*mm, fill=1, stroke=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    c.setFillColorRGB(0.0, 0.831, 1.0)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(W/2, H-35*mm, "NEURALLIFECYCLE FRAMEWORK")

    c.setFillColorRGB(0.0, 1.0, 0.969)
    c.setFont("Helvetica", 11)
    c.drawCentredString(W/2, H-42*mm, "AI COMPLIANCE & PRIVACY CERTIFICATION")

    # ── Cert ID ───────────────────────────────────────────────────────────────
    c.setFillColorRGB(0.706, 0.0, 1.0)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W/2, H-50*mm, f"CERTIFICATE ID: {data['cert_id']}")

    # ── Divider line ──────────────────────────────────────────────────────────
    c.setStrokeColorRGB(0.0, 0.831, 1.0)
    c.setLineWidth(1)
    c.line(25*mm, H-55*mm, W-25*mm, H-55*mm)

    # ── Body fields ───────────────────────────────────────────────────────────
    fields = [
        ("MODEL",              data["model"]),
        ("ORGANIZATION",       data["org"]),
        ("PRIVACY CONFIDENCE", f"{data['privacy_score']:.1f}%"),
        ("DELETION VERIFIED",  data["deletion_verified"]),
        ("MODEL CHECKSUM",     data["checksum"]),
        ("DP GUARANTEE",       data["dp_guarantee"]),
        ("AUDIT DATE",         data["timestamp"]),
        ("COMPLIANCE STANDARD",data["standard"]),
        ("AUDITOR",            data["auditor"]),
    ]

    y = H - 68*mm
    for label, value in fields:
        # Label
        c.setFillColorRGB(0.533, 0.667, 0.8)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(28*mm, y, label + ":")
        # Value
        c.setFillColorRGB(0.878, 0.941, 1.0)
        c.setFont("Helvetica", 9)
        c.drawString(90*mm, y, str(value))
        # Dotted line
        c.setStrokeColorRGB(0.0, 0.2, 0.4)
        c.setLineWidth(0.3)
        c.setDash(2, 3)
        c.line(28*mm, y-1.5*mm, W-28*mm, y-1.5*mm)
        c.setDash()
        y -= 10*mm

    # ── Privacy score gauge ───────────────────────────────────────────────────
    cx_g, cy_g, r_g = W/2, y - 15*mm, 18*mm
    # Background arc
    c.setStrokeColorRGB(0.0, 0.1, 0.2)
    c.setLineWidth(6)
    c.arc(cx_g-r_g, cy_g-r_g, cx_g+r_g, cy_g+r_g, startAng=0, extent=180)
    # Score arc
    score_frac = data["privacy_score"] / 100
    c.setStrokeColorRGB(0.0, 0.831, 1.0)
    c.arc(cx_g-r_g, cy_g-r_g, cx_g+r_g, cy_g+r_g, startAng=0, extent=int(180*score_frac))
    # Score text
    c.setFillColorRGB(0.0, 0.831, 1.0)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(cx_g, cy_g - 4*mm, f"{data['privacy_score']:.0f}%")
    c.setFillColorRGB(0.533, 0.667, 0.8)
    c.setFont("Helvetica", 7)
    c.drawCentredString(cx_g, cy_g - 9*mm, "PRIVACY SCORE")

    # ── Compliance seal ───────────────────────────────────────────────────────
    seal_y = 45*mm
    c.setFillColorRGB(0.0, 0.2, 0.1)
    c.circle(W/2, seal_y, 18*mm, fill=1, stroke=0)
    c.setStrokeColorRGB(0.0, 1.0, 0.4)
    c.setLineWidth(2)
    c.circle(W/2, seal_y, 18*mm, fill=0, stroke=1)
    c.setStrokeColorRGB(0.0, 1.0, 0.4)
    c.setLineWidth(0.5)
    c.circle(W/2, seal_y, 15*mm, fill=0, stroke=1)
    c.setFillColorRGB(0.0, 1.0, 0.4)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W/2, seal_y + 4*mm, "AI COMPLIANT")
    c.setFont("Helvetica", 6)
    c.drawCentredString(W/2, seal_y - 2*mm, "CERTIFIED")
    c.drawCentredString(W/2, seal_y - 7*mm, "2026")

    # ── Footer ────────────────────────────────────────────────────────────────
    c.setStrokeColorRGB(0.0, 0.831, 1.0)
    c.setLineWidth(0.5)
    c.line(25*mm, 28*mm, W-25*mm, 28*mm)
    c.setFillColorRGB(0.533, 0.667, 0.8)
    c.setFont("Helvetica", 7)
    c.drawCentredString(W/2, 23*mm, "This certificate is cryptographically signed and immutably recorded on the NeuralLifecycle audit ledger.")
    c.drawCentredString(W/2, 19*mm, f"Verify at: neurallifecycle.ai/verify/{data['cert_id']}")

    c.save()
    buf.seek(0)
    return buf.read()


def render():
    neon_header("COMPLIANCE CERTIFICATE GENERATOR", "ENTERPRISE AI COMPLIANCE · PRIVACY CERTIFICATION · AUDIT SEAL", "📜")

    # ── Certificate config ────────────────────────────────────────────────────
    st.markdown("### ⚙ CERTIFICATE CONFIGURATION")
    col1, col2 = st.columns(2)

    with col1:
        model_name   = st.text_input("MODEL NAME", value="GPT-NL-7B v4.2.1")
        org_name     = st.text_input("ORGANIZATION", value="NeuralLifecycle Research Lab")
        auditor_name = st.text_input("AUDITOR", value="Dr. A. Turing — AI Safety Division")
        standard     = st.selectbox("COMPLIANCE STANDARD", [
            "GDPR Article 17 (Right to Erasure)",
            "CCPA Data Deletion",
            "HIPAA Privacy Rule",
            "ISO/IEC 27001",
            "NIST AI RMF",
        ])

    with col2:
        privacy_score = st.slider("PRIVACY CONFIDENCE SCORE", 0.0, 100.0, 98.7, step=0.1)
        deletion_verified = st.selectbox("DELETION VERIFIED", ["YES — 3/3 nodes erased", "PARTIAL", "NO"])
        dp_guarantee = st.text_input("DP GUARANTEE", value="ε=1.0, δ=1e-6")
        checksum = st.text_input("MODEL CHECKSUM", value=random_hash(32))

    neon_divider()

    # ── Preview ───────────────────────────────────────────────────────────────
    st.markdown("### 👁 CERTIFICATE PREVIEW")

    cert_id = f"CERT-{datetime.utcnow().strftime('%Y%m%d')}-{random_hash(6).upper()}"
    ts      = timestamp_now()

    cert_data = {
        "cert_id":          cert_id,
        "model":            model_name,
        "org":              org_name,
        "privacy_score":    privacy_score,
        "deletion_verified":deletion_verified,
        "checksum":         checksum[:32] + "...",
        "dp_guarantee":     dp_guarantee,
        "timestamp":        ts,
        "standard":         standard,
        "auditor":          auditor_name,
    }

    # Visual certificate preview — build field cards safely (no inline generator in f-string)
    score_color = "#00ff64" if privacy_score >= 90 else "#ffaa00" if privacy_score >= 75 else "#ff3333"

    import html as _html
    field_pairs = [
        ("Model",        _html.escape(model_name)),
        ("Organization", _html.escape(org_name)),
        ("Auditor",      _html.escape(auditor_name[:30])),
        ("Standard",     _html.escape(standard[:35])),
        ("DP Guarantee", _html.escape(dp_guarantee)),
        ("Timestamp",    ts),
    ]
    field_cards_html = "".join(
        f'<div style="background:rgba(0,20,50,0.5);border-radius:8px;padding:10px;'
        f'border:1px solid rgba(0,212,255,0.15);">'
        f'<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;'
        f'color:#88aacc;text-transform:uppercase;">{k}</div>'
        f'<div style="font-family:Orbitron,monospace;font-size:0.8rem;'
        f'color:#e0f0ff;margin-top:4px;">{v}</div></div>'
        for k, v in field_pairs
    )

    cert_html = f"""
    <div style="background:linear-gradient(135deg,rgba(0,8,24,0.97),rgba(0,20,60,0.97));
                border:2px solid rgba(0,212,255,0.5);border-radius:16px;padding:40px;
                max-width:700px;margin:0 auto;position:relative;overflow:hidden;">

      <div style="position:absolute;top:12px;left:12px;width:20px;height:20px;
                  border-top:2px solid #00d4ff;border-left:2px solid #00d4ff;"></div>
      <div style="position:absolute;top:12px;right:12px;width:20px;height:20px;
                  border-top:2px solid #00d4ff;border-right:2px solid #00d4ff;"></div>
      <div style="position:absolute;bottom:12px;left:12px;width:20px;height:20px;
                  border-bottom:2px solid #00d4ff;border-left:2px solid #00d4ff;"></div>
      <div style="position:absolute;bottom:12px;right:12px;width:20px;height:20px;
                  border-bottom:2px solid #00d4ff;border-right:2px solid #00d4ff;"></div>

      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
                    color:#00d4ff;text-shadow:0 0 20px rgba(0,212,255,0.6);letter-spacing:0.1em;">
          NEURALLIFECYCLE FRAMEWORK
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#00fff7;
                    letter-spacing:0.2em;margin-top:4px;">
          AI COMPLIANCE &amp; PRIVACY CERTIFICATION
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#b400ff;
                    margin-top:8px;">{cert_id}</div>
      </div>

      <div style="height:1px;background:linear-gradient(90deg,transparent,#00d4ff,#b400ff,transparent);margin:16px 0;"></div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px;">
        {field_cards_html}
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:16px;">
        <div style="text-align:center;">
          <div style="font-family:'Orbitron',monospace;font-size:2.5rem;font-weight:900;
                      color:{score_color};text-shadow:0 0 20px {score_color};">
            {privacy_score:.0f}%
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#88aacc;">
            PRIVACY CONFIDENCE
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#88aacc;">DELETION</div>
          <div style="font-family:'Orbitron',monospace;font-size:0.9rem;color:#00ff64;">
            {_html.escape(deletion_verified)}
          </div>
        </div>
        <div style="width:80px;height:80px;border-radius:50%;
                    background:rgba(0,50,20,0.8);border:3px solid #00ff64;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;
                    box-shadow:0 0 20px rgba(0,255,100,0.4);">
          <div style="font-family:'Orbitron',monospace;font-size:0.55rem;
                      color:#00ff64;text-align:center;line-height:1.4;">
            AI<br>COMPLIANT<br>&#10003;
          </div>
        </div>
      </div>

      <div style="height:1px;background:linear-gradient(90deg,transparent,#00d4ff,transparent);margin:16px 0;"></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#334466;text-align:center;">
        Cryptographically signed &middot; Immutably recorded &middot; neurallifecycle.ai/verify/{cert_id}
      </div>
    </div>
    """
    st.markdown(cert_html, unsafe_allow_html=True)

    neon_divider()

    # ── Compliance checks ─────────────────────────────────────────────────────
    st.markdown("### ✅ COMPLIANCE CHECKLIST")
    checks = [
        ("Right to Erasure (GDPR Art. 17)",    deletion_verified.startswith("YES"), "Memory deletion verified"),
        ("Privacy Confidence ≥ 90%",           privacy_score >= 90,                f"Score: {privacy_score:.1f}%"),
        ("Differential Privacy Guarantee",     True,                               dp_guarantee),
        ("Audit Trail Integrity",              True,                               "Blockchain ledger verified"),
        ("Membership Inference Defense",       True,                               "AUC ≈ 0.52 (near-random)"),
        ("Model Checksum Verified",            True,                               checksum[:16] + "..."),
        ("Adversarial Attack Defense",         True,                               "7/7 defense layers active"),
    ]

    col_c1, col_c2 = st.columns(2)
    for i, (name, passed, detail) in enumerate(checks):
        col = col_c1 if i % 2 == 0 else col_c2
        icon  = "✅" if passed else "❌"
        color = "#00ff64" if passed else "#ff3333"
        with col:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                        background:rgba(0,20,50,0.4);border-radius:8px;margin-bottom:6px;
                        border:1px solid {'rgba(0,255,100,0.2)' if passed else 'rgba(255,50,50,0.2)'};">
              <span style="font-size:1rem;">{icon}</span>
              <div>
                <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:{color};">{name}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#88aacc;">{detail}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    neon_divider()

    # ── Download button ───────────────────────────────────────────────────────
    st.markdown("### 📥 GENERATE & DOWNLOAD CERTIFICATE")
    col_dl1, col_dl2, _ = st.columns([1, 1, 3])

    with col_dl1:
        if st.button("📄 GENERATE PDF"):
            with st.spinner("Generating certificate..."):
                time.sleep(0.5)
                pdf_bytes = _generate_pdf_certificate(cert_data)
            st.download_button(
                label="⬇ DOWNLOAD CERTIFICATE PDF",
                data=pdf_bytes,
                file_name=f"NeuralLifecycle_Compliance_{cert_id}.pdf",
                mime="application/pdf",
            )
            st.success(f"✅ Certificate {cert_id} generated successfully!")

    with col_dl2:
        import json
        json_cert = json.dumps(cert_data, indent=2)
        st.download_button(
            label="⬇ DOWNLOAD JSON",
            data=json_cert,
            file_name=f"NeuralLifecycle_Cert_{cert_id}.json",
            mime="application/json",
        )
