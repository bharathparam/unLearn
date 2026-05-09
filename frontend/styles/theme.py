"""
NeuralLifecycle Framework - Global CSS Theme
Cyberpunk / Glassmorphism / Dark AI Lab aesthetic
"""

GLOBAL_CSS = """
<style>
/* ─── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ─── Root Variables ────────────────────────────────────────────────────────── */
:root {
  --electric-blue:  #00d4ff;
  --neon-purple:    #b400ff;
  --cyan:           #00fff7;
  --magenta:        #ff00aa;
  --dark-navy:      #020818;
  --card-bg:        rgba(0, 20, 50, 0.55);
  --glass-border:   rgba(0, 212, 255, 0.25);
  --glow-blue:      0 0 20px rgba(0,212,255,0.6), 0 0 40px rgba(0,212,255,0.3);
  --glow-purple:    0 0 20px rgba(180,0,255,0.6), 0 0 40px rgba(180,0,255,0.3);
  --glow-cyan:      0 0 20px rgba(0,255,247,0.6), 0 0 40px rgba(0,255,247,0.3);
  --glow-magenta:   0 0 20px rgba(255,0,170,0.6), 0 0 40px rgba(255,0,170,0.3);
}

/* ─── Global Reset & Background ────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--dark-navy) !important;
  color: #e0f0ff !important;
  font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,255,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(180,0,255,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 50% 60% at 50% 50%, rgba(0,255,247,0.04) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

/* ─── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(2,8,24,0.97) 0%, rgba(5,15,40,0.97) 100%) !important;
  border-right: 1px solid var(--glass-border) !important;
  box-shadow: 4px 0 30px rgba(0,212,255,0.1) !important;
}

[data-testid="stSidebar"] * {
  font-family: 'Rajdhani', sans-serif !important;
}

/* ─── Sidebar Radio Buttons ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio label {
  color: #88aacc !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.05em;
  transition: all 0.3s ease;
  padding: 4px 8px;
  border-radius: 4px;
}
[data-testid="stSidebar"] .stRadio label:hover {
  color: var(--electric-blue) !important;
  text-shadow: var(--glow-blue);
}

/* ─── Main Content Area ─────────────────────────────────────────────────────── */
[data-testid="stMain"] {
  background: transparent !important;
}

.main .block-container {
  padding-top: 1rem !important;
  max-width: 1400px !important;
}

/* ─── Headings ──────────────────────────────────────────────────────────────── */
h1, h2, h3 {
  font-family: 'Orbitron', monospace !important;
  letter-spacing: 0.08em !important;
}

h1 { color: var(--electric-blue) !important; text-shadow: var(--glow-blue); }
h2 { color: var(--cyan) !important; text-shadow: var(--glow-cyan); }
h3 { color: var(--neon-purple) !important; text-shadow: var(--glow-purple); }

/* ─── Metric Cards ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--card-bg) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 4px 24px rgba(0,212,255,0.1), inset 0 1px 0 rgba(255,255,255,0.05) !important;
  transition: all 0.3s ease !important;
}
[data-testid="stMetric"]:hover {
  border-color: var(--electric-blue) !important;
  box-shadow: var(--glow-blue) !important;
  transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] { color: #88aacc !important; font-size: 0.75rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--electric-blue) !important; font-family: 'Orbitron', monospace !important; font-size: 1.6rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ─── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(180,0,255,0.15)) !important;
  border: 1px solid var(--electric-blue) !important;
  color: var(--electric-blue) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
  transition: all 0.3s ease !important;
  backdrop-filter: blur(8px) !important;
  text-transform: uppercase !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.35), rgba(180,0,255,0.35)) !important;
  box-shadow: var(--glow-blue) !important;
  transform: translateY(-2px) !important;
  border-color: var(--cyan) !important;
}

/* ─── Inputs & Selects ──────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
  background: rgba(0,20,50,0.7) !important;
  border: 1px solid var(--glass-border) !important;
  color: #e0f0ff !important;
  border-radius: 8px !important;
  font-family: 'Share Tech Mono', monospace !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--electric-blue) !important;
  box-shadow: var(--glow-blue) !important;
}

/* ─── Sliders ───────────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--electric-blue) !important;
  box-shadow: var(--glow-blue) !important;
}

/* ─── Progress Bars ─────────────────────────────────────────────────────────── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--electric-blue), var(--neon-purple)) !important;
  box-shadow: var(--glow-blue) !important;
}

/* ─── Tabs ──────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(0,20,50,0.5) !important;
  border-radius: 10px !important;
  border: 1px solid var(--glass-border) !important;
  gap: 4px !important;
  padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  color: #88aacc !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  border-radius: 6px !important;
  transition: all 0.3s ease !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(180,0,255,0.2)) !important;
  color: var(--electric-blue) !important;
  text-shadow: var(--glow-blue) !important;
  border: 1px solid var(--glass-border) !important;
}

/* ─── Expanders ─────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--card-bg) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 8px !important;
  color: var(--electric-blue) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.75rem !important;
}

/* ─── Dividers ──────────────────────────────────────────────────────────────── */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--electric-blue), var(--neon-purple), transparent) !important;
  margin: 1.5rem 0 !important;
}

/* ─── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,20,50,0.3); }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--electric-blue), var(--neon-purple)); border-radius: 3px; }

/* ─── Plotly Charts ─────────────────────────────────────────────────────────── */
.js-plotly-plot .plotly .modebar {
  background: rgba(0,20,50,0.8) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 6px !important;
}

/* ─── Dataframes ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--glass-border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

/* ─── Animations ────────────────────────────────────────────────────────────── */
@keyframes pulse-glow {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
@keyframes scan-line {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
}
@keyframes rotate-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
@keyframes gradient-shift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes flicker {
  0%, 95%, 100% { opacity: 1; }
  96% { opacity: 0.4; }
  97% { opacity: 1; }
  98% { opacity: 0.6; }
}

/* ─── Glass Card Component ──────────────────────────────────────────────────── */
.glass-card {
  background: var(--card-bg);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 24px;
  backdrop-filter: blur(16px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}
.glass-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--electric-blue), transparent);
}
.glass-card:hover {
  border-color: rgba(0,212,255,0.5);
  box-shadow: var(--glow-blue), 0 8px 32px rgba(0,0,0,0.4);
  transform: translateY(-3px);
}

/* ─── Status Badge ──────────────────────────────────────────────────────────── */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 20px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}
.status-online {
  background: rgba(0,255,100,0.15);
  border: 1px solid rgba(0,255,100,0.4);
  color: #00ff64;
}
.status-warning {
  background: rgba(255,200,0,0.15);
  border: 1px solid rgba(255,200,0,0.4);
  color: #ffc800;
}
.status-critical {
  background: rgba(255,50,50,0.15);
  border: 1px solid rgba(255,50,50,0.4);
  color: #ff3232;
}

/* ─── Hero Section ──────────────────────────────────────────────────────────── */
.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(1.8rem, 4vw, 3.2rem);
  font-weight: 900;
  background: linear-gradient(135deg, var(--electric-blue), var(--neon-purple), var(--cyan));
  background-size: 200% 200%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradient-shift 4s ease infinite;
  letter-spacing: 0.1em;
  line-height: 1.2;
}
.hero-subtitle {
  font-family: 'Share Tech Mono', monospace;
  color: #88aacc;
  font-size: 0.9rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

/* ─── Terminal Log ──────────────────────────────────────────────────────────── */
.terminal-log {
  background: rgba(0,5,15,0.9);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 8px;
  padding: 16px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem;
  color: #00ff64;
  max-height: 200px;
  overflow-y: auto;
  line-height: 1.6;
}
.terminal-log .log-error { color: #ff4444; }
.terminal-log .log-warn  { color: #ffaa00; }
.terminal-log .log-info  { color: #00d4ff; }
.terminal-log .log-success { color: #00ff64; }

/* ─── Neon Divider ──────────────────────────────────────────────────────────── */
.neon-divider {
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--electric-blue) 30%, var(--neon-purple) 70%, transparent);
  border-radius: 2px;
  margin: 1.5rem 0;
  box-shadow: 0 0 10px rgba(0,212,255,0.5);
}

/* ─── Sidebar Logo ──────────────────────────────────────────────────────────── */
.sidebar-logo {
  font-family: 'Orbitron', monospace;
  font-size: 1.1rem;
  font-weight: 900;
  background: linear-gradient(135deg, var(--electric-blue), var(--neon-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 0.05em;
  text-align: center;
  padding: 8px 0;
}

/* ─── Hide Streamlit Branding ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
"""

PARTICLE_BG_JS = """
<canvas id="neural-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;pointer-events:none;opacity:0.35;"></canvas>
<script>
(function(){
  const canvas = document.getElementById('neural-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let W = canvas.width  = window.innerWidth;
  let H = canvas.height = window.innerHeight;
  window.addEventListener('resize', () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; });

  const N = 80;
  const nodes = Array.from({length: N}, () => ({
    x: Math.random()*W, y: Math.random()*H,
    vx: (Math.random()-0.5)*0.4, vy: (Math.random()-0.5)*0.4,
    r: Math.random()*2+1,
    hue: Math.random() > 0.5 ? 195 : 280,
    pulse: Math.random()*Math.PI*2
  }));

  function draw(){
    ctx.clearRect(0,0,W,H);
    const t = Date.now()*0.001;

    // Draw connections
    for(let i=0;i<N;i++){
      for(let j=i+1;j<N;j++){
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx*dx+dy*dy);
        if(dist < 140){
          const alpha = (1 - dist/140)*0.4;
          const grad = ctx.createLinearGradient(nodes[i].x,nodes[i].y,nodes[j].x,nodes[j].y);
          grad.addColorStop(0, `hsla(${nodes[i].hue},100%,60%,${alpha})`);
          grad.addColorStop(1, `hsla(${nodes[j].hue},100%,60%,${alpha})`);
          ctx.beginPath();
          ctx.strokeStyle = grad;
          ctx.lineWidth = 0.6;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    nodes.forEach(n => {
      n.pulse += 0.03;
      const glow = Math.sin(n.pulse)*0.4+0.6;
      const grad = ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r*4);
      grad.addColorStop(0, `hsla(${n.hue},100%,70%,${glow})`);
      grad.addColorStop(1, `hsla(${n.hue},100%,70%,0)`);
      ctx.beginPath();
      ctx.fillStyle = grad;
      ctx.arc(n.x, n.y, n.r*4, 0, Math.PI*2);
      ctx.fill();

      ctx.beginPath();
      ctx.fillStyle = `hsla(${n.hue},100%,80%,${glow})`;
      ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
      ctx.fill();

      n.x += n.vx; n.y += n.vy;
      if(n.x<0||n.x>W) n.vx*=-1;
      if(n.y<0||n.y>H) n.vy*=-1;
    });

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""
