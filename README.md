# ⬡ NeuralLifecycle Framework (unLearn Engine)

**Ultra-modern 3D Cyberpunk AI Research Dashboard & ROME Editing Engine**

> *"A live operating system for controlling, editing, and verifying AI memories."*

---

## 🚀 Quick Start

### 1. Setup Backend (unLearn Engine)
```bash
# Create virtual environment
setup_env.bat

# Activate environment
rome_env\Scripts\activate

# Start the ROME API Server (Port 8000)
python api_server.py
```

### 2. Setup Frontend (NeuralLifecycle)
```bash
# Install dependencies
pip install streamlit plotly altair pandas numpy scipy scikit-learn reportlab pillow httpx

# Run the dashboard (Port 8501/8502)
python -m streamlit run frontend/app.py
```

Open **http://localhost:8502** in your browser.

---

## 📄 Platform Pages

| Page | Description |
|------|-------------|
| ⬡ Dashboard | Live AI system status, metrics, heatmaps, logs |
| 🧠 Neural Memory Mapping | **Full 3D transformer visualization** — travel inside the AI brain |
| 🗑 AI Unlearning Console | ROME surgical editing, memory dissolution animation |
| 🔒 Privacy Verification Lab | MIA attack simulation, ROC curves, privacy radar |
| ⚔ Adversarial Attack Simulator | Prompt injection, jailbreak detection, defense activation |
| 📜 Compliance Certificate | PDF certificate generator with privacy seal |
| 🏗 System Architecture | Sankey flow, 3D pipeline, component metrics |

---

## 🛠 Tech Stack

- **Backend (unLearn)**: Python, FastAPI, PyTorch, ROME (Rank-One Model Editing), Qwen2.5-1.5B
- **Frontend**: Streamlit, Plotly, Custom CSS/JS (Glassmorphism + Cyberpunk)
- **ML Concepts**: LoRA, ROME, Causal Tracing, Membership Inference Attacks (MIA)
- **Compliance**: PDF generation via ReportLab

---

## 🎨 Design Aesthetics

- **Glassmorphism** dark cards with neon borders
- **Cyberpunk** color palette: Electric Blue · Neon Purple · Cyan · Magenta
- **Orbitron** + **Rajdhani** + **Share Tech Mono** fonts
- **Animated particle neural network** background (HTML5 Canvas)
- **Plotly 3D** interactive visualizations throughout

---

## 📁 Repository Structure

- `frontend/`: Streamlit dashboard source code
- `api_server.py`: Production ROME + MIA API Server
- `rome_core.py`: Core ROME algorithm implementation
- `model_loader.py`: Optimized model loading for 8GB VRAM
- `hparams.py`: Hyperparameters for model editing

---

## 📜 License

MIT License - AI Safety Research Platform © 2026
