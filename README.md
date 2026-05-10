# ⬡ NeuralLifecycle Framework

<div align="center">

![NeuralLifecycle](https://img.shields.io/badge/NeuralLifecycle-v4.2.1-00d4ff?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-b400ff?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00ff64?style=for-the-badge)

**An advanced AI research operating system for neural memory editing, unlearning verification, and privacy auditing.**

</div>

---

## 🧠 What is NeuralLifecycle?

NeuralLifecycle is a cinematic, hackathon-grade dashboard built on **Streamlit** that provides a complete lifecycle management system for large language models. It enables researchers to:

- **Edit AI memories** in real-time using the **ROME** (Rank-One Model Editing) algorithm
- **Verify forgetting** with a full **Membership Inference Attack (MIA)** suite
- **Audit privacy compliance** and generate formal **Neural Privacy Compliance Certificates**

---

## ✨ Features

### 🧠 Neural Memory Mapping (ROME Editor)
- **Live 3D AI Brain Visualization** — Volumetric neural web with animated synaptic pathways
- **Causal Tracing** — Forward-pass propagation wave animation
- **Brain Surgery Mode** — Cinematic extraction and implantation animation during ROME edits
- **Memory Editing** — Inject new facts directly into model weights (MLP matrices)
- **Memory Deletion** — Sever synaptic pathways to erase specific knowledge
- **Before/After Verification** — Automatically re-queries the model to confirm the edit

### 🛡️ Integrated MIA Defense Scan (Post-Op Audit)
- **Auto-Audit Toggle** — Automatically triggers an MIA scan after every ROME edit
- **Privacy Score** — Real-time confidence score for how well a memory was forgotten
- **Neural Privacy Compliance Certificate** — Formal, audit-grade certificate with:
  - Unique Serial Number
  - Privacy Confidence Score
  - Compliance Verdict (✓ CERTIFIED / ⚠ MONITORING)
  - Auditor Narrative Summary (Gemini-powered)
- **Leakage Probability** — Measures semantic information leakage post-edit
- **Attack Success Rate** — Reports how many of 24 adversarial prompts succeeded

### 🔬 Neural Verification Engine
- **Multi-metric leakage scoring** — String similarity, token overlap, N-gram match, semantic proximity
- **24-prompt adversarial attack suite** — Categories: Direct Probe, Roleplay, Semantic Entrapment
- **Forgetting Delta** — Quantifies the change in leakage before and after unlearning
- **Risk Classification** — NEGLIGIBLE / LOW / MEDIUM / HIGH / CRITICAL
- **Audit Report Archive** — Persistent compliance report storage

---

## 🏗️ Architecture

```
NeuralAI/
└── frontend/               # Streamlit Dashboard
    ├── app.py              # Main app entry point, navigation, sidebar
    ├── run.bat             # Windows launcher
    ├── pages/
    │   ├── neural_memory.py        # ROME editor + MIA integration (Page 1)
    │   └── verification_engine.py  # Standalone verification lab (Page 2)
    ├── styles/
    │   └── theme.py        # Global CSS (glassmorphism, cyberpunk palette)
    └── utils/
        └── helpers.py      # Shared UI components (neon_header, neon_divider, etc.)
```

---

## 🔌 Backend APIs

The frontend connects to two separate backend services:

| Service | Default URL | Description |
|---|---|---|
| **ROME Backend** | `https://postlabially-overinstructive-aurore.ngrok-free.dev` | Qwen-1.5B model editing & inference |
| **MIA Engine** | `https://unsenile-subtransversally-julien.ngrok-free.dev` | Membership Inference Attack verification |

> Both URLs can be overridden from the sidebar in the dashboard.

### Key Endpoints

#### ROME Backend
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Run a forward pass and get model output |
| `POST` | `/edit` | Apply a ROME weight edit |
| `POST` | `/restore` | Restore original model weights |

#### MIA Engine
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/verify` | Run full MIA verification |
| `POST` | `/report` | Run verification + generate compliance report |
| `POST` | `/attack` | Launch 24-prompt adversarial attack suite |
| `GET` | `/attack/prompts` | List all attack prompt templates |
| `GET` | `/reports` | List all saved audit reports |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/bharathparam/unLearn.git
cd unLearn

# Install dependencies
pip install streamlit httpx plotly numpy pandas

# Navigate to the frontend
cd frontend
```

### Running the Dashboard

```bash
# Option 1: Using the batch file (Windows)
run.bat

# Option 2: Directly with Python
python -m streamlit run app.py --server.port 8502
```

Then open your browser and go to: **[http://localhost:8502](http://localhost:8502)**

---

## 🎮 How to Use

### 1. Edit a Neural Memory (ROME)
1. Navigate to **🧠 Neural Memory Mapping**
2. Enter a factual prompt (e.g., `"The capital of France is"`)
3. Click **⚡ EXECUTE FORWARD PASS** — watch the 3D brain light up
4. Enter the **Subject** (`France`) and **New Target** (`Berlin`)
5. Click **⚔️ EDIT MEMORY** — the cinematic surgery animation begins
6. After the edit, the model is automatically re-queried to confirm

### 2. Run a Security Audit (MIA)
After a successful edit:
- **Auto-Audit** (enabled by default): The MIA scan runs automatically and displays a **Privacy Certificate**
- **Manual Audit**: Scroll to **🛡️ POST-OP MIA DEFENSE SCAN** and click **INITIATE SECURITY AUDIT**

### 3. Restore Original Weights
Click **🧹 RESTORE ORIGINAL WEIGHTS** in the sidebar to reset the model to its original state.

---

## 🎨 Design System

The dashboard uses a **dark cyberpunk aesthetic** with:
- **Glassmorphism** cards with neon borders
- **Orbitron** typeface for headings (sci-fi monospace)
- **Share Tech Mono** for data/terminal output
- Color palette: `#00d4ff` (cyan), `#b400ff` (purple), `#00ff64` (green), `#ff00aa` (magenta)
- Animated particle background

---

## 🌿 Branch

Active development is on the [`neuralai-frontend`](https://github.com/bharathparam/unLearn/tree/neuralai-frontend) branch.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ for AI Safety Research | NeuralLifecycle © 2026</sub>
</div>
