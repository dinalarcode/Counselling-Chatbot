# 📖 Chatbot Konseling Berbasis Alkitab

A Bible-based Christian counseling chatbot that combines **multi-label intent classification** (LABAN/IndoBERT), **RAG-powered Bible verse retrieval** (FAISS + LLM reranker), and **LLM-driven empathetic responses** to guide users through a structured counseling session — all in **Bahasa Indonesia**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation \& Setup](#installation--setup)
- [How to Run](#how-to-run)
- [API Endpoints / Usage](#api-endpoints--usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [License](#license)

---

## Overview

### Problem

Many Indonesian-speaking individuals seek emotional and spiritual support but lack access to professional Christian counselors. Existing chatbots often provide generic responses without contextual Bible verse grounding or structured counseling flow.

### Solution

This chatbot implements a **6-stage counseling state machine** (`pembukaan → pembahasan → intervensi → solusi → relaksasi → penutupan`) with an emergency branch (`bantuan_profesional`) for crisis detection. Each stage is powered by:

1. **LABAN Classifier** — A fine-tuned IndoBERT multi-label model that detects 10 emotional intents (e.g., *Perasaan Sedih dan Kehilangan*, *Perasaan Takut dan Kecemasan*).
2. **FAISS Vector Search + LLM Reranker** — Retrieves contextually relevant Bible verses from the full Terjemahan Baru (TB) using Augmented Vector Indexing (AVI) with chapter-level theological summaries.
3. **LLM Counselor** — Generates empathetic, stage-appropriate responses via Groq (Llama 3.3 70B), Google Gemini, or OpenAI, orchestrated through LangChain.

### Key Features

- 🧠 **Multi-label intent classification** with 10 Indonesian emotional intents
- 📜 **31,000+ Bible verses** indexed with theological context enrichment (AVI)
- 🔄 **Automatic session progression** with transition signal detection
- 🚨 **Crisis detection** via keyword safety net + classifier for suicidal ideation
- 📊 **Session-level verse diversity** — book and exact-verse exclusion prevents repetition
- 🌐 **Multi-provider LLM** — switch between Groq, Gemini, OpenAI, or Ollama in one config
- 💻 **Flask web interface** with a clean chat UI

---

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────────────────┐
│  SessionManager (State Machine)                  │
│  ┌──────────────────────────────────────────────┐ │
│  │  Stage: pembukaan → pembahasan → intervensi  │ │
│  │         → solusi → relaksasi → penutupan     │ │
│  │  Branch: bantuan_profesional (emergency)     │ │
│  └──────────────────────────────────────────────┘ │
│                      │                            │
│                      ▼                            │
│  ┌──────────────────────────────────────────────┐ │
│  │  RAGEngine                                   │ │
│  │  ├─ LABAN Classifier (IndoBERT)              │ │
│  │  ├─ QnA Knowledge Base (FAISS)               │ │
│  │  ├─ Bible Verse Retrieval                    │ │
│  │  │   ├─ Layer 1: Query Expansion (synonyms)  │ │
│  │  │   ├─ Layer 2: FAISS Semantic Search       │ │
│  │  │   ├─ Layer 2.5: Diversity Filter          │ │
│  │  │   └─ Layer 3: LLM Reranker               │ │
│  │  └─ LLM Response Generation (LangChain)      │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
    │
    ▼
Flask Web UI (JSON API)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.10 – 3.13 | Tested on 3.13 |
| **CUDA** (optional) | 12.8+ | GPU acceleration for IndoBERT inference |
| **Git** | any | For cloning the repository |
| **pip** | latest | Python package manager |

### Required API Keys (at least one LLM provider)

| Provider | Environment Variable | Sign-up |
|---|---|---|
| **Groq** (default) | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| **Google Gemini** | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| **OpenAI** | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |

> **Note:** Only the API key for your chosen provider (`CHATBOT_LLM_PROVIDER` in `config.py`) is required. Groq is the default and offers a generous free tier.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dinalarcode/chatbot_konseling_berbasis_Alkitab.git
cd chatbot_konseling_berbasis_Alkitab
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv cbenv
cbenv\Scripts\activate

# Linux / macOS
python3 -m venv cbenv
source cbenv/bin/activate
```

### 3. Install PyTorch (with CUDA support)

Install PyTorch **first** to ensure CUDA compatibility:

```bash
# GPU (CUDA 12.8) — Recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Python Dependencies

```bash
pip install -r requirement.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> Only the key for your active provider is required. See step 6 to choose a provider.

### 6. Configure the LLM Provider (Optional)

Edit `config.py` to switch between LLM providers:

```python
class config:
    # Change this to "groq", "gemini", "openai", or "ollama"
    CHATBOT_LLM_PROVIDER = "groq"

    # Each provider has its own model setting:
    GROQ_CHATBOT_MODEL    = "llama-3.3-70b-versatile"
    GEMINI_CHATBOT_MODEL  = "gemini-2.5-flash"
    OPENAI_CHATBOT_MODEL  = "gpt-5.4-mini"
```

### 7. Verify Required Data Files

Ensure these files/directories exist before running:

| Path | Description | How to Obtain |
|---|---|---|
| `checkpoint/IndoBERT_multi_label_zsl.pt` | Trained LABAN classifier weights | Included in repo or train via `models/multilabel/run_trainer.py` |
| `data/faiss_bible_index/` | Pre-built FAISS Bible index | Auto-built on first run from the enriched CSV |
| `data/faiss_qna_index/` | Pre-built FAISS QnA index | Auto-built on first run from `data/dataset_qna.csv` |
| `data/verse_retrieval/alkitab_tb_enriched_groq.csv` | AVI-enriched Bible verses | Generate via the AVI pipeline (see below) |

#### Building the AVI (Augmented Vector Indexing) Pipeline (if needed)

If `data/verse_retrieval/alkitab_tb_enriched_groq.csv` does not exist:

```bash
# Step 1: Generate chapter-level theological summaries via LLM
python data/verse_retrieval/generate_chapter_summaries.py

# Step 2: Prepend summaries to each verse to create the enriched CSV
python data/verse_retrieval/prepare_enriched_bible.py
```

> **Note:** The FAISS index is built automatically on first startup if `data/faiss_bible_index/` doesn't exist. This takes ~15–45 minutes (one-time cost).

---

## How to Run

### Start the Development Server

```bash
python app.py
```

The Flask server starts at:

```
http://127.0.0.1:5000
```

Open this URL in your browser to access the chat interface.

### Console Output

The terminal will show initialization progress and real-time debug info:

```
Initializing RAGEngine and SessionManager...
Memuat Bible FAISS index dari disk...
Bible index dimuat. Total dokumen: 31103
Memuat QnA FAISS index dari disk...
QnA index dimuat. Total dokumen: 536
Initialization complete.
 * Running on http://127.0.0.1:5000
```

During conversations, timing and debug data are printed to the console:

```
[TIMING] classifier.predict: 12.3ms
[TIMING] search_qna: 5.1ms
[TIMING] retrieve_verse_with_llm: 1423.5ms
[TIMING] chain.invoke (LLM): 2105.8ms

--- Technical Debug Info ---
User Input: Saya merasa sedih dan kehilangan arah
Stage: pembahasan -> intervensi
Intents: ['Menyatakan Perasaan Sedih dan Kehilangan']
---
```

---

## API Endpoints / Usage

The application exposes a REST API consumed by the built-in web UI. All endpoints accept and return JSON.

### `GET /`

Renders the main chat interface (HTML page).

```bash
curl http://127.0.0.1:5000/
```

---

### `POST /chat`

Send a user message and receive a counseling response.

**Request:**

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Saya merasa sangat sedih dan kehilangan arah hidup"}'
```

**Response:**

```json
{
  "response": "Saya bisa merasakan betapa beratnya perasaan yang kamu alami saat ini...",
  "show_professional_button": false
}
```

| Field | Type | Description |
|---|---|---|
| `response` | `string` | The chatbot's counseling response |
| `show_professional_button` | `boolean` | `true` when the session enters `bantuan_profesional` stage (crisis detected) |

---

### `POST /reset`

Reset the counseling session to the initial state.

```bash
curl -X POST http://127.0.0.1:5000/reset
```

**Response:**

```json
{
  "status": "success",
  "message": "Session reset."
}
```

---

### `POST /shutdown`

Shut down the Flask server.

```bash
curl -X POST http://127.0.0.1:5000/shutdown
```

---

### Special Commands (via Chat)

These commands are sent as regular chat messages:

| Command | Description |
|---|---|
| `bible test` | Activate Bible Test Mode — bypasses LLM counselor, returns raw FAISS + LLM reranker verse results for evaluation |
| `exit test` | Deactivate Bible Test Mode and return to normal counseling |

---

## Project Structure

```
The_Chatbot/
├── app.py                          # Flask application entry point
├── config.py                       # Centralized configuration (LLM, model, hyperparams)
├── requirement.txt                 # Python dependencies
├── .env                            # API keys (not committed)
│
├── core/                           # Core engine modules
│   ├── rag_engine.py               # RAG pipeline: classifier → FAISS → LLM response
│   ├── session_manager.py          # 6-stage counseling state machine
│   ├── vector_db.py                # FAISS index management, LLM reranker, diversity filter
│   └── knowledge_base.py           # Legacy knowledge base (template-based)
│
├── models/
│   └── multilabel/
│       ├── bert_model.py           # IndoBERT architecture (LABAN classifier)
│       ├── run_trainer.py          # Training script for multi-label classifier
│       └── predict.py              # Inference wrapper for intent classification
│
├── data/
│   ├── alkitab_tb.csv              # Raw Bible TB data (31,000+ verses)
│   ├── dataset_qna.csv             # QnA knowledge base for example answers
│   ├── dataset_multiintent.csv     # Multi-label intent training data
│   ├── intent_content.csv          # Intent definitions and content
│   ├── intent_content_augmented.csv# Augmented intent data
│   ├── data_preparation.py         # Data loading and tokenization
│   ├── scrape_alkitab.py           # Bible scraping script (alkitab.mobi)
│   ├── faiss_bible_index/          # Pre-built FAISS index for Bible verses
│   ├── faiss_qna_index/            # Pre-built FAISS index for QnA
│   ├── verse_retrieval/            # AVI pipeline
│   │   ├── generate_chapter_summaries.py  # LLM-based chapter summarization
│   │   ├── prepare_enriched_bible.py      # Enriched verse preparation
│   │   └── alkitab_tb_enriched_groq.csv   # Final enriched Bible data
│   └── augmentation/               # Data augmentation pipeline
│       ├── augment_engine.py       # LLM-based data augmentation
│       ├── augment_multilabel.py   # Multi-label augmentation logic
│       └── seeds.json              # Augmentation seed data
│
├── evaluation/                     # Evaluation and benchmarking
│   ├── eval_ragas.py               # RAGAS evaluation framework
│   ├── eval_cosine_heatmap.py      # Cosine similarity heatmap analysis
│   ├── eval_seen_unseen.py         # Seen/unseen data evaluation
│   └── compare_embed_models.py     # Embedding model comparison
│
├── checkpoint/                     # Trained model weights (.pt files)
├── templates/
│   └── index.html                  # Chat interface HTML
├── static/
│   ├── style.css                   # Chat UI styling
│   └── script.js                   # Frontend chat logic
└── documentation/                  # Project documentation
```

---

## Evaluation

The project includes several evaluation tools under `evaluation/`:

| Script | Purpose |
|---|---|
| `eval_ragas.py` | End-to-end RAG evaluation using the RAGAS framework (faithfulness, relevance, etc.) |
| `eval_seen_unseen.py` | Evaluates classifier performance on seen vs. unseen intent data |
| `eval_cosine_heatmap.py` | Generates cosine similarity heatmaps between intents and Bible verses |
| `compare_embed_models.py` | Benchmarks different embedding models for verse retrieval quality |

### Running RAGAS Evaluation

```bash
python evaluation/eval_ragas.py
```

> Requires `GEMINI_API_KEY` in `.env` (uses Gemini as the RAGAS judge by default).

---

## License

This project is developed as an undergraduate thesis (Tugas Akhir). Please contact the author for usage terms and permissions.
