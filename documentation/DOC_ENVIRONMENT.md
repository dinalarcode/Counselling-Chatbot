# Technical Documentation: Development Environment

---

## 1. Hardware Specifications

| Component | Specification |
|---|---|
| **Laptop** | HP VICTUS |
| **Processor (CPU)** | Intel Core 12th Gen (Family 6 Model 154 Stepping 3), x86_64 |
| **Memory (RAM)** | 16 GB DDR4 |
| **Graphics (GPU)** | NVIDIA GeForce RTX 3050 Laptop GPU |
| **CUDA Compute** | CUDA 12.8 |
| **Storage** | SSD (project stored on D: drive) |

> The RTX 3050 Laptop GPU is used for training the LABAN multi-label classifier and computing FAISS embeddings. All model training runs on CUDA; inference can fall back to CPU if CUDA is unavailable.

---

## 2. Operating System

| Item | Value |
|---|---|
| **OS** | Windows 11 |
| **Build** | 10.0.26100-SP0 |
| **Architecture** | 64-bit (AMD64) |

---

## 3. Python & Virtual Environment

| Item | Value |
|---|---|
| **Python Version** | 3.14.3 |
| **Compiler** | MSC v.1944 64-bit (AMD64) |
| **Virtual Environment** | `cbenv` (created via `python -m venv cbenv`) |
| **Activation** | `cbenv\Scripts\activate` |
| **Package Manager** | pip (via `cbenv\Scripts\pip`) |

---

## 4. Core Libraries & Versions

### 4.1 Deep Learning & NLP

| Library | Version | Purpose |
|---|---|---|
| `torch` | 2.11.0+cu128 | PyTorch deep learning framework (CUDA 12.8 build) |
| `torchaudio` | 2.11.0+cu128 | Audio processing (installed with PyTorch bundle) |
| `torchvision` | 0.26.0+cu128 | Vision processing (installed with PyTorch bundle) |
| `transformers` | 4.56.2 | HuggingFace Transformers — model loading, tokenization |
| `tokenizers` | 0.22.2 | Fast tokenizer backend for HuggingFace |
| `sentence-transformers` | 5.4.1 | Sentence embedding models (MiniLM for FAISS) |
| `sentencepiece` | 0.2.1 | Subword tokenization (required by some transformer models) |
| `safetensors` | 0.7.0 | Safe model weight serialization |

### 4.2 LangChain Ecosystem

| Library | Version | Purpose |
|---|---|---|
| `langchain` | 1.2.17 | Core LangChain framework for RAG pipeline |
| `langchain-core` | 1.3.2 | LangChain core abstractions |
| `langchain-community` | 0.4.1 | Community integrations (FAISS vector store) |
| `langchain-huggingface` | 1.2.2 | HuggingFace embedding integration |
| `langchain-google-genai` | 4.2.2 | Google Gemini LLM integration |
| `langchain-groq` | 1.1.2 | Groq LLM integration (RAGAS evaluation) |
| `langchain-text-splitters` | 1.1.2 | Text chunking utilities |
| `langgraph` | 1.1.10 | LangGraph workflow orchestration |

### 4.3 Machine Learning & Data Science

| Library | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.8.0 | MultiLabelBinarizer, train_test_split, F1/precision/recall metrics |
| `pandas` | 3.0.2 | DataFrame operations for dataset management |
| `numpy` | 2.4.4 | Numerical array operations |
| `scipy` | 1.17.1 | Scientific computing (sparse matrices, distance metrics) |
| `matplotlib` | 3.10.9 | Training curve visualization, heatmap generation |

### 4.4 Web Framework

| Library | Version | Purpose |
|---|---|---|
| `Flask` | 3.1.3 | Web server for chatbot UI |
| `Jinja2` | 3.1.6 | HTML template engine (used by Flask) |

### 4.5 NLP Preprocessing

| Library | Version | Purpose |
|---|---|---|
| `PySastrawi` | 1.2.0 | Indonesian stopword removal and stemming |

### 4.6 External API Clients

| Library | Version | Purpose |
|---|---|---|
| `google-genai` | 1.75.0 | Google Generative AI SDK |
| `groq` | 1.2.0 | Groq API client (LLM-based augmentation + RAGAS judge) |
| `openai` | 2.36.0 | OpenAI-compatible API client (used by Groq) |

### 4.7 Evaluation

| Library | Version | Purpose |
|---|---|---|
| `ragas` | 0.4.3 | RAG evaluation framework (faithfulness, relevancy, correctness) |
| `datasets` | 4.8.5 | HuggingFace Datasets (required by RAGAS) |

### 4.8 Utilities

| Library | Version | Purpose |
|---|---|---|
| `beautifulsoup4` | 4.14.3 | HTML parsing (Bible scraping from alkitab.mobi) |
| `requests` | 2.33.1 | HTTP client for API calls and web scraping |
| `python-dotenv` | 1.2.2 | Environment variable loading from `.env` file |
| `tqdm` | 4.67.1 | Progress bars for training and scraping loops |
| `huggingface_hub` | 0.36.2 | Model downloading from HuggingFace Hub |

---

## 5. External APIs

| API | Provider | Model Used | Purpose |
|---|---|---|---|
| **Google Gemini** | Google AI | `gemini-2.5-flash` | Primary LLM for generating counseling responses |
| **Groq** | Groq Inc. | `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` | Data augmentation (utterance generation) and RAGAS LLM-as-a-judge |

API keys are stored in `.env` file at the project root:
```
GOOGLE_API_KEY=<key>
GROQ_API_KEY=<key>
```

---

## 6. Pre-trained Models

| Model | HuggingFace ID | Dimension | Purpose |
|---|---|---|---|
| **IndoBERT** | `indobenchmark/indobert-base-p1` | 768 | LABAN multi-label intent classifier (dual-encoder backbone) |
| **MiniLM Multilingual** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | FAISS embedding model for Bible verse + QnA retrieval |

> IndoBERT was selected as the LABAN backbone after a comparative evaluation of four transformer paradigms (see `evaluation/compare_embed_models.py`). With a Test F1-Micro of **0.9318**, Test Precision of **0.9602**, and Test Recall of **0.9051**, IndoBERT outperformed all other candidates including IndoBERTweet (F1-Micro: 0.8991), mBERT, and MiniLM-multi. MiniLM was chosen for retrieval because it is specifically trained for cross-lingual semantic similarity.

---

## 7. Project Directory Structure

```
The_Chatbot/
├── app.py                          # Flask web server
├── config.py                       # Centralized hyperparameters
├── requirement.txt                 # Pip dependency list
├── .env                            # API keys (not in version control)
├── cbenv/                          # Python virtual environment
├── core/
│   ├── rag_engine.py               # RAG pipeline orchestrator
│   ├── session_manager.py          # Counseling stage management
│   ├── vector_db.py                # FAISS index builder + retrieval
│   └── knowledge_base.py           # Knowledge base utilities
├── models/
│   └── multilabel/
│       ├── bert_model.py           # LABAN dual-encoder architecture
│       ├── run_trainer.py          # Training pipeline
│       └── predict.py              # Inference / predictor class
├── data/
│   ├── dataset_qna.csv             # Counseling QnA pairs
│   ├── dataset_multiintent.csv     # Original multi-intent utterances
│   ├── alkitab_tb.csv              # Full Indonesian Bible (TB)
│   ├── data_preparation.py         # Tokenization + data splitting
│   ├── scrape_alkitab.py           # Bible web scraper
│   ├── faiss_bible_index/          # Pre-built FAISS index (Bible)
│   ├── faiss_qna_index/            # Pre-built FAISS index (QnA)
│   └── augmentation/
│       ├── augment_engine.py       # Single-label LLM augmentation
│       ├── augment_multilabel.py   # Multi-label combination augmentation
│       ├── run_augmentation.py     # CLI runner
│       ├── merge_to_training.py    # Merge augmented → training format
│       └── dataset_multiintent_augmented.csv
├── evaluation/
│   ├── compare_embed_models.py     # Backbone comparison
│   ├── eval_seen_unseen.py         # Zero-shot evaluation
│   ├── eval_cosine_heatmap.py      # Embedding space visualization
│   ├── ragas_manager/             # RAGAS RAG evaluation (ragas_engine.py + listof_ragaslist.py)
│   └── eval_verse_retrieval.py     # Verse retrieval accuracy
├── checkpoint/                     # Saved model weights
├── templates/                      # Flask HTML templates
└── documentation/                  # Technical documentation (this folder)
```

---

*Document generated for the Biblical Counseling Chatbot (Tugas Akhir) project — 2026-05-31*
