# Chatbot Konseling Alkitab — AI Code Context

---

## Project Overview

- **Type:** Tugas Akhir (undergraduate thesis) — Python-based chatbot
- **Domain:** Biblical counseling (Konseling Kristen / LABAN Method)
- **Language:** Bahasa Indonesia (all UI, training data, LLM responses); English for all variable/function/class names
- **Methodology:**
  - Multi-label intent classification using a fine-tuned transformer (LABAN architecture — dual BERT encoders)
  - RAG (Retrieval-Augmented Generation) pipeline: FAISS vector search → Gemini LLM
  - 6-stage structured counseling session managed by a state machine (`SessionManager`)
- **Interface:** Flask web app (single-page chat UI, Cream/White theme)

---

## Project Progress & Final Deliverables

### Final Objectives / Project Deliverables

The thesis evaluation requires the following empirical results and artifacts:

1. **Multi-label Classification Performance** — Precision, Recall, F1 (micro/macro/weighted) on both *seen* and *unseen* test sets (zero-shot learning evaluation across transformer variants)
2. **Embedding Model Comparison** — Benchmarking the following models as classifiers under the LABAN architecture:
   - `indolem/indobert-base-uncased` (IndoBERT)
   - `indobenchmark/indobert-lite-base-p1` (IndoBERT-Lite)
   - `indolem/indobertweet-base-uncased` (IndoBERTweet) ← **currently active**
   - `intfloat/multilingual-e5-small` (mE5)
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - `sentence-transformers/all-MiniLM-L6-v2`
   - `distilbert-base-multilingual-cased`
3. **Cosine Similarity Heatmap** — Per-intent pair cosine similarity visualization for the multi-label embedding space
4. **RAGAS Evaluation** — RAG quality metrics (faithfulness, answer relevancy, context precision, context recall) for the retrieval pipeline

### Current Progress Achieved

- [x] LABAN architecture implemented (`bert_model.py` — dual encoder: utterance + label)
- [x] Training pipeline complete (`run_trainer.py` — 80/10/10 split, AdamW, BCELoss, per-epoch F1 validation, best-checkpoint saving)
- [x] IndoBERTweet checkpoint trained post-augmentation → `checkpoint/IndoBERT_multi_label_zsl.pt`
- [x] Partial embedding comparison: training metrics PNGs exist for `indobertweet`, `indobert-lite`, `distilbert`, `multilingual-e5-small`
- [x] Bible FAISS index built from full TB Bible (`alkitab_tb.csv`, ~31 k verses)
- [x] QnA FAISS index persisted to disk (`data/faiss_qna_index/`)
- [x] RAGEngine with Gemini 2.5-Flash (LangChain); 6-stage SessionManager
- [x] Flask web UI deployed locally
- [x] Data augmentation: ~300-400 samples/intent via Groq Llama-3; multi-label combinations
- [ ] Full embedding comparison table (all 6+ models benchmarked, numbers not yet aggregated)
- [ ] Cosine similarity heatmap (not yet generated)
- [ ] RAGAS evaluation (not yet run)
- [ ] Streaming LLM response (SSE) — pending timing measurement decision
- [ ] Timing instrumentation cleanup before final submission

---

## Tech Stack

| Component | Package / Tool | Version / Notes |
|-----------|---------------|----------------|
| **Language** | Python | 3.10+ (no walrus operator, no 3.12-only features) |
| **LLM** | Google Gemini 2.5-Flash | via `langchain-google-genai` |
| **Classifier Backbone** | HuggingFace `transformers` | IndoBERTweet (active); see `config.py` for alternatives |
| **Vector Store** | FAISS | via `langchain_community` (Bible + QnA indexes) |
| **Embeddings (RAG)** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | used in FAISS retrieval |
| **Web Framework** | Flask | single-page, Fetch-based |
| **Training** | PyTorch | AdamW optimizer, BCEWithLogitsLoss |
| **Data Augmentation** | Groq API (Llama-3) | only needed for generating new training data |
| **Package Runner** | pip (inside `cbenv`) | **always activate `cbenv` before any command** |
| **GPU** | CUDA (recommended) | `config.device = 'cuda'`; CPU fallback supported |

> **Dependency note:** `anthropic`, `openai`, `aiml`, `beautifulsoup4` are present in `requirement.txt` but are **not used** in the live pipeline (legacy / scraper). Audit before final submission.

---

## System Actors & Roles

| Actor | Description | Entry Point |
|-------|-------------|-------------|
| **User (Konseli)** | Person seeking biblical counseling | Browser → `http://127.0.0.1:5000/` |
| **Chatbot (Konselor)** | Gemini-powered counselor guided by LABAN stages | `core/rag_engine.py` |
| **Researcher / Developer** | Trains models, runs evaluations | CLI (`run_trainer.py`, `predict.py`, etc.) |

> No login / authentication system. Single anonymous session per browser tab; reset via `/reset` endpoint or page reload.

---

## Folder Structure

```
The_Chatbot/
├── app.py                      Flask entry — /chat (POST), /reset (POST), / (GET)
├── config.py                   Global config singleton `opt`; MODEL_NAME, hidden_size, thresold, etc.
├── CLAUDE.md                   AI coding context (this file)
├── HANDOFF.md                  Full project handoff & task tracker
├── requirement.txt             pip dependencies
├── test.py                     CUDA / PyTorch smoke test
├── .env                        API keys — NOT committed (GEMINI_API_KEY, GROQ_API_KEY)
│
├── core/
│   ├── rag_engine.py           RAGEngine — orchestrates full pipeline; [TIMING] instrumentation
│   ├── vector_db.py            VectorDBManager — FAISS build/load/search (Bible + QnA)
│   ├── session_manager.py      SessionManager — 6-stage state machine, intent accumulation
│   ├── knowledge_base.py       LEGACY — template-based engine (dead code, not used in live app)
│   ├── test_rag.py             Integration test for RAGEngine
│   └── test_knowledge_base.py  Integration test for legacy engine
│
├── models/multilabel/
│   ├── bert_model.py           BertEmbedding — LABAN dual-encoder architecture
│   ├── predict.py              Predictor class — loads checkpoint, threshold-based inference
│   ├── run_trainer.py          Full training pipeline (split, train, validate, save best)
│   └── *.png                   Training metrics PNGs (one per model experiment)
│
├── data/
│   ├── data_preparation.py     LABANDataset (PyTorch Dataset), tokenization, DataLoaders
│   ├── alkitab_tb.csv          Full Terjemahan Baru Bible (~31 k verses)
│   ├── dataset_multiintent.csv Original multi-label training data
│   ├── dataset_qna.csv         QnA pairs for RAG injection
│   ├── intent_content.csv      Per-intent example sentences (pre-augmentation)
│   ├── intent_content_augmented.csv  Per-intent examples (post-augmentation)
│   ├── dataset_ayat.csv        Small curated Bible verse set (legacy)
│   ├── scrape_alkitab.py       One-time scraper for alkitab.mobi (beautifulsoup4)
│   ├── data_exploration.ipynb  EDA notebook
│   ├── faiss_bible_index/      Persisted FAISS index — Bible (auto-built on first run, ~10-30 min)
│   ├── faiss_qna_index/        Persisted FAISS index — QnA (built on first run)
│   └── augmentation/
│       ├── augment_engine.py       Groq API wrapper
│       ├── augment_multilabel.py   LLM augmentation for intent pairs/triples
│       ├── run_augmentation.py     Single-label augmentation runner
│       ├── merge_to_training.py    Merges augmented CSV into main dataset
│       ├── parse_seeds.py          Seed sentence parser
│       ├── write_output.py         Output writer helper
│       ├── seeds.json              Seed sentences per intent
│       ├── dataset_multiintent_augmented.csv  ← ACTIVE training dataset
│       └── raw_generated/          Raw Groq outputs before merging
│
├── templates/
│   └── index.html              Single-page chat UI (Gemini-inspired, Cream/White)
├── static/
│   ├── style.css               Cream/White responsive layout; typing indicator
│   └── script.js               Fetch-based chat; /reset on page load; error display
└── checkpoint/
    └── IndoBERT_multi_label_zsl.pt   Trained weights — git-ignored (never commit)
```

---

## Design System & Tokens

> Web UI only (Flask front-end). No dedicated design system library.

| Token | Value | Usage |
|-------|-------|-------|
| Primary background | `#FFFDF7` (Cream) | Page background |
| Chat surface | `#FFFFFF` (White) | Chat bubble backgrounds |
| Accent / Send button | Soft teal/sage (see `style.css`) | CTA elements |
| Font | System sans-serif (no Google Fonts import) | Body text |
| Layout | Single-column centered, max-width ~700 px | Chat container |
| Transition | Landing page → chatbox on first message | JS-driven class toggle |

---

## URL Structure

| Route | Method | Function |
|-------|--------|----------|
| `/` | GET | Renders `index.html` (landing + chat UI) |
| `/chat` | POST | Accepts `{ message: str }` JSON; returns `{ response: str }` |
| `/reset` | POST | Resets `SessionManager` state; called on page load automatically |

---

## Important Rules

- **Language split:** Variable/function/class names → English. UI text, LLM prompts, training data → Bahasa Indonesia.
- **Config singleton:** Always use `opt` (from `config.py`). Never re-instantiate `config` anywhere else.
- **Typo `thresold`:** This is intentional and canonical. Do NOT rename without updating every reference (`config.py`, `predict.py`, any test using it).
- **Virtual environment:** Always activate `cbenv` before running any Python command.
- **Active training dataset:** `data/augmentation/dataset_multiintent_augmented.csv` — this is the file `run_trainer.py` reads. Not `dataset_multiintent.csv`.
- **No hardcoded values:** Model name, batch size, thresholds, device must always come from `config.py → opt`.
- **Strict module separation:** `core/` (RAG, session, vector DB) and `models/multilabel/` (BERT classifier) must not cross-import each other.
- **BERT skip in `pembukaan`:** `RAGEngine.SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan'})` — do not remove; intentional optimization.
- **`[TIMING]` prints** in `rag_engine.py` are diagnostic only — remove before final submission.

---

## Critical Architectural Decisions (Guardrails)

### Multi-label Classifier (LABAN)
- **Dual-encoder design** (`bert_model.py`): one BERT instance for utterances, one for intent labels. This doubles VRAM usage — this is by design, not a bug.
- **Output dimension = number of intents (8).** Adding or removing an intent changes the final linear layer output size, invalidates the checkpoint, and requires full re-training. **STOP — ask user before doing this.**
- **`hidden_size` must match `MODEL_NAME` output dimension** — no automatic guard. Must be updated manually in `config.py` when switching models.
- **Checkpoint format:** PyTorch `.pt` file at `checkpoint/IndoBERT_multi_label_zsl.pt`. Never commit binary weights to Git.

### The 8 Intent Labels (fixed)

Defined via `MultiLabelBinarizer` fitted on training CSV in `data/data_preparation.py`:

1. Kekhawatiran dan Kecemasan
2. Perasaan Percaya
3. Perasaan Sedih dan Kehilangan
4. Perasaan Sebelum Menghadapi Kejadian
5. Perasaan Takut dan Kecemasan
6. Perasaan tidak Berharga dan Rendah Diri
7. Perasaan tidak Berdaya
8. Rasa Syukur dan Apresiasi

### Counseling Stage Machine

| # | Stage | Min Turns | Max Turns | Bible Verses | Classifier |
|---|-------|-----------|-----------|:---:|:---:|
| 1 | `pembukaan` | 1 | 2 | ✗ | **Skipped** |
| 2 | `pembahasan` | 1 | 4 | ✗ | ✓ (accumulates `primary_intents`) |
| 3 | `intervensi` | 1 | 3 | ✗ | ✓ |
| 4 | `solusi` | 1 | 3 | ✓ | ✓ |
| 5 | `relaksasi` | 1 | 3 | ✓ | ✓ |
| 6 | `penutupan` | 1 | ∞ | ✗ | ✓ |

- Stage advances automatically if max turns exceeded or user sends a regex-matched transition signal.
- `primary_intents` are snapshotted at `pembahasan → intervensi` transition; `override_intents` ensures accumulated intents (not just current-turn intents) drive Bible verse selection.
- **Do not alter stage transition logic** without explicit user confirmation — it affects therapeutic flow.

### RAG Pipeline
- Bible verse retrieval: semantic FAISS search → keyword re-rank → random tiebreaker (ensures verse diversity).
- Verse injection is **only** active in `solusi` and `relaksasi` stages.
- `VectorDBManager` holds both `faiss_bible_index` and `faiss_qna_index` — both are persisted to disk; first-time build of Bible index can take 10–30 min.

### Config Quick-Reference

| Key | Current Value | Notes |
|-----|---------------|-------|
| `MODEL_NAME` | `indolem/indobertweet-base-uncased` | Active backbone |
| `hidden_size` | `768` | Must match MODEL_NAME output dim |
| `thresold` | `0.5` | Intentional typo — do not rename |
| `max_len` | `50` | Max token length for classifier |
| `BATCH_SIZE` | `16` | Training batch size |
| `epochs` | `50` | Training epochs |
| `LEARNING_RATE` | `2e-5` | AdamW LR |

**Commented-out MODEL_NAME alternatives in `config.py`:**
- `indobenchmark/indobert-lite-base-p1` (hidden_size=768)
- `sentence-transformers/all-MiniLM-L6-v2` (hidden_size=384)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (hidden_size=384)
- `intfloat/multilingual-e5-small` (hidden_size=384)
- `distilbert-base-multilingual-cased` (hidden_size=768)

---

## Git Workflow

- **Commit format:** `<type> <short description>` — e.g., `multilabel augmented`, `fix vector_db keyword dedup`
- **Never commit:** `.env`, `checkpoint/*.pt`, `cbenv/`, `__pycache__/`, FAISS index directories
- **Branch:** single `main` branch (thesis project, no PR flow)
- **Before committing:** Remove `[TIMING]` diagnostic prints from `rag_engine.py`

---

## Latest Refactoring

**Session: 2026-05-15 — Runtime Optimization**

- **QnA FAISS index persisted to disk** (`data/faiss_qna_index/`) — eliminates 5–15 s rebuild on every startup
- **Regex patterns pre-compiled** → `SessionManager._COMPILED_SIGNALS` at class load (not per-turn)
- **Redundant keyword extraction removed** in `VectorDBManager.retrieve_verse()` — `_get_keyword_list()` called once, reused for query build and re-ranking
- **Conditional BERT skip** → `RAGEngine.SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan'})` skips forward pass where classification has no effect
- **`[TIMING]` instrumentation added** to `RAGEngine.generate_response()` — diagnostic only, must be removed before submission
- **Code quality:** Core pipeline is stable. `core/knowledge_base.py` is dead code (not used in live app; referenced only by `test_knowledge_base.py`).

---

## Known Issues & Guardrails

| # | Issue | Severity | Location | Action |
|---|-------|:--------:|----------|--------|
| 1 | Typo `thresold` | Low | `config.py:12` | Canonical — do NOT rename |
| 2 | `hidden_size` hardcoded | Medium | `config.py:14` | Update manually when switching MODEL_NAME |
| 3 | `.env` not in `.gitignore` | **High** | `.gitignore` | Rotate keys before any public sharing |
| 4 | `session_ended` never set to `True` | Medium | `session_manager.py` | Chatbot continues past `penutupan` |
| 5 | Bible FAISS first-build: 10–30 min | Medium | `vector_db.py` | Expected; loads fast on subsequent runs |
| 6 | `knowledge_base.py` is dead code | Low | `core/` | Remove or move to `archive/` before submission |
| 7 | LABAN loads 2 full BERT models | Medium | `bert_model.py` | By design (dual encoder) — doubles VRAM |
| 8 | Max-turn force-advance | Low | `session_manager.py` | User may be cut off at `pembahasan` max (4 turns) |
| 9 | Groq key needed for augmentation only | Info | `data/augmentation/` | Not needed for chatbot runtime |

---

## Task List

### ✅ Completed

- [x] LABAN multi-label classifier (architecture, training, inference)
- [x] IndoBERTweet checkpoint trained and saved
- [x] Bible FAISS index (full TB, ~31 k verses, hybrid retrieval)
- [x] QnA FAISS index persisted to disk
- [x] RAGEngine + Gemini 2.5-Flash integration
- [x] 6-stage SessionManager with auto-progression and transition signals
- [x] Flask web UI (Cream/White theme, typing indicator, /reset on load)
- [x] Stage-specific behavioral prompts for all 6 counseling phases
- [x] Bible verse injection restricted to `solusi` and `relaksasi` only
- [x] `override_intents` mechanism for accumulated intent-driven verse selection
- [x] Single-label augmentation pipeline (Groq Llama-3, ~300-400/intent)
- [x] Multi-label combination augmentation (`augment_multilabel.py`)
- [x] Augmented dataset merged (`dataset_multiintent_augmented.csv`)
- [x] Runtime optimizations (QnA cache, regex precompile, BERT skip, keyword dedup)
- [x] Per-phase timing instrumentation added

---

### ⏳ In Progress

- [ ] **Measure timing output** — Run app, send messages through all stages, read `[TIMING]` console lines to identify bottleneck (Gemini API vs BERT vs FAISS)
- [ ] **Streaming LLM response (SSE)** — Implement only if `chain.invoke > 70%` of total time; requires changes to `app.py`, `rag_engine.py`, `script.js`

---

### 🎯 Thesis Evaluation (Required Before Submission)

- [ ] **Full embedding model comparison** — Run `run_trainer.py` for each of the 6+ MODEL_NAME variants; aggregate Precision/Recall/F1 into a comparison table
- [ ] **Seen / Unseen test split** — Confirm zero-shot evaluation protocol; ensure test set contains utterances for unseen intent combinations
- [ ] **Cosine similarity heatmap** — Generate per-intent-pair cosine similarity matrix visualization from label embeddings
- [ ] **RAGAS evaluation** — Run RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) over a sample of chatbot conversations

---

### 🔧 Cleanup (Before Final Submission)

- [ ] Remove all `[TIMING]` diagnostic prints from `rag_engine.py`
- [ ] Delete or archive `core/knowledge_base.py` and `core/test_knowledge_base.py`
- [ ] Audit and trim `requirement.txt` (remove unused: `anthropic`, `openai`, `aiml`, `beautifulsoup4`)
- [ ] Add "Mulai Sesi Baru" reset button to UI (`static/script.js` + `index.html`)
- [ ] Implement user-friendly Indonesian error message in `script.js` (replace raw 500 error string)
- [ ] Decide and implement `session_ended` behavior in `session_manager.py` (restart vs. closed-session notice)
- [ ] Add `.env` to `.gitignore` and rotate all API keys before any public sharing

---

## Stop Conditions

Stop and ask the user before proceeding when:

1. **F1 score on validation set drops** below the previously recorded best — do not silently accept regression.
2. **Counseling stage logic in `session_manager.py`** is to be altered — stage transitions affect therapeutic flow and require human judgment.
3. **An intent label is to be added or removed** — changes model output dimension, invalidates checkpoint, requires full re-train.
4. **Bible translation or verse source** is ambiguous (which TB edition, which verse to prefer).
5. **A dependency major version bump** could break the `transformers` / `chromadb` / FAISS API.
6. **Any change touches `.env` secrets** (API keys, LLM credentials).
7. **Switching `MODEL_NAME`** — confirm new `hidden_size`, re-run training; never run inference with a mismatched checkpoint.
