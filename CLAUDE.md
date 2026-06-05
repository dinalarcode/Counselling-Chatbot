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
   - `indolem/indobertweet-base-uncased` (IndoBERTweet)
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
- [x] Full embedding comparison table (all 6 models benchmarked) → `evaluation/results/backbone_comparison_summary.csv`
- [x] Seen/Unseen zero-shot evaluation (3 splits) → `evaluation/results/seen_unseen_summary.csv`
- [x] Cosine similarity heatmap (label-vs-label, centroid-vs-label, per-sample utterance-vs-label)
- [x] `BibleTestSession` added to `app.py` — sandbox mode for Cohen's Kappa verse retrieval testing (uses LLM reranker internally)
- [x] LLM reranker integrated into Bible verse retrieval (`retrieve_verse_with_llm()`) — FAISS top-5 → Gemini selects best; `BIBLICAL_SYNONYMS` dict added for query expansion
- [x] `MODEL_NAME` switched to `indobenchmark/indobert-base-p1` (IndoBERT) in `config.py` after backbone comparison confirmed it as top performer
- [x] **LLM config centralized** — `config.py` now has `CHATBOT_LLM_PROVIDER` ("groq"/"gemini"/"openai") + per-provider model/key/temperature; `_build_chatbot_llm()` factory in `rag_engine.py`; `eval_ragas.py` updated to use `opt.RAGAS_JUDGE_*`; active chatbot LLM switched to OpenAI (`gpt-5.4-mini`)
- [/] RAGAS evaluation — script ready (`eval_ragas.py`), template at `evaluation/data/ragas_testset.csv`, pending manual `reference` column fill + `--evaluate` run
- [ ] Streaming LLM response (SSE) — pending timing measurement decision
- [ ] Timing instrumentation cleanup before final submission

---

## Tech Stack

| Component | Package / Tool | Version / Notes |
|-----------|---------------|----------------|
| **Language** | Python | 3.10+ (no walrus operator, no 3.12-only features) |
| **LLM** | Multi-provider: Google Gemini 2.5-Flash / Groq llama-3.3-70b / OpenAI gpt-5.4-mini | via `langchain-google-genai`, `langchain-groq`, `langchain-openai`; active provider set in `opt.CHATBOT_LLM_PROVIDER` (currently `"openai"`) |
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
│                               BibleTestSession uses LLM reranker (retrieve_verse_with_llm) internally
│                               Also contains BibleTestSession (Cohen's Kappa sandbox — "bible test" mode)
├── config.py                   Global config singleton `opt`; MODEL_NAME, hidden_size, thresold, etc.
├── CLAUDE.md                   AI coding context (this file)
├── HANDOFF.md                  Full project handoff & task tracker
├── requirement.txt             pip dependencies
├── test.py                     CUDA / PyTorch smoke test
├── .env                        API keys — NOT committed (GEMINI_API_KEY, GROQ_API_KEY); already in .gitignore
│
├── core/
│   ├── rag_engine.py           RAGEngine — orchestrates full pipeline; [TIMING] instrumentation (remove before submission)
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
├── evaluation/
│   ├── compare_embed_models.py  LABAN backbone comparison (6 models, identical hyperparams)
│   ├── eval_seen_unseen.py      Zero-shot seen/unseen label evaluation (3 splits)
│   ├── eval_cosine_heatmap.py   Cosine similarity heatmap (label-vs-label, centroid-vs-label, per-sample)
│   ├── eval_ragas.py            RAGAS evaluation (--generate template, --evaluate with Groq judge)
│   ├── data/                    Test data for evaluations
│   │   └── ragas_testset.csv    50-sample test template (user fills 'reference' column)
│   └── results/                 Auto-created output CSVs and PNGs
│       ├── backbone_comparison_summary.csv      ✅ DONE — 6-model benchmark results
│
├── documentation/
│   ├── LABAN_DOCUMENTATION.md/.pdf              Thesis write-up: LABAN architecture
│   ├── RAGAS_DOCUMENTATION.md/.pdf              Thesis write-up: RAGAS evaluation
│   ├── COSINE_HEATMAP_DOCUMENTATION.md/.pdf     Thesis write-up: cosine heatmap
│   ├── FAISS_VERSE_RETRIEVAL_DOCUMENTATION.md   LLM reranker design rationale
│   ├── DOC_DATA_COLLECTION.md                   Data collection methodology
│   ├── DOC_ENVIRONMENT.md                       Environment setup
│   └── DOC_PREPROCESSING.md                    Data preprocessing
│       ├── backbone_comparison_per_intent.csv   ✅ DONE — per-intent breakdown
│       ├── *_training_curve.png                 ✅ DONE — 6 training curve plots
│       ├── seen_unseen_summary.csv              ✅ DONE — 3-split ZSL results
│       ├── seen_unseen_per_intent.csv           ✅ DONE — per-intent ZSL breakdown
│       ├── seen_unseen_aggregate.csv            ✅ DONE — aggregated ZSL stats
│       ├── cosine_label_vs_label.png            ✅ DONE
│       ├── cosine_centroid_vs_label.png         ✅ DONE
│       ├── cosine_utterance_vs_label.png        ✅ DONE
│       └── cosine_similarity_metrics.csv        ✅ DONE
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

### The 10 Intent Labels (fixed)

Defined via `MultiLabelBinarizer` fitted on training CSV in `data/data_preparation.py`:

1. Mengisyaratkan Butuh Bantuan Profesional
2. Mengisyaratkan Gejala Fisik
3. Menyatakan Perasaan Benci dan Jijik
4. Menyatakan Perasaan Marah dan Frustasi
5. Menyatakan Perasaan Percaya
6. Menyatakan Perasaan Sebelum Menghadapi Kejadian
7. Menyatakan Perasaan Sedih dan Kehilangan
8. Menyatakan Perasaan Takut dan Kecemasan
9. Menyatakan Rasa Syukur dan Apresiasi
10. Menyatakan Reaksi Terkejut dan Tidak Terduga

### Counseling Stage Machine

| # | Stage | Min Turns | Max Turns | Bible Verses | Classifier | Special |
|---|-------|-----------|-----------|:---:|:---:|---|
| 1 | `pembukaan` | 1 | 2 | ✗ | **Skipped** | |
| 2 | `pembahasan` | 1 | 4 | ✗ | ✓ (accumulates `primary_intents`) | |
| 3 | `intervensi` | 1 | 3 | ✗ | ✓ | |
| 4 | `solusi` | 1 | 3 | ✓ | ✓ | CBT guidance if physical symptoms |
| 5 | `relaksasi` | 1 | 3 | ✓ | ✓ | CBT guidance if physical symptoms |
| 6 | `penutupan` | 1 | ∞ | ✗ | ✓ | Offers professional referral |
| B | `bantuan_profesional` | — | — | ✓ | **Skipped** | Branch stage (see below) |

- Stage advances automatically if max turns exceeded or user sends a regex-matched transition signal.
- `primary_intents` are snapshotted at `pembahasan → intervensi` transition; `override_intents` ensures accumulated intents (not just current-turn intents) drive Bible verse selection.
- **Do not alter stage transition logic** without explicit user confirmation — it affects therapeutic flow.

#### Physical Symptom CBT Guidance
- When `Mengisyaratkan Gejala Fisik` is detected at any point during the session, the `has_physical_symptoms` flag is set.
- During `solusi` and `relaksasi`, extra CBT-based guidance is injected into the LLM prompt: deep breathing, mindfulness, relaxation strategies, and behavioral activation.

#### Bantuan Profesional (Branch Stage)
- **Emergency skip:** If `Mengisyaratkan Butuh Bantuan Profesional` is detected at ANY stage, the session immediately jumps to `bantuan_profesional`.
- **Penutupan offer:** At `penutupan`, the chatbot offers professional referral. If the user accepts (detected via transition signals), the session transitions to `bantuan_profesional`.
- In this stage: a Bible verse about seeking help is retrieved, an empathetic message is generated, and a "Hubungi Konselor Profesional" button is shown in the UI.
- After this stage, `session_ended = True`.

### RAG Pipeline
- **Bible verse retrieval** (updated): `BIBLICAL_SYNONYMS` query expansion → FAISS `similarity_search_with_score` (top-k=5) → Gemini LLM reranker (`retrieve_verse_with_llm()`) picks the best verse. Falls back to top FAISS result if LLM output is unparseable.
- `BIBLICAL_SYNONYMS` in `vector_db.py`: maps each of the 10 LABAN intents to formal TB biblical vocabulary so informal user slang gets expanded to words that actually appear in the Bible.
- Verse injection is active in `solusi`, `relaksasi`, **and `bantuan_profesional`** stages (see `BIBLE_VERSE_STAGES`).
- `VectorDBManager` holds both `faiss_bible_index` and `faiss_qna_index` — both persisted to disk; first-time build of Bible index can take 10–30 min.

### Config Quick-Reference

| Key | Current Value | Notes |
|-----|---------------|-------|
| `MODEL_NAME` | `indobenchmark/indobert-base-p1` | Active backbone (switched after comparison — IndoBERT was best) |
| `hidden_size` | `768` | Must match MODEL_NAME output dim |
| `thresold` | `0.5` | Intentional typo — do not rename |
| `max_len` | `50` | Max token length for classifier |
| `BATCH_SIZE` | `16` | Training batch size |
| `epochs` | `50` | Training epochs |
| `LEARNING_RATE` | `2e-5` | AdamW LR |
| `CHATBOT_LLM_PROVIDER` | `"openai"` | Active chatbot LLM: `"groq"` / `"gemini"` / `"openai"` |
| `OPENAI_CHATBOT_MODEL` | `"gpt-5.4-mini"` | OpenAI model (active) |
| `GEMINI_CHATBOT_MODEL` | `"gemini-2.5-flash"` | Gemini model (when provider=gemini) |
| `GROQ_CHATBOT_MODEL` | `"llama-3.3-70b-versatile"` | Groq model (when provider=groq) |
| `CHATBOT_TEMPERATURE` | `0.7` | Shared chatbot temperature |
| `RAGAS_JUDGE_MODEL` | `"gemini-2.5-flash"` | Fixed Gemini judge for RAGAS |
| `RAGAS_JUDGE_TEMPERATURE` | `0.0` | Deterministic judging |

**Commented-out MODEL_NAME alternatives in `config.py`:**
- `indolem/indobertweet-base-uncased` (hidden_size=768) ← previously active
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
- **`[TIMING]` instrumentation added** to `RAGEngine.generate_response()` — diagnostic only, **must be removed before submission**
- **Code quality:** Core pipeline is stable. `core/knowledge_base.py` is dead code (not used in live app; referenced only by `test_knowledge_base.py`).

**Session: 2026-06-02 — Evaluation Completed**

- **Backbone comparison completed** — All 6 transformer variants benchmarked at 50 epochs (identical hyperparams). Results in `evaluation/results/backbone_comparison_summary.csv`. IndoBERT is top performer (F1=0.9318); IndoBERTweet (F1=0.8991) was previously active.
- **Seen/Unseen ZSL evaluation completed** — 3 splits run. F1-Macro Seen=0.8997, Unseen=0.0 across all splits (model cannot predict truly unseen labels — expected behavior for LABAN without label expansion).
- **`BibleTestSession` added to `app.py`** — type `bible test` in chat UI to activate sandbox mode for Cohen's Kappa inter-rater reliability evaluation of verse relevance.

**Session: 2026-06-03 — LLM Reranker for Bible Retrieval**

- **`retrieve_verse_with_llm()` added to `VectorDBManager`** — replaces the old keyword re-rank + random tiebreaker. New pipeline: `BIBLICAL_SYNONYMS` query expansion → FAISS top-5 → Gemini LLM picks best verse from candidates. Falls back to top FAISS result if LLM output is unparseable.
- **`BIBLICAL_SYNONYMS` dict** added to `vector_db.py` — maps each of the 10 LABAN intents to formal TB biblical vocabulary (e.g., "kesel" → "murka", "capek" → "lelah") for richer FAISS queries.
- **`MODEL_NAME` switched to `indobenchmark/indobert-base-p1`** in `config.py` — IndoBERT replaces IndoBERTweet as the active backbone after the comparison confirmed it as the best performer.
- **`documentation/` folder created** — 10 thesis write-up files (LABAN, RAGAS, cosine heatmap, FAISS retrieval, data collection, environment, preprocessing).
- **`bantuan_profesional` added to `BIBLE_VERSE_STAGES`** in `rag_engine.py` — verse retrieval now also runs in this branch stage.

**Session: 2026-06-04 — LLM Config Centralized (Multi-Provider)**

- **`config.py` expanded** — added `CHATBOT_LLM_PROVIDER` ("groq" | "gemini" | "openai"), per-provider model names (`GROQ_CHATBOT_MODEL`, `GEMINI_CHATBOT_MODEL`, `OPENAI_CHATBOT_MODEL`), API key env var names, shared `CHATBOT_TEMPERATURE`, and separate `RAGAS_JUDGE_*` settings (always Gemini for deterministic judging).
- **`_build_chatbot_llm()` factory added to `rag_engine.py`** — reads `opt.CHATBOT_LLM_PROVIDER` and instantiates the correct LangChain LLM. Switching providers is now a single config change with no code edits.
- **`eval_ragas.py` updated** — RAGAS judge block now reads from `opt.RAGAS_JUDGE_MODEL`, `opt.RAGAS_JUDGE_API_KEY_ENV`, `opt.RAGAS_JUDGE_TEMPERATURE`. No more hardcoded Groq key in the script.
- **Active chatbot LLM set to OpenAI** — `CHATBOT_LLM_PROVIDER = "openai"`, `OPENAI_CHATBOT_MODEL = "gpt-5.4-mini"`. To switch back to Gemini or Groq, change `CHATBOT_LLM_PROVIDER` in `config.py` only.
- **`augment_engine.py` unchanged** — uses direct HTTP requests to Groq; not affected by LangChain-based centralization.

---

## Known Issues & Guardrails

| # | Issue | Severity | Location | Action |
|---|-------|:--------:|----------|--------|
| 1 | Typo `thresold` | Low | `config.py:12` | Canonical — do NOT rename |
| 2 | `hidden_size` hardcoded | Medium | `config.py:14` | Update manually when switching MODEL_NAME |
| 3 | ~~`.env` not in `.gitignore`~~ | ~~High~~ | ~~`.gitignore`~~ | ✅ RESOLVED — `.env` is already in `.gitignore` |
| 4 | `session_ended` not set in normal flow | Low | `session_manager.py` | Set to `True` after `bantuan_profesional`; normal penutupan flow remains open-ended |
| 5 | Bible FAISS first-build: 10–30 min | Medium | `vector_db.py` | Expected; loads fast on subsequent runs |
| 6 | `knowledge_base.py` is dead code | Low | `core/` | Remove or move to `archive/` before submission |
| 7 | LABAN loads 2 full BERT models | Medium | `bert_model.py` | By design (dual encoder) — doubles VRAM |
| 8 | Max-turn force-advance | Low | `session_manager.py` | User may be cut off at `pembahasan` max (4 turns) |
| 9 | Groq key needed for augmentation only | Info | `data/augmentation/` | Not needed for chatbot runtime |
| 10 | `[TIMING]` prints still in `rag_engine.py` | Medium | `core/rag_engine.py` | Remove all `_t0/_t1/_t2/_t3/_t4` + print lines before final submission |

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

- [x] **LABAN backbone comparison** — ✅ DONE. Results in `evaluation/results/backbone_comparison_summary.csv`.
  - Best: **IndoBERT** (F1-micro=0.9318, P=0.9602, R=0.9051) vs active model **IndoBERTweet** (F1-micro=0.8991, P=0.9356, R=0.8653)
  - IndoBERT-Lite failed badly (F1=0.3659) — numerical instability
  - All 6 training curves saved as `*_training_curve.png`
- [x] **Seen / Unseen zero-shot evaluation** — ✅ DONE. Results in `evaluation/results/seen_unseen_summary.csv`.
  - 3 splits, F1-Macro Seen avg=0.8997 (±0.008), F1-Macro Unseen=0.0 (model cannot generalize to truly unseen labels)
  - F1-Macro All avg=0.6298 (±0.006) — confirms LABAN requires seen labels at inference time
- [x] **Cosine similarity heatmap** — ✅ DONE. Results in `evaluation/results/`.
  - `cosine_label_vs_label.png` — label embedding separability; gap=1.0540
  - `cosine_centroid_vs_label.png` — utterance centroid alignment; gap=0.6048
  - `cosine_utterance_vs_label.png` — per-sample utterance alignment; gap=0.4195
  - `cosine_similarity_metrics.csv` — per-intent detailed metrics
- [/] **RAGAS evaluation** — Script ready, template generated. **Pending: user must fill `reference` column.**
  - Phase 1 (`--generate`) ✅ done → template at `evaluation/data/ragas_testset.csv` (50 samples)
  - Phase 2 (`--evaluate`) ⏳ blocked — user must first fill `reference` column with ideal responses
  - Output (when run): `evaluation/results/ragas_per_sample.csv`, `ragas_summary.csv`, `ragas_per_stage.csv`

---

### 🔧 Cleanup (Before Final Submission)

- [ ] Remove all `[TIMING]` diagnostic prints from `rag_engine.py` (lines with `_t0/_t1/_t2/_t3/_t4` + `[TIMING]` prints)
- [ ] Delete or archive `core/knowledge_base.py` and `core/test_knowledge_base.py`
- [ ] Audit and trim `requirement.txt` (remove unused: `anthropic`, `openai`, `aiml`, `beautifulsoup4`)
- [ ] Add "Mulai Sesi Baru" reset button to UI (`static/script.js` + `index.html`)
- [x] ~~Improve server error display~~ — `app.py` already returns `"Terjadi kesalahan internal. Silakan coba lagi."` for 500 errors ✅
- [ ] Decide and implement `session_ended` behavior in `session_manager.py` (restart vs. closed-session notice)
- [x] ~~Add `.env` to `.gitignore`~~ — `.env` is already in `.gitignore` (resolved)

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
