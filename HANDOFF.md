# HANDOFF.md — Biblical Counseling Chatbot (Tugas Akhir)

> This file is the living reference for any agent or developer picking up this project.
> Update the relevant section whenever a task is completed or a new issue is discovered.

---

## 1. Project Summary

A Python-based **biblical counseling chatbot** built as a Tugas Akhir (thesis) project.
The system conducts structured multi-turn counseling conversations using the **LABAN method** for intent classification and a **RAG pipeline** to inject contextually relevant Bible verses and Q&A answers into Gemini LLM responses.

**Core Tech Stack:**
- LLM: Multi-provider (Groq / Gemini / OpenAI) — active provider set in `config.py → opt.CHATBOT_LLM_PROVIDER` (currently `openai`)
- Chatbot Model: `gpt-5.4-mini` (OpenAI); switchable to Gemini 2.5-Flash or Groq llama-3.3-70b via config
- Intent Classifier: Fine-tuned `indobenchmark/indobert-base-p1` (IndoBERT) with LABAN multi-label architecture
- Vector Store: FAISS (via `langchain_community`) for Bible verses + QnA pairs
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Web Interface: Flask + vanilla JS/CSS (single-page chat UI)
- Language: Bahasa Indonesia (all training data, responses, UI)

---

## 2. Architecture Overview

```
User (Browser)
    ↓ POST /chat
app.py  (Flask)
    ↓
SessionManager.chat()          [core/session_manager.py]
    ├─ Stage tracking (6 linear + 1 branch: pembukaan → ... → penutupan → bantuan_profesional)
    ├─ Transition signal detection (regex per stage)
    ├─ Intent accumulation across turns (Counter → primary_intents)
    ├─ Physical symptom tracking (has_physical_symptoms flag)
    ├─ Professional help keyword safety net (17 compiled regex patterns)
    ↓
RAGEngine.generate_response()  [core/rag_engine.py]
    ├─ Intent Classification — models/multilabel/predict.py (IndoBERTweet + LABAN)
    │      SKIPPED in: pembukaan, bantuan_profesional
    ├─ CBT guidance injection (solusi/relaksasi when physical symptoms detected)
    ├─ QnA FAISS search — core/vector_db.py
    ├─ Bible Verse Retrieval — core/vector_db.py
    │      ONLY in: relaksasi, solusi, bantuan_profesional
    │      Algorithm: BIBLICAL_SYNONYMS query expansion → FAISS top-5 → Gemini LLM reranker picks best verse
    ├─ Prompt assembly (LangChain PromptTemplate)
    ↓
Gemini 2.5-Flash API call
    ↓
Response → user (+ show_professional_button flag if bantuan_profesional)
```

**Counseling Stage Flow:**

| # | Stage | Min Turns | Max Turns | Bible Verses | Classifier | Special |
|---|-------|-----------|-----------|--------------|------------|--------|
| 1 | pembukaan | 1 | 2 | No | **Skipped** | |
| 2 | pembahasan | 1 | 4 | No | Yes (accumulates primary_intents) | |
| 3 | intervensi | 1 | 3 | No | Yes | |
| 4 | solusi | 1 | 3 | Yes | Yes | CBT guidance if physical symptoms |
| 5 | relaksasi | 1 | 3 | Yes | Yes | CBT guidance if physical symptoms |
| 6 | penutupan | 1 | ∞ | No | Yes | Offers professional referral |
| B | bantuan_profesional | — | — | Yes | Skipped | Branch stage + button |

Auto-advances if max turns exceeded or user sends a transition signal phrase.

**Branch flows:**
- **Emergency:** `Mengisyaratkan Butuh Bantuan Profesional` detected at any stage → immediate jump to `bantuan_profesional`
- **Penutupan offer:** User accepts professional referral offer at `penutupan` → transition to `bantuan_profesional`

---

## 3. File Map

```
The_Chatbot/
├── app.py                    Flask entry point; /chat, /reset, / endpoints
├── config.py                 Global config singleton (opt); MODEL_NAME, hidden_size, thresold, etc.
├── CLAUDE.md                 Coding standards + stop conditions for AI agents
├── HANDOFF.md                This file
├── requirement.txt           Python dependencies
├── test.py                   Minimal CUDA/PyTorch smoke test
├── .env                      API keys (NOT committed to git — see §8)
│
├── core/
│   ├── rag_engine.py         RAGEngine class — orchestrates full response pipeline
│   ├── vector_db.py          VectorDBManager — FAISS build/search for Bible + QnA
│   ├── session_manager.py    SessionManager — stage state machine + intent accumulation
│   ├── knowledge_base.py     LEGACY — template-based engine, superseded by RAGEngine
│   ├── test_rag.py           Integration test for RAGEngine
│   └── test_knowledge_base.py Integration test for legacy KnowledgeBaseEngine
│
├── models/multilabel/
│   ├── bert_model.py         BertEmbedding (LABAN architecture — dual BERT encoders)
│   ├── predict.py            predictor class — loads checkpoint, runs inference
│   └── run_trainer.py        Full training pipeline (dataset split, AdamW, BCELoss, F1 tracking)
│
├── data/
│   ├── data_preparation.py   LABANDataset (PyTorch Dataset), tokenization, DataLoaders
│   ├── alkitab_tb.csv        Full Bible in Terjemahan Baru (~31k verses)
│   ├── dataset_multiintent.csv  Original multi-intent training data
│   ├── dataset_qna.csv       QnA pairs for example-answer retrieval
│   ├── intent_content.csv    Per-intent example sentences
│   ├── augmentation/
│   │   ├── augment_multilabel.py   LLM augmentation for intent combinations (Groq)
│   │   ├── run_augmentation.py     Single-label augmentation runner
│   │   ├── augment_engine.py       Groq API wrapper
│   │   ├── merge_to_training.py    Merges augmented CSV into main dataset
│   │   ├── dataset_multiintent_augmented.csv  Active training dataset (post-augmentation)
│   │   └── seeds.json              Seed sentences per intent
│   └── faiss_bible_index/    Persisted FAISS index for Bible (auto-built on first run)
│   (faiss_qna_index/ will appear after first run post-optimization)
│
├── documentation/
│   ├── LABAN_DOCUMENTATION.md/.pdf       Full LABAN architecture write-up (thesis chapter)
│   ├── RAGAS_DOCUMENTATION.md/.pdf       RAGAS evaluation methodology write-up
│   ├── COSINE_HEATMAP_DOCUMENTATION.md/.pdf  Cosine heatmap analysis write-up
│   ├── FAISS_VERSE_RETRIEVAL_DOCUMENTATION.md  LLM reranker design write-up
│   ├── DOC_DATA_COLLECTION.md            Data collection methodology
│   ├── DOC_ENVIRONMENT.md                Environment setup documentation
│   └── DOC_PREPROCESSING.md              Data preprocessing documentation
│
├── templates/
│   └── index.html            Single-page chat UI
├── static/
│   ├── style.css             Cream/white theme, responsive layout, professional button
│   └── script.js             Fetch-based chat, typing indicator, professional button, /reset
├── evaluation/
│   ├── compare_embed_models.py  LABAN backbone comparison (6 models, identical hyperparams)
│   ├── eval_seen_unseen.py      Zero-shot seen/unseen label evaluation (3 splits)
│   ├── eval_cosine_heatmap.py   Cosine similarity heatmap evaluation
│   ├── eval_ragas.py            RAGAS RAG quality evaluation (--generate / --evaluate)
│   ├── data/                    Test data for evaluations
│   │   └── ragas_testset.csv    50-sample test set (question, stage, reference, intent_override)
│   └── results/                 Auto-created output CSVs and PNGs
└── checkpoint/
    └── IndoBERT_multi_label_zsl.pt   Trained classifier weights (git-ignored)
```

---

## 4. The 10 Intent Labels

Defined in `data/data_preparation.py` via `MultiLabelBinarizer` fitted on training data.
Current intents (from training CSV):

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

> **STOP CONDITION (CLAUDE.md):** Adding or removing an intent changes the model output dimension and requires full re-training. Never do this without explicit user confirmation.

---

## 5. What Has Been Done

### 5.1 Core Pipeline (Completed)
- [x] LABAN multi-label intent classifier — architecture (`bert_model.py`), training (`run_trainer.py`), inference (`predict.py`)
- [x] IndoBERTweet checkpoint trained and saved to `checkpoint/IndoBERT_multi_label_zsl.pt`
- [x] Bible FAISS index built from `alkitab_tb.csv` with hybrid semantic+keyword retrieval
- [x] QnA FAISS index built from `dataset_qna.csv` for example-answer injection
- [x] RAGEngine with Gemini 2.5-Flash integration via LangChain
- [x] 6-stage SessionManager with automatic progression, turn limits, and transition signals
- [x] Flask web app with chat UI (index.html, style.css, script.js)
- [x] Stage-specific behavioral prompt instructions for each counseling phase
- [x] Bible verse injection restricted to `solusi` and `relaksasi` stages only
- [x] Primary intents snapshotted at pembahasan→intervensi transition for accurate verse retrieval
- [x] `override_intents` mechanism so accumulated intents (not just current turn) drive verse selection

### 5.2 Data Augmentation (Completed)
- [x] Single-label augmentation pipeline (`run_augmentation.py` + Groq Llama-3) targeting 300-400 samples per intent
- [x] Multi-label combination augmentation (`augment_multilabel.py`) for intent pairs/triples
- [x] Augmented dataset merged into `data/augmentation/dataset_multiintent_augmented.csv` (active training source)

### 5.3 Runtime Optimizations
- [x] QnA FAISS index persisted to disk — eliminates rebuild on every startup
- [x] Regex patterns pre-compiled at class load in `SessionManager._COMPILED_SIGNALS`
- [x] Redundant keyword extraction removed in `VectorDBManager.retrieve_verse()`
- [x] Conditional BERT skip for `pembukaan` and `bantuan_profesional` stages
- [x] Per-phase timing instrumentation (`[TIMING]` lines in console)
- [x] Sastrawi dynamic stopwords (809 words + colloquial extras) replaced hardcoded 80-word list

### 5.4 New Intent Features (Session 2026-05-20)
- [x] **Physical symptom CBT guidance** — `has_physical_symptoms` flag tracked across session; injects CBT coping strategies into solusi/relaksasi LLM prompts
- [x] **Bantuan profesional branch stage** — emergency skip when `Mengisyaratkan Butuh Bantuan Profesional` detected (classifier OR keyword safety net)
- [x] **Keyword safety net** — 17 compiled regex patterns catch suicidal/hopeless expressions the classifier may miss
- [x] **Penutupan referral offer** — chatbot offers professional help at penutupan; user acceptance transitions to bantuan_profesional
- [x] **"Hubungi Konselor Profesional" button** — dummy button in UI (placeholder for future live-chat)
- [x] **Bible verse fix for bantuan_profesional** — hardcoded verse query ensures FAISS retrieves encouraging verse
- [x] **Stale Bible FAISS index fixed** — dimension mismatch (768 vs 384) diagnosed and rebuilt with current MiniLM model

### 5.5 Evaluation Framework (Session 2026-05-20)
- [x] `evaluation/` directory created
- [x] `compare_embed_models.py` — LABAN backbone comparison across 6 transformer models (IndoBERTweet, IndoBERT-Lite, IndoBERT, MiniLM-L6-v2, Multilingual-E5-small, DistilBERT-multilingual)
- [x] `eval_seen_unseen.py` — Zero-shot evaluation with 3 seen/unseen splits; computes F1-Macro Seen, F1-Macro Unseen, F1-Macro All

### 5.6 Cosine Similarity Heatmap Evaluation (Session 2026-05-25)
- [x] `eval_cosine_heatmap.py` -- Assesses semantic consistency between utterance embeddings and label embeddings
- [x] Three heatmaps generated: Label-vs-Label, Utterance Centroid-vs-Label, Per-Sample Utterance-vs-Label
- [x] Per-intent metrics CSV with alignment gaps and separability scores
- [x] Key findings: Label separation gap = 1.0540, Centroid alignment gap = 0.6048, Per-sample alignment gap = 0.4195

### 5.7 RAGAS RAG Quality Evaluation (Session 2026-05-26)
- [x] `eval_ragas.py` -- Two-phase RAGAS evaluation for the chatbot's RAG pipeline
  - Phase 1 (`--generate`): Randomly samples 50 QnA pairs (seed=42), auto-assigns counseling stages via content heuristics, saves template CSV with empty `reference` column
  - Phase 2 (`--evaluate`): Runs full RAG pipeline per sample, evaluates with RAGAS metrics using Groq LLM judge
- [x] Test template generated at `evaluation/data/ragas_testset.csv` (50 samples across pembukaan/pembahasan/intervensi/solusi/relaksasi/penutupan)
- [x] Metrics: Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference, ContextRecall
- [x] LLM Judge: Groq `llama-3.1-8b-instant` (avoids self-judging bias with Gemini generator)
- [x] Dependencies: `ragas==0.4.3`, `langchain-groq==1.1.2`, `datasets==4.8.5`
- [ ] **Pending**: User must fill `reference` column in `ragas_testset.csv` with ideal responses (including Bible verses for solusi/relaksasi), then run `--evaluate`
- Output: `ragas_per_sample.csv`, `ragas_summary.csv`, `ragas_per_stage.csv`

### 5.8 Backbone Comparison Completed (Session 2026-06-02)
- [x] `compare_embed_models.py` fully executed — all 6 transformer models trained and benchmarked
- [x] Results in `evaluation/results/backbone_comparison_summary.csv`:

| Model | F1-micro | Precision | Recall | Training Time |
|-------|----------|-----------|--------|---------------|
| **IndoBERT** | **0.9318** | **0.9602** | **0.9051** | 1284 s |
| IndoBERTweet (active) | 0.8991 | 0.9356 | 0.8653 | 1232 s |
| Multilingual-E5 | 0.9234 | 0.9425 | 0.9051 | 1030 s |
| DistilBERT-multi | 0.8967 | 0.9229 | 0.8720 | 996 s |
| MiniLM-L6-v2 | 0.8944 | 0.9108 | 0.8786 | 2491 s |
| IndoBERT-Lite | 0.3659 | 0.2351 | 0.8256 | 845 s |

- [x] 6 training curve PNGs saved to `evaluation/results/`
- [x] Per-intent breakdown in `backbone_comparison_per_intent.csv`

### 5.9 Seen/Unseen ZSL Evaluation Completed (Session 2026-06-02)
- [x] `eval_seen_unseen.py` fully executed — 3 splits completed
- [x] Results in `evaluation/results/seen_unseen_summary.csv` and `seen_unseen_aggregate.csv`:
  - **F1-Macro Seen avg: 0.8997** (±0.008) — model classifies seen labels well
  - **F1-Macro Unseen avg: 0.0** — model cannot predict labels it was not trained on
  - **F1-Macro All avg: 0.6298** (±0.006) — confirms LABAN requires seen label embeddings at inference
- [x] Aggregate statistics in `seen_unseen_aggregate.csv`

### 5.10 BibleTestSession Added to app.py (Session 2026-06-02)
- [x] **BibleTestSession** class added to `app.py` as a sandbox bypass mode
- [x] Activate by typing `bible test` in chat UI; deactivate with `exit test`
- [x] Runs LABAN classification + FAISS retrieval **with LLM reranker** (Gemini picks best verse from top-5 FAISS candidates)
- [x] Prints intent scores + retrieved verse to console for manual recording into Cohen's Kappa evaluation sheet
- [x] Used for inter-rater reliability evaluation of verse relevance quality

### 5.11 LLM Config Centralized (Session 2026-06-04)
- [x] **Multi-provider LLM support** — `config.py` now has `CHATBOT_LLM_PROVIDER` ("groq" | "gemini" | "openai") + per-provider model/key/temperature settings
- [x] **`_build_chatbot_llm()` factory** added to `rag_engine.py` — reads `opt.CHATBOT_LLM_PROVIDER` and constructs the appropriate LangChain LLM (ChatGroq / ChatGoogleGenerativeAI / ChatOpenAI). Eliminates hardcoded Gemini usage.
- [x] **`eval_ragas.py` updated** — RAGAS judge LLM now reads `opt.RAGAS_JUDGE_MODEL`, `opt.RAGAS_JUDGE_API_KEY_ENV`, `opt.RAGAS_JUDGE_TEMPERATURE` from `config.py` (no more hardcoded Groq key). Judge remains Gemini for quality.
- [x] **Active chatbot LLM switched to OpenAI** — `CHATBOT_LLM_PROVIDER = "openai"`, `OPENAI_CHATBOT_MODEL = "gpt-5.4-mini"` in `config.py`. Gemini and Groq configs are retained as commented-in alternatives.
- [x] **`augment_engine.py` unchanged** — uses direct `requests` to Groq API (not LangChain); no centralization needed since augmentation is a standalone tool with its own API key argument.

---

## 6. What Is Ongoing

### 6.1 RAGAS Evaluation — Phase 2 Blocked on Manual Data Entry
`eval_ragas.py --generate` has been run and the template is at `evaluation/data/ragas_testset.csv` (50 samples).
The `reference` column is currently empty. The user must manually fill in ideal reference answers for each row
(including relevant Bible verses for rows with stage=`solusi` or `relaksasi`), then run:
```
python evaluation/eval_ragas.py --evaluate
```

### 6.2 Performance Measurement (Started, Not Yet Measured)
The timing instrumentation (`[TIMING]` prints) was added but the app has not been profiled with real traffic. The actual bottleneck — whether it is Gemini API latency, BERT inference, or FAISS search — has not yet been measured.

**Next step:** Run the app, send messages through several stages, and read the terminal output:
```
[TIMING] classifier.predict: XXXms
[TIMING] search_qna: XXXms
[TIMING] retrieve_verse: XXXms    (only in solusi/relaksasi)
[TIMING] chain.invoke (LLM): XXXms
[TIMING] total generate_response: XXXms
```
The distribution of these numbers determines what to optimize next (see §7.1).

### 6.3 Legacy Code Not Yet Cleaned Up
`core/knowledge_base.py` is a template-based engine that predates the current RAG approach. It is imported by `core/test_knowledge_base.py` but is **not used in the live app**. It has not been removed because the test script still references it.

---

## 7. What Needs To Be Done (To Finish the Project)

Priority order: **P1 = blocking or high impact**, **P2 = important**, **P3 = polish**

### P1 — Performance

#### 7.1 Decide on LLM Streaming Based on Timing Data
After running the timing measurement (§6.1):
- **If `chain.invoke` > 70% of total time:** The Gemini API is the real bottleneck. The fix is streaming the response via Flask SSE (Server-Sent Events) so text appears word-by-word. This requires:
  - Change `app.py /chat` to use `Response(stream_with_context(...), mimetype='text/event-stream')`
  - Change `rag_engine.py` to use `self.chain.stream(...)` instead of `.invoke(...)`
  - Update `script.js` to use `EventSource` or `fetch` with `ReadableStream`
  - This is a medium-complexity change but gives the most visible UX improvement
- **If BERT inference > 30% of total:** Consider caching label embeddings inside `predictor` so `self.intent_ids` / `self.intent_mask` are computed once (they already are — confirm this is actually done correctly in predict.py line 24-25)
- **If FAISS search > 20% of total:** `candidate_count = max(k * 10, 20)` = 20 for k=1. Reducing this to 10 would halve search time with minimal quality impact

#### 7.2 Remove Timing Instrumentation Before Final Submission
The `[TIMING]` print lines in `core/rag_engine.py` are for diagnosis only. Remove all `_t0`, `_t1`, etc. and their print statements before final demo/submission. Keep only the structure changes (QnA cache, regex precompile, BERT skip, keyword dedup).

### P2 — Correctness & Quality

#### 7.3 Evaluate Classifier F1 After Augmentation
The model was trained after the augmentation (`8669c2a multilabel augmented`). The training metrics PNG should be reviewed:
- File: `models/multilabel/indolem-indobertweet-base-uncased_afteraugmented_training_metrics.png`
- Check: Did F1 score improve or regress compared to pre-augmentation?
- If regressed: investigate augmentation data quality; the Groq-generated sentences may introduce noise

#### 7.4 Remove or Quarantine Legacy Knowledge Base
`core/knowledge_base.py` and `core/test_knowledge_base.py` are dead code in the current pipeline. Either:
- Delete both files (clean option)
- Or move to `archive/` folder with a comment
Do NOT leave unused imports pointing to it in the active pipeline.

#### 7.5 Multi-label Augmentation — Enable Remaining Combinations
In `data/augmentation/augment_multilabel.py` lines 64-78, most intent combinations are commented out. Only two are active. If the classifier struggles with co-occurring emotions (e.g., sadness + fear), enabling more combinations and re-running augmentation + training will improve accuracy.
Steps: uncomment the target combinations → run `python data/augmentation/augment_multilabel.py` → merge output → re-train.

#### 7.6 Validate Session End Behavior
The session has no explicit "session ended" flag after `penutupan`. Currently:
- After penutupan response, the chatbot keeps accepting messages (continues in penutupan state)
- `session_ended` field exists on SessionManager but is never set to `True`
- Decide: should a follow-up message restart the session, or display a "session closed" notice?
- Implement the chosen behavior in `session_manager.py:chat()` and update `static/script.js` to handle a reset or end-of-session UI state

### P3 — Polish & Thesis Readiness

#### 7.7 Add Session Reset Button to UI
Currently `static/script.js` calls `/reset` on page load only. A visible "Mulai Sesi Baru" button in the UI would allow users to restart without refreshing the page, which is important for user testing (thesis evaluation).

#### 7.8 ~~Improve Error Display in UI~~ — ✅ DONE
`app.py` already returns `"Terjadi kesalahan internal. Silakan coba lagi."` for 500 errors. No further action needed.

#### 7.9 Write Test Cases for Stage Transitions
`core/test_rag.py` tests retrieval but not the full conversation flow. For thesis evaluation, add a simple script that runs a simulated conversation end-to-end and asserts:
- Correct stage at each expected turn
- Bible verses appear only in solusi/relaksasi
- Session reaches penutupan within expected turn range

#### 7.10 Clean Up `requirement.txt`
Several packages may be unused in the final pipeline: `anthropic`, `openai` (DeepSeek fallback commented out), `aiml`, `beautifulsoup4` (only used in scraper). Audit and trim before submission to reduce install time.

---

## 8. Known Issues & Constraints

| # | Issue | Severity | File | Notes |
|---|-------|----------|------|-------|
| 1 | **Typo `thresold`** | Low | `config.py:12` | Intentional per CLAUDE.md — do NOT rename without updating all references |
| 2 | **`hidden_size = 768` hardcoded** | Medium | `config.py:14` | Must manually update if MODEL_NAME is changed. No validation guard. |
| 3 | ~~**`.env` not in `.gitignore`**~~ | ~~High~~ | ~~`.gitignore`~~ | ✅ RESOLVED — `.env` is already in `.gitignore` |
| 4 | **`session_ended` behavior** | Low | `session_manager.py` | Set to `True` after bantuan_profesional; normal penutupan flow remains open-ended |
| 5 | **Bible index first-build: 10-30 min** | Medium | `vector_db.py:63` | Expected behavior, documented. Once built, loads fast from disk. |
| 6 | **`knowledge_base.py` is dead code** | Low | `core/knowledge_base.py` | Not referenced in live app; confuses code readers |
| 7 | **BERT loads two full models** | Medium | `bert_model.py:32-34` | LABAN needs dual encoders by design — label encoder + utterance encoder. This is correct architecture, not a bug, but doubles VRAM/RAM usage. |
| 8 | **Max-turn force advance** | Low | `session_manager.py:260` | User can be cut off mid-explanation if they exceed pembahasan max (4 turns) |
| 9 | **Groq API key needed for augmentation** | Low | `data/augmentation/` | Not needed for running the chatbot; only for generating new training data |
| 10 | **Bible index must match embed model** | High | `data/faiss_bible_index/` | If EMBED_MODEL in config.py changes, delete the index folder and let it rebuild. Dimension mismatch causes silent FAISS failures. |
| 11 | **`[TIMING]` prints in rag_engine.py** | Medium | `core/rag_engine.py` | Remove all `_t0/_t1/_t2/_t3/_t4` + print statements before final submission |

---

## 9. How to Run

```bash
# 1. Activate virtual environment
cbenv\Scripts\activate        # Windows
# source cbenv/bin/activate   # Mac/Linux

# 2. Run the chatbot (starts Flask on http://127.0.0.1:5000)
python app.py

# 3. Train the intent classifier (requires GPU recommended)
python models/multilabel/run_trainer.py

# 4. Test intent prediction interactively
python models/multilabel/predict.py

# 5. Test RAG pipeline
python core/test_rag.py

# 6. Run evaluation — LABAN backbone comparison
python evaluation/compare_embed_models.py

# 7. Run evaluation — Seen/Unseen zero-shot
python evaluation/eval_seen_unseen.py

# 8. Run augmentation (dry run first to preview)
python data/augmentation/run_augmentation.py --dry_run
```

**First-time setup note:** On first `python app.py`, if `data/faiss_bible_index/` does not exist, the Bible FAISS index will build from scratch (~10-30 minutes). This only happens once. Subsequent starts load from disk in ~5-10 seconds.

---

## 10. Config Quick-Reference

All values live in `config.py` and are accessed via the `opt` singleton.

| Key | Current Value | What it controls |
|-----|---------------|-----------------|
| `MODEL_NAME` | `indobenchmark/indobert-base-p1` | Transformer backbone for classifier (switched to IndoBERT after backbone comparison) |
| `hidden_size` | `768` | Must match MODEL_NAME output dim |
| `thresold` | `0.5` | Intent detection cutoff (note typo) |
| `max_len` | `50` | Max token length for classifier input |
| `BATCH_SIZE` | `16` | Training batch size |
| `epochs` | `50` | Training epochs |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `CHATBOT_LLM_PROVIDER` | `"openai"` | Active chatbot LLM provider: `"groq"` / `"gemini"` / `"openai"` |
| `OPENAI_CHATBOT_MODEL` | `"gpt-5.4-mini"` | OpenAI model name (active when provider=openai) |
| `GEMINI_CHATBOT_MODEL` | `"gemini-2.5-flash"` | Gemini model name (active when provider=gemini) |
| `GROQ_CHATBOT_MODEL` | `"llama-3.3-70b-versatile"` | Groq model name (active when provider=groq) |
| `CHATBOT_TEMPERATURE` | `0.7` | Shared temperature for chatbot LLM |
| `RAGAS_JUDGE_MODEL` | `"gemini-2.5-flash"` | Fixed Gemini model for RAGAS judging |
| `RAGAS_JUDGE_TEMPERATURE` | `0.0` | Deterministic judging |

**Switching models:** Comment out current `MODEL_NAME`, add new one, update `hidden_size` to match, re-run training. Never run inference with a mismatched checkpoint.

---

## 11. Session Log

| Date | Work Done |
|------|-----------|
| 2026-05-15 | Runtime optimization session: timing instrumentation, QnA FAISS persistence, regex precompile, keyword dedup, BERT skip |
| 2026-05-19 | Intent schema sync (8→10 intents), training CSV casing fix, Tikhonov regularization for LABAN gram matrix |
| 2026-05-20 | Physical symptom CBT guidance, bantuan_profesional branch stage, keyword safety net, Sastrawi stopwords, Bible FAISS index rebuilt, evaluation framework (backbone comparison + seen/unseen) |
| 2026-05-25 | Cosine similarity heatmap evaluation (eval_cosine_heatmap.py) — 3 heatmaps + metrics CSV; label separation gap = 1.0540, centroid alignment gap = 0.6048 |
| 2026-05-26 | RAGAS evaluation framework (eval_ragas.py) — two-phase pipeline with Groq judge; template CSV generated (50 samples); pending user `reference` fill |
| 2026-06-02 | Backbone comparison completed (all 6 models, IndoBERT best @ F1=0.9318); Seen/Unseen ZSL evaluation completed (F1-Seen=0.8997, Unseen=0.0); BibleTestSession added to app.py for Cohen's Kappa verse evaluation |
| 2026-06-03 | LLM reranker added to Bible verse retrieval (`retrieve_verse_with_llm()` in vector_db.py) — FAISS top-5 candidates → Gemini picks best; BIBLICAL_SYNONYMS query expansion dict added (10 intents → formal biblical vocab); MODEL_NAME switched to IndoBERT (`indobert-base-p1`) in config.py post-comparison; `documentation/` folder created with 10 thesis write-up files |
| 2026-06-04 | LLM config centralized into `config.py` — multi-provider support (Groq/Gemini/OpenAI) via `CHATBOT_LLM_PROVIDER`; `_build_chatbot_llm()` factory added to `rag_engine.py`; `eval_ragas.py` updated to use `opt.RAGAS_JUDGE_*`; active chatbot LLM switched to OpenAI (`gpt-5.4-mini`) |
