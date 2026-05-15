# HANDOFF.md — Biblical Counseling Chatbot (Tugas Akhir)

> This file is the living reference for any agent or developer picking up this project.
> Update the relevant section whenever a task is completed or a new issue is discovered.

---

## 1. Project Summary

A Python-based **biblical counseling chatbot** built as a Tugas Akhir (thesis) project.
The system conducts structured multi-turn counseling conversations using the **LABAN method** for intent classification and a **RAG pipeline** to inject contextually relevant Bible verses and Q&A answers into Gemini LLM responses.

**Core Tech Stack:**
- LLM: Google Gemini 2.5-Flash via LangChain (`langchain-google-genai`)
- Intent Classifier: Fine-tuned `indolem/indobertweet-base-uncased` with LABAN multi-label architecture
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
    ├─ Stage tracking (6 stages: pembukaan → pembahasan → intervensi → solusi → relaksasi → penutupan)
    ├─ Transition signal detection (regex per stage)
    ├─ Intent accumulation across turns (Counter → primary_intents)
    ↓
RAGEngine.generate_response()  [core/rag_engine.py]
    ├─ Intent Classification — models/multilabel/predict.py (IndoBERTweet + LABAN)
    │      SKIPPED in: pembukaan
    ├─ QnA FAISS search — core/vector_db.py
    ├─ Bible Verse Retrieval — core/vector_db.py
    │      ONLY in: relaksasi, solusi
    │      Algorithm: semantic FAISS search → keyword re-rank → random tiebreaker
    ├─ Prompt assembly (LangChain PromptTemplate)
    ↓
Gemini 2.5-Flash API call
    ↓
Response → user
```

**Counseling Stage Flow:**

| # | Stage | Min Turns | Max Turns | Bible Verses | Classifier |
|---|-------|-----------|-----------|--------------|------------|
| 1 | pembukaan | 1 | 2 | No | **Skipped** |
| 2 | pembahasan | 1 | 4 | No | Yes (accumulates primary_intents) |
| 3 | intervensi | 1 | 3 | No | Yes |
| 4 | solusi | 1 | 3 | Yes | Yes |
| 5 | relaksasi | 1 | 3 | Yes | Yes |
| 6 | penutupan | 1 | ∞ | No | Yes |

Auto-advances if max turns exceeded or user sends a transition signal phrase.

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
├── templates/
│   └── index.html            Single-page chat UI
├── static/
│   ├── style.css             Cream/white theme, responsive layout
│   └── script.js             Fetch-based chat, typing indicator, /reset on page load
└── checkpoint/
    └── IndoBERT_multi_label_zsl.pt   Trained classifier weights (git-ignored)
```

---

## 4. The 8 Intent Labels

Defined in `data/data_preparation.py` via `MultiLabelBinarizer` fitted on training data.
Current intents (from training CSV):

1. Kekhawatiran dan Kecemasan
2. Perasaan Percaya
3. Perasaan Sedih dan Kehilangan
4. Perasaan Sebelum Menghadapi Kejadian
5. Perasaan Takut dan Kecemasan
6. Perasaan tidak Berharga dan Rendah Diri
7. Perasaan tidak Berdaya
8. Rasa Syukur dan Apresiasi

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

### 5.3 Runtime Optimizations — Done in Latest Session
- [x] **QnA FAISS index persisted to disk** (`data/faiss_qna_index/`) — eliminates rebuild on every startup; saves 5-15s per restart
- [x] **Regex patterns pre-compiled** at class load in `SessionManager._COMPILED_SIGNALS` — avoids re-compilation on every turn
- [x] **Redundant keyword extraction removed** in `VectorDBManager.retrieve_verse()` — `_get_keyword_list()` called once, result reused for both query building and re-ranking
- [x] **Conditional BERT skip** — `RAGEngine.SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan'})` skips the BERT forward pass in the opening stage where intents have no effect on response or verse retrieval
- [x] **Per-phase timing instrumentation** added to `RAGEngine.generate_response()` — prints `[TIMING]` lines to console so bottlenecks can be measured

---

## 6. What Is Ongoing

### 6.1 Performance Measurement (Started, Not Yet Measured)
The timing instrumentation (`[TIMING]` prints) was added but the app has not been run since. The actual bottleneck — whether it is Gemini API latency, BERT inference, or FAISS search — has not yet been measured with real traffic.

**Next step:** Run the app, send messages through several stages, and read the terminal output:
```
[TIMING] classifier.predict: XXXms
[TIMING] search_qna: XXXms
[TIMING] retrieve_verse: XXXms    (only in solusi/relaksasi)
[TIMING] chain.invoke (LLM): XXXms
[TIMING] total generate_response: XXXms
```
The distribution of these numbers determines what to optimize next (see §7.1).

### 6.2 Legacy Code Not Yet Cleaned Up
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

#### 7.8 Improve Error Display in UI
If the Flask backend returns a 500 error, `script.js` currently shows the raw error string in a chat bubble. Replace with a user-friendly Indonesian message like "Maaf, terjadi kendala. Silakan coba lagi."

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
| 3 | **`.env` not in `.gitignore`** | High | `.gitignore` | API keys visible in repo history. Rotate all keys before any public sharing. |
| 4 | **`session_ended` never set** | Medium | `session_manager.py:136` | Field exists but is never flipped to `True`; session continues past penutupan |
| 5 | **Bible index first-build: 10-30 min** | Medium | `vector_db.py:63` | Expected behavior, documented. Once built, loads fast from disk. |
| 6 | **`knowledge_base.py` is dead code** | Low | `core/knowledge_base.py` | Not referenced in live app; confuses code readers |
| 7 | **BERT loads two full models** | Medium | `bert_model.py:32-34` | LABAN needs dual encoders by design — label encoder + utterance encoder. This is correct architecture, not a bug, but doubles VRAM/RAM usage. |
| 8 | **Max-turn force advance** | Low | `session_manager.py:260` | User can be cut off mid-explanation if they exceed pembahasan max (4 turns) |
| 9 | **Groq API key needed for augmentation** | Low | `data/augmentation/` | Not needed for running the chatbot; only for generating new training data |

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

# 6. Run augmentation (dry run first to preview)
python data/augmentation/run_augmentation.py --dry_run
```

**First-time setup note:** On first `python app.py`, if `data/faiss_bible_index/` does not exist, the Bible FAISS index will build from scratch (~10-30 minutes). This only happens once. Subsequent starts load from disk in ~5-10 seconds.

---

## 10. Config Quick-Reference

All values live in `config.py` and are accessed via the `opt` singleton.

| Key | Current Value | What it controls |
|-----|---------------|-----------------|
| `MODEL_NAME` | `indolem/indobertweet-base-uncased` | Transformer backbone for classifier |
| `hidden_size` | `768` | Must match MODEL_NAME output dim |
| `thresold` | `0.5` | Intent detection cutoff (note typo) |
| `max_len` | `50` | Max token length for classifier input |
| `BATCH_SIZE` | `16` | Training batch size |
| `epochs` | `50` | Training epochs |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |

**Switching models:** Comment out current `MODEL_NAME`, add new one, update `hidden_size` to match, re-run training. Never run inference with a mismatched checkpoint.

---

## 11. Session Log

| Date | Work Done |
|------|-----------|
| 2026-05-15 | Runtime optimization session: added timing instrumentation; persisted QnA FAISS index to disk; pre-compiled SessionManager regex patterns; removed redundant keyword extraction in `retrieve_verse`; added conditional BERT skip for `pembukaan` stage via `SKIP_CLASSIFICATION_STAGES` |
