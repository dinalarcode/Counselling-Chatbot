# Chatbot Konseling Alkitab — AI Code Context

> For refactoring history, full known-issue detail, and ZSL thesis framing → read `CONTRIBUTING.md`.

---

## Project Overview

- **Type:** Tugas Akhir (undergraduate thesis) — Python chatbot
- **Domain:** Biblical counseling (Konseling Kristen / LABAN Method)
- **Language:** UI/prompts/training data → Bahasa Indonesia. Variable/function/class names → English.
- **Architecture:** Multi-label LABAN classifier (dual BERT encoders) + RAG pipeline (FAISS → LLM) + 6-stage `SessionManager`
- **Interface:** Flask single-page web app (Cream/White theme, `http://127.0.0.1:5000/`)

---

## Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Python | 3.10+ | No walrus operator, no 3.12-only features |
| LLM | OpenAI / Gemini / Groq | Active: `openai`; switch via `opt.CHATBOT_LLM_PROVIDER` in `config.py` |
| Classifier | HuggingFace `transformers` | Active backbone: `indobenchmark/indobert-base-p1` |
| Vector Store | FAISS | `langchain_community`; Bible + QnA indexes |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | FAISS retrieval |
| Web | Flask | Fetch-based, single-page |
| Training | PyTorch | AdamW, BCEWithLogitsLoss |
| Env | `cbenv` | Always activate before any Python command |

> Unused in live pipeline (legacy): `anthropic`, `aiml`, `beautifulsoup4` — remove from `requirement.txt` before submission.

---

## Folder Structure

```
The_Chatbot/
├── app.py                      Flask entry — /chat (POST), /reset (POST), / (GET); BibleTestSession
├── config.py                   Config singleton `opt` — all tunables live here
├── CONTRIBUTING.md             Refactoring log, known-issue detail, ZSL framing
├── requirement.txt             pip dependencies
├── .env                        API keys (git-ignored)
├── core/
│   ├── rag_engine.py           RAGEngine — full pipeline orchestration; [TIMING] instrumentation (remove)
│   ├── vector_db.py            VectorDBManager — FAISS build/load/search; BIBLICAL_SYNONYMS
│   ├── session_manager.py      SessionManager — 6-stage state machine
│   └── knowledge_base.py       LEGACY dead code — remove before submission
├── models/multilabel/
│   ├── bert_model.py           BertEmbedding — LABAN dual-encoder
│   ├── predict.py              Predictor — threshold-based inference
│   └── run_trainer.py          Training pipeline
├── data/
│   ├── alkitab_tb.csv          Full TB Bible (~31 k verses)
│   ├── dataset_qna.csv         QnA pairs for RAG
│   ├── faiss_bible_index/      Persisted FAISS Bible index (first build: 15–45 min)
│   ├── faiss_qna_index/        Persisted FAISS QnA index
│   ├── verse_retrieval/        AVI pipeline (chapter summaries → enriched CSV)
│   │   ├── alkitab_tb_enriched.csv   Active Bible source for FAISS
│   │   └── chapter_summaries_ollamaQwen.csv  Active summaries
│   └── augmentation/
│       └── dataset_multiintent_augmented.csv  ← ACTIVE training dataset
├── evaluation/
│   ├── compare_embed_models.py  Backbone comparison (6 models)
│   ├── eval_seen_unseen.py      ZSL seen/unseen evaluation
│   ├── eval_cosine_heatmap.py   Cosine similarity heatmap
│   ├── eval_ragas.py            RAGAS evaluation
│   └── results/                 Output CSVs and PNGs (all ✅ DONE except RAGAS Phase 2)
├── templates/index.html         Chat UI
├── static/style.css + script.js Chat UI assets
└── checkpoint/
    └── IndoBERT_multi_label_zsl.pt  Trained weights — git-ignored
```

---

## Important Rules

- **Config singleton:** Use `opt` from `config.py` only. Never re-instantiate.
- **Typo `thresold`:** Canonical — do NOT rename (`config.py`, `predict.py`, tests).
- **Active training data:** `data/augmentation/dataset_multiintent_augmented.csv` (not `dataset_multiintent.csv`).
- **No hardcoded values:** Model name, batch size, thresholds, device → always from `opt`.
- **Module separation:** `core/` and `models/multilabel/` must not cross-import.
- **BERT skip:** `RAGEngine.SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan'})` — intentional, do not remove.
- **`[TIMING]` prints:** Diagnostic only — remove all `_t0/_t1/_t2/_t3/_t4` + print lines before submission.
- **Never commit:** `.env`, `checkpoint/*.pt`, `cbenv/`, `__pycache__/`, FAISS index dirs.

---

## Critical Architecture

### LABAN Classifier
- **Dual encoder** (`bert_model.py`): one BERT for utterances, one for labels. Doubles VRAM — by design.
- **Output dim = 10 intents (fixed).** Adding/removing an intent invalidates the checkpoint and requires re-training. **STOP — ask user first.**
- **`hidden_size` must match `MODEL_NAME` output dim** — update manually in `config.py` when switching.
- Checkpoint: `checkpoint/IndoBERT_multi_label_zsl.pt` — never commit.

### The 10 Intent Labels (fixed)
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

| # | Stage | Max Turns | Bible Verse | Classifier | Notes |
|---|-------|-----------|:-----------:|:----------:|-------|
| 1 | `pembukaan` | 2 | ✗ | **Skipped** | |
| 2 | `pembahasan` | 5 | ✗ | ✓ | Accumulates `primary_intents`; exploration only |
| 3 | `intervensi` | 3 | ✗ | ✓ | Exploration/validation only; no solutions |
| 4 | `solusi` | 3 | ✗ | ✓ | CBT if physical symptoms; consent asked on final turn |
| 5 | `relaksasi` | 3 | ✓* | ✓ | 4-7-8 / 5-4-3-2-1 technique; anti-loop rule |
| 6 | `penutupan` | ∞ | ✗ | ✓ | Offers professional referral |
| B | `bantuan_profesional` | — | ✗ | **Skipped** | Emergency branch; `session_ended=True` |

*Bible verse in `relaksasi` only if `spiritual_consent is True`. `BIBLE_VERSE_STAGES = ['relaksasi']`.

- **Spiritual consent** (`spiritual_consent`): tri-state `None`/`True`/`False`. Asked once on final `solusi` turn. Captured stickily. If never asked → no verses (safe fail). Pass `spiritual_consent=True` explicitly in non-session callers (e.g. `eval_ragas.py`).
- **Do not alter stage transition logic** without explicit user confirmation.

### RAG Pipeline
- Retrieval: `BIBLICAL_SYNONYMS` query expansion → FAISS top-10 → LLM reranker (`retrieve_verse_with_llm()`) → best verse. Falls back to top FAISS result.
- Bible FAISS source: `data/verse_retrieval/alkitab_tb_enriched.csv` (AVI-enriched). If index absent, run AVI Steps 1→2 first.
- QnA FAISS: `data/faiss_qna_index/` (persisted).

### Config Quick-Reference

| Key | Value | Notes |
|-----|-------|-------|
| `MODEL_NAME` | `indobenchmark/indobert-base-p1` | Best from comparison |
| `hidden_size` | `768` | Must match MODEL_NAME |
| `thresold` | `0.5` | Intentional typo — do not rename |
| `CHATBOT_LLM_PROVIDER` | `"openai"` | `"groq"` / `"gemini"` / `"openai"` |
| `OPENAI_CHATBOT_MODEL` | `"gpt-5.4-mini"` | Active model |
| `CHATBOT_TEMPERATURE` | `0.7` | Shared |
| `RAGAS_JUDGE_MODEL` | `"gemini-2.5-flash"` | Fixed for RAGAS |

---

## Design System (Web UI)

| Token | Value |
|-------|-------|
| Background | `#FFFDF7` (Cream) |
| Chat surface | `#FFFFFF` |
| Accent | Soft teal/sage (see `style.css`) |
| Font | System sans-serif |
| Layout | Single-column, max-width ~700 px |

Routes: `GET /` → `index.html` | `POST /chat` → `{response: str}` | `POST /reset` → resets session.

---

## Known Issues (summary — see CONTRIBUTING.md for full detail)

| # | Issue | Action |
|---|-------|--------|
| 1 | Typo `thresold` | Canonical — do not rename |
| 2 | `hidden_size` hardcoded | Update manually when switching MODEL_NAME |
| 5 | Bible FAISS first-build: 15–45 min | Expected; needs AVI pipeline first |
| 10 | `[TIMING]` prints | Remove before submission |
| 12 | `bantuan_profesional` no verse (consent=None) | Intentional fail-safe |
| 14 | New `generate_response()` callers need `spiritual_consent` | Pass `True` when verses required |

---

## Task Status

### ✅ Done
- LABAN classifier, training, inference
- All 6 backbone comparisons + cosine heatmap
- ZSL seen/unseen evaluation — corrected (`eval_seen_unseen.py`): LABAN has no `nn.Linear` head; both "Model A" and "Model B" run the *same* gram-inverse projection (`w = sqrt(H)·G⁻¹·b`). The only variable is the label basis fed into it — Rezim A = fixed basis (10 or 7 seen), Rezim B = dynamically extended basis (7 seen + 3 unseen, encoded on-the-fly by the label encoder). Raw cosine similarity removed entirely. Script runs clean end-to-end (fixed a Windows cp1252 console crash on box-drawing chars via `sys.stdout.reconfigure(encoding="utf-8")`).
  - **Results (test set n=399, seed=42):** Produksi (10 label): Rezim A = Rezim B, F1-Macro **0.8909** (identical by construction — no unseen labels to differentiate the two regimes). Riset ZSL (7 seen/3 unseen, avg 3 splits): Rezim A unseen = **0.0** (structural — no column in G for unseen labels), Rezim B unseen = **0.8913** F1-Macro (proof of architectural ZSL — extending the basis, not swapping the math).
- `BAB_4.5_Evaluasi_Klasifikasi_Multi_Intent.md` fully rewritten to match: removed the false "Model A (Linear) >> Model B (Cosine)" narrative, replaced with single-mechanism/dual-basis framing, injected real metrics into all tables (4.5.1–4.5.6, summary table), deploy justification (§4.5.2.1) now rests on clinical determinism + RAG static intent mapping rather than a nonexistent accuracy edge.
- Bible + QnA FAISS indexes (enriched AVI)
- RAGEngine + LLM reranker + multi-provider LLM config
- 6-stage SessionManager with spiritual consent, ephemeral summarization, technique extraction
- Flask UI with privacy modal

### ⏳ In Progress
- Measure `[TIMING]` output → decide if SSE streaming needed (implement only if LLM > 70% of total time)

### 🎯 RAGAS (pending)
- Phase 1 ✅ → `evaluation/data/ragas_testset.csv` (50 samples)
- Phase 2 ⏳ → fill `reference` column, then run `python eval_ragas.py --evaluate`

### 🔧 Before Submission
- [ ] Remove all `[TIMING]` prints from `rag_engine.py`
- [ ] Delete `core/knowledge_base.py` + `core/test_knowledge_base.py`
- [ ] Trim `requirement.txt` (remove `anthropic`, `aiml`, `beautifulsoup4`)
- [ ] Add "Mulai Sesi Baru" reset button to UI
- [ ] Rebuild Bible FAISS index if enriched CSV updated (delete `data/faiss_bible_index/` then restart)

---

## Stop Conditions

Stop and ask the user before:
1. F1 score drops below previously recorded best.
2. Altering `session_manager.py` stage transition logic.
3. Adding or removing an intent label (invalidates checkpoint, requires re-train).
4. Bible translation / verse source is ambiguous.
5. Major dependency version bump (`transformers`, FAISS, LangChain).
6. Any change touching `.env` secrets.
7. Switching `MODEL_NAME` (confirm `hidden_size`, re-train required).

---

## Git Workflow

- DONT COMMIT ANYTHING INTO GITHUB
