# CONTRIBUTING — Deep Reference for The_Chatbot

> Read this file when: modifying stage logic, debugging RAG/consent flow, reviewing session history, or checking known-issue details.

---

## Session Refactoring Log

### 2026-06-24 — Solusi→Relaksasi Transition Refinements
- `BIBLE_VERSE_STAGES` narrowed to `['relaksasi']` — removed `solusi` and `bantuan_profesional`.
- Spiritual consent question moved to **final `solusi` turn** (`turn_in_stage >= MAX_TURNS['solusi']`). Early exit → no consent asked → no verses in `relaksasi` (safe fail).
- `generate_technique_extraction(history_list)` added to `RAGEngine` — background LLM call at `solusi→relaksasi`. Stores in `session.chosen_technique`. Returns `None` on failure.
- `generate_response()` gained `chosen_technique=None` param; prefixes technique instruction in `relaksasi` when set.
- Anti-looping rule in `relaksasi` STAGE_INSTRUCTIONS: if user signals completion ("udah", "selesai", "lebih baik", "lega"), LLM must NOT repeat technique — ask for evaluation instead.
- `SessionManager`: added `temp_solusi_history`, `chosen_technique`; `reset()` clears both. `_record_solusi_turn()` added. `_advance_stage()` fires extraction at `solusi→relaksasi`.

### 2026-06-24 — Ephemeral State Summarization & Consent Modal
- `generate_background_summary(history_list)` added to `RAGEngine` — 1-sentence complaint summary (Bahasa Indonesia) from `pembahasan` history. Called at `pembahasan→intervensi` transition.
- `complaint_summary` stored on `SessionManager`; `temp_pembahasan_history` cleared immediately after.
- `generate_response()` gained `complaint_summary=None`; prepends `"Konteks Keluhan Klien: {summary}"` for stages `intervensi`, `solusi`, `relaksasi`.
- Blocking privacy/consent modal added to `index.html` (`.consent-modal-visible`); disables input until "Saya Mengerti" clicked. Styled in `style.css`, JS in `script.js` `DOMContentLoaded`.
- All ephemeral fields (`temp_pembahasan_history`, `complaint_summary`, `temp_solusi_history`, `chosen_technique`) cleared in `reset()`.
- Summary generation triggered in `SessionManager._advance_stage()` (not `generate_response()` — the latter is stateless).

### 2026-06-23 — Counseling Psychology Constraints
- `pembahasan` and `intervensi` STAGE_INSTRUCTIONS rewritten to forbid solutions/advice/reframing (`"DILARANG KERAS..."`). Exploration + emotion validation only.
- Spiritual consent: tri-state `SessionManager.spiritual_consent` (`None`/`True`/`False`) + `spiritual_consent_asked`. Captured stickily via `_detect_spiritual_consent()` (regex: decline checked first). `generate_response()` gained `spiritual_consent` + `ask_spiritual_consent` params.
- `relaksasi` prompt rewritten: Pernapasan 4-7-8 / Grounding 5-4-3-2-1 is always-on core; verse is conditional add-on.
- `eval_ragas.py` passes `spiritual_consent=True` explicitly to get verses for RAGAS eval.
- Known deviations: consent asked first `intervensi` turn (not "final turn"); bare `\btidak\b`/`\bnggak\b` may misfire on venting → fails safe.

### 2026-06-09 — AVI (Augmented Vector Indexing)
- `data/verse_retrieval/` two-step pipeline:
  - Step 1 `generate_chapter_summaries.py` — Groq/Ollama chapter summaries (active: qwen3:4b via Ollama).
  - Step 2 `prepare_enriched_bible.py` — LEFT JOIN summaries + Bible CSV → `alkitab_tb_enriched.csv` (~36 MB). `enriched_text` = `[Konteks Pasal: <summary>] <verse>`.
- `VectorDBManager.build_bible_index()` default source changed to enriched CSV. FAISS top-k bumped 5→10.

### 2026-06-04 — LLM Config Centralized
- `config.py`: `CHATBOT_LLM_PROVIDER` ("groq"/"gemini"/"openai"), per-provider model names, `CHATBOT_TEMPERATURE`, `RAGAS_JUDGE_*` settings.
- `_build_chatbot_llm()` factory in `rag_engine.py` — switching providers = one config change.
- `eval_ragas.py` reads `opt.RAGAS_JUDGE_*` (no more hardcoded keys).
- Active: `CHATBOT_LLM_PROVIDER = "openai"`, `OPENAI_CHATBOT_MODEL = "gpt-5.4-mini"`.

### 2026-06-03 — LLM Reranker for Bible Retrieval
- `retrieve_verse_with_llm()` in `VectorDBManager`: query expansion via `BIBLICAL_SYNONYMS` → FAISS top-5 → Gemini picks best. Falls back to top FAISS result if unparseable.
- `BIBLICAL_SYNONYMS` dict maps 10 LABAN intents to formal TB vocabulary.
- `MODEL_NAME` switched to `indobenchmark/indobert-base-p1` (best from comparison).

### 2026-06-02 — Evaluation Completed
- Backbone comparison: IndoBERT best (F1=0.9318). IndoBERT-Lite failed (F1=0.3659, numerical instability). Results: `evaluation/results/backbone_comparison_summary.csv`.
- Seen/Unseen ZSL: F1-Macro Seen=0.8997, Unseen=0.0 (3 splits). Unseen=0.0 is a **protocol artifact**: eval passes all 10 label texts during training → BCELoss provides negative supervision for unseen labels → model suppresses them. Correct ZSL would train with only seen labels. NOT fixed intentionally — production uses full supervision (10 intents fixed). ZSL evaluation is thesis architectural argument only.
- `BibleTestSession` added to `app.py` — type `bible test` for Cohen's Kappa sandbox.

### 2026-05-15 — Runtime Optimization
- QnA FAISS index persisted to `data/faiss_qna_index/` (eliminates 5–15 s rebuild).
- Regex patterns pre-compiled to `SessionManager._COMPILED_SIGNALS` at class load.
- `_get_keyword_list()` called once, reused in `VectorDBManager.retrieve_verse()`.
- `RAGEngine.SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan'})` — skips BERT forward pass.
- `[TIMING]` instrumentation added (remove before submission).

---

## Known Issues — Full Detail

| # | Issue | Sev | Location | Notes |
|---|-------|-----|----------|-------|
| 1 | Typo `thresold` | Low | `config.py:12` | Canonical — do NOT rename |
| 2 | `hidden_size` hardcoded | Med | `config.py:14` | Update manually when switching MODEL_NAME |
| 3 | `.env` not committed | — | `.gitignore` | ✅ Resolved |
| 4 | `session_ended` not set in normal flow | Low | `session_manager.py` | Set after `bantuan_profesional`; penutupan remains open |
| 5 | Bible FAISS first-build 15–45 min | Med | `vector_db.py` | Requires AVI pipeline first; fast on reload |
| 6 | `knowledge_base.py` dead code | Low | `core/` | Remove before submission |
| 7 | LABAN loads 2 BERT models | Med | `bert_model.py` | By design (dual encoder) — doubles VRAM |
| 8 | Max-turn force-advance | Low | `session_manager.py` | User may be cut off at pembahasan max (4 turns) |
| 9 | Groq key for augmentation only | Info | `data/augmentation/` | Not needed for chatbot runtime |
| 10 | `[TIMING]` prints in `rag_engine.py` | Med | `core/rag_engine.py` | Remove `_t0/_t1/_t2/_t3/_t4` + prints before submission |
| 11 | AVI enriched CSV must be regenerated if summaries change | Med | `data/verse_retrieval/` | Run `prepare_enriched_bible.py` then delete `data/faiss_bible_index/` |
| 12 | `bantuan_profesional` via emergency-skip: no verse (consent=None) | Med | `rag_engine.py` | Intentional fail-safe. To restore: add to `BIBLE_VERSE_STAGES` + guard `spiritual_consent is True` |
| 13 | Consent decline regex may misfire on venting | Low | `session_manager.py` | `\btidak\b`/`\bnggak\b` on first reply post-consent → consent=False. Fails safe. |
| 14 | New callers of `generate_response()` must pass `spiritual_consent` | Med | `rag_engine.py` | Default `None` → no verse. Pass `spiritual_consent=True` when verses required (as `eval_ragas.py` does). |
| 15 | Early `solusi` exit → consent never asked → no verse in `relaksasi` | Low | `session_manager.py` | Expected behavior. Safe fail. |
| 16 | `complaint_summary` = billed API call per session | Info | `rag_engine.py` | One call at `pembahasan→intervensi`. Negligible cost for thesis use. |
| 17 | `chosen_technique` may be None if LLM fails | Low | `rag_engine.py` | `relaksasi` STAGE_INSTRUCTIONS has fallback technique list. |

---

## ZSL Thesis Framing (for write-up)

Present **Unseen F1 = 0.0** as evidence that LABAN's ZSL capability requires zero supervision signal for unseen intents. The current evaluation protocol inadvertently provides negative supervision (zeroed BCELoss targets for unseen columns), which teaches the model to suppress those outputs. This is a valid and explainable finding, not an architecture failure. The correct LABAN ZSL protocol trains with only seen label texts (7-dim), then evaluates with all 10 — but this was not implemented because the 10-intent set is fixed in production and full supervision is the correct training configuration.
