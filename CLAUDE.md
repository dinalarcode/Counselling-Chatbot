# CLAUDE.md

## 1. Project Overview
- A Python-based biblical counseling chatbot that uses a RAG (Retrieval-Augmented Generation) pipeline for context-aware responses, the LABAN method for multi-label intent classification powered by a fine-tuned IndoBERTweet transformer (`indolem/indobertweet-base-uncased`), a ChromaDB vector store for Bible verse retrieval, and a Flask web interface for interaction.

---

## 2. Essential Commands
- **Run the app:** `python app.py`
- **Train the classifier:** `python models/multilabel/run_trainer.py`
- **Run intent prediction (single input):** `python test.py`
- **Test RAG pipeline:** `python core/test_rag.py`
- **Test knowledge base:** `python core/test_knowledge_base.py`

---

## 3. Coding Standards
- Use Python 3.10+; no walrus operator or 3.12-only features.
- All configuration (model name, batch size, thresholds, device) must live in `config.py` — never hardcode values inside modules.
- Class names use `PascalCase`; functions and variables use `snake_case`.
- Variable names must be in English, even though training data and responses may be in Indonesian (Bahasa Indonesia).
- Do not modify `thresold` (sic) in `config.py` without re-evaluating precision/recall on the validation set first.
- When adding a new transformer model, add its `MODEL_NAME` as a commented-out line in `config.py` and update `hidden_size` to match the model's output dimension.
- Keep `core/` (RAG, session, vector DB) and `models/multilabel/` (BERT classifier) strictly separated — no cross-imports.
- Always use `opt` (the singleton from `config.py`) when accessing config values, never instantiate `config` again.

---

## 4. Workflow Rules
- Before switching the active `MODEL_NAME` in `config.py`, confirm the new `hidden_size` and re-run training; never run prediction with a mismatched checkpoint.
- After any change to `core/vector_db.py` or `core/knowledge_base.py`, run `python core/test_knowledge_base.py` and `python core/test_rag.py` to verify retrieval integrity.
- After any change to `models/multilabel/bert_model.py` or `run_trainer.py`, re-train and review the saved metrics PNG before committing.
- New Bible verse data or QnA data must be placed in the `data/` directory; update `core/knowledge_base.py` to point to the new source.
- Checkpoint files go in `checkpoint/`; never commit binary `.pt` or `.bin` model weights to Git.
- Always activate the virtual environment (`cbenv`) before running any command.

---

## 5. Stop Conditions
Stop and ask for input when:
- The F1 score on the validation set drops below the previously recorded best (regression in classification quality).
- A requested change would alter the counseling stage logic inside `core/session_manager.py` — stage transitions affect the therapeutic flow and require human judgment.
- A new intent label needs to be added or removed from the training dataset — this changes the model output dimension and the entire pipeline.
- There is ambiguity about which Bible translation or verse source to use for retrieval.
- A dependency upgrade could break the transformer/tokenizer API (e.g., major `transformers` or `chromadb` version bumps).
- The user's query or a code change touches `.env` secrets (API keys, LLM credentials).
