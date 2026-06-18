# Intent Classification Adjustment Guide

This document outlines the standard operating procedure for adding, removing, or renaming the emotion intents used by the LABAN classifier (IndoBERT) and the subsequent RAG pipeline in the Biblical Counseling Chatbot.

## 1. Data & Annotation Updates

To modify the intent classes, you must first update the source data where the labels are defined and mapped to training utterances.

- **`data/dataset_multiintent.csv`**: Update the core dataset. The `Intent` column contains the labels separated by `; ` (e.g., `Intent A; Intent B`). Modify these strings to reflect your new or renamed intents.
- **`data/intent_content.csv`**: Update the intent definitions and descriptive metadata if you are tracking intent explanations here.
- **`data/augmentation/seeds.json`**: If you are using the LLM augmentation pipeline to generate synthetic training data, update the intent keys and example seed utterances in this JSON file.
- **`data/augmentation/dataset_multiintent_augmented.csv`**: After updating `seeds.json`, rerun the augmentation scripts (`augment_engine.py` / `augment_multilabel.py`) to generate a new augmented dataset reflecting the updated intent classes.

## 2. Model Retraining

The LABAN classifier (`IndoBERT`) must be retrained to recognize the new intent space. The `MultiLabelBinarizer` dynamically infers the number of classes from the CSV, so you do not need to manually change neural network dimensions in the code—but you *must* generate a new weight file.

1. **Verify Data Path**: Ensure `data/data_preparation.py` points to your updated dataset (either the original or the augmented one).
2. **Execute Retraining**: Run `python models/multilabel/run_trainer.py`.
3. **Verify Output**: Confirm that a new model checkpoint is saved to `checkpoint/IndoBERT_multi_label_zsl.pt`. The `predict.py` wrapper will automatically load the new dimensions based on the updated `listof_intent` generated during data preparation.

## 3. Codebase & Configuration Adjustments

Several core backend components rely on hardcoded mappings of the intent strings. These must match the exact strings used in your CSV files.

- **`core/vector_db.py` -> `BIBLICAL_SYNONYMS`**: This dictionary maps each exact intent string to formal biblical synonyms used in the FAISS vector search. You **must** update the keys in this dictionary to match any new, renamed, or removed intents.
- **`core/session_manager.py`**: Check the following hardcoded constants. If you rename or remove these specific intents, you must update these variables so the state machine's emergency branching and CBT logic do not break:
  - `PROFESSIONAL_INTENT = 'Mengisyaratkan Butuh Bantuan Profesional'`
  - `PHYSICAL_SYMPTOM_INTENT = 'Mengisyaratkan Gejala Fisik'`

## 4. LLM Prompt Adjustments

The detected intents are passed directly to the LLMs. Therefore, the semantic naming of your intents directly influences both verse retrieval and the generated counseling response.

- **RAG Engine Prompt (`core/rag_engine.py`)**: The detected intent names are injected directly into the LLM system prompt via the `{detected_intents}` variable (`Emosi/Niat Terdeteksi (Multilabel): {detected_intents}`). Ensure your new intent names are highly descriptive and natural (e.g., "Menyatakan Perasaan Takut dan Kecemasan") so the LLM counselor understands exactly what emotion to empathize with.
- **FAISS Retrieval Query (`core/vector_db.py`)**: The raw intent strings (plus their biblical synonyms) are concatenated to form the FAISS search query. If a new intent is added, make sure its name contains semantically meaningful words in Indonesian, as these words will directly influence which Bible verses are retrieved.
