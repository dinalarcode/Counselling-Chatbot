# Technical Documentation: Data Collection & Annotation

---

## 1. Data Source Overview

All original data in this project originates from **two sources**:

| Source | Content | Format |
|---|---|---|
| `Data 300825.xlsx` | Real counseling conversations from a university | Excel spreadsheet |
| `alkitab.mobi` | Full Indonesian Bible (Terjemahan Baru) | Web page (scraped) |

The Excel file (`Data 300825.xlsx`) contains existing counseling conversation data from a university counseling service. This data was extracted and restructured into two CSV files used by the system: `dataset_qna.csv` and `dataset_multiintent.csv`.

---

## 2. Intent Label Definitions

The project uses **10 intent labels** to classify the emotional state expressed in a counseling client's utterance. These labels were designed to cover the full spectrum of emotions encountered in pastoral/biblical counseling contexts.

| # | Intent Label | Description | Example Utterance |
|---|---|---|---|
| 1 | **Mengisyaratkan Butuh Bantuan Profesional** | Permintaan bantuan konseling, rujukan, atau pengakuan tidak mampu menangani sendirian | "Aku rasa aku butuh bicara sama psikolog deh" |
| 2 | **Mengisyaratkan Gejala Fisik** | Keluhan fisik seperti sakit kepala, sesak napas, gemetar, susah tidur, atau kelelahan | "Tiap malam aku susah tidur dan sering pusing" |
| 3 | **Menyatakan Perasaan Benci dan Jijik** | Ekspresi kebencian, jijik, penolakan, rasa enggan, atau perasaan muak | "Aku benci banget sama dia, muak lihat mukanya" |
| 4 | **Menyatakan Perasaan Marah dan Frustasi** | Ekspresi kemarahan, frustrasi, kesal, jengkel, atau ketidaksabaran | "Kesel banget sih, udah capek ngomong tapi nggak didengerin" |
| 5 | **Menyatakan Perasaan Percaya** | Ekspresi kepercayaan, keterbukaan, kemauan mencoba saran, atau kesediaan mengikuti proses | "Aku percaya ini bisa membaik, mau coba sarannya" |
| 6 | **Menyatakan Perasaan Sebelum Menghadapi Kejadian** | Antisipasi, persiapan mental, niat atau rencana sebelum menghadapi tantangan | "Besok aku harus presentasi, deg-degan banget" |
| 7 | **Menyatakan Perasaan Sedih dan Kehilangan** | Ekspresi kesedihan, kehilangan, duka, merasa tidak berharga, putus asa, atau kesepian | "Rasanya sepi banget, kayak nggak ada yang peduli" |
| 8 | **Menyatakan Perasaan Takut dan Kecemasan** | Ekspresi takut, khawatir, cemas, panik, gugup, atau perasaan terancam | "Aku takut banget kalau ternyata gagal lagi" |
| 9 | **Menyatakan Rasa Syukur dan Apresiasi** | Ekspresi rasa syukur, lega, apresiasi, perasaan membaik, atau terima kasih | "Alhamdulillah, aku merasa lebih baik sekarang" |
| 10 | **Menyatakan Reaksi Terkejut dan Tidak Terduga** | Ekspresi kaget, tidak percaya, bingung atas sesuatu yang tak terduga | "Serius?! Aku nggak nyangka sama sekali" |

### Multi-Label Annotation

A single utterance can express **multiple intents simultaneously**. For example:

```
Utterance: "Aku sedih banget kehilangan dia, tapi juga takut harus jalan sendiri"
Intent:    Menyatakan Perasaan Sedih dan Kehilangan; Menyatakan Perasaan Takut dan Kecemasan
```

Intents are separated by semicolons (`;`) in the CSV files. The system converts these into **multi-hot binary vectors** during preprocessing (see DOC_PREPROCESSING.md).

---

## 3. Dataset 1: Counseling QnA (`dataset_qna.csv`)

### Purpose
Provides example counselor responses for the RAG pipeline. When a user sends a message, the system searches this dataset for the most similar question and uses the corresponding answer as context for the LLM.

### Source & Collection
Extracted from `Data 300825.xlsx`. Contains real counseling question-answer pairs from a university counseling service.

### Format

| Column | Type | Description |
|---|---|---|
| `question` | string | The counseling client's utterance |
| `answer` | string | The counselor's response |

### Statistics
- **Total records**: 367 pairs
- **Encoding**: UTF-8-sig
- **Index**: `data/faiss_qna_index/` (FAISS L2 index using MiniLM embeddings)

---

## 4. Dataset 2: Multi-Intent Utterances (`dataset_multiintent.csv`)

### Purpose
Training data for the LABAN multi-label intent classifier. Each utterance is annotated with one or more of the 10 intent labels.

### Source & Collection
Extracted from `Data 300825.xlsx`. Contains client utterances from real counseling conversations, annotated with intent labels.

### Format

| Column | Type | Description |
|---|---|---|
| `question` | string | The counseling client's utterance |
| `Intent` | string | Semicolon-separated intent labels (e.g., `"Sedih dan Kehilangan; Takut dan Kecemasan"`) |

### Statistics
- **Encoding**: UTF-8-sig

---

## 5. Dataset 3: Augmented Multi-Intent (`dataset_multiintent_augmented.csv`)

### Purpose
Expanded version of `dataset_multiintent.csv` used for LABAN training. Contains all original data plus LLM-generated utterances to balance the intent distribution.

### Source & Collection
Generated via a **4-phase LLM augmentation pipeline**:

### Phase 1: Seed Collection
Seed sentences for each intent were extracted from the original dataset and stored in `data/augmentation/seeds.json`. These serve as few-shot examples for the LLM.

### Phase 2: Single-Label LLM Augmentation
Script: `data/augmentation/run_augmentation.py`

- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **Method**: Few-shot prompting — the LLM receives 5–8 seed examples per intent and generates new utterances matching that intent
- **Target**: 300–400 samples per intent (configurable per-intent targets)
- **Batching**: 20 sentences per API call with rate-limiting delays
- **Output**: Raw generated text saved to `data/augmentation/raw_generated/*.txt`

### Phase 3: Multi-Label Combination Augmentation
Script: `data/augmentation/augment_multilabel.py`

- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **Method**: Generates utterances that express **2–3 intents simultaneously** (e.g., "Sedih + Takut")
- **Format**: Pipe-delimited (`|`) output parsed into individual sentences
- **Combinations**: Configured in `COMBINATION_CONFIG` — each entry specifies which intents to combine and how many samples to generate

### Phase 4: Merge to Training Format
Script: `data/augmentation/merge_to_training.py`

Converts the wide-format augmented data (one column per intent) back to the `(question, Intent)` CSV format used by the training pipeline.

### Deduplication

Both augmentation scripts apply **Jaccard similarity-based deduplication** to prevent near-duplicate utterances:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
Threshold = 0.75
```

If a newly generated sentence has ≥ 75% token overlap with any existing sentence, it is discarded. Deduplication checks against:
1. All original seed sentences
2. All previously generated sentences in the current batch
3. All accumulated sentences from prior augmentation runs

### Format

Same format as `dataset_multiintent.csv`:

| Column | Type | Description |
|---|---|---|
| `question` | string | Client utterance (original or LLM-generated) |
| `Intent` | string | Semicolon-separated intent labels |

---

## 6. Dataset 4: Alkitab Terjemahan Baru (`alkitab_tb.csv`)

### Purpose
The Bible verse knowledge base for the RAG pipeline. When the counseling session reaches the `solusi` or `relaksasi` stage, the system retrieves a relevant Bible verse from this dataset via FAISS semantic search.

### Source & Collection
Web-scraped from [alkitab.mobi](http://alkitab.mobi/tb/) using BeautifulSoup.

Script: `data/scrape_alkitab.py`

**Scraping Process**:
1. Iterates through all **66 books** of the Bible (39 Old Testament + 27 New Testament)
2. For each book, scrapes every chapter page from `http://alkitab.mobi/tb/{book_abbr}/{chapter}/`
3. Parses HTML to extract verse numbers and text content
4. Applies rate limiting (0.5 second delay between requests)
5. Saves all verses to a single CSV file

### Format

| Column | Type | Description |
|---|---|---|
| `book_abbr` | string | Book abbreviation (e.g., `"Mzm"`, `"Yoh"`) |
| `book_name` | string | Full book name (e.g., `"Mazmur"`, `"Yohanes"`) |
| `chapter` | int | Chapter number |
| `verse` | int | Verse number |
| `text` | string | The verse text in Indonesian (Terjemahan Baru) |
| `reference` | string | Formatted reference (e.g., `"Mazmur 34:18"`) |

### Statistics
- **Total verses**: 31,102
- **Books**: 66 (complete Protestant canon)
- **Translation**: Terjemahan Baru (TB) — the most widely used Indonesian Bible translation
- **File size**: ~5.7 MB
- **Encoding**: UTF-8
- **Index**: `data/faiss_bible_index/` (FAISS L2 index using MiniLM embeddings)

---

## 7. Auxiliary Data Files

| File | Purpose |
|---|---|
| `data/intent_content.csv` | Wide-format CSV with one column per intent — used as intermediate format during augmentation |
| `data/intent_content_augmented.csv` | Augmented version of `intent_content.csv` (post-augmentation, pre-merge) |
| `data/dataset_ayat.csv` | Legacy Bible verse dataset (superseded by `alkitab_tb.csv`) |
| `data/augmentation/seeds.json` | Seed sentences per intent for few-shot LLM prompting |

---

## 8. Data Pipeline Summary

```
Data 300825.xlsx (real university counseling data)
    │
    ├──→ dataset_qna.csv (367 QnA pairs)
    │       └──→ FAISS QnA Index (faiss_qna_index/)
    │
    └──→ dataset_multiintent.csv (original utterances + intents)
            │
            ├──→ seeds.json (seed extraction)
            │       │
            │       ├──→ run_augmentation.py (single-label LLM augmentation)
            │       └──→ augment_multilabel.py (multi-label combination augmentation)
            │               │
            │               └──→ intent_content_augmented.csv
            │                       │
            │                       └──→ merge_to_training.py
            │                               │
            └───────────────────────────────┘
                    │
                    └──→ dataset_multiintent_augmented.csv (final training data)

alkitab.mobi (web)
    │
    └──→ scrape_alkitab.py
            │
            └──→ alkitab_tb.csv (31,102 verses)
                    └──→ FAISS Bible Index (faiss_bible_index/)
```

---

*Document generated for the Biblical Counseling Chatbot (Tugas Akhir) project — 2026-05-31*
