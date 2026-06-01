# Technical Documentation: Data Pre-Processing

---

## Overview

This document covers the 5 pre-processing stages that transform raw CSV data into model-ready inputs. Each stage is described with its input, process, output, and the source code that implements it.

```
Raw CSV Files
    │
    ├── [Stage 1] Tokenization          → Token IDs + attention masks
    ├── [Stage 2] Multi-Hot Encoding    → 10 binary intent columns
    ├── [Stage 3] Data Splitting        → 80% train / 10% val / 10% test
    ├── [Stage 4] Stopword Removal      → Clean keywords for FAISS queries
    └── [Stage 5] FAISS Index Building  → L2-indexed vector database
```

---

## Stage 1: Tokenization

**File**: `data/data_preparation.py` (lines 25–43)
**When**: Every time the training pipeline is loaded

### Input
- Raw utterance text from `dataset_multiintent_augmented.csv` (column: `question`)
- Raw intent label text from the 10 intent names

### Process

The tokenizer converts raw Indonesian text into fixed-length sequences of token IDs that the LABAN model can process.

```python
from transformers import AutoTokenizer
from config import opt

tokenizer = AutoTokenizer.from_pretrained(opt.MODEL_NAME)
# opt.MODEL_NAME = 'indolem/indobertweet-base-uncased'
```

**Utterance Tokenization** (per sample, inside `LABANDataset.__getitem__`):

```python
encoding = self.tokenizer(
    question,
    add_special_tokens=True,     # Add [CLS] and [SEP] tokens
    max_length=opt.max_len,      # max_len = 50
    padding='max_length',        # Pad to exactly 50 tokens
    truncation=True,             # Truncate if longer than 50
    return_attention_mask=True,  # Binary mask: 1=real token, 0=padding
    return_tensors='pt',         # Return PyTorch tensors
)
```

**Intent Label Tokenization** (done once at startup):

```python
tokenized_intent = tokenizer(
    listof_intent,    # List of 10 intent name strings
    padding=True,     # Pad all labels to the same length
    truncation=True,
    return_tensors='pt'
)
```

### Output

| Output | Shape | Description |
|---|---|---|
| `input_ids` (utterance) | `(batch_size, 50)` | Integer token IDs |
| `attention_mask` (utterance) | `(batch_size, 50)` | Binary mask (1=real, 0=pad) |
| `intent_ids` (labels) | `(10, max_label_len)` | Token IDs for all 10 intent names |
| `intent_mask` (labels) | `(10, max_label_len)` | Attention mask for intent tokens |

### Why `max_len = 50`?

Counseling utterances in Indonesian are typically short (1–3 sentences). A maximum length of 50 tokens captures >95% of utterances without truncation, while keeping memory usage low during training.

---

## Stage 2: Multi-Hot Encoding

**File**: `data/data_preparation.py` (lines 29–49)
**When**: Dataset loading

### Input
- `Intent` column from CSV: semicolon-separated string
  - Example: `"Menyatakan Perasaan Sedih dan Kehilangan; Menyatakan Perasaan Takut dan Kecemasan"`

### Process

**Step 2a**: Split the semicolon-separated string into a list:

```python
dsqi['intent_list'] = dsqi['Intent'].apply(lambda x: x.split('; '))
# Result: ['Menyatakan Perasaan Sedih dan Kehilangan', 'Menyatakan Perasaan Takut dan Kecemasan']
```

**Step 2b**: Apply `MultiLabelBinarizer` to convert lists into binary vectors:

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(dsqi['intent_list'])
listof_intent = list(mlb.classes_)  # Sorted alphabetically
```

**Step 2c**: Create a DataFrame with binary columns:

```python
df_inten = pd.DataFrame(encoded_labels, columns=mlb.classes_).astype('float32')
dsqi_encoded = pd.concat([dsqi['question'], df_inten], axis=1)
```

### Output

A DataFrame with 11 columns (1 text + 10 binary):

| question | Mengisyaratkan Butuh... | Mengisyaratkan Gejala... | ... | Menyatakan Reaksi... |
|---|---|---|---|---|
| "aku sedih dan takut" | 0.0 | 0.0 | ... | 0.0 |
| "kepalaku pusing" | 0.0 | 1.0 | ... | 0.0 |

Each row has one or more `1.0` values corresponding to the detected intents. This multi-hot format is used directly as the training target for `BCEWithLogitsLoss`.

### Why `float32`?

PyTorch's `BCEWithLogitsLoss` requires float targets (not integers). Converting to `float32` ensures compatibility and avoids type casting during training.

---

## Stage 3: Data Splitting

**File**: `data/data_preparation.py` (lines 102–120)
**When**: Dataset loading

### Input
- `dsqi_encoded` DataFrame from Stage 2 (all rows with multi-hot labels)

### Process

```python
from sklearn.model_selection import train_test_split

# Split 1: 80% train, 20% temp
df_train, df_temp = train_test_split(dsqi_encoded, test_size=0.2, random_state=42)

# Split 2: temp → 50/50 → 10% validation, 10% test
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
```

### Output

| Split | Proportion | Purpose |
|---|---|---|
| `df_train` | 80% | Model training |
| `df_val` | 10% | Per-epoch validation (checkpoint selection) |
| `df_test` | 10% | Final evaluation (reported in thesis) |

Each split is wrapped in a `LABANDataset` and loaded via `DataLoader`:

```python
train_data = LABANDataset(dataframe=df_train, tokenizer=tokenizer)
val_data   = LABANDataset(dataframe=df_val, tokenizer=tokenizer)
test_data  = LABANDataset(dataframe=df_test, tokenizer=tokenizer)

train_dataload = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataload   = DataLoader(val_data,   batch_size=16, shuffle=False)
test_dataload  = DataLoader(test_data,  batch_size=16, shuffle=False)
```

### Why `random_state=42`?

A fixed random seed ensures that the exact same samples end up in train/val/test across all runs. This is critical for reproducible evaluation results.

---

## Stage 4: Stopword Removal

**File**: `core/vector_db.py` (lines 10–21)
**When**: Every FAISS query (verse retrieval)

### Input
- Raw user utterance text (e.g., `"saya merasa sangat sedih banget kehilangan semangat hidup"`)

### Process

**Step 4a**: Load Indonesian stopwords from Sastrawi library:

```python
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

_sastrawi_factory = StopWordRemoverFactory()
STOPWORDS_ID = set(_sastrawi_factory.get_stop_words())
```

**Step 4b**: Extend with colloquial/informal words not covered by Sastrawi:

```python
STOPWORDS_ID.update({
    'kak', 'nggak', 'gak', 'dong', 'sih', 'nih', 'deh', 'lho', 'kan',
    'kok', 'banget', 'kayak', 'gimana', 'gitu', 'udah', 'terus',
    'aja', 'emang', 'doang', 'cuma', 'tuh', 'yah', 'wah',
    'gatau', 'gapaham', 'gajelas', 'gamau', 'gaada', 'gabisa',
})
```

**Step 4c**: Extract keywords (words ≥ 3 characters, not in stopword set):

```python
def _get_keyword_list(self, text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS_ID]
```

### Output

```
Input:  "saya merasa sangat sedih banget kehilangan semangat hidup"
Output: ["sedih", "kehilangan", "semangat", "hidup"]
         ↑ "saya", "merasa", "sangat", "banget" are stopwords → removed
```

These keywords are appended to the FAISS search query alongside detected intents and biblical synonyms, helping the keyword re-ranking layer (Layer 3) in the verse retrieval pipeline.

---

## Stage 5: FAISS Index Construction

**File**: `core/vector_db.py` (lines 80–147)
**When**: First startup (built once, loaded from disk on subsequent runs)

### 5.1 Bible FAISS Index

#### Input
- `data/alkitab_tb.csv` — 31,102 Bible verses with metadata

#### Process

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Convert each verse into a LangChain Document
documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row['text'],         # The verse text (embedded by MiniLM)
        metadata={
            "book_abbr": row['book_abbr'],
            "book_name": row['book_name'],
            "chapter":   row['chapter'],
            "verse":     row['verse'],
            "text":      row['text'],
            "reference": row['reference'],  # e.g., "Mazmur 34:18"
        }
    )
    documents.append(doc)

# Build FAISS index (this embeds all 31,102 verses → ~10-30 minutes)
bible_db = FAISS.from_documents(documents, embeddings)

# Save to disk for fast loading on next startup
bible_db.save_local('data/faiss_bible_index')
```

#### Output
- `data/faiss_bible_index/` directory containing:
  - `index.faiss` — The FAISS L2 index (31,102 vectors × 384 dimensions)
  - `index.pkl` — Metadata mapping (verse text, reference, book info)
- Subsequent startups load from disk in seconds: `FAISS.load_local('data/faiss_bible_index')`

### 5.2 QnA FAISS Index

#### Input
- `data/dataset_qna.csv` — 367 counseling QnA pairs

#### Process

Same as Bible index, but each document's `page_content` is formatted as:
```python
page_content = f"Pertanyaan: {question}"
metadata = {"answer": answer}
```

#### Output
- `data/faiss_qna_index/` directory (367 vectors × 384 dimensions)

---

## Pre-Processing Pipeline Summary

```
dataset_multiintent_augmented.csv
    │
    ├── [Stage 1] AutoTokenizer (IndoBERTweet)
    │     → input_ids (batch, 50)
    │     → attention_mask (batch, 50)
    │
    ├── [Stage 2] MultiLabelBinarizer
    │     → 10 binary columns (float32)
    │
    └── [Stage 3] train_test_split (80/10/10)
          → train_dataload, val_dataload, test_dataload
              → LABANDataset → DataLoader (batch=16)


User Input (at runtime)
    │
    ├── [Stage 4] Stopword Removal (Sastrawi + colloquial)
    │     → Clean keywords for FAISS query
    │
    └── [Stage 5] FAISS Search
          → Embedded query (MiniLM, 384-dim)
          → L2 similarity search against pre-built index
          → Top-k candidates
```

| Stage | Happens | File | Key Tool |
|---|---|---|---|
| 1. Tokenization | Training time | `data_preparation.py` | `AutoTokenizer` (IndoBERTweet) |
| 2. Multi-Hot Encoding | Training time | `data_preparation.py` | `MultiLabelBinarizer` (scikit-learn) |
| 3. Data Splitting | Training time | `data_preparation.py` | `train_test_split` (scikit-learn) |
| 4. Stopword Removal | Runtime (every query) | `vector_db.py` | PySastrawi + custom list |
| 5. FAISS Indexing | First startup | `vector_db.py` | FAISS + MiniLM (384-dim) |

---

*Document generated for the Biblical Counseling Chatbot (Tugas Akhir) project — 2026-05-31*
