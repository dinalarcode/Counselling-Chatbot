# How FAISS Bible Verse Retrieval Works

## Overview

The chatbot retrieves Bible verses using a **3-layer pipeline**: Query Construction → FAISS Semantic Search → Keyword Re-Ranking. The entire system lives in `core/vector_db.py` inside the `VectorDBManager` class.

```
User Utterance
    ↓
LABAN Classifier → detected intents
    ↓
┌─────────────────────────────────────────────────────┐
│            retrieve_verse(intents, user_input)       │
│                                                      │
│  Layer 1: Query Construction                         │
│    intents + keywords(user_input) → combined query   │
│                                                      │
│  Layer 2: FAISS Semantic Search                      │
│    combined query → embed → L2 search → 20 candidates│
│                                                      │
│  Layer 3: Keyword Re-Ranking + Tiebreaker            │
│    count keyword hits → sort → random tiebreak → top1│
└─────────────────────────────────────────────────────┘
    ↓
Bible Verse (reference + text)
```

---

## Layer 1: Query Construction

**Goal**: Build a single search query that captures both the emotional context (intents) and the specific topic (user's words).

**Input**:
- `intents` — list of detected intent strings from LABAN (e.g., `["Menyatakan Perasaan Sedih dan Kehilangan"]`)
- `user_input` — raw user utterance (e.g., `"saya merasa sangat sedih kehilangan semangat hidup"`)

**Process**:
1. Extract keywords from `user_input` by removing Indonesian stopwords (Sastrawi library + custom colloquial words like "banget", "aja", "gak", etc.)
2. Concatenate: `intent_part + keyword_part`

**Example**:
```
intents    = ["Menyatakan Perasaan Sedih dan Kehilangan"]
user_input = "saya merasa sangat sedih kehilangan semangat hidup"

→ stopwords removed: ["sedih", "kehilangan", "semangat", "hidup"]
→ combined_query = "Menyatakan Perasaan Sedih dan Kehilangan sedih kehilangan semangat hidup"
```

---

## Layer 2: FAISS Semantic Search

**Goal**: Find the most semantically similar Bible verses to the combined query.

### How the Index is Built (one-time)

1. Load `data/alkitab_tb.csv` — contains **31,102 verses** from the entire Indonesian Bible (Terjemahan Baru).
2. Each verse text is embedded into a **384-dimensional vector** using the `paraphrase-multilingual-MiniLM-L12-v2` sentence transformer model.
3. All 31,102 vectors are stored in a **FAISS index** (Facebook AI Similarity Search).
4. The index is saved to disk at `data/faiss_bible_index/` so it only needs to be built once.

### How Search Works (every query)

1. The `combined_query` string is embedded into the same 384-dimensional vector space using the same MiniLM model.
2. FAISS computes the **L2 (Euclidean) distance** between the query vector and all 31,102 verse vectors.
3. The top 20 closest verses are returned as candidates.

```
combined_query → MiniLM → query_vector (384-dim)
                              ↓
                    FAISS L2 Search against 31,102 verses
                              ↓
                    Top 20 candidates (sorted by distance)
```

> **Note**: Lower L2 distance = more semantically similar. A verse with score 0.41 is closer to the query than a verse with score 0.85.

### Why MiniLM (not IndoBERTweet)?

The FAISS index uses a **different model** than the LABAN classifier:

| Component | Model | Dimension | Purpose |
|---|---|---|---|
| LABAN Classifier | `indolem/indobertweet-base-uncased` | 768 | Intent classification |
| FAISS Index | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Semantic similarity search |

MiniLM is chosen for retrieval because it is specifically trained for **cross-lingual semantic similarity** (matching Indonesian queries to Indonesian Bible text), whereas IndoBERTweet is trained for **token-level understanding** (better for classification).

---

## Layer 3: Keyword Re-Ranking + Tiebreaker

**Goal**: Refine the 20 FAISS candidates by checking if the verse text actually contains words from the user's utterance.

### Step 3a: Keyword Hit Counting

For each of the 20 candidate verses, count how many extracted keywords appear in the verse text:

```
Candidate: Mazmur 34:18 — "Tuhan itu dekat kepada orang-orang yang patah hati..."
Keywords from user: ["sedih", "kehilangan", "semangat", "hidup"]

→ "sedih" in verse? No  → 0
→ "kehilangan" in verse? No  → 0
→ "semangat" in verse? No  → 0
→ "hidup" in verse? No  → 0
→ keyword_hits = 0

Candidate: Pengkhotbah 11:10 — "Buanglah kesedihan dari hatimu... kemudaan dan fajar hidup..."
→ "hidup" in verse? Yes → 1
→ keyword_hits = 1
```

### Step 3b: Sort

All 20 candidates are re-sorted by two criteria:
1. **keyword_hits** — descending (more keyword matches = better)
2. **faiss_score** — ascending (lower L2 distance = better)

```
Sort key: (-keyword_hits, faiss_score)

Result:
  Rank 1: keyword_hits=2, faiss_score=0.41  ← BEST
  Rank 2: keyword_hits=1, faiss_score=0.39
  Rank 3: keyword_hits=1, faiss_score=0.45
  Rank 4: keyword_hits=0, faiss_score=0.38
  ...
```

### Step 3c: Random Tiebreaker

If multiple candidates share the same top keyword_hits count, one is randomly selected from that group. This adds variety across sessions.

---

## Data Flow Diagram

```
alkitab_tb.csv (31,102 verses)
        ↓ [one-time build]
    MiniLM Embedding
        ↓
    FAISS Index (saved to disk)
        ↓ [every query]
    retrieve_verse(intents, user_input)
        │
        ├── [Layer 1] Combine intents + keywords → query string
        │
        ├── [Layer 2] Embed query → FAISS L2 search → 20 candidates
        │
        └── [Layer 3] Count keyword hits → sort → tiebreak → top 1
                │
                ↓
        { reference: "Mazmur 34:18", text: "Tuhan itu dekat..." }
```

---

## Key Files

| File | Role |
|---|---|
| `core/vector_db.py` | `VectorDBManager` class — builds index, runs retrieval |
| `data/alkitab_tb.csv` | Source data — 31,102 Bible verses (TB translation) |
| `data/faiss_bible_index/` | Pre-built FAISS index on disk (loaded at startup) |
| `config.py` → `EMBED_MODEL` | Embedding model name (`paraphrase-multilingual-MiniLM-L12-v2`) |

---

*Document generated for the Biblical Counseling Chatbot (Tugas Akhir) project — 2026-05-27*
