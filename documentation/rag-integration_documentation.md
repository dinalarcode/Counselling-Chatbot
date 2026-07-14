# BAB 4.6 — Integrasi Sistem: Klasifikasi Multi-Intent, Retrieval Augmented Generation, dan Orkestrasi LLM

---

## 4.6.1 Gambaran Arsitektur Integrasi

Sistem chatbot konseling Alkitab yang dibangun dalam penelitian ini merupakan integrasi dari beberapa komponen terpisah yang bekerja secara berurutan dalam setiap giliran percakapan (*conversation turn*). Alur data berjalan dari lapisan antarmuka pengguna (Flask) menuju lapisan manajemen sesi (*SessionManager*), lapisan klasifikasi multi-intent (LABAN + IndoBERT), lapisan pengambilan konteks berbasis vektor (FAISS), dan terakhir lapisan pembangkitan respons berbasis LLM (RAGEngine + LangChain). Setiap komponen bertanggung jawab atas satu fungsi yang terdefinisi dengan jelas, dan seluruh alur ini dieksekusi dalam satu panggilan fungsi `chat()` untuk setiap pesan pengguna.

Secara ringkas, alur tersebut dapat digambarkan sebagai berikut:

```
Input Pengguna (Browser)
    ↓ POST /chat
app.py — Flask routing
    ↓
SessionManager.chat()         — manajemen tahap & akumulasi intent
    ↓
RAGEngine.generate_response()
    ├── [1] predictor.predict()           — klasifikasi LABAN (IndoBERT)
    ├── [2] VectorDBManager.search_qna()  — retrieval contoh jawaban
    ├── [3] VectorDBManager.retrieve_verse_with_llm()  — AVI Bible retrieval + LLM reranker
    └── [4] LangChain chain.invoke()      — pembangkitan respons LLM
        ↓
Respons JSON → Browser
```

Inisialisasi seluruh komponen berat (pemuatan model, pembangunan indeks FAISS) dilakukan **sekali** saat server Flask pertama kali dijalankan, sehingga setiap permintaan percakapan selanjutnya hanya memerlukan inferensi, bukan pemuatan ulang model.

---

## 4.6.2 Integrasi Model Klasifikasi LABAN

### Pemuatan Awal dan Inisialisasi

Seluruh komponen sistem diinisialisasi di level modul dalam `app.py` saat server pertama kali dinyalakan. Ini memastikan bahwa bobot model IndoBERT, tokenizer, dan indeks FAISS sudah tersedia di memori GPU sebelum permintaan pertama tiba.

**File: `app.py`**
```python
from core.rag_engine import RAGEngine
from core.session_manager import SessionManager

# Initialize global instances for the chatbot
print("Initializing RAGEngine and SessionManager...")
try:
    rag = RAGEngine()
    chatbot_session = SessionManager(rag)
    bible_test_session = BibleTestSession(rag)
    print("Initialization complete.")
except Exception as e:
    print(f"Error during initialization: {e}")
    rag = None
    chatbot_session = None
```

Di dalam konstruktor `RAGEngine.__init__()`, tiga komponen utama diinisialisasi: LLM, classifier (LABAN), dan VectorDBManager.

**File: `core/rag_engine.py`**
```python
def __init__(self):
    # Initialize the LLM (provider set in config.py → opt.CHATBOT_LLM_PROVIDER)
    self.llm = _build_chatbot_llm()

    # Initialize the Multilabel Classifier (Predictor)
    self.classifier = predictor()

    # Initialize the Vector DB Manager (Knowledge Base)
    self.vector_db = VectorDBManager()
    # Build indices if not built
    self.vector_db.build_bible_index('data/verse_retrieval/alkitab_tb_enriched_groq.csv')
    self.vector_db.build_qna_index('data/dataset_qna.csv')
    # ... [kode lainnya disingkat] ...
    # Create the LangChain processing chain
    self.chain = self.prompt_template | self.llm
```

### Proses Klasifikasi Per Giliran

Pada setiap giliran percakapan, teks input pengguna diumpankan ke dalam `predictor.predict()`. Kelas `predictor` (didefinisikan dalam `models/multilabel/predict.py`) memuat bobot checkpoint IndoBERT yang telah di-*fine-tune* dan melakukan tokenisasi serta inferensi melalui arsitektur LABAN. Hasilnya adalah sebuah kamus yang berisi daftar intent yang terdeteksi beserta skor probabilitasnya.

**File: `models/multilabel/predict.py`**
```python
class predictor:
    def __init__(self, model_path='checkpoint/IndoBERT_multi_label_zsl.pt', thresold=0.5):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(opt.MODEL_NAME)
        self.thresold = thresold
        # kerangka model
        jumlah_inten = len(listof_intent)
        self.model = BertEmbedding(num_labels=jumlah_inten)
        # memuat hasil train ke kerangka
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        # ...
    
    def predict(self, text):
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=opt.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        utterance_ids = encoding['input_ids'].to(self.device)
        utterance_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(
                utterance_ids=utterance_ids,
                utterance_mask=utterance_mask,
                label_ids=self.intent_ids,
                Label_mask=self.intent_mask
            )
        probz = torch.sigmoid(logits).squeeze().cpu().numpy()
        # ...
        return {
            "pertanyaan": text,
            "inten_terdeteksi": detected_intents,
            "apakah_multilabel": len(detected_intents) > 1,
            "skore": scores
        }
```

`opt.MODEL_NAME` merujuk ke `config.py` yang menetapkan `'indobenchmark/indobert-base-p1'` sebagai backbone produksi aktif, hasil seleksi berdasarkan komparasi empiris empat paradigma backbone (lihat Sub-bab 4.5.2).

Perlu diperhatikan bahwa klasifikasi **tidak selalu dilakukan** pada setiap giliran. `RAGEngine` mendefinisikan himpunan tahap di mana klasifikasi dilewati karena hasilnya tidak berdampak pada alur respons:

**File: `core/rag_engine.py`**
```python
SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan', 'bantuan_profesional'})
# ...
if current_stage in self.SKIP_CLASSIFICATION_STAGES:
    detected_intents = []
    intents_str = "Umum"
    print(f"[TIMING] classifier.predict: skipped (stage={current_stage})")
else:
    prediction = self.classifier.predict(user_input)
    detected_intents = prediction.get('inten_terdeteksi', [])
    intents_str = ", ".join(detected_intents) if detected_intents else "Umum"
```

Optimasi ini menghemat waktu inferensi GPU (sekitar 150–300ms per giliran) pada tahap-tahap di mana hasil klasifikasi tidak digunakan.

### Akumulasi Intent dan State Machine Sesi

Setelah intent terdeteksi pada suatu giliran, `SessionManager` mengakumulasi intent tersebut ke dalam sebuah `Counter` yang mencatat frekuensi kemunculan setiap intent sepanjang percakapan. Akumulasi ini menjadi mekanisme inti yang memungkinkan retrieval ayat Alkitab di tahap-tahap selanjutnya (terutama `relaksasi`) tetap relevan dengan keluhan yang paling sering muncul selama sesi *pembahasan*, meskipun pengguna mungkin sudah tidak lagi mengungkapkan masalahnya secara eksplisit.

**File: `core/session_manager.py`**
```python
def __init__(self, rag_engine):
    # ...
    self.accumulated_intents = Counter()  # intent → frequency
    self.primary_intents = []  # top intents derived dari pembahasan
    # ...

# Di dalam chat():
    # Accumulate intents from this turn
    turn_intents = result['context_used']['intents']
    for intent in turn_intents:
        self.accumulated_intents[intent] += 1

    # Track physical symptom intent across the entire session
    if self.PHYSICAL_SYMPTOM_INTENT in turn_intents:
        self.has_physical_symptoms = True

    # Update primary intents (top intents by frequency)
    self._update_primary_intents()
```

Metode `_update_primary_intents()` mengambil tiga intent paling sering muncul dan menyimpannya dalam atribut `primary_intents`:

**File: `core/session_manager.py`**
```python
def _update_primary_intents(self):
    """Update the primary intents list based on accumulated frequency."""
    if self.accumulated_intents:
        # Sort by frequency (descending), take top 3
        sorted_intents = self.accumulated_intents.most_common(3)
        self.primary_intents = [intent for intent, _ in sorted_intents]
```

Ketika sesi bertransisi dari tahap `pembahasan` ke `intervensi`, *snapshot* `primary_intents` diambil dan dibekukan. Pada tahap `relaksasi`, daftar intent ini dioper sebagai `override_intents` ke `generate_response()`, sehingga retrieval ayat didasarkan pada keluhan historis (bukan hanya input giliran saat ini):

**File: `core/session_manager.py`**
```python
# Di dalam chat():
    # Determine which intents to use for bible verse retrieval.
    override_intents = None
    if self.current_stage == 'relaksasi' and self.primary_intents:
        override_intents = self.primary_intents

    result = self.rag_engine.generate_response(
        user_input,
        current_stage=self.current_stage,
        override_intents=override_intents,
        has_physical_symptoms=self.has_physical_symptoms,
        excluded_books=self.used_verse_books,
        excluded_verses=self.used_verses,
        spiritual_consent=self.spiritual_consent,
        # ... [parameter lainnya disingkat] ...
    )
```

### Transisi Antar Tahap (Stage Machine)

`SessionManager` mengimplementasikan mesin tahap konseling dengan enam tahap linear: `pembukaan → pembahasan → intervensi → solusi → relaksasi → penutupan`, dan satu tahap cabang: `bantuan_profesional`. Setiap transisi dikendalikan oleh kombinasi dari tiga mekanisme: (1) jumlah giliran minimum (*minimum turns*), (2) sinyal transisi berbasis ekspresi reguler yang dicocokkan terhadap input pengguna, dan (3) batas maksimum giliran sebagai *safety net* paksa.

**File: `core/session_manager.py`**
```python
STAGE_ORDER = ['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']

MIN_TURNS = {
    'pembukaan': 1,
    'pembahasan': 3,  # clinical requirement: minimum 3 exploration turns
    'intervensi': 1,
    'solusi': 1,
    'relaksasi': 1,
    'penutupan': 1,
}

MAX_TURNS = {
    'pembukaan': 2,
    'pembahasan': 5,  # extended ceiling to allow richer exploration
    'intervensi': 3,
    'solusi': 3,
    'relaksasi': 3,
    'penutupan': 99,  # never force-exit penutupan
}
```

Mekanisme deteksi sinyal transisi menggunakan pola *regex* yang dikompilasi satu kali saat kelas dimuat, untuk efisiensi pencocokan di setiap giliran:

**File: `core/session_manager.py`**
```python
# Pre-compiled regex patterns for fast matching (compiled once at class load)
_COMPILED_SIGNALS = {
    stage: [re.compile(p) for p in patterns]
    for stage, patterns in TRANSITION_SIGNALS.items()
}

def _detect_transition_signal(self, user_input, stage):
    compiled = self._COMPILED_SIGNALS.get(stage, [])
    text_lower = user_input.lower().strip()
    for pattern in compiled:
        if pattern.search(text_lower):
            return True
    return False
```

Selain sinyal normal, sistem juga memiliki jalur darurat (*emergency skip*) ke `bantuan_profesional` yang dipicu oleh dua lapisan deteksi secara bersamaan: (a) classifier LABAN yang mendeteksi intent `'Mengisyaratkan Butuh Bantuan Profesional'` pada tahap awal (*early stages*), dan (b) 17 pola kata kunci berbasis *regex* yang mendeteksi ekspresi bunuh diri atau menyerah tanpa bergantung pada classifier:

**File: `core/session_manager.py`**
```python
PROFESSIONAL_KEYWORDS = [
    re.compile(r'\bmati\s*(aja|saja)\b', re.IGNORECASE),
    re.compile(r'\bbunuh\s*diri\b', re.IGNORECASE),
    re.compile(r'\bmengakhiri\s*(hidup|hidupku|nyawa)\b', re.IGNORECASE),
    # ... [kode lainnya disingkat] ...
]

EARLY_STAGES = {'pembukaan', 'pembahasan', 'intervensi'}
classifier_positive = self.PROFESSIONAL_INTENT in turn_intents
should_skip_to_professional = (
    keyword_triggered
    or (classifier_positive and self.current_stage in EARLY_STAGES)
) and not self.in_professional_stage
```

Rancangan ini memastikan bahwa kasus krisis selalu tertangani, bahkan jika model classifier melewatkan ekspresi informal atau slang.

---

## 4.6.3 Pemanfaatan FAISS untuk Retrieval (RAG)

Sistem menggunakan dua indeks FAISS yang berbeda, masing-masing dengan fungsi yang terpisah: indeks QnA untuk retrieval *few-shot example* dan indeks Alkitab untuk retrieval ayat kontekstual.

### Indeks QnA — Contoh Respons Terdahulu

Indeks QnA dibangun dari `data/dataset_qna.csv`, di mana setiap pasang pertanyaan-jawaban direpresentasikan sebagai satu dokumen. Model embedding yang digunakan untuk membangun **kedua** indeks FAISS adalah `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (didefinisikan sebagai `opt.EMBED_MODEL` dalam `config.py`), yang berbeda dari backbone LABAN yang digunakan untuk klasifikasi intent.

**File: `core/vector_db.py`**
```python
class VectorDBManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=opt.EMBED_MODEL)
        self.bible_db = None
        self.qna_db = None

# Di dalam build_qna_index():
        page_content = f"Pertanyaan: {question}"
        doc = Document(page_content=page_content, metadata={"answer": answer})
        documents.append(doc)
    # ...
    self.qna_db = FAISS.from_documents(documents, self.embeddings)
    self.qna_db.save_local(index_dir)

def search_qna(self, query, k=1):
    results = self.qna_db.similarity_search(query, k=k)
    return results
```

Hasil pencarian QnA digunakan dalam prompt LLM sebagai variabel `{example_answer}` — sebuah panduan nada dan konten yang diambil dari basis pengetahuan konseling yang sudah ada. Ini adalah komponen RAG pertama yang dieksekusi dalam setiap giliran.

### Indeks Alkitab — Augmented Vector Indexing (AVI)

Indeks Alkitab dibangun dari file `alkitab_tb_enriched_groq.csv`, yang merupakan hasil pipeline *Augmented Vector Indexing* (AVI). Berbeda dari pengindeksan ayat mentah, setiap dokumen dalam indeks ini mengandung teks *enriched* yang menggabungkan ringkasan konteks pasal (dihasilkan oleh LLM) dengan teks ayat aslinya. Tujuannya adalah agar proses *semantic embedding* dapat menangkap konteks teologis pasal secara holistik, bukan hanya kata-kata literal dari satu ayat yang terisolasi.

**File: `core/vector_db.py`**
```python
# Use enriched_text (chapter context + verse) for embedding;
# fall back to raw text if enriched_text is missing
enriched_text = str(row.get('enriched_text', '')).strip()
page_content = enriched_text if enriched_text and enriched_text != 'nan' else text

doc = Document(
    page_content=page_content,   # ← enriched_text untuk embedding
    metadata={
        "book_abbr": book_abbr,
        "book_name": book_name,
        "chapter": chapter,
        "verse": verse,
        "text": text,            # ← teks ayat asli untuk ditampilkan
        "reference": reference,
    }
)
```

### Tiga-Lapisan Retrieval Ayat dengan LLM Reranker

Retrieval ayat Alkitab mengikuti pipeline berlapis yang dirancang untuk memaksimalkan relevansi kontekstual sekaligus menghindari pengulangan yang monoton lintas sesi.

**Layer 1 — Query Construction (Ekspansi Sinonim Alkitabiah):** Query FAISS tidak dibangun dari teks input pengguna secara langsung, karena bahasa informal pengguna (misalnya "kesel banget", "capek") kemungkinan besar tidak muncul dalam teks Alkitab TB yang menggunakan bahasa formal. Sebagai gantinya, setiap intent yang terdeteksi dipetakan ke sinonim-sinonim Alkitabiah yang relevan melalui kamus `BIBLICAL_SYNONYMS`:

**File: `core/vector_db.py`**
```python
BIBLICAL_SYNONYMS = {
    "Menyatakan Perasaan Sedih dan Kehilangan": (
        "dukacita ratap tangis meratap bersedih perkabungan "
        "berkabung kehilangan air mata patah hati remuk jiwa"
        "duka ratapan hancur hati penghiburan dihibur sukacita memulihkan"
    ),
    "Menyatakan Perasaan Takut dan Kecemasan": (
        "takut ketakutan gentar ngeri kecemasan gemetar "
        "cemas waswas kuatir gelisah"
        "damai berani aman perlindungan tempat perlindungan penolong"
    ),
    # ... [10 intent lainnya disingkat] ...
}

# Di dalam _faiss_search():
intent_part = " ".join(intents)
synonym_parts = []
for intent in intents:
    synonyms = BIBLICAL_SYNONYMS.get(intent, "")
    if synonyms:
        synonym_parts.append(synonyms)
synonym_part = " ".join(synonym_parts)
combined_query = f"{intent_part} {synonym_part}".strip()
```

**Layer 2 — FAISS Semantic Search dengan Oversample dan Filter Diversitas:** Query gabungan digunakan untuk mencari di indeks FAISS dengan faktor *oversample* 3× (misalnya, jika diinginkan `k=10` kandidat akhir, FAISS mengambil `30` terlebih dahulu). Hasil kemudian difilter untuk memastikan diversitas: ayat yang sudah pernah digunakan dalam sesi ini (*session-level verse exclusion*), ayat dari kitab yang sudah diberikan (*book-level exclusion*), dan lebih dari dua ayat dari pasal yang sama (*per-chapter cap*) dibuang.

**File: `core/vector_db.py`**
```python
OVERSAMPLE_FACTOR = 3     # fetch 3× more from FAISS before filtering
MAX_PER_CHAPTER   = 2     # at most 2 verses per chapter in the final k candidates

# Di dalam _faiss_search():
oversample_k = k * OVERSAMPLE_FACTOR
candidates_with_scores = self.bible_db.similarity_search_with_score(
    combined_query, k=oversample_k
)
# ...
for doc, score in candidates_with_scores:
    reference = doc.metadata.get("reference", "")
    book_abbr = doc.metadata.get("book_abbr", "")

    if reference in excluded_verses:
        continue  # session-level verse exclusion
    if book_abbr in excluded_books:
        continue  # session-level book exclusion
    # ...
    current_count = chapter_count.get(chapter_key, 0)
    if current_count >= MAX_PER_CHAPTER:
        continue  # call-level diversity cap
    # ...
```

**Layer 3 — LLM Reranker:** Dari pool kandidat yang telah difilter, LLM (yang sama dengan LLM untuk pembangkitan respons) dipanggil dengan prompt khusus untuk memilih **satu** ayat terbaik berdasarkan konteks percakapan pengguna. Ini adalah komponen kritis yang membedakan sistem ini dari retrieval FAISS murni: model bahasa memiliki pemahaman semantik yang lebih kaya tentang konteks emosional dan teologis daripada kesamaan vektor semata.

**File: `core/vector_db.py`**
```python
def retrieve_verse_with_llm(self, intents, user_input, llm, k=10, ...):
    # Step 1+2+2.5: Get FAISS candidates (with diversity filters)
    candidates = self._faiss_search(intents, user_input, k=k, ...)

    # Build numbered candidate list for the LLM
    candidate_lines = []
    for i, c in enumerate(candidates, 1):
        candidate_lines.append(f"{i}. {c['reference']} — \"{c['text']}\"")
    candidate_list_str = "\n".join(candidate_lines)

    # Build the reranker prompt
    prompt = (
        "You are the Bible verse selector assistant for the Bible Terjemahan Baru (TB).\n\n"
        "The context of the user's counseling conversation:\n"
        f'"{user_input}"\n\n'
        "Here is the list of candidate verses:\n"
        f"{candidate_list_str}\n\n"
        "Instructions:\n"
        "1. Choose ONE verse that best suits the context of the conversation above.\n"
        # ... [instruksi lainnya disingkat] ...
        "5. Output format: REFERENCE|VERSE TEXT\n"
        "6. Output ONLY one line and DO NOT translate it to english.\n"
    )

    response = llm.invoke(prompt)
    raw_output = response.content.strip()

    # Parse pipe-delimited output: REFERENSI|TEKS
    if "|" in raw_output:
        parts = raw_output.split("|", 1)
        ref = parts[0].strip()
        text = parts[1].strip().strip('"').strip("'")
        # Validate: reference must exist in our candidates
        for c in candidates:
            if c["reference"] == ref:
                return [{"reference": ref, "text": c["text"], "book_abbr": c.get("book_abbr", "")}]
```

### Gerbang Persetujuan Spiritual (*Spiritual Consent Gate*)

Retrieval dan injeksi ayat Alkitab ke dalam prompt LLM **tidak dilakukan secara otomatis**. Sistem mengimplementasikan mekanisme *tri-state consent*: nilai `None` (belum dijawab), `True` (setuju), dan `False` (menolak). Pertanyaan persetujuan diajukan sekali oleh LLM pada akhir tahap `solusi`, dan jawaban pengguna dideteksi melalui pola *regex*. Hanya ketika `spiritual_consent is True` retrieval ayat diaktifkan:

**File: `core/rag_engine.py`**
```python
# Conditional RAG: verses are only retrieved when the user has explicitly
# given spiritual consent (True). Without it, we bypass RAG entirely and
# provide pure psychological counseling — no spiritual value imposition.
intents_for_verse = override_intents if override_intents else detected_intents
if current_stage in self.BIBLE_VERSE_STAGES and intents_for_verse and spiritual_consent is True:
    verse_results = self.vector_db.retrieve_verse_with_llm(
        intents=intents_for_verse,
        user_input=user_input,
        llm=self.llm,
        k=10,
        excluded_books=excluded_books,
        excluded_verses=excluded_verses
    )
```

---

## 4.6.4 Orkestrasi LLM dengan LangChain (RAGEngine)

### Pemilihan Provider LLM yang Dapat Dikonfigurasi

Sistem mengabstraksi pemilihan provider LLM melalui fungsi pabrik (*factory function*) `_build_chatbot_llm()`. Provider aktif ditentukan oleh nilai `opt.CHATBOT_LLM_PROVIDER` dalam `config.py` (nilai default: `"groq"`), dan seluruh kredensial API dimuat dari file `.env` melalui `python-dotenv`. Desain ini memungkinkan penggantian provider (misalnya dari Groq ke Gemini atau OpenAI) hanya dengan mengubah satu baris di `config.py` tanpa modifikasi kode apapun.

**File: `core/rag_engine.py`**
```python
def _build_chatbot_llm():
    """Build LLM instance based on opt.CHATBOT_LLM_PROVIDER."""
    provider = opt.CHATBOT_LLM_PROVIDER.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv(opt.GROQ_API_KEY_ENV)
        return ChatGroq(
            model=opt.GROQ_CHATBOT_MODEL,
            api_key=api_key,
            temperature=opt.CHATBOT_TEMPERATURE,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv(opt.GEMINI_API_KEY_ENV)
        return ChatGoogleGenerativeAI(
            model=opt.GEMINI_CHATBOT_MODEL,
            google_api_key=api_key,
            temperature=opt.CHATBOT_TEMPERATURE,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        # ... [kode lainnya disingkat] ...

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5:3b", base_url="http://localhost:11434", ...)
    # ...
```

### Template Prompt dan Struktur Konteks

Prompt yang dikirim ke LLM dibangun melalui `langchain_core.prompts.PromptTemplate` dengan lima variabel dinamis yang diisi ulang setiap giliran: `user_input`, `detected_intents`, `stage_instruction`, `example_answer`, dan `bible_section`. Struktur template dirancang agar LLM menerima semua konteks yang diperlukan (niat emosional pengguna, panduan perilaku tahap konseling, contoh jawaban dari basis pengetahuan, dan ayat Alkitab jika relevan) dalam satu prompt yang koheren.

**File: `core/rag_engine.py`**
```python
self.prompt_template = PromptTemplate(
    input_variables=[
        "user_input", "detected_intents",
        "stage_instruction", "example_answer", "bible_section"
    ],
    template="""Anda adalah seorang psikolog dan konselor Kristen yang berempati, hangat, dan bijaksana.
Tugas Anda adalah merespons curhatan atau pertanyaan pengguna dengan cara yang suportif dan natural.

Informasi Konteks:
- Panduan Perilaku Anda Saat Ini: {stage_instruction}
- Input Pengguna: "{user_input}"
- Emosi/Niat Terdeteksi (Multilabel): {detected_intents}

Bahan Inspirasi dari Knowledge Base:
- Contoh Respons Terdahulu (Gunakan sebagai inspirasi nada dan konten): 
"{example_answer}"

{bible_section}

Instruksi:
1. Berikan respons yang terasa natural, berempati, dan seperti percakapan sungguhan dengan psikolog.
2. Ikuti "Panduan Perilaku" yang diberikan untuk menentukan pendekatan dan nada bicara Anda.
# ... [instruksi 3-7 disingkat] ...

Respons Anda:
"""
)

# Create the LangChain processing chain
self.chain = self.prompt_template | self.llm
```

Operator `|` adalah sintaks *LCEL (LangChain Expression Language)* yang menghubungkan `PromptTemplate` ke LLM dalam satu *pipeline* yang dapat dikomposisi. Ketika `chain.invoke()` dipanggil, LangChain secara otomatis memformat template, mengirimkan hasil ke LLM, dan mengembalikan objek respons.

### Injeksi Konteks Bertahap (Stage-Aware Prompt Assembly)

Variabel `stage_instruction` tidak hanya berisi teks instruksi perilaku statis untuk setiap tahap, tetapi juga dapat diperkaya secara dinamis dengan beberapa lapisan konteks tambahan berdasarkan kondisi sesi. Proses perakitan instruksi ini terjadi dalam `generate_response()` sebelum prompt diinvokasi:

**File: `core/rag_engine.py`**
```python
# Get stage-specific behavioral instruction
stage_instruction = self.STAGE_INSTRUCTIONS.get(
    current_stage,
    self.STAGE_INSTRUCTIONS["pembahasan"]  # fallback
)

# Inject ephemeral complaint summary into later stages
if complaint_summary and current_stage in ('intervensi', 'solusi', 'relaksasi'):
    stage_instruction = f"Konteks Keluhan Klien: {complaint_summary}\n\n" + stage_instruction

# Inject the chosen relaxation technique for the relaksasi stage
if chosen_technique and current_stage == 'relaksasi':
    stage_instruction = (
        f"Klien telah memilih teknik {chosen_technique}. "
        f"Pandu klien HANYA dengan teknik tersebut. "
    ) + stage_instruction

# Pembahasan turn-3+ injection
if current_stage == 'pembahasan' and turn_in_stage >= 3:
    stage_instruction = stage_instruction + " " + self.PEMBAHASAN_TURN3_PROMPT

# Inject CBT physical symptom guidance when applicable
if has_physical_symptoms and current_stage in ('solusi', 'relaksasi'):
    stage_instruction = stage_instruction + " " + self.PHYSICAL_SYMPTOM_CBT_GUIDANCE

# Ask for spiritual consent at the FINAL turn of the solusi stage
if ask_spiritual_consent and current_stage == 'solusi':
    stage_instruction = stage_instruction + " " + self.SPIRITUAL_CONSENT_PROMPT
```

Mekanisme ini memungkinkan LLM yang pada dasarnya *stateless* (tidak memiliki memori lintas giliran) untuk tetap mengetahui konteks yang diperlukan — ringkasan keluhan, teknik yang dipilih, dan panduan perilaku yang tepat — tanpa harus mengirimkan seluruh riwayat percakapan dalam setiap prompt.

### Ringkasan Percakapan dan Ekstraksi Teknik oleh LLM

Pada dua momen transisi kritis, sistem menggunakan LLM itu sendiri untuk mengekstrak informasi ringkasan yang kemudian disimpan dalam sesi dan diinjeksikan ke prompt-prompt selanjutnya.

Transisi pertama terjadi saat sesi bergerak dari `pembahasan` ke `intervensi`: LLM dipanggil satu kali dengan seluruh riwayat percakapan `pembahasan` untuk menghasilkan satu kalimat ringkasan keluhan utama klien (`complaint_summary`).

Transisi kedua terjadi saat sesi bergerak dari `solusi` ke `relaksasi`: LLM dipanggil satu kali dengan riwayat percakapan `solusi` untuk mengekstrak nama teknik relaksasi yang disepakati (`chosen_technique`, misalnya `"Pernapasan 4-7-8"`).

**File: `core/session_manager.py`**
```python
def _advance_stage(self):
    """Move to the next stage and reset turn count."""
    # ...
    # When advancing past pembahasan, snapshot intents and generate background summary
    if self.current_stage == 'intervensi':
        self._snapshot_primary_intents()
        if self.temp_pembahasan_history:
            self.complaint_summary = self.rag_engine.generate_background_summary(
                self.temp_pembahasan_history
            )
            self.temp_pembahasan_history = []  # immediately free ephemeral data

    # When advancing to relaksasi, extract the chosen relaxation technique
    if self.current_stage == 'relaksasi':
        if self.temp_solusi_history:
            self.chosen_technique = self.rag_engine.generate_technique_extraction(
                self.temp_solusi_history
            )
            self.temp_solusi_history = []  # immediately free ephemeral data
```

Kedua riwayat sementara (`temp_pembahasan_history` dan `temp_solusi_history`) dibebaskan segera setelah ringkasan dihasilkan, agar tidak mengonsumsi memori yang tidak perlu sepanjang sisa sesi.

### Pemanggilan LLM Akhir dan Pembentukan Respons

Setelah seluruh konteks dirakit, `chain.invoke()` dieksekusi sebagai langkah terakhir dalam `generate_response()`:

**File: `core/rag_engine.py`**
```python
# 4. Build the bible section dynamically
bible_section = ""
if bible_verses:
    verse_display = f"{bible_reference} (TB) \"{bible_verses}\""
    bible_section = (
        "- Ayat Alkitab Relevan (Sertakan dan jelaskan maknanya dengan lembut "
        "dalam konteks masalah pengguna):\n"
        f'{verse_display}'
    )

# 5. Format the inputs and invoke the LLM Chain
response = self.chain.invoke({
    "user_input": user_input,
    "detected_intents": intents_str,
    "stage_instruction": stage_instruction,
    "example_answer": example_answer,
    "bible_section": bible_section
})

return {
    "response": response.content,
    "context_used": {
        "intents": detected_intents,
        "example_answer": example_answer,
        "bible_verses": f"{bible_reference} - {bible_verses}" if bible_verses else "",
        "bible_reference": bible_reference,
        "bible_book_abbr": bible_book_abbr,
        "stage": current_stage
    }
}
```

Nilai `response.content` berisi teks respons yang dihasilkan LLM, sedangkan `context_used` mengembalikan seluruh konteks yang digunakan — termasuk intent yang terdeteksi, contoh jawaban, dan referensi ayat Alkitab — untuk keperluan *debugging* dan pencatatan di konsol server.

---

## 4.6.5 Antarmuka Flask (Route `/chat`)

Permintaan dari antarmuka pengguna (browser) diterima oleh Flask melalui *endpoint* `POST /chat`. Rute ini bersifat sinkron: respons hanya dikembalikan setelah seluruh pipeline (klasifikasi → retrieval → pembangkitan LLM) selesai dieksekusi. Flask mengembalikan respons dalam format JSON dengan tiga kunci: `response` (teks respons chatbot), `show_professional_button` (sinyal ke frontend untuk menampilkan tombol referral konselor profesional), dan `session_ended` (sinyal untuk menonaktifkan *input* pengguna saat sesi selesai).

**File: `app.py`**
```python
@app.route("/chat", methods=["POST"])
def chat():
    """Receive user message and return chatbot response."""
    if not chatbot_session:
        return jsonify({"error": "Chatbot is not initialized correctly. Check terminal logs."}), 500

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_input = data["message"].strip()

    # [Bible Test Mode gate — disingkat] ...

    # Process the message through the session manager
    try:
        result = chatbot_session.chat(user_input)
        
        # Technical details (debug data) are printed to the console.
        print("\n--- Technical Debug Info ---")
        print(f"User Input: {user_input}")
        print(f"Stage: {result['debug']['stage']} -> {result['debug']['new_stage']}")
        print(f"Intents: {result['debug']['intents_this_turn']}")
        if result['debug'].get('bible_verses'):
            print(f"Bible Verses Used: {result['debug']['bible_verses']}")
        print("----------------------------\n")
        
        return jsonify({
            "response": result["response"],
            "show_professional_button": result.get("show_professional_button", False),
            "session_ended": result.get("session_ended", False),
        })
    
    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "Terjadi kesalahan internal. Silakan coba lagi."}), 500
```

Informasi *debug* internal — termasuk tahap sebelum dan sesudah transisi, intent yang terdeteksi giliran ini, dan referensi ayat Alkitab yang digunakan — dicatat ke konsol server dan **tidak dikirimkan** ke klien. Pemisahan ini menjaga respons API tetap bersih dan hanya mengandung data yang relevan bagi pengguna.

Rute `/reset` memungkinkan pengguna memulai sesi konseling baru tanpa me-*restart* server, dengan memanggil `SessionManager.reset()` yang mengembalikan seluruh atribut sesi ke nilai awalnya tanpa memuat ulang model atau indeks FAISS.

**File: `app.py`**
```python
@app.route("/reset", methods=["POST"])
def reset():
    """Reset the chatbot session."""
    if chatbot_session:
        chatbot_session.reset()
        print("\n--- Session Reset ---")
    return jsonify({"status": "success", "message": "Session reset."})
```

---

## Ringkasan Alur Data Per Giliran

Tabel berikut merangkum komponen, fungsi, dan keluaran dari setiap langkah yang terjadi dalam satu giliran percakapan (dari saat `POST /chat` diterima hingga respons JSON dikembalikan):

| # | Komponen | Fungsi | Keluaran |
|:---:|:---|:---|:---|
| 1 | `app.py` → `/chat` | Menerima JSON, validasi input, routing | `user_input` (string) |
| 2 | `SessionManager.chat()` | Manajemen tahap, akumulasi intent, override | Panggilan ke `generate_response()` dengan parameter sesi |
| 3 | `predictor.predict()` | Inferensi LABAN (IndoBERT) + sigmoid threshold | `detected_intents` (list), `scores` (dict) |
| 4 | `VectorDBManager.search_qna()` | FAISS semantic search pada indeks QnA | `example_answer` (string) |
| 5 | `VectorDBManager.retrieve_verse_with_llm()` | AVI FAISS search + LLM reranker (hanya jika `relaksasi` dan `spiritual_consent=True`) | `bible_reference`, `bible_verses` (string) |
| 6 | Perakitan `stage_instruction` | Injeksi ringkasan, teknik, CBT, consent | `stage_instruction` (string gabungan) |
| 7 | `chain.invoke()` | Invokasi LLM melalui LangChain LCEL pipeline | `response.content` (string respons final) |
| 8 | `app.py` → `jsonify()` | Pembentukan respons JSON | `{"response": ..., "show_professional_button": ..., "session_ended": ...}` |

---

*Script Utama: `app.py`, `core/rag_engine.py`, `core/session_manager.py`, `core/vector_db.py`, `models/multilabel/predict.py`*
*Backbone Aktif Klasifikasi: IndoBERT (`indobenchmark/indobert-base-p1`, 768-dim)*
*Backbone Embedding FAISS: MiniLM-multi (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384-dim)*
*Framework: PyTorch + HuggingFace Transformers + LangChain + Flask*
