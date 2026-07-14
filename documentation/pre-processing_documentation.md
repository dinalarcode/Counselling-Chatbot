# BAB 4.3 — Hasil Pre-Processing Data

---

## 4.3.1 Pre-process Dataset

Sebelum data mentah dapat digunakan untuk melatih model klasifikasi LABAN maupun untuk membangun basis pengetahuan RAG, data tersebut terlebih dahulu melalui serangkaian tahap pra-pemrosesan (*pre-processing*) yang dilakukan secara *offline*. Tahap ini mencakup tokenisasi teks, pengubahan label menjadi representasi biner, pembagian dataset, serta pembangunan indeks vektor FAISS.

### Tokenisasi Ucapan dan Label Intent

Tokenisasi dilakukan pada saat pipeline pelatihan dimuat, diimplementasikan pada `data/data_preparation.py`. Proses ini mengubah teks berbahasa Indonesia mentah menjadi rangkaian *token ID* berpanjang tetap yang dapat diproses oleh model LABAN, menggunakan `AutoTokenizer` dari HuggingFace sesuai backbone yang ditetapkan pada `opt.MODEL_NAME`.

**File: `data/data_preparation.py`**
```python
from transformers import AutoTokenizer
from config import opt

tokenizer = AutoTokenizer.from_pretrained(opt.MODEL_NAME)
```

Setiap ucapan klien (kolom `question` pada `dataset_multiintent_augmented.csv`) ditokenisasi di dalam `LABANDataset.__getitem__` dengan konfigurasi berikut:

```python
encoding = self.tokenizer(
    question,
    add_special_tokens=True,     # Menambahkan token [CLS] dan [SEP]
    max_length=opt.max_len,      # max_len = 50
    padding='max_length',        # Padding hingga tepat 50 token
    truncation=True,             # Pemotongan jika lebih dari 50 token
    return_attention_mask=True,  # Mask biner: 1 = token asli, 0 = padding
    return_tensors='pt',
)
```

Selain ucapan klien, 10 nama label intent juga ditokenisasi satu kali di awal proses (bukan per-sampel), karena arsitektur *dual encoder* LABAN memerlukan representasi token dari label itu sendiri:

```python
tokenized_intent = tokenizer(
    listof_intent,    # Daftar 10 nama intent
    padding=True,
    truncation=True,
    return_tensors='pt'
)
```

Keluaran dari tahap ini adalah sebagai berikut:

| Keluaran | Dimensi | Deskripsi |
|---|---|---|
| `input_ids` (ucapan) | `(batch_size, 50)` | Token ID dalam bentuk integer |
| `attention_mask` (ucapan) | `(batch_size, 50)` | Mask biner (1 = token asli, 0 = padding) |
| `intent_ids` (label) | `(10, max_label_len)` | Token ID untuk seluruh 10 nama intent |
| `intent_mask` (label) | `(10, max_label_len)` | Attention mask untuk token intent |

Nilai `max_len = 50` dipilih karena ucapan konseling berbahasa Indonesia umumnya singkat (1–3 kalimat). Panjang maksimum 50 token telah mencakup lebih dari 95% ucapan tanpa perlu pemotongan, sekaligus menjaga penggunaan memori tetap rendah selama pelatihan.

### Pengubahan Label ke Representasi Multi-Hot

Kolom `Intent` pada berkas CSV berupa string yang dipisahkan tanda titik koma (mis. `"Menyatakan Perasaan Sedih dan Kehilangan; Menyatakan Perasaan Takut dan Kecemasan"`). Representasi ini perlu diubah menjadi vektor biner agar dapat digunakan sebagai target pelatihan.

Langkah pertama adalah memecah string tersebut menjadi daftar label:

```python
dsqi['intent_list'] = dsqi['Intent'].apply(lambda x: x.split('; '))
```

Selanjutnya, `MultiLabelBinarizer` dari scikit-learn digunakan untuk mengonversi daftar label menjadi vektor biner:

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(dsqi['intent_list'])
listof_intent = list(mlb.classes_)  # Terurut alfabetis
```

Hasilnya kemudian digabungkan kembali dengan kolom `question` menjadi satu DataFrame:

```python
df_inten = pd.DataFrame(encoded_labels, columns=mlb.classes_).astype('float32')
dsqi_encoded = pd.concat([dsqi['question'], df_inten], axis=1)
```

Keluaran tahap ini adalah DataFrame dengan 11 kolom (1 kolom teks dan 10 kolom biner), di mana setiap baris memiliki satu atau lebih nilai `1.0` sesuai intent yang terdeteksi pada ucapan tersebut. Format multi-hot ini langsung digunakan sebagai target pelatihan untuk fungsi *loss* `BCEWithLogitsLoss`. Tipe data `float32` dipilih karena `BCEWithLogitsLoss` pada PyTorch mensyaratkan target bertipe *float*, bukan integer, sehingga menghindari kebutuhan konversi tipe data tambahan selama pelatihan.

### Pembagian Dataset (Train/Validation/Test)

Setelah dataset memiliki label multi-hot, dataset dibagi menjadi tiga subset menggunakan `train_test_split` dari scikit-learn, dilakukan dalam dua tahap pembagian:

```python
from sklearn.model_selection import train_test_split

# Pembagian 1: 80% train, 20% sisa
df_train, df_temp = train_test_split(dsqi_encoded, test_size=0.2, random_state=42)

# Pembagian 2: sisa dibagi 50/50 → 10% validasi, 10% test
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
```

| Subset | Proporsi | Kegunaan |
|---|---|---|
| `df_train` | 80% | Pelatihan model |
| `df_val` | 10% | Validasi per-epoch (pemilihan checkpoint) |
| `df_test` | 10% | Evaluasi akhir (dilaporkan dalam skripsi) |

Setiap subset kemudian dibungkus dalam objek `LABANDataset` dan dimuat melalui `DataLoader`:

```python
train_data = LABANDataset(dataframe=df_train, tokenizer=tokenizer)
val_data   = LABANDataset(dataframe=df_val, tokenizer=tokenizer)
test_data  = LABANDataset(dataframe=df_test, tokenizer=tokenizer)

train_dataload = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataload   = DataLoader(val_data,   batch_size=16, shuffle=False)
test_dataload  = DataLoader(test_data,  batch_size=16, shuffle=False)
```

Nilai `random_state=42` digunakan secara konsisten agar sampel yang sama selalu berada pada subset train/val/test yang sama di setiap kali dijalankan, sehingga hasil evaluasi dapat direproduksi.

### Pembangunan Indeks FAISS

Selain data pelatihan classifier, dua basis pengetahuan RAG — ayat Alkitab dan pasangan QnA konseling — juga melalui pra-pemrosesan berupa pembangunan indeks vektor FAISS. Proses ini dilakukan sekali pada saat server pertama kali dijalankan (*first startup*) dan hasilnya disimpan ke disk untuk dimuat ulang secara cepat pada proses berikutnya.

**Indeks Alkitab**, dibangun dari `data/alkitab_tb.csv` (31.102 ayat), menggunakan model embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row['text'],
        metadata={
            "book_abbr": row['book_abbr'],
            "book_name": row['book_name'],
            "chapter":   row['chapter'],
            "verse":     row['verse'],
            "text":      row['text'],
            "reference": row['reference'],
        }
    )
    documents.append(doc)

bible_db = FAISS.from_documents(documents, embeddings)
bible_db.save_local('data/faiss_bible_index')
```

Keluaran dari proses ini adalah direktori `data/faiss_bible_index/` yang berisi `index.faiss` (indeks L2 berisi 31.102 vektor berdimensi 384) dan `index.pkl` (pemetaan metadata ayat). Pada startup berikutnya, indeks ini dimuat langsung dari disk dalam hitungan detik melalui `FAISS.load_local()`.

**Indeks QnA** dibangun dengan proses yang sama dari `data/dataset_qna.csv` (367 pasangan tanya-jawab), dengan `page_content` diformat sebagai `f"Pertanyaan: {question}"` dan jawaban disimpan sebagai metadata, menghasilkan direktori `data/faiss_qna_index/` berisi 367 vektor berdimensi 384.

---

## 4.3.2 Pre-process Input Pengguna

Berbeda dengan pra-pemrosesan dataset yang dilakukan sekali secara *offline*, ucapan pengguna diproses secara *real-time* pada setiap giliran percakapan. Pra-pemrosesan ini difokuskan pada pembersihan teks untuk kebutuhan kueri pencarian FAISS pada retrieval ayat Alkitab, diimplementasikan pada `core/vector_db.py`.

### Penghapusan Stopword

Ucapan pengguna mentah — misalnya `"saya merasa sangat sedih banget kehilangan semangat hidup"` — umumnya memuat banyak kata fungsi dan partikel informal yang tidak berkontribusi pada makna inti kalimat. Kata-kata tersebut perlu dihapus agar kueri pencarian FAISS lebih terfokus pada kata kunci yang relevan.

Tahap pertama adalah memuat daftar stopword bahasa Indonesia dari pustaka Sastrawi:

```python
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

_sastrawi_factory = StopWordRemoverFactory()
STOPWORDS_ID = set(_sastrawi_factory.get_stop_words())
```

Karena daftar stopword bawaan Sastrawi tidak mencakup kata-kata percakapan sehari-hari (kolokial) yang umum digunakan pengguna, daftar tersebut diperluas secara manual:

```python
STOPWORDS_ID.update({
    'kak', 'nggak', 'gak', 'dong', 'sih', 'nih', 'deh', 'lho', 'kan',
    'kok', 'banget', 'kayak', 'gimana', 'gitu', 'udah', 'terus',
    'aja', 'emang', 'doang', 'cuma', 'tuh', 'yah', 'wah',
    'gatau', 'gapaham', 'gajelas', 'gamau', 'gaada', 'gabisa',
})
```

Selanjutnya, kata kunci diekstraksi dengan mengambil kata-kata yang terdiri dari minimal 3 karakter dan bukan merupakan stopword:

```python
def _get_keyword_list(self, text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS_ID]
```

Hasil dari proses ini dapat diilustrasikan sebagai berikut:

```
Input:  "saya merasa sangat sedih banget kehilangan semangat hidup"
Output: ["sedih", "kehilangan", "semangat", "hidup"]
         ↑ "saya", "merasa", "sangat", "banget" adalah stopword → dihapus
```

Kata kunci bersih hasil tahap ini kemudian digabungkan ke dalam kueri pencarian FAISS bersama dengan intent yang terdeteksi dan sinonim Alkitabiah, membantu lapisan re-ranking berbasis kata kunci pada pipeline retrieval ayat Alkitab.

### Ringkasan Alur Pra-Pemrosesan

Secara keseluruhan, dua jalur pra-pemrosesan berjalan pada waktu yang berbeda: satu kali secara *offline* untuk dataset, dan berulang secara *real-time* untuk setiap input pengguna.

```
dataset_multiintent_augmented.csv (offline, sekali per-training)
    │
    ├── [1] AutoTokenizer
    │     → input_ids (batch, 50)
    │     → attention_mask (batch, 50)
    │
    ├── [2] MultiLabelBinarizer
    │     → 10 kolom biner (float32)
    │
    └── [3] train_test_split (80/10/10)
          → train_dataload, val_dataload, test_dataload


Input Pengguna (runtime, setiap giliran percakapan)
    │
    └── [4] Penghapusan Stopword (Sastrawi + kolokial)
          → Kata kunci bersih untuk kueri FAISS
```

| Tahap | Terjadi Pada | File | Alat Utama |
|---|---|---|---|
| Tokenisasi | Waktu pelatihan | `data_preparation.py` | `AutoTokenizer` |
| Multi-Hot Encoding | Waktu pelatihan | `data_preparation.py` | `MultiLabelBinarizer` (scikit-learn) |
| Pembagian Dataset | Waktu pelatihan | `data_preparation.py` | `train_test_split` (scikit-learn) |
| Pembangunan Indeks FAISS | Startup pertama | `vector_db.py` | FAISS + MiniLM (384 dimensi) |
| Penghapusan Stopword | Runtime (setiap giliran) | `vector_db.py` | PySastrawi + daftar kolokial kustom |

---

*Script Utama: `data/data_preparation.py`, `core/vector_db.py`*
*Dataset Sumber: `data/augmentation/dataset_multiintent_augmented.csv`, `data/alkitab_tb.csv`, `data/dataset_qna.csv`*
