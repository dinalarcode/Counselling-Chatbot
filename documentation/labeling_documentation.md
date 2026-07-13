# BAB 4.2 — Hasil Pengumpulan dan Pelabelan Data

---

## 4.2.1 Ekstraksi Data Konseling

Seluruh data konseling asli yang digunakan dalam penelitian ini bersumber dari satu berkas induk, yaitu `Data 300825.xlsx`, sebuah spreadsheet Excel yang memuat data percakapan konseling nyata dari layanan konseling sebuah universitas. Berkas ini kemudian diekstraksi dan direstrukturisasi menjadi dua berkas CSV terpisah yang digunakan pada dua komponen sistem yang berbeda: `dataset_qna.csv` untuk kebutuhan retrieval RAG, dan `dataset_multiintent.csv` untuk kebutuhan pelatihan model klasifikasi multi-intent LABAN.

### Dataset QnA Konseling (`dataset_qna.csv`)

Dataset ini menyediakan contoh respons konselor yang digunakan oleh pipeline RAG. Ketika pengguna mengirimkan sebuah pesan, sistem mencari pertanyaan paling mirip dalam dataset ini dan menggunakan jawaban yang berpadanan sebagai konteks tambahan bagi LLM.

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `question` | string | Ucapan (*utterance*) klien konseling |
| `answer` | string | Respons konselor |

**Statistik:**
- **Total data**: 367 pasangan tanya-jawab
- **Encoding**: UTF-8-sig
- **Indeks turunan**: `data/faiss_qna_index/` (indeks FAISS L2 menggunakan embedding MiniLM)

### Dataset Multi-Intent (`dataset_multiintent.csv`)

Dataset ini menjadi data pelatihan awal bagi classifier multi-label LABAN. Setiap ucapan klien diberi anotasi satu atau lebih dari 10 label intent yang telah dirancang untuk mencakup spektrum emosi yang ditemui dalam konteks konseling pastoral/Alkitabiah.

| # | Label Intent | Deskripsi | Contoh Ucapan |
|---|---|---|---|
| 1 | **Mengisyaratkan Butuh Bantuan Profesional** | Permintaan bantuan konseling, rujukan, atau pengakuan tidak mampu menangani sendirian | "Aku rasa aku butuh bicara sama psikolog deh" |
| 2 | **Mengisyaratkan Gejala Fisik** | Keluhan fisik seperti sakit kepala, sesak napas, gemetar, susah tidur, atau kelelahan | "Tiap malam aku susah tidur dan sering pusing" |
| 3 | **Menyatakan Perasaan Benci dan Jijik** | Ekspresi kebencian, jijik, penolakan, rasa enggan, atau perasaan muak | "Aku benci banget sama dia, muak lihat mukanya" |
| 4 | **Menyatakan Perasaan Marah dan Frustasi** | Ekspresi kemarahan, frustrasi, kesal, jengkel, atau ketidaksabaran | "Kesel banget sih, udah capek ngomong tapi nggak didengerin" |
| 5 | **Menyatakan Perasaan Percaya** | Ekspresi kepercayaan, keterbukaan, kemauan mencoba saran, atau kesediaan mengikuti proses | "Aku percaya ini bisa membaik, mau coba sarannya" |
| 6 | **Menyatakan Perasaan Sebelum Menghadapi Kejadian** | Antisipasi, persiapan mental, niat, atau rencana sebelum menghadapi tantangan | "Besok aku harus presentasi, deg-degan banget" |
| 7 | **Menyatakan Perasaan Sedih dan Kehilangan** | Ekspresi kesedihan, kehilangan, duka, merasa tidak berharga, putus asa, atau kesepian | "Rasanya sepi banget, kayak nggak ada yang peduli" |
| 8 | **Menyatakan Perasaan Takut dan Kecemasan** | Ekspresi takut, khawatir, cemas, panik, gugup, atau perasaan terancam | "Aku takut banget kalau ternyata gagal lagi" |
| 9 | **Menyatakan Rasa Syukur dan Apresiasi** | Ekspresi rasa syukur, lega, apresiasi, perasaan membaik, atau terima kasih | "Alhamdulillah, aku merasa lebih baik sekarang" |
| 10 | **Menyatakan Reaksi Terkejut dan Tidak Terduga** | Ekspresi kaget, tidak percaya, bingung atas sesuatu yang tak terduga | "Serius?! Aku nggak nyangka sama sekali" |

Satu ucapan dapat mengekspresikan **lebih dari satu intent sekaligus** (anotasi multi-label), sebagaimana diilustrasikan berikut ini:

```
Ucapan: "Aku sedih banget kehilangan dia, tapi juga takut harus jalan sendiri"
Intent:  Menyatakan Perasaan Sedih dan Kehilangan; Menyatakan Perasaan Takut dan Kecemasan
```

Dalam berkas CSV, label-label intent tersebut dipisahkan dengan tanda titik koma (`;`). Selama tahap prapemrosesan, string ini dikonversi menjadi vektor biner *multi-hot*.

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `question` | string | Ucapan klien (asli maupun hasil augmentasi LLM) |
| `Intent` | string | Label intent yang dipisahkan tanda titik koma, mis. `"Sedih dan Kehilangan; Takut dan Kecemasan"` |

**Statistik:** Encoding UTF-8-sig.

---

## 4.2.2 Penyesuaian Data Ayat Alkitab

### Sumber dan Metode Pengumpulan

Data ayat Alkitab yang menjadi basis pengetahuan (*knowledge base*) pipeline RAG diperoleh melalui *web scraping* terhadap situs **[alkitab.mobi](http://alkitab.mobi/tb/)**, sebuah platform digital yang menyediakan teks Alkitab Terjemahan Baru (TB) berbahasa Indonesia secara daring. Terjemahan Baru (TB) dipilih karena merupakan versi Alkitab bahasa Indonesia yang paling banyak digunakan oleh gereja-gereja Protestan di Indonesia, sehingga relevan dengan konteks pengguna target sistem konseling ini.

Seluruh proses scraping diimplementasikan dalam satu skrip Python, `data/scrape_alkitab.py`, yang dirancang untuk berjalan satu kali dan menghasilkan berkas basis data ayat Alkitab lengkap dalam format CSV. Skrip ini sepenuhnya berjalan secara lokal tanpa ketergantungan pada layanan berbayar maupun API eksternal, dengan memanfaatkan pustaka **`requests`** untuk mengirimkan permintaan HTTP GET (dilengkapi header `User-Agent` dan *timeout* 15 detik) serta **`BeautifulSoup`** (`bs4`) untuk mem-*parsing* struktur DOM halaman HTML.

```python
response = requests.get(url, timeout=15, headers={
    "User-Agent": "Mozilla/5.0 (Bible Research Bot - Academic Project)"
})
```

Pola URL yang digunakan untuk mengakses setiap pasal mengikuti format `http://alkitab.mobi/tb/{singkatan_buku}/{nomor_pasal}/` — misalnya `http://alkitab.mobi/tb/Mzm/23/` untuk Mazmur pasal 23. Cakupan scraping meliputi **66 buku** Alkitab (39 Perjanjian Lama dan 27 Perjanjian Baru) sesuai kanon Protestan, mencakup total **1.189 pasal**.

### Mekanisme Ekstraksi dan Rate Limiting

Fungsi utama `run_scraper()` melakukan iterasi bersarang atas seluruh buku dan pasal. Setelah setiap pasal selesai diproses, skrip menunggu **0,5 detik** (`DELAY_SECONDS = 0.5`) sebelum melanjutkan ke pasal berikutnya guna menghindari pemblokiran oleh server sumber sekaligus menjaga kelancaran server yang bukan milik peneliti.

```python
for book_abbr, book_name, num_chapters in BOOKS:
    for chapter in range(1, num_chapters + 1):
        verses = scrape_chapter(book_abbr, chapter)
        ...
        time.sleep(DELAY_SECONDS)
```

Setiap halaman pasal diproses oleh fungsi `scrape_chapter()`. Situs alkitab.mobi merepresentasikan setiap ayat sebagai satu paragraf `<p>` yang memuat elemen `<span class="reftext">` (nomor ayat) beserta teks ayat itu sendiri. Paragraf yang bukan ayat — seperti paragraf tersembunyi, paragraf *loading/error*, dan judul seksi — dilewati secara eksplisit sebelum teks ayat diekstraksi dan dinormalisasi *whitespace*-nya menggunakan ekspresi reguler.

Skrip ini bersifat *fault-tolerant*: apabila sebuah pasal gagal diambil (karena *timeout*, kesalahan jaringan, atau respons HTTP non-2xx), proses tidak berhenti melainkan mencatat pasal tersebut dalam daftar `failed_chapters` dan melanjutkan ke pasal berikutnya, dengan daftar kegagalan dilaporkan di akhir proses untuk pengulangan manual apabila diperlukan.

### Struktur Data Keluaran (`alkitab_tb.csv`)

Hasil scraping disimpan dalam satu berkas CSV di direktori `data/`, memuat enam kolom sebagai berikut:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `book_abbr` | string | Singkatan buku sesuai konvensi alkitab.mobi (mis. `"Mzm"`, `"Yoh"`) |
| `book_name` | string | Nama buku lengkap dalam bahasa Indonesia (mis. `"Mazmur"`, `"Yohanes"`) |
| `chapter` | integer | Nomor pasal |
| `verse` | integer | Nomor ayat dalam pasal tersebut |
| `text` | string | Teks ayat dalam bahasa Indonesia (Terjemahan Baru) |
| `reference` | string | Referensi ayat terformat (mis. `"Mazmur 23:1"`), dibangun secara programatik dari `book_name`, `chapter`, dan `verse` sebagai *human-readable identifier* |

**Statistik:**
- **Total ayat**: 31.102 ayat
- **Cakupan**: 66 buku (kanon Protestan lengkap)
- **Terjemahan**: Terjemahan Baru (TB)
- **Ukuran berkas**: ~5,7 MB
- **Encoding**: UTF-8

### Penyesuaian Menuju Indeks Vektor

Data mentah `alkitab_tb.csv` tidak langsung digunakan dalam format CSV oleh mesin RAG, melainkan diproses lebih lanjut menjadi indeks vektor FAISS yang disimpan di `data/faiss_bible_index/`. Setiap teks ayat diubah menjadi vektor numerik (*embedding*) menggunakan model embedding MiniLM, sehingga memungkinkan pencarian semantik — yaitu pencarian ayat yang maknanya paling dekat dengan suatu kueri, bukan hanya yang mengandung kata-kata yang sama secara harfiah. Dengan demikian, `alkitab_tb.csv` berfungsi sebagai *single source of truth* untuk seluruh konten Alkitab dalam sistem, di mana integritas hasil scraping — termasuk ketepatan nomor ayat dan kelengkapan referensi — secara langsung menentukan kualitas retrieval ayat pada tahap `solusi` dan `relaksasi` dalam alur konseling.

---

## 4.2.3 Augmentasi Data Train Klasifikasi

Dataset `dataset_multiintent.csv` hasil ekstraksi awal (Sub-bab 4.2.1) memiliki distribusi label yang tidak seimbang antar-intent. Untuk mengatasi hal ini, dataset tersebut diperluas menjadi **`dataset_multiintent_augmented.csv`** — dataset aktif yang digunakan untuk pelatihan model LABAN — melalui sebuah pipeline augmentasi berbasis LLM yang terdiri atas empat fase.

### Fase 1: Pengumpulan Seed

Kalimat contoh (*seed*) untuk setiap intent diekstraksi dari dataset asli dan disimpan dalam `data/augmentation/seeds.json`. Kalimat-kalimat ini berfungsi sebagai contoh *few-shot* bagi LLM pada fase augmentasi berikutnya.

### Fase 2: Augmentasi LLM Single-Label

Dijalankan melalui skrip `data/augmentation/run_augmentation.py`, menggunakan LLM Groq API (`llama-3.1-8b-instant`) dengan metode *few-shot prompting* — LLM menerima 5–8 contoh seed per intent dan menghasilkan ucapan baru yang sesuai dengan intent tersebut. Target generasi berkisar 300–400 sampel per intent (target dapat dikonfigurasi per-intent), diproses secara *batch* 20 kalimat per panggilan API disertai jeda untuk *rate-limiting*. Keluaran mentah disimpan pada `data/augmentation/raw_generated/*.txt`.

### Fase 3: Augmentasi Kombinasi Multi-Label

Dijalankan melalui skrip `data/augmentation/augment_multilabel.py`, menggunakan LLM yang sama untuk menghasilkan ucapan yang mengekspresikan **2–3 intent sekaligus** (misalnya kombinasi "Sedih + Takut"). Keluaran diformat dengan pemisah pipa (`|`) yang kemudian diuraikan menjadi kalimat-kalimat individual. Kombinasi intent yang dihasilkan dikonfigurasi melalui `COMBINATION_CONFIG`, yang menentukan pasangan intent mana yang dikombinasikan beserta jumlah sampel yang dihasilkan.

### Fase 4: Penggabungan ke Format Pelatihan

Dijalankan melalui skrip `data/augmentation/merge_to_training.py`, yang mengonversi data augmentasi berformat lebar (satu kolom per intent) kembali ke format `(question, Intent)` yang digunakan oleh pipeline pelatihan.

### Deduplikasi Berbasis Kesamaan Jaccard

Kedua skrip augmentasi (Fase 2 dan Fase 3) menerapkan deduplikasi berbasis kesamaan Jaccard untuk mencegah ucapan-ucapan yang nyaris duplikat:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
Threshold = 0,75
```

Apabila sebuah kalimat baru hasil generasi memiliki tumpang tindih token ≥ 75% dengan kalimat yang sudah ada, kalimat tersebut dibuang. Pemeriksaan deduplikasi dilakukan terhadap tiga sumber: (1) seluruh kalimat seed asli, (2) seluruh kalimat yang telah dihasilkan dalam *batch* saat ini, dan (3) seluruh kalimat akumulasi dari proses augmentasi sebelumnya.

### Ringkasan Alur Augmentasi

```
dataset_multiintent.csv (utterance asli + intent)
    │
    ├──→ seeds.json (ekstraksi seed)
    │        │
    │        ├──→ run_augmentation.py       (augmentasi single-label)
    │        └──→ augment_multilabel.py     (augmentasi kombinasi multi-label)
    │                 │
    │                 └──→ intent_content_augmented.csv
    │                          │
    │                          └──→ merge_to_training.py
    │                                   │
    └───────────────────────────────────┘
              │
              └──→ dataset_multiintent_augmented.csv  (data pelatihan aktif)
```

Berkas `dataset_multiintent_augmented.csv` inilah yang menjadi sumber data pelatihan aktif bagi classifier LABAN sebagaimana ditetapkan dalam `config.py`, menggantikan `dataset_multiintent.csv` yang hanya berperan sebagai data seed awal.

---

*Sumber Data: `Data 300825.xlsx` (data konseling universitas), `alkitab.mobi` (Alkitab Terjemahan Baru)*
*Script Utama: `data/scrape_alkitab.py`, `data/augmentation/run_augmentation.py`, `data/augmentation/augment_multilabel.py`, `data/augmentation/merge_to_training.py`*
*Dataset Aktif: `data/augmentation/dataset_multiintent_augmented.csv` (pelatihan klasifikasi), `data/alkitab_tb.csv` (basis ayat Alkitab)*
