# BAB 4.7.2 — Evaluasi RAGAS (Retrieval-Augmented Generation Assessment)

---

## 4.7.2.1 Latar Belakang dan Tujuan Evaluasi

Evaluasi kualitas sistem *Retrieval-Augmented Generation* (RAG) pada chatbot konseling Alkitab tidak dapat diukur secara memadai hanya melalui metrik klasifikasi intent (lihat Sub-bab 4.5). Diperlukan pengukuran terpisah terhadap dua komponen inti pipeline RAG: (1) kualitas *retriever* — apakah konteks yang diambil dari indeks FAISS relevan dan cukup, dan (2) kualitas *generator* — apakah respons LLM benar-benar berpijak pada konteks tersebut dan menjawab kebutuhan pengguna. Untuk kebutuhan ini, penelitian menggunakan **RAGAS** (Es et al., 2024), sebuah kerangka evaluasi *open-source* yang menilai kedua komponen menggunakan pendekatan **LLM-as-a-Judge**.

RAGAS dipilih karena tidak bergantung pada metrik kesamaan teks permukaan seperti BLEU atau ROUGE, melainkan menilai kualitas semantik melalui model bahasa independen yang membaca input pengguna, konteks yang diambil, respons yang dihasilkan, dan jawaban rujukan (*reference*).

---

## 4.7.2.2 Empat Metrik yang Dievaluasi

| Metrik | Aspek yang Dinilai | Komponen Pipeline |
|---|---|---|
| **Faithfulness** | Apakah klaim dalam respons yang dihasilkan didukung oleh konteks yang diambil (bebas halusinasi)? | Generator |
| **Answer Relevancy** | Apakah respons secara langsung menjawab pertanyaan/keluhan pengguna? | Generator |
| **Context Precision** | Apakah dokumen yang diambil relevan dengan query pengguna? | Retriever |
| **Context Recall** | Apakah konteks yang diambil memuat informasi yang cukup untuk menghasilkan jawaban rujukan? | Retriever |

Setiap metrik dinilai pada rentang **0.0–1.0**. Perhitungan dilakukan oleh LLM *judge*, bukan oleh pipeline chatbot itu sendiri, untuk menghindari bias evaluasi diri (*self-evaluation bias*).

---

## 4.7.2.3 Adaptasi RAGAS terhadap Arsitektur Sistem

RAGAS versi standar mengasumsikan pipeline RAG generik dengan satu sumber retrieval dan dataset evaluasi yang sudah lengkap. Chatbot ini memiliki empat karakteristik yang memerlukan penyesuaian terhadap alur evaluasi standar tersebut.

### a. Tidak Ada Dataset Rujukan Siap Pakai

RAGAS membutuhkan empat kolom: `user_input`, `retrieved_contexts`, `response`, dan `reference`. Karena "jawaban ideal" konseling bersifat kontekstual terhadap tahap sesi, dataset ini dibangun melalui dua fase pada `evaluation/eval_ragas.py`:

- **Fase 1 (`--generate`):** Mengambil `RAGAS_TESTSET_SIZE` (30) sampel acak (`random_state=42`) dari `data/dataset_qna.csv`, memberi label tahap konseling otomatis melalui heuristik regex (`classify_stage()`), lalu menyimpan templat kosong ke `evaluation/data/ragas_testset.csv`.
- **Fase 2 (manual):** Kolom `reference` (jawaban ideal per pertanyaan) diisi secara manual sebelum evaluasi dijalankan.

### b. Retrieval Dua Sumber (Dual FAISS)

Berbeda dari asumsi satu *vector store*, chatbot mengambil konteks dari dua indeks FAISS independen:

| Sumber | Fungsi | Aktif pada Tahap |
|---|---|---|
| **Indeks QnA** | Contoh jawaban konselor sebagai inspirasi nada respons | Seluruh tahap |
| **Indeks Alkitab (AVI)** | Ayat Alkitab paling relevan sebagai bimbingan spiritual | `solusi`, `relaksasi`, `bantuan_profesional` (bergantung *spiritual consent*) |

Pada Fase 2, `retrieved_contexts` dibangun dengan menjalankan `RAGEngine.generate_response()` secara langsung per sampel, sehingga konteks yang dievaluasi adalah hasil retrieval aktual, bukan data statis yang disuntikkan manual.

### c. Perilaku Bergantung Tahap Konseling

`SessionManager` menjalankan enam tahap dengan aturan perilaku berbeda (lihat Sub-bab 4.6.2). Skor RAGAS global tidak dapat mengungkap apakah chatbot melanggar aturan tahap tertentu (misalnya menyisipkan ayat Alkitab pada tahap `pembukaan`). Karena itu, `eval_ragas.py` menghasilkan agregasi tambahan per tahap (`ragas_per_stage.csv`) di samping skor global.

### d. Judge LLM Independen dari Generator

LLM generator chatbot (`opt.CHATBOT_LLM_PROVIDER`) dan LLM *judge* RAGAS dikonfigurasi sebagai model yang **terpisah** untuk menghindari bias evaluasi diri:

| Peran | Model Aktif | Sumber Konfigurasi |
|---|---|---|
| **Generator** (chatbot) | Groq `llama-3.3-70b-versatile` | `opt.CHATBOT_LLM_PROVIDER = "groq"`, `opt.GROQ_CHATBOT_MODEL` |
| **Judge** (evaluator RAGAS) | OpenAI `gpt-4o-mini` | `opt.RAGAS_JUDGE_MODEL`, diinisialisasi via `langchain_openai.ChatOpenAI` |

Karena kedua model berasal dari penyedia (*provider*) dan keluarga model yang berbeda, penilaian tidak dipengaruhi oleh kecenderungan model menilai keluarannya sendiri secara positif.

---

## 4.7.2.4 Prosedur Eksekusi

```bash
cbenv\Scripts\activate

# Fase 1 — hasilkan templat testset
python evaluation/eval_ragas.py --generate

# (manual) isi kolom `reference` pada evaluation/data/ragas_testset.csv

# Fase 2 — jalankan evaluasi penuh
python evaluation/eval_ragas.py --evaluate
```

Fase 2 memvalidasi bahwa seluruh kolom `reference` telah terisi, menjalankan pipeline RAG penuh (klasifikasi → retrieval dual-FAISS → generasi LLM) untuk setiap sampel, menyusun `EvaluationDataset` RAGAS, lalu menjalankan keempat metrik dengan *judge* `gpt-4o-mini`. Tiga berkas keluaran disimpan ke `evaluation/results/`:

| Berkas | Isi | Jumlah Baris |
|---|---|---|
| `ragas_per_sample.csv` | Skor per sampel untuk seluruh kasus uji | 30 |
| `ragas_summary.csv` | Statistik agregat (mean, median, std, min, max) | 4 (satu per metrik) |
| `ragas_per_stage.csv` | Rata-rata skor per tahap konseling | Hingga 6 (satu per tahap) |

---

## 4.7.2.5 Hasil Evaluasi

Evaluasi dijalankan terhadap **30 sampel** (`n=30`, `random_state=42`). Hasil agregat global tercantum pada Tabel 4.7.2.1.

**Tabel 4.7.2.1 — Ringkasan Skor RAGAS Global (`ragas_summary.csv`)**

| Metrik | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| Faithfulness | 0.2450 | 0.20 | 0.2682 | 0.00 | 0.80 |
| Answer Relevancy | 0.0153 | 0.00 | 0.0673 | 0.00 | 0.3575 |
| Context Precision | 0.8667 | 0.9999 | 0.3457 | 0.00 | 0.9999 |
| Context Recall | 0.7000 | 0.8333 | 0.3646 | 0.00 | 1.00 |

**Tabel 4.7.2.2 — Skor RAGAS per Tahap Konseling (`ragas_per_stage.csv`)**

| Tahap | n | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---:|---:|---:|---:|---:|
| `pembahasan` | 24 | 0.2000 | 0.0000 | 0.8750 | 0.7083 |
| `pembukaan` | 2 | 0.7083 | 0.0512 | 0.5000 | 1.0000 |
| `penutupan` | 1 | 0.0000 | 0.0000 | 0.9999 | 0.0000 |
| `relaksasi` | 1 | 0.0000 | 0.3575 | 0.9999 | 0.0000 |
| `solusi` | 2 | 0.5667 | 0.0000 | 0.9999 | 1.0000 |

### Interpretasi

**Context Precision (0.8667) dan Context Recall (0.7000)** tergolong baik hingga sangat baik. Ini mengindikasikan bahwa lapisan *retriever* — embedding `paraphrase-multilingual-MiniLM-L12-v2`, ekspansi sinonim Alkitabiah (`BIBLICAL_SYNONYMS`), dan *LLM reranker* pada `retrieve_verse_with_llm()` — secara konsisten mengambil dokumen yang relevan terhadap query pengguna.

**Faithfulness (0.2450) dan Answer Relevancy (0.0153)** berada jauh di bawah ambang yang dapat diterima. Nilai ini **tidak dapat ditafsirkan sebagai chatbot berhalusinasi secara luas atau memberi jawaban tidak relevan**, karena pemeriksaan silang terhadap `ragas_per_sample.csv` diperlukan sebelum penarikan kesimpulan tersebut. Kandidat penyebab yang lebih mungkin, mengingat arsitektur sistem:

1. **Gaya respons konseling versus asumsi RAGAS.** Metrik Faithfulness RAGAS mengekstraksi klaim faktual diskrit dari respons dan mencocokkannya terhadap konteks. Respons konseling bersifat empatik, terbuka, dan reflektif (mis. "Saya mengerti perasaanmu...") — bukan pernyataan faktual yang dapat diverifikasi terhadap satu dokumen sumber tunggal. Struktur respons semacam ini secara sistematis dinilai rendah oleh metrik yang dirancang untuk QA faktual.
2. **Domain Bahasa Indonesia.** RAGAS dan model *judge* `gpt-4o-mini` dirancang serta diuji dominan pada teks Bahasa Inggris; ekstraksi klaim dan penilaian relevansi pada teks Bahasa Indonesia informal berisiko kurang presisi.
3. **`reference` tidak selalu selaras dengan gaya respons aktual chatbot,** sehingga Answer Relevancy — yang membandingkan makna semantik respons terhadap `user_input` melalui pertanyaan buatan (*generated questions*) — dapat menghasilkan skor rendah jika respons chatbot lebih panjang/kontekstual daripada yang diasumsikan pertanyaan buatan tersebut.

**Variasi antar tahap** juga signifikan: `pembukaan` mencatat Faithfulness tertinggi (0.7083), yang wajar karena respons pada tahap ini bersifat sapaan singkat dengan klaim minimal untuk diverifikasi. Tahap `pembahasan` (n=24, mayoritas sampel) mendominasi rata-rata global dan mencatat Answer Relevancy 0.0000 di seluruh sampelnya — pola yang konsisten dengan poin 1 di atas, bukan indikasi kegagalan retrieval, karena Context Precision pada tahap yang sama tetap tinggi (0.8750).

Ukuran sampel per tahap (`n=1` untuk `relaksasi` dan `penutupan`) terlalu kecil untuk kesimpulan statistik yang kuat pada tahap-tahap tersebut dan sebaiknya dibaca sebagai indikasi awal, bukan hasil final.

---

## 4.7.2.6 Saran Pengembangan untuk Skor RAGAS Ideal

Berdasarkan pola hasil pada Sub-bab 4.7.2.5, berikut rekomendasi teknis dan konkret untuk pengembangan lanjutan, disusun berdasarkan komponen arsitektur yang relevan.

### a. Faithfulness dan Answer Relevancy (Generator)

1. **Perkaya kolom `reference` pada `ragas_testset.csv`** agar mencerminkan gaya respons empatik-konseling yang sesungguhnya (bukan jawaban faktual singkat), sehingga metrik Answer Relevancy — yang bergantung pada kemiripan semantik terhadap pertanyaan buatan dari `reference` — tidak salah menilai respons yang secara klinis tepat tetapi secara struktural berbeda dari asumsi RAGAS.
2. **Pertimbangkan metrik RAGAS alternatif yang lebih sesuai untuk domain dialogis/konseling**, seperti `ResponseRelevancy` dengan *embedding* multibahasa kustom, atau metrik berbasis rubrik (`AspectCritic`) yang dapat didefinisikan khusus untuk menilai empati dan kesesuaian tahap, alih-alih memaksakan metrik faktual QA generik.
3. **Uji *judge* LLM alternatif yang lebih kuat pada Bahasa Indonesia** (mis. `gpt-4o` penuh, atau model dengan performa multibahasa lebih tinggi) untuk memvalidasi apakah skor rendah bersumber dari keterbatasan ekstraksi klaim oleh `gpt-4o-mini`, bukan dari kualitas respons generator itu sendiri.
4. **Perketat instruksi *prompt template*** (`core/rag_engine.py`, `self.prompt_template`) agar LLM generator secara eksplisit merujuk kembali ke `example_answer` dan `bible_section` dalam responsnya (mis. parafrase langsung sebagian isi ayat), meningkatkan keterlacakan klaim terhadap konteks yang diambil.

### b. Context Precision dan Context Recall (Retriever)

Skor retriever sudah baik (0.87 dan 0.70), namun beberapa peningkatan tetap relevan mengingat n=30 masih tergolong kecil:

1. **Perluas ukuran dan keragaman `ragas_testset.csv`** (`opt.RAGAS_TESTSET_SIZE`) di atas 30 sampel, dengan proporsi tahap yang lebih seimbang (saat ini `pembahasan` mendominasi 24/30 sampel), agar estimasi Context Precision/Recall per tahap — khususnya `solusi` dan `relaksasi` yang melibatkan retrieval Alkitab — lebih dapat diandalkan secara statistik.
2. **Tinjau ulang cakupan `data/verse_retrieval/alkitab_tb_enriched_groq.csv` (AVI)** untuk pasal-pasal dengan ringkasan kontekstual yang tipis, karena Context Recall bergantung pada kelengkapan informasi dalam dokumen yang diambil, bukan hanya relevansi topikalnya.
3. **Evaluasi kamus `BIBLICAL_SYNONYMS`** (`core/vector_db.py`) untuk intent yang belum tercakup skenario RAGAS ini, guna memastikan ekspansi query tetap efektif di luar sampel uji saat ini.

### c. Metodologi Evaluasi

1. **Jalankan analisis `ragas_per_sample.csv` secara manual** untuk mengidentifikasi pola kegagalan konkret (mis. apakah skor Faithfulness rendah terkonsentrasi pada respons panjang, atau pada intent tertentu), sebagai dasar prioritas perbaikan yang lebih terarah daripada perbaikan menyeluruh.
2. **Tambahkan metrik pelengkap non-RAGAS untuk validasi kepatuhan tahap** (mis. pemeriksaan otomatis apakah ayat Alkitab muncul di luar `BIBLE_VERSE_STAGES`), karena RAGAS tidak dirancang untuk menilai kepatuhan terhadap aturan *state machine* `SessionManager`.
3. **Ulangi evaluasi Fase 2 setelah setiap perubahan signifikan** pada `prompt_template`, `BIBLICAL_SYNONYMS`, atau `MODEL_NAME` embedding, untuk memastikan perbaikan pada satu metrik tidak menurunkan metrik lain (*regression check*).

---

*Skrip evaluasi: `evaluation/eval_ragas.py` | Hasil: `evaluation/results/ragas_summary.csv`, `ragas_per_stage.csv`, `ragas_per_sample.csv`*
*Generator: Groq `llama-3.3-70b-versatile` | Judge: OpenAI `gpt-4o-mini` | Ukuran sampel: n=30, seed=42*
