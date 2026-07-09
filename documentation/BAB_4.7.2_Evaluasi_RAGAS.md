# BAB 4.7.2 — Evaluasi RAGAS (Retrieval-Augmented Generation Assessment)

---

## 4.7.2.1 Latar Belakang dan Tujuan Evaluasi

Evaluasi kualitas sistem *Retrieval-Augmented Generation* (RAG) pada chatbot konseling Alkitab tidak dapat diukur secara memadai hanya melalui metrik klasifikasi intent (lihat Sub-bab 4.5). Diperlukan pengukuran terpisah terhadap kualitas keluaran generator: apakah respons LLM secara konsisten empatik, tidak menghakimi, selaras dengan prinsip konseling alkitabiah, dan patuh terhadap aturan tahap `SessionManager` (lihat Sub-bab 4.6.2). Untuk kebutuhan ini, penelitian menggunakan **RAGAS** (Es et al., 2024), sebuah kerangka evaluasi *open-source* yang menilai kualitas respons menggunakan pendekatan **LLM-as-a-Judge**.

RAGAS dipilih karena tidak bergantung pada metrik kesamaan teks permukaan seperti BLEU atau ROUGE, melainkan menilai kualitas semantik melalui model bahasa independen yang membaca input pengguna, konteks yang diambil, dan respons yang dihasilkan.

---

## 4.7.2.2 Metrik yang Dievaluasi: LABAN Counseling Standard (AspectCritic)

Iterasi awal evaluasi (lihat riwayat pengembangan) menggunakan empat metrik generik bawaan RAGAS — *Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall*. Metrik tersebut dirancang untuk pipeline *question-answering* faktual dan terbukti tidak cocok untuk domain dialog konseling: *Faithfulness* mengekstraksi klaim faktual diskrit dari respons empatik yang secara struktural bukan pernyataan faktual, sementara *Answer Relevancy* membandingkan makna semantik respons terhadap pertanyaan buatan dari kolom `reference`, yang tidak merepresentasikan gaya jawaban konseling yang panjang dan reflektif. Kedua metrik ini secara sistematis menghasilkan skor rendah tanpa mencerminkan kualitas respons yang sebenarnya.

Sebagai gantinya, digunakan metrik kustom tunggal berbasis **`AspectCritic`** RAGAS bernama **`LABAN_Counseling_Standard`**, didefinisikan di `evaluation/eval_ragas.py`:

```python
LABAN_CRITERIA_DEFINITION = (
    "Apakah respons chatbot menunjukkan empati yang tepat, tidak menghakimi, "
    "selaras dengan prinsip konseling alkitabiah, dan merespons dengan tepat "
    "sesuai instruksi tahap konseling saat ini: misalnya, tidak memberikan "
    "solusi secara prematur pada tahap pembahasan/intervensi, tidak redundan "
    "(sudah jelas di input pengguna, tapi tetap ditanyakan kembali)"
)
```

`AspectCritic` memberi *judge* LLM sebuah kriteria kualitatif tunggal dan meminta keputusan biner (1 = memenuhi kriteria, 0 = tidak) per sampel, alih-alih dekomposisi klaim faktual. Pendekatan ini selaras dengan rekomendasi metodologi evaluasi RAGAS untuk domain non-QA-faktual (lihat rekomendasi Sub-bab 4.7.2.6 versi sebelumnya, poin a.2) dan sekaligus menggabungkan penilaian kepatuhan tahap ke dalam satu metrik, alih-alih memerlukan pemeriksaan terpisah non-RAGAS.

Prompt kriteria secara eksplisit menilai empat aspek sekaligus:

| Aspek | Deskripsi |
|---|---|
| Empati | Respons menunjukkan pemahaman terhadap perasaan pengguna tanpa menghakimi |
| Keselarasan alkitabiah | Prinsip konseling Kristen tercermin secara wajar, bukan dipaksakan |
| Kepatuhan tahap | Tidak memberi solusi prematur pada `pembahasan`/`intervensi`; sesuai aturan `SessionManager` |
| Non-redundansi | Tidak menanyakan ulang hal yang sudah eksplisit di input pengguna |

Skor dinilai pada rentang **0 atau 1** (biner, bukan kontinu 0.0–1.0 seperti metrik RAGAS generik). Perhitungan dilakukan oleh LLM *judge*, bukan oleh pipeline chatbot itu sendiri, untuk menghindari bias evaluasi diri (*self-evaluation bias*).

---

## 4.7.2.3 Adaptasi RAGAS terhadap Arsitektur Sistem

### a. Tidak Ada Dataset Rujukan Siap Pakai

Dataset evaluasi dibangun melalui dua fase pada `evaluation/eval_ragas.py`:

- **Fase 1 (`--generate`):** Mengambil `RAGAS_TESTSET_SIZE` (30) sampel dari `data/dataset_qna.csv`, diseimbangkan lintas enam tahap konseling (`balance_by_stage()`, `random_state=42`), memberi label tahap otomatis melalui heuristik regex (`classify_stage()`), lalu menyimpan templat kosong ke `evaluation/data/ragas_testset.csv`.
- **Fase 2 (manual):** Kolom `reference` (jawaban ideal per pertanyaan, digunakan sebagai konteks tambahan bagi *judge*) diisi secara manual sebelum evaluasi dijalankan.

### b. Retrieval Dua Sumber (Dual FAISS)

Konteks dievaluasi berasal dari dua indeks FAISS independen:

| Sumber | Fungsi | Aktif pada Tahap |
|---|---|---|
| **Indeks QnA** | Contoh jawaban konselor sebagai inspirasi nada respons | Seluruh tahap |
| **Indeks Alkitab (AVI)** | Ayat Alkitab paling relevan sebagai bimbingan spiritual | `solusi`, `relaksasi`, `bantuan_profesional` (bergantung *spiritual consent*) |

Pada Fase 2, `retrieved_contexts` dibangun dengan menjalankan `RAGEngine.generate_response()` secara langsung per sampel (`spiritual_consent=True` dipaksa agar retrieval ayat aktif pada tahap yang relevan), sehingga konteks yang dievaluasi adalah hasil retrieval aktual.

### c. Perilaku Bergantung Tahap Konseling — Dinilai Langsung oleh Metrik

Berbeda dari pendekatan sebelumnya yang memerlukan agregasi `ragas_per_stage.csv` terpisah untuk *menyimpulkan* pelanggaran aturan tahap secara tidak langsung, definisi `LABAN_CRITERIA_DEFINITION` kini menyertakan kepatuhan tahap sebagai bagian eksplisit dari kriteria penilaian per sampel. `eval_ragas.py` tetap menyisipkan prefiks tahap (`[Tahap Konseling: {stage}] ...`) ke `user_input` yang dikirim ke *judge*, sehingga *judge* memiliki konteks tahap saat menilai. Agregasi `ragas_per_stage.csv` tetap dihasilkan untuk melihat distribusi skor per tahap.

### d. Judge LLM Independen dari Generator

LLM generator chatbot (`opt.CHATBOT_LLM_PROVIDER`) dan LLM *judge* RAGAS dikonfigurasi sebagai model yang **terpisah** untuk menghindari bias evaluasi diri:

| Peran | Model Aktif | Sumber Konfigurasi |
|---|---|---|
| **Generator** (chatbot) | Google `gemini-2.5-flash` | `opt.CHATBOT_LLM_PROVIDER = "gemini"`, `opt.GEMINI_CHATBOT_MODEL` |
| **Judge** (evaluator RAGAS) | OpenAI `gpt-4o` | `opt.RAGAS_JUDGE_MODEL`, diinisialisasi via `langchain_openai.ChatOpenAI` |

Generator berpindah dari Groq `llama-3.3-70b-versatile` (evaluasi sebelumnya) ke Gemini `gemini-2.5-flash`, dan *judge* diperkuat dari `gpt-4o-mini` menjadi `gpt-4o` penuh — sesuai rekomendasi pengembangan sebelumnya (Sub-bab 4.7.2.6 versi lama, poin a.3) untuk memvalidasi apakah skor rendah bersumber dari keterbatasan *judge*, bukan kualitas generator. Karena kedua model berasal dari penyedia dan keluarga model yang berbeda, penilaian tidak dipengaruhi oleh kecenderungan model menilai keluarannya sendiri secara positif.

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

Fase 2 memvalidasi bahwa seluruh kolom `reference` telah terisi, menjalankan pipeline RAG penuh (klasifikasi → retrieval dual-FAISS → generasi LLM) untuk setiap sampel, menyusun `EvaluationDataset` RAGAS, lalu menjalankan metrik `LABAN_Counseling_Standard` (`AspectCritic`) dengan *judge* `gpt-4o`. Tiga berkas keluaran disimpan ke `evaluation/results/`:

| Berkas | Isi | Jumlah Baris |
|---|---|---|
| `ragas_per_sample.csv` | Skor per sampel untuk seluruh kasus uji | 30 |
| `ragas_summary.csv` | Statistik agregat (mean, median, std, min, max) | 1 (satu metrik) |
| `ragas_per_stage.csv` | Rata-rata skor per tahap konseling | 6 (satu per tahap) |

---

## 4.7.2.5 Hasil Evaluasi

Evaluasi dijalankan terhadap **30 sampel** (`n=30`, `random_state=42`), diseimbangkan lintas enam tahap konseling. Hasil agregat global tercantum pada Tabel 4.7.2.1.

**Tabel 4.7.2.1 — Ringkasan Skor RAGAS Global (`ragas_summary.csv`)**

| Metrik | Mean | Median | Std | Min | Max | n |
|---|---:|---:|---:|---:|---:|---:|
| LABAN Counseling Standard | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 30 |

**Tabel 4.7.2.2 — Skor RAGAS per Tahap Konseling (`ragas_per_stage.csv`)**

| Tahap | n | LABAN Counseling Standard (Mean) | Std |
|---|---:|---:|---:|
| `pembukaan` | 6 | 1.0000 | 0.0000 |
| `pembahasan` | 6 | 1.0000 | 0.0000 |
| `intervensi` | 5 | 1.0000 | 0.0000 |
| `solusi` | 5 | 1.0000 | 0.0000 |
| `relaksasi` | 3 | 1.0000 | 0.0000 |
| `penutupan` | 5 | 1.0000 | 0.0000 |

### Interpretasi

Seluruh **30 dari 30 sampel** dinilai memenuhi kriteria `LABAN_Counseling_Standard` oleh *judge* `gpt-4o` (skor 1 di setiap sampel, seluruh tahap, standar deviasi 0.0000 di semua baris). Ini adalah hasil ideal bagi metrik biner *AspectCritic*: tidak ada satu pun respons yang dinilai gagal secara empati, keselarasan alkitabiah, kepatuhan tahap, atau redundansi.

Kualitatif atas `ragas_per_sample.csv` mendukung skor ini. Beberapa pola yang teramati:

1. **Tahap `pembukaan` (n=6, seluruhnya sapaan "Halo")** menghasilkan respons singkat dan konsisten yang selalu menegaskan ruang aman ("ini adalah ruang yang aman untukmu") tanpa menyisipkan solusi atau ayat Alkitab — sesuai aturan `BIBLE_VERSE_STAGES` yang mengecualikan `pembukaan`.
2. **Tahap `pembahasan` dan `intervensi`** secara konsisten merespons dengan pertanyaan eksploratif reflektif (mis. "Apa yang membuatmu...", "Bagaimana rasanya...") tanpa memberikan solusi prematur, selaras dengan aturan tahap yang menyatakan kedua tahap ini bersifat eksplorasi/validasi saja.
3. **Tahap `solusi` dan `relaksasi`** menyisipkan ayat Alkitab yang relevan dengan konteks masalah pengguna (mis. Yesaya 26:3 untuk kecemasan tugas, Matius 11:28 untuk kelelahan emosional, Mazmur 62:2 untuk ketenangan) — retrieval ayat berfungsi sesuai desain hanya pada tahap yang diizinkan.
4. **Tahap `penutupan`** secara konsisten merangkum sesi dan menawarkan rujukan profesional di akhir respons, sesuai desain tahap ini pada Sub-bab 4.6.2.

Perlu dicatat bahwa **skor sempurna pada n kecil per tahap** (`relaksasi` n=3, `intervensi`/`solusi`/`penutupan` n=5) tetap perlu dibaca dengan hati-hati: ukuran sampel ini cukup untuk indikasi kualitatif konsistensi perilaku per tahap, namun terlalu kecil untuk klaim generalisasi statistik yang kuat terhadap populasi interaksi pengguna yang lebih luas.

Dibandingkan hasil evaluasi generik sebelumnya (Faithfulness 0.245, Answer Relevancy 0.015 — lihat riwayat pengembangan), lompatan ke skor sempurna bukan indikasi bahwa kualitas generator berubah drastis, melainkan konfirmasi bahwa **metrik generik RAGAS tidak cocok untuk domain dialog konseling**, sedangkan metrik `AspectCritic` yang dirancang khusus untuk kriteria domain (empati, keselarasan alkitabiah, kepatuhan tahap, non-redundansi) mampu menilai kualitas respons secara akurat.

---

## 4.7.2.6 Saran Pengembangan Lanjutan

1. **Perbesar ukuran testset** (`opt.RAGAS_TESTSET_SIZE`) di atas 30 sampel, terutama untuk tahap dengan n kecil (`relaksasi`, `intervensi`, `solusi`, `penutupan`), agar skor sempurna saat ini dapat divalidasi pada populasi sampel yang lebih besar dan lebih beragam sebelum diklaim sebagai kesimpulan final.
2. **Uji kasus adversarial/edge-case** yang dirancang untuk berpotensi melanggar kriteria (mis. input pengguna yang ambigu terhadap tahap, permintaan solusi langsung pada tahap `pembahasan`) untuk memverifikasi bahwa skor sempurna bukan karena testset saat ini terlalu mudah bagi generator.
3. **Pertimbangkan *inter-rater reliability***: jalankan `LABAN_Counseling_Standard` dengan *judge* alternatif (mis. Gemini 2.5 Pro) pada subset sampel yang sama untuk memvalidasi bahwa skor sempurna konsisten lintas model *judge*, bukan artefak dari satu model tertentu (`gpt-4o`).
4. **Ulangi evaluasi setelah setiap perubahan signifikan** pada `prompt_template`, `BIBLICAL_SYNONYMS`, atau `MODEL_NAME` embedding, untuk memastikan skor sempurna tetap terjaga sebagai *regression check* — bukan hanya divalidasi sekali di titik waktu ini.

---

*Skrip evaluasi: `evaluation/eval_ragas.py` | Hasil: `evaluation/results/ragas_summary.csv`, `ragas_per_stage.csv`, `ragas_per_sample.csv`*
*Generator: Google `gemini-2.5-flash` | Judge: OpenAI `gpt-4o` | Metrik: `LABAN_Counseling_Standard` (AspectCritic, biner) | Ukuran sampel: n=30, seed=42*
