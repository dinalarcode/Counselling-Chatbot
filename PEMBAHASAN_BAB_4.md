# Kerangka Berpikir Pembahasan Bab 4 (Analisis & Evaluasi Sistem)

Sub-bab ini berfokus pada analisis mendalam mengenai alasan mekanistis (WHY) di balik pilihan arsitektur, modifikasi komponen, serta kegagalan atau keberhasilan hasil pengujian sistem Chatbot Konseling Alkitab.

---

## 1. Analisis Lingkungan, Pengumpulan, dan Pra-pemrosesan Data (Review Sub-bab 4.1 – 4.3)
### Fokus Utama: Rasionalisasi Modifikasi Data dan Pengayaan Konteks Semantik

* **Rasionalisasi Modifikasi Augmentasi (`dataset_multiintent_augmented.csv`):**
    * **WHY:** Arsitektur *dual-encoder* melatih kecocokan ruang semantik antara kueri dan label teks. Dataset asli kekurangan variasi linguistik non-formal dari pengguna yang sedang mengalami krisis emosional. Modifikasi berupa augmentasi data mutlak diperlukan untuk mencegah model mengalami *overfitting* pada struktur kalimat yang kaku.
    * **Pembahasan Hasil:** Augmentasi memperluas batas keputusan semantik model, meningkatkan ketahanan klasifikasi saat diuji dengan kalimat dunia nyata (*seen labels*).
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Analisis dampak teknik augmentasi data teks terhadap performa generalisasi model berbasis transformer pada domain konseling/psikologi]`

* **Pipa Enkapsulasi Enriched AVI (`alkitab_tb_enriched.csv`):**
    * **WHY:** Pencarian literal menggunakan kata kunci gagal menangkap esensi teologis dari keluhan pengguna. Modifikasi dilakukan dengan mengintegrasikan ringkasan bab (*chapter summaries*) berbasis Ollama Qwen ke dalam setiap ayat sebelum diindeks oleh FAISS. Langkah ini menyuntikkan konteks tingkat makro (topik bab) ke dalam pencarian tingkat mikro (teks ayat).
    * **Pembahasan Hasil:** Indeks FAISS mampu mengembalikan ayat yang relevan secara kontekstual/tematik, bukan sekadar ayat yang memiliki kesamaan kata secara harafiah.
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Peningkatan akurasi Information Retrieval (IR) pada dokumen keagamaan menggunakan pengayaan konteks berbasis ringkasan teks otomatis]`

---

## 2. Analisis Mekanisme Klasifikasi LABAN & Eksperimen Zero-Shot Learning (Review Sub-bab 4.4 – 4.5)
### Fokus Utama: Pembuktian Struktur Dual-Encoder vs Kepala Linier Kaku

* **Eksperimen Skenario Produksi (10 Seen Intents):**
    * **WHY:** Membandingkan keandalan akurasi teoretis arsitektur *Dual-Encoder LABAN* (Model B) terhadap model klasifikasi konvensional dengan *Linear Head* (Model A). Modifikasi *Dual-Encoder* bertujuan mempertahankan kelenturan pemetaan teks ke dalam ruang vektor bersama tanpa mengunci dimensi keluaran secara permanen.
    * **Pembahasan Hasil:** Model dengan *Linear Head* (Model A) menunjukkan performa akurasi superior pada data yang sudah pernah dilihat (*seen*) karena parameter bobotnya teroptimasi secara penuh untuk mendeteksi 10 kelas spesifik tersebut.
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Perbandingan performa model klasifikasi multi-label konvensional dengan arsitektur dual-encoder berbasis representasi semantik]`

* **Eksperimen Skenario Riset (7 Seen, 3 Unseen Intents):**
    * **WHY:** Membuktikan hipotesis kegagalan struktural Model A dan kapabilitas Zero-Shot Learning (ZSL) pada Model B. 
    * **Pembahasan Hasil:** Model A mencetak skor akurasi mutlak **0.0** pada *unseen intents*. Hal ini terjadi karena lapisan linear terakhirnya terikat mati pada dimensi kelas latih, sehingga secara matematis mustahil memproyeksikan probabilitas ke kelas baru. Sebaliknya, Model B berhasil mendapatkan skor **> 0.0**. Keberhasilan mekanistis ini bertumpu pada fungsi `BertEmbedding` yang mengekstrak langsung nilai `pooler_output` dari teks input dan teks label secara terpisah, lalu mempertemukannya melalui perhitungan *pure cosine similarity*. Model B tidak mengklasifikasikan objek, melainkan mengukur kedekatan jarak semantik dua buah teks.
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Analisis fungsionalitas representasi pooler_output dan metrik kesamaan kosinus dalam mendukung kapabilitas Zero-Shot Learning]`

---

## 3. Analisis Mekanisme Integrasi RAG dan State Machine Konseling (Review Sub-bab 4.6)
### Fokus Utama: Pengendalian Alur Dialog, Kendali Etis, dan Mitigasi Halusinasi

* **Arsitektur 6-Stage `SessionManager`:**
    * **WHY:** Sesi konseling pastoral membutuhkan struktur bertahap demi keselamatan psikologis pengguna. Modifikasi alur percakapan dikunci melalui mesin status (*state machine*) untuk memastikan transisi dari eksplorasi emosi ke pencarian solusi berjalan secara linear dan terukur.
    * **Pembahasan Hasil:** Batasan *Max Turns* per tahap mencegah sistem terjebak dalam putaran percakapan berulang (*looping*), sementara aturan CBT otomatis diaktifkan pada tahap `solusi` apabila terdeteksi klasifikasi gejala fisik.
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Implementasi finite state machine dalam penanganan alur dialog chatbot konseling kesehatan mental]`

* **Kendali Etis Spiritual (`spiritual_consent`):**
    * **WHY:** Menampilkan ayat Alkitab kepada pengguna yang sedang mengalami krisis keagamaan atau trauma spiritual tanpa persetujuan dapat memperburuk kondisi psikologis mereka (*spiritual bypass*). Modifikasi dilakukan dengan menerapkan tri-state variabel (`None`/`True`/`False`) yang ditangkap secara lengket (*sticky*) pada akhir tahap `solusi`.
    * **Pembahasan Hasil:** Sistem mengisolasi pemanggilan dokumen teologis hanya pada tahap `relaksasi` (`BIBLE_VERSE_STAGES = ['relaksasi']`). Jika persetujuan bernilai `False` atau `None`, sistem secara aman mengabaikan pencarian ayat (*safe fail*), melindungi pengguna dari intervensi spiritual yang tidak diinginkan.
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Pentingnya persetujuan etis intervensi spiritual dalam pengembangan asisten virtual keagamaan]`

* **Pipa RAG Teraugmentasi (`BIBLICAL_SYNONYMS` & LLM Reranker):**
    * **WHY:** Pencarian FAISS mentah sering kali mengembalikan ayat yang kurang tepat akibat keterbatasan representasi jarak vektor tunggal. Modifikasi dilakukan melalui rantai orkestrasi hibrida: Ekspansi kueri dengan `BIBLICAL_SYNONYMS` -> Pencarian FAISS Top-10 -> Penyaringan ulang oleh LLM Reranker (`retrieve_verse_with_llm()`).
    * **Pembahasan Hasil:** LLM Reranker berfungsi sebagai filter penalaran kontekstual akhir yang memitigasi halusinasi teologis, memastikan ayat teratas yang dikirimkan ke pengguna memiliki validitas eksegesis yang tinggi.
    * **Placeholder Penelitian Terkait:** `[VALIDASI_PAPER: Reduksi halusinasi teologis LLM menggunakan kombinasi perluasan kueri sinonim, FAISS, dan komponen reranking]`