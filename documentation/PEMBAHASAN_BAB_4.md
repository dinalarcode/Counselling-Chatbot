# Pembahasan Bab 4 (Analisis dan Evaluasi Sistem)

Sub-bab ini menganalisis alasan mekanistis (*WHY*) di balik setiap keputusan arsitektural dan modifikasi pipeline yang dijelaskan pada Sub-bab 4.1–4.6, kemudian menghubungkannya secara langsung dengan hasil evaluasi kuantitatif. Struktur pembahasan mengikuti tiga klaster analitis: (1) lingkungan data dan pra-pemrosesan, (2) mekanisme klasifikasi LABAN dan kapabilitas *Zero-Shot Learning*, (3) integrasi RAG dan mesin status konseling.

---

## 1. Analisis Lingkungan, Pengumpulan, dan Pra-pemrosesan Data (Review Sub-bab 4.1–4.3)

### 1.1 Augmentasi Dataset Multi-Intent (`dataset_multiintent_augmented.csv`)

**WHY:** Arsitektur *dual-encoder* LABAN memproyeksikan ucapan pengguna ke ruang vektor label melalui *encoder* yang dilatih pada distribusi linguistik dataset. Dataset mentah menangkap kalimat konseling yang cenderung terstruktur dan homogen secara sintaksis, sementara pengguna nyata yang mengalami tekanan emosional menulis dengan register informal — singkatan, partikel tidak baku, tanda baca yang hilang. Tanpa augmentasi, batas keputusan (*decision boundary*) yang dipelajari model akan terlalu sempit dan gagal menggeneralisasi ke variasi kalimat yang tidak identik dengan pola dataset asli. Augmentasi bukan penambahan volume data semata, melainkan perluasan ruang variasi linguistik agar *encoder* utterance belajar invarian terhadap gaya penulisan, bukan menghafal frasa spesifik.

**Pembahasan Hasil:** Dampak augmentasi terlihat langsung pada metrik F1-Macro skenario produksi (10 *seen* intents) sebesar **0.8909** (Tabel 4.5.4) — angka yang dicapai pada model yang dilatih di atas dataset teraugmentasi, bukan dataset asli. Nilai Sample Alignment Gap sebesar **0.419** (Tabel 4.5.12) turut mengonfirmasi bahwa model tetap mampu memisahkan ucapan individual dari label yang salah meskipun variasi linguistik dalam dataset augmentasi meningkat — bukti bahwa augmentasi memperluas ketahanan tanpa merusak koherensi geometris ruang vektor.

`[VALIDASI_PAPER: Analisis dampak teknik augmentasi data teks terhadap performa generalisasi model berbasis transformer pada domain konseling/psikologi]`

### 1.2 Pipeline Augmented Vector Indexing / AVI (`alkitab_tb_enriched.csv`)

**WHY:** Pencarian FAISS murni mengukur kemiripan vektor antara kueri dan teks ayat tunggal. Teks ayat Alkitab yang diindeks secara harfiah kehilangan konteks pasal — sebuah ayat pendek seperti "Janganlah takut" tidak membawa informasi tentang situasi naratif yang melatarinya. Modifikasi menyuntikkan ringkasan pasal (dihasilkan oleh LLM Ollama Qwen) ke dalam teks yang di-*embed*, sehingga representasi vektor ayat membawa konteks makro (tema pasal) sekaligus mikro (teks ayat). Tanpa langkah ini, *retrieval* akan sering mengembalikan ayat yang secara leksikal mirip kueri namun secara teologis tidak relevan dengan situasi pengguna.

**Pembahasan Hasil:** `core/vector_db.py` mengonfirmasi bahwa `page_content` yang diindeks FAISS adalah `enriched_text` (gabungan ringkasan pasal + teks ayat), dengan *fallback* ke teks mentah hanya jika `enriched_text` kosong. Peningkatan relevansi kontekstual ini tidak diukur melalui metrik F1 (karena bersifat *retrieval*, bukan klasifikasi), namun secara struktural menjadi prasyarat bagi Layer 3 (LLM Reranker, Sub-bab 4.6.3) untuk memilih ayat yang tepat dari kandidat yang sudah relevan secara tematik sejak tahap FAISS.

`[VALIDASI_PAPER: Peningkatan akurasi Information Retrieval (IR) pada dokumen keagamaan menggunakan pengayaan konteks berbasis ringkasan teks otomatis]`

---

## 2. Analisis Mekanisme Klasifikasi LABAN dan Zero-Shot Learning (Review Sub-bab 4.4–4.5)

### 2.1 Koreksi Kerangka Analitis: Satu Mekanisme, Dua Basis Label

Dokumentasi awal proyek ini (draf pertama Sub-bab 4.5) sempat membingkai perbandingan sebagai "*Linear Head* konvensional (Model A) melawan *cosine similarity* murni (Model B)". Kerangka tersebut **secara faktual keliru** dan telah dikoreksi (lihat CLAUDE.md § Task Status, BAB 4.5.1). Kode sumber `models/multilabel/bert_model.py` tidak pernah memuat `nn.Linear` sebagai kepala klasifikasi, dan mekanisme skor bukan *cosine similarity* karena tidak ada normalisasi magnitudo vektor:

```python
gram   = torch.mm(clusters, clusters.permute(1, 0))                 # Gram matrix (10,10)
gram   = gram + torch.eye(gram.size(0), device=gram.device) * 1e-4   # regularisasi Tikhonov
weight = torch.mm(pooled_output, clusters.permute(1, 0))             # (1,10)
logits = torch.mm(weight, torch.inverse(gram)) * np.sqrt(opt.hidden_size)
```

Persamaan ini adalah proyeksi *least-squares* melalui invers Gram: $\mathbf{w} = \sqrt{H} \cdot \mathbf{G}^{-1} \mathbf{b}$. **Model A dan Model B menjalankan persamaan yang persis sama.** Satu-satunya variabel pembeda adalah **basis label** yang membentuk matriks $\mathbf{G}$ dan vektor $\mathbf{b}$ pada saat inferensi:

- **Rezim A (basis tetap):** $\mathbf{G}$ dibangun hanya dari 10 (atau 7 *seen*) label yang telah ditentukan di muka.
- **Rezim B (basis diperluas):** $\mathbf{G}$ dibangun dari 7 label *seen* **ditambah** 3 label *unseen* yang di-*encode* secara *on-the-fly* oleh *label encoder* saat prediksi — tanpa retraining.

Bagian pembahasan berikut ditulis di atas kerangka yang telah dikoreksi ini.

### 2.2 WHY Dual-Encoder Dipertahankan (Bukan Klasifikasi Konvensional)

**WHY:** Klasifikasi multi-label konvensional mengunci dimensi keluaran pada jumlah kelas yang diketahui saat training — menambah atau mengurangi intent berarti melatih ulang seluruh model. Arsitektur LABAN sengaja menghindari pengurungan ini dengan memperlakukan label sebagai teks yang di-*encode* oleh *encoder* kedua (`self.bertlabelencoder`), terpisah dari *encoder* ucapan (`self.bert`). Karena logit dihasilkan dari proyeksi terhadap `clusters` (representasi label) yang dibentuk pada saat inferensi — bukan bobot yang dibekukan dari waktu training — jumlah dan komposisi label yang diproyeksikan dapat berubah tanpa mengubah satu parameter pun di dalam *encoder*. Ini adalah syarat arsitektural minimum yang harus dipenuhi agar ZSL mungkin terjadi.

**Pembahasan Hasil:** Skenario produksi (10 *seen* intents, Tabel 4.5.4) membuktikan bahwa fleksibilitas ini tidak mengorbankan performa: Rezim A dan Rezim B menghasilkan angka **identik** di seluruh metrik (F1-Macro **0.8909**, Precision-Macro 0.9239, Recall-Macro 0.8612). Kesetaraan ini bukan koinsidensi — ketika tidak ada label *unseen* yang diperkenalkan, basis diperluas Rezim B secara matematis kolaps menjadi basis tetap Rezim A.

`[VALIDASI_PAPER: Perbandingan performa mekanisme proyeksi label-aware (gram-inverse) dengan pipeline klasifikasi multi-label konvensional yang menggunakan kepala linear tetap]`

### 2.3 WHY Rezim A Gagal Struktural pada Label Unseen

**WHY:** Regularisasi Tikhonov menjamin $\mathbf{G}_\varepsilon$ selalu *invertible*, tetapi tidak dapat menciptakan informasi yang tidak ada. Ketika Rezim A membangun $\mathbf{G}$ hanya dari 7 label *seen*, matriks tersebut berukuran $7 \times 7$ — tidak ada baris atau kolom yang berkorespondensi dengan label *unseen* apa pun. Proyeksi $\mathbf{w} = \sqrt{H}\cdot\mathbf{G}^{-1}\mathbf{b}$ secara matematis hanya dapat menghasilkan skor untuk label yang eksis sebagai kolom dalam $\mathbf{G}$. Kegagalan Rezim A pada label *unseen* bukan soal *neuron* yang hilang (karena tidak ada *neuron* dalam mekanisme ini sama sekali) — melainkan karena label tersebut **tidak pernah menjadi bagian dari basis proyeksi** yang dievaluasi.

**Pembahasan Hasil:** Tabel 4.5.5 mencatat Rezim A mencetak **0.0** di seluruh metrik (F1-Macro, F1-Micro, Precision-Macro, Recall-Macro) pada kelompok 3 intent *unseen*, rata-rata dari 3 *split* independen (Tabel 4.5.3). Nilai nol yang konsisten lintas *split* — bukan sekadar rendah — mengonfirmasi bahwa ini adalah batasan struktural *closed-set*, bukan kegagalan generalisasi yang bersifat probabilistik.

`[VALIDASI_PAPER: Batasan structural closed-set classification pada arsitektur berbasis proyeksi basis label tetap]`

### 2.4 WHY Rezim B Membuktikan ZSL Arsitektural

**WHY:** Rezim B memperluas $\mathbf{G}$ menjadi $10 \times 10$ dengan menyisipkan *embedding* label *unseen* — dihasilkan oleh *label encoder* yang sama, tanpa parameter baru, tanpa *fine-tuning* tambahan. Kapabilitas ZSL yang muncul bukan berasal dari mekanisme skor yang berbeda (keduanya tetap gram-inverse), melainkan murni dari kemampuan *label encoder* mengubah **teks deskriptif apa pun** menjadi vektor 768-dimensi yang dapat langsung diproyeksikan. Inilah nilai tambah nyata dari memperlakukan label sebagai teks (filosofi inti LABAN, Sub-bab 4.4.1) dan bukan sebagai indeks kelas numerik.

**Pembahasan Hasil:** Rezim B mencetak F1-Macro **0.8913** pada kelompok *unseen* yang identik dengan yang gagal total di Rezim A (Tabel 4.5.5) — kontras **0.0 vs 0.8913** pada label yang sama adalah bukti langsung bahwa perluasan basis, bukan pergantian mekanisme, adalah sumber kapabilitas ZSL. Nilai F1-Macro *unseen* Rezim B (0.8913) bahkan melampaui F1-Macro *seen* Rezim B pada *split* yang sama (0.8907) — mengindikasikan bahwa performa proyeksi tidak menurun akibat perluasan basis, karena regularisasi Tikhonov menjaga stabilitas numerik invers Gram terlepas dari ukuran matriks.

`[VALIDASI_PAPER: Analisis mekanisme label-as-text encoding dalam mendukung kapabilitas Zero-Shot Learning tanpa retraining]`

### 2.5 WHY IndoBERT Dipilih sebagai Backbone Produksi

**WHY:** Komparasi empat *backbone* (`compare_embed_models.py`) diuji di bawah dua sumbu hipotesis: ragam korpus bahasa Indonesia (IndoBERT formal vs. IndoBERTweet informal) dan cakupan bahasa (spesifik-Indonesia vs. multibahasa vs. distilasi ringan). Hipotesis awal memprediksi keunggulan IndoBERTweet karena register korpus Twitter-nya lebih dekat dengan gaya bahasa informal pengguna chatbot konseling. Namun, arsitektur *dual-encoder* LABAN memproyeksikan **dua** representasi — ucapan **dan** label. Teks label intent bersifat deskriptif dan formal (mis. "Menyatakan Perasaan Takut dan Kecemasan"), sehingga kesesuaian register *backbone* terhadap sisi label ternyata lebih menentukan daripada kesesuaian terhadap sisi input.

**Pembahasan Hasil:** IndoBERT mencatat Test F1-Micro **0.9243**, melampaui IndoBERTweet (0.9120) meski IndoBERTweet dilatih pada 409 juta *tweet* berbahasa Indonesia informal (Tabel 4.5.8). mBERT (0.9178) mendekati performa IndoBERT namun memerlukan waktu training ~4,8× lebih lama (6.514,6 dtk vs. 1.367,7 dtk) — tidak efisien untuk selisih hanya 0,65 poin. MiniLM-multi mencatat F1-Micro terendah (0.8854) *sekaligus* waktu training terlama (17.153,0 dtk), karena arsitekturnya sebagai *sentence transformer* tidak dioptimalkan untuk *fine-tuning* tingkat token yang dibutuhkan proyeksi Gram. Hasil ini membalikkan hipotesis awal dan menetapkan `indobenchmark/indobert-base-p1` sebagai *backbone* produksi berdasarkan bukti empiris, bukan asumsi register.

`[VALIDASI_PAPER: Pengaruh kesesuaian register korpus pra-pelatihan backbone terhadap sisi label versus sisi input pada arsitektur dual-encoder]`

### 2.6 WHY Cosine Heatmap Digunakan Sebagai Diagnostik Geometri (Bukan Mekanisme Klasifikasi)

**WHY:** F1-score yang tinggi dapat dicapai melalui jalur yang tidak sehat — bias *threshold*, *co-occurrence* label yang tidak disengaja, atau optimasi yang mengeksploitasi properti statistik data validasi tanpa benar-benar membangun representasi semantik yang koheren. *Cosine similarity* di sini dipakai sebagai **alat ukur diagnostik pasca-hoc** atas geometri ruang vektor yang dibangun `pooler_output`, bukan sebagai mekanisme skor yang menghasilkan logit (yang tetap gram-inverse, lihat § 2.1–2.2). Perbedaan peran ini penting: metrik geometris memverifikasi *kesehatan* representasi, sementara proyeksi Gram invers yang tetap menjadi jalur keputusan aktual.

**Pembahasan Hasil:** Tiga lapisan heatmap (`eval_cosine_heatmap.py`) menunjukkan Label Separation Gap **1.054** (off-diagonal rata-rata **negatif**, −0.054 — label saling "berlawanan arah", jauh melampaui ortogonalitas), Centroid Alignment Gap **0.601**, dan Sample Alignment Gap **0.419** (Tabel 4.5.10–4.5.12). Penurunan bertahap dari Gap Label (1.054) ke Gap Centroid (0.601) ke Gap Sampel (0.419) adalah pola yang **diharapkan** — setiap lapisan menambah sumber variasi (agregasi sampel individual paling variatif) — dan seluruh nilai tetap jauh di atas nol, mengonfirmasi bahwa F1-Macro tinggi pada § 2.2 adalah konsekuensi dari proyeksi semantik yang koheren, bukan artefak statistik.

`[VALIDASI_PAPER: Validasi geometris ruang vektor sebagai pelengkap metrik F1 pada evaluasi model klasifikasi multi-label berbasis embedding]`

### 2.7 WHY Model Produksi Tetap Menggunakan Basis Tetap (Rezim A), Bukan ZSL Aktif

**WHY:** Meskipun Rezim B terbukti mampu ZSL, sistem produksi (`models/multilabel/predict.py`) tetap menjalankan basis tetap 10 label. Keputusan ini bukan kompromi akurasi — Tabel 4.5.4 membuktikan Rezim A dan B identik pada 10 label produksi — melainkan keputusan operasional: daftar intent yang tertutup dan stabil mencegah *drift* perilaku klasifikasi yang tidak terduga pada sesi konseling klinis, di mana pemetaan intent ke instruksi tahap (`RAGEngine.STAGE_INSTRUCTIONS`) dan sinonim Alkitabiah (`BIBLICAL_SYNONYMS`) bersifat statis dan sudah divalidasi terhadap 10 label yang persis itu. Menambah label secara dinamis di produksi akan memutus korespondensi antara intent yang terdeteksi dan logika hilir (*downstream logic*) yang bergantung padanya.

**Pembahasan Hasil:** Justifikasi ini tercermin dalam Ringkasan Evaluasi (Tabel 4.5.13): status Rezim A pada skenario produksi ditandai "✅ Konfigurasi Produksi" tepat karena kesetaraan F1-nya dengan Rezim B, bukan superioritasnya.

`[VALIDASI_PAPER: Trade-off antara fleksibilitas Zero-Shot Learning dan determinisme klinis pada sistem klasifikasi intent untuk aplikasi konseling kesehatan mental]`

---

## 3. Analisis Integrasi RAG dan State Machine Konseling (Review Sub-bab 4.6)

### 3.1 WHY Mesin Status 6-Tahap Dikunci Secara Linear

**WHY:** Konseling pastoral berbasis metode LABAN mensyaratkan eksplorasi emosi mendahului pencarian solusi — melompat langsung ke solusi tanpa validasi perasaan berisiko membuat pengguna merasa tidak didengar, sebuah kegagalan umum pada chatbot konseling yang tidak berstruktur. `SessionManager` mengunci urutan `pembukaan → pembahasan → intervensi → solusi → relaksasi → penutupan` melalui kombinasi tiga mekanisme independen: `MIN_TURNS` (mencegah transisi prematur, mis. `pembahasan` minimal 3 giliran sebagai *clinical requirement*), deteksi sinyal transisi berbasis *regex* yang dikompilasi sekali saat kelas dimuat (efisiensi pencocokan per giliran), dan `MAX_TURNS` sebagai *safety net* paksa (mis. `pembahasan` maksimal 5 giliran) agar sesi tidak terjebak dalam eksplorasi tanpa akhir.

**Pembahasan Hasil:** Kombinasi tiga mekanisme ini tidak diuji melalui metrik F1 (karena bukan tugas klasifikasi), namun validitasnya tercermin secara tidak langsung: `SKIP_CLASSIFICATION_STAGES = frozenset({'pembukaan', 'bantuan_profesional'})` menghemat 150–300ms inferensi GPU per giliran pada tahap yang tidak memerlukan klasifikasi — bukti bahwa arsitektur *state machine* memungkinkan optimasi selektif tanpa mengorbankan korektnesf alur.

`[VALIDASI_PAPER: Implementasi finite state machine dengan batas giliran ganda (minimum-maksimum) dalam penanganan alur dialog chatbot konseling kesehatan mental]`

### 3.2 WHY Akumulasi Intent Dipertahankan Lintas Tahap

**WHY:** Retrieval ayat Alkitab di tahap `relaksasi` tidak boleh bergantung semata pada input satu giliran terakhir, karena pada tahap tersebut pengguna sering kali tidak lagi mengulang keluhannya secara eksplisit (fokus sudah beralih ke teknik relaksasi). `SessionManager` mengakumulasi intent yang terdeteksi ke `Counter` sepanjang sesi, membekukan 3 intent paling sering muncul (`primary_intents`) tepat saat sesi keluar dari `pembahasan`, dan mengoper daftar beku ini sebagai `override_intents` ke `generate_response()` saat `relaksasi`. Mekanisme ini memastikan retrieval ayat tetap merepresentasikan keluhan historis utama, bukan sinyal terbaru yang mungkin sudah tidak relevan.

**Pembahasan Hasil:** Karena `override_intents` menggantikan `detected_intents` giliran-saat-ini hanya pada tahap `relaksasi`, konsistensi tematik ayat yang diambil bergantung langsung pada akurasi klasifikasi selama `pembahasan` — menghubungkan kembali kualitas keputusan pada § 2.2 (F1-Macro produksi 0.8909) dengan kualitas *retrieval* pada RAG.

`[VALIDASI_PAPER: Strategi state persistence intent lintas sesi untuk retrieval konten kontekstual pada chatbot konseling multi-turn]`

### 3.3 WHY Persetujuan Spiritual Bersifat Tri-State dan Sticky

**WHY:** Menampilkan ayat Alkitab kepada pengguna yang sedang mengalami krisis keagamaan tanpa persetujuan eksplisit berisiko memperparah kondisi psikologis (*spiritual bypass*) — pengguna yang trauma secara spiritual dapat merasa dipaksa menerima kerangka religius yang justru menjadi sumber masalahnya. Variabel `spiritual_consent` dirancang tri-state (`None`/`True`/`False`, bukan Boolean biner) karena keadaan "belum ditanya" secara semantik berbeda dari "menolak" — keduanya harus sama-sama **mencegah** retrieval ayat (*fail-safe* default), namun hanya `None` yang memicu pertanyaan persetujuan pada giliran akhir tahap `solusi`. Setelah dijawab, nilai ditangkap secara *sticky* agar pengguna tidak ditanya berulang kali sepanjang sesi.

**Pembahasan Hasil:** Gerbang kondisional di `core/rag_engine.py` — `if current_stage in self.BIBLE_VERSE_STAGES and intents_for_verse and spiritual_consent is True` — hanya mengizinkan pemanggilan `retrieve_verse_with_llm()` ketika ketiga syarat terpenuhi sekaligus. Karena `BIBLE_VERSE_STAGES = ['relaksasi']` adalah satu-satunya tahap yang diizinkan, dan pemeriksaan `is True` (bukan *truthy check*) secara eksplisit menolak `None` maupun `False`, sistem tidak pernah menyuntikkan konten teologis pada jalur mana pun yang tidak melalui persetujuan eksplisit — termasuk pada cabang darurat `bantuan_profesional`, yang memang tidak pernah menyentuh tahap `relaksasi` sama sekali.

`[VALIDASI_PAPER: Kerangka etis informed consent untuk intervensi konten spiritual pada asisten virtual berbasis AI di ranah kesehatan mental]`

### 3.4 WHY Retrieval Ayat Melalui Tiga Lapisan, Bukan FAISS Tunggal

**WHY:** Kesamaan vektor FAISS murni adalah metrik jarak — ia tidak "memahami" mengapa satu ayat lebih tepat dari ayat lain secara pastoral, hanya bahwa representasinya secara numerik berdekatan. Bergantung pada Top-1 FAISS berisiko tinggi menghasilkan ayat yang relevan secara leksikal namun keliru secara eksegesis (*theological hallucination* jika langsung diteruskan ke LLM generatif tanpa verifikasi). Modifikasi merancang tiga lapisan berurutan yang masing-masing menambah presisi: (1) ekspansi kueri via `BIBLICAL_SYNONYMS` mengatasi kesenjangan register antara bahasa informal pengguna dan bahasa formal Alkitab TB; (2) *oversampling* 3× dengan filter diversitas (per-sesi, per-kitab, maks. 2 ayat per-pasal) mencegah pengulangan ayat yang monoton lintas sesi; (3) LLM Reranker sebagai penyaring akhir yang menilai kecocokan kontekstual-naratif — kapasitas yang tidak dimiliki kesamaan vektor.

**Pembahasan Hasil:** Validasi ayat pada Layer 3 (`retrieve_verse_with_llm()`) memuat pengecekan eksplisit `if c["reference"] == ref` — LLM hanya boleh memilih dari kandidat yang telah melalui Layer 1 dan 2, bukan menghasilkan referensi bebas. Desain ini membatasi ruang halusinasi LLM pada *pemilihan* (dari daftar terbatas dan terverifikasi), bukan pada *generasi bebas* referensi Alkitab — mengurangi risiko LLM mengarang ayat yang tidak eksis.

`[VALIDASI_PAPER: Reduksi halusinasi teologis LLM menggunakan kombinasi perluasan kueri sinonim, FAISS oversampling dengan filter diversitas, dan komponen reranking berbasis kandidat terbatas]`

### 3.5 WHY Ringkasan dan Ekstraksi Teknik Dijalankan sebagai Panggilan LLM Terpisah dan Efemeral

**WHY:** LLM generatif dalam arsitektur ini bersifat *stateless* per panggilan — tidak ada memori bawaan lintas giliran. Mengirim seluruh riwayat percakapan `pembahasan` (bisa mencapai 5 giliran) ke dalam setiap prompt tahap-tahap berikutnya akan memboroskan token context window tanpa nilai tambah proporsional. Sebagai gantinya, sistem memanggil LLM **satu kali** di titik transisi kritis (`pembahasan→intervensi` dan `solusi→relaksasi`) untuk mengekstrak satu kalimat ringkasan (`complaint_summary`) atau nama teknik (`chosen_technique`), lalu segera membebaskan riwayat sementara (`temp_pembahasan_history`, `temp_solusi_history`) begitu ekstraksi selesai.

**Pembahasan Hasil:** Pembebasan riwayat sementara segera setelah ekstraksi (bukan menunggu akhir sesi) adalah keputusan manajemen memori yang eksplisit dalam `_advance_stage()` — memastikan representasi konteks yang dibawa ke tahap selanjutnya tetap ringkas (satu kalimat, bukan transkrip penuh) sekaligus mengurangi jejak data personal yang disimpan lebih lama dari yang diperlukan.

`[VALIDASI_PAPER: Strategi ringkasan episodik (ephemeral summarization) untuk manajemen konteks pada LLM stateless dalam sistem dialog multi-tahap]`

---

## Sintesis Lintas Klaster

Ketiga klaster analisis di atas saling bergantung secara kausal, bukan berdiri sendiri: kualitas augmentasi data (§1.1) menentukan kualitas proyeksi Gram invers (§2.2), yang menentukan `detected_intents` yang diakumulasi `SessionManager` (§3.2), yang pada akhirnya menentukan kueri yang dibentuk `BIBLICAL_SYNONYMS` untuk *retrieval* ayat yang telah diperkaya AVI (§1.2, §3.4). Kegagalan atau ketidaktepatan pada lapisan mana pun akan menjalar ke lapisan berikutnya — sebuah argumen langsung mengapa evaluasi Bab 4 dirancang berlapis (klasifikasi → geometri → integrasi *end-to-end*) dan bukan diukur hanya pada satu titik akhir pipeline.

`[VALIDASI_PAPER: Evaluasi end-to-end pada sistem chatbot konseling berlapis — dampak propagasi kesalahan antar-komponen dari klasifikasi hingga generasi respons]`

---

*Sumber: `documentation/BAB_4.4_Klasifikasi_Multi_Intent.md`, `documentation/BAB_4.5_Evaluasi_Klasifikasi_Multi_Intent.md`, `documentation/BAB_4.6_Integrasi_Sistem_RAG.md`, `CLAUDE.md`*
*Kode Rujukan: `models/multilabel/bert_model.py`, `models/multilabel/predict.py`, `core/session_manager.py`, `core/rag_engine.py`, `core/vector_db.py`*
