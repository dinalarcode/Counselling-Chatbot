# BAB 5 — KESIMPULAN DAN SARAN

---

## 5.1 Kesimpulan

Penelitian ini telah menyelesaikan seluruh tahapan metodologi yang dirancang, mulai dari studi literatur, pengumpulan dan pelabelan data, pra-pemrosesan, pengembangan model klasifikasi multi-intent, evaluasi model, hingga integrasi model ke dalam sistem chatbot konseling berbasis *Retrieval-Augmented Generation* (RAG). Model klasifikasi dibangun dengan mengadaptasi arsitektur *Label-Aware BERT Attention Network* (LABAN) ke domain Bahasa Indonesia menggunakan backbone IndoBERT, kemudian diintegrasikan dengan pipeline retrieval ayat Alkitab dan orkestrasi *Large Language Model* (LLM) melalui LangChain di atas antarmuka web Flask. Kesimpulan penelitian diuraikan berdasarkan ketiga rumusan masalah sebagai berikut.

---

### 5.1.1 Menjawab Rumusan Masalah Pertama — Pengembangan Dataset

Rumusan masalah pertama, yaitu *bagaimana mengembangkan dataset untuk Multi-Label Intent Classification pada Chatbot konseling berbasis religius*, telah terjawab. Penelitian ini berhasil membangun tiga aset data utama dari data konseling universitas nyata dan hasil *web scraping* Alkitab Terjemahan Baru (TB):

- **Dataset klasifikasi multi-intent** dengan **10 label intent** konseling (mis. "Menyatakan Perasaan Sedih dan Kehilangan", "Menyatakan Perasaan Takut dan Kecemasan"), diformat multi-label dengan pemisah titik koma (`;`).
- **Dataset QnA konseling** berisi **367 pasangan tanya-jawab** sebagai sumber *few-shot example* bagi RAG.
- **Basis pengetahuan ayat Alkitab** berisi **31.102 ayat** yang mencakup **66 buku** dan **1.189 pasal** kanon Protestan.

Untuk mengatasi ketidakseimbangan distribusi label, dataset intent diperluas melalui pipeline augmentasi berbasis LLM (Groq `llama-3.1-8b-instant`) dengan metode *few-shot prompting* dan deduplikasi berbasis kesamaan Jaccard pada ambang $\ge 0{,}75$, menghasilkan `dataset_multiintent_augmented.csv` sebagai data pelatihan aktif. Dataset kemudian dibagi menjadi tiga subset dengan proporsi **80% train / 10% validasi / 10% test** (`random_state=42`), menghasilkan *test set* independen berisi **403 sampel**. Dengan demikian, dataset yang dikembangkan telah memenuhi kebutuhan pelatihan model multi-label sekaligus menyediakan basis pengetahuan spiritual bagi pipeline RAG.

---

### 5.1.2 Menjawab Rumusan Masalah Kedua — Pengembangan Model LABAN

Rumusan masalah kedua, yaitu *bagaimana mengembangkan model LABAN untuk Multi-label Intent Classification*, telah terjawab melalui adaptasi arsitektur LABAN dan dibuktikan secara kuantitatif oleh evaluasi tiga lapisan. Arsitektur asli dimodifikasi pada empat aspek: penggantian backbone ke IndoBERT (`indobenchmark/indobert-base-p1`), pemangkasan arsitektur menjadi mode `normal` + `zero-shot`, penambahan **regularisasi Tikhonov** ($G_\varepsilon = G + \varepsilon I,\ \varepsilon = 10^{-4}$) untuk menjamin invertibilitas matriks Gram, serta penerapan fungsi aktivasi **Sigmoid** dengan *threshold* $0{,}5$ agar beberapa intent dapat aktif secara serentak.

Hasil evaluasi yang dicapai adalah sebagai berikut:

- **Komparasi backbone** (empat paradigma, 50 *epoch*, seed=42, *controlled experiment*) menetapkan **IndoBERT** sebagai backbone produksi dengan **Test F1-Micro 0,9243**, **Precision-Micro 0,9468**, dan **Recall-Micro 0,9029** — mengungguli IndoBERTweet (0,9120), mBERT (0,9178), dan MiniLM-multi (0,8854).
- **Performa klasifikasi produksi** pada 10 intent mencapai **F1-Macro 0,8909**, **F1-Micro 0,8975**, **Precision-Macro 0,9239**, dan **Recall-Macro 0,8612**.
- **Kemampuan *Zero-Shot Learning* (ZSL)** terbukti nyata: pada tiga intent *unseen* (rata-rata tiga split), model produksi *closed-set* memperoleh F1-Macro **0,0** secara struktural, sedangkan model LABAN *dual-encoder* mencapai **F1-Macro 0,8913** (F1-Micro 0,9001; Precision-Macro 0,9247; Recall-Macro 0,8614) tanpa pelatihan ulang.
- **Validitas geometris** ruang vektor dikonfirmasi melalui analisis *cosine similarity*: **Label Separation Gap 1,054**, **Centroid Alignment Gap 0,601**, dan **Sample Alignment Gap 0,419** — memastikan performa F1 yang tinggi merupakan konsekuensi proyeksi semantik yang koheren, bukan artefak statistik.

Dengan capaian tersebut, model LABAN yang dikembangkan terbukti mampu melakukan klasifikasi multi-intent secara akurat sekaligus mempertahankan kapabilitas generalisasi terhadap intent baru.

---

### 5.1.3 Menjawab Rumusan Masalah Ketiga — Integrasi ke Sistem Chatbot RAG

Rumusan masalah ketiga, yaitu *bagaimana mengintegrasikan model Multi-label Intent Classification ke sistem Chatbot RAG agar lebih dinamis namun tetap faktual dan kontekstual*, telah terjawab melalui pembangunan sistem terintegrasi yang dieksekusi dalam satu alur `chat()` per giliran percakapan. Komponen-komponen yang berhasil diintegrasikan meliputi:

- **Mesin tahap konseling (*state machine*) enam tahap** — `pembukaan → pembahasan → intervensi → solusi → relaksasi → penutupan` — beserta cabang darurat `bantuan_profesional` yang dipicu oleh dua lapisan deteksi (classifier LABAN dan 17 pola *regex* ekspresi krisis), menjamin kasus berisiko selalu ditautkan ke konselor profesional.
- **Retrieval berbasis dua indeks FAISS** (QnA dan Alkitab) menggunakan model *embedding* MiniLM-multilingual 384 dimensi, dengan indeks Alkitab dibangun lewat *Augmented Vector Indexing* (AVI) yang memperkaya *embedding* dengan konteks pasal.
- **Retrieval ayat tiga lapis** — konstruksi kueri via sinonim Alkitabiah, pencarian FAISS dengan *oversample* $3\times$ dan filter diversitas, lalu **LLM reranker** yang memilih satu ayat terbaik. Keluaran reranker divalidasi agar hanya referensi yang benar-benar ada dalam daftar kandidat yang diloloskan, sehingga sistem terjaga dari halusinasi (**faktual**).
- **Gerbang persetujuan spiritual (*tri-state consent*)** yang memastikan ayat hanya diinjeksikan setelah pengguna menyetujui, menjaga sistem tetap sensitif secara etis dan kontekstual.
- **Orkestrasi LLM melalui LangChain (LCEL)** dengan pemilihan provider yang dapat dikonfigurasi (Groq sebagai *default*, serta Gemini, OpenAI, dan Ollama), dan perakitan prompt bertahap (*stage-aware prompt assembly*) yang membuat respons **dinamis** sesuai tahap konseling dan intent yang terdeteksi.

Integrasi ini membuktikan bahwa intent hasil klasifikasi LABAN dapat dijadikan penentu konteks retrieval dan penyusun prompt LLM, sehingga chatbot menghasilkan respons yang dinamis, berpijak pada basis pengetahuan (faktual), dan relevan secara kontekstual.

Kualitas respons chatbot secara menyeluruh dievaluasi menggunakan kerangka **RAGAS** dengan pendekatan *LLM-as-a-Judge*. Mengingat metrik generik RAGAS (*Faithfulness*, *Answer Relevancy*) terbukti tidak sesuai untuk domain dialog konseling yang bersifat empatik dan reflektif, digunakan metrik kustom berbasis `AspectCritic` bernama **`LABAN_Counseling_Standard`** yang menilai empat aspek sekaligus (empati, keselarasan alkitabiah, kepatuhan tahap, dan non-redundansi) secara biner (1 = memenuhi, 0 = tidak). Evaluasi dijalankan atas **90 sampel** (`seed=42`) — diseimbangkan lintas enam tahap dan memuat kasus *adversarial* — dengan generator Google `gemini-2.5-flash` dan *judge* independen OpenAI `gpt-4o`. Hasilnya, **89 dari 90 sampel (98,9%)** dinilai memenuhi kriteria, dengan **skor rata-rata global 0,9889** (median 1,0000; std 0,1054). Seluruh tahap memperoleh skor sempurna 1,0000 kecuali tahap `pembahasan` (mean 0,9773), dan seluruh kasus *adversarial* berhasil lulus — membuktikan chatbot mampu menahan diri dari pemberian solusi prematur sesuai aturan tahap `SessionManager`.

Selain evaluasi otomatis, dilakukan pula **evaluasi eksternal oleh manusia** untuk dua fungsi kritis sistem:

- **Relevansi retrieval ayat Alkitab** diukur menggunakan **Cohen's Kappa** dengan **3 responden** atas **30 kasus uji** (pasangan input pengguna dan ayat hasil retrieval), dinilai secara biner (1 = relevan, 0 = tidak relevan). Distribusi penilaian relevan adalah 17/30 (Responden 1), 22/30 (Responden 2), dan 17/30 (Responden 3), menghasilkan rata-rata **Cohen's Kappa 0,5346443353**. Berdasarkan kriteria Landis & Koch (1977), nilai ini tergolong **persetujuan moderat** (*moderate agreement*, rentang 0,41–0,60), yang berarti retrieval ayat sudah berfungsi di atas kesepakatan acak namun **belum mencapai ambang reliabilitas yang kuat** ($\kappa \ge 0{,}60$) sehingga masih menyisakan ruang perbaikan pada penyelarasan semantik antara intent dan ayat.
- **Fungsi konseling secara medis/psikologis** diukur menggunakan **Content Validity Index (CVI)** oleh **1 orang pakar (psikolog sebagai verifikator ahli)**, yang menelaah satu sesi konseling utuh berisi **18 pasangan pertukaran** pengguna–chatbot lintas seluruh tahap, menggunakan skala 1–4 (1 = tidak relevan hingga 4 = sangat relevan). Pakar menyatakan setuju (skor 3 atau 4) pada **14 dari 18 item**, menghasilkan skor **CVI 0,7777777778**. Nilai ini berada tepat pada ambang **validitas isi yang baik** menurut Polit & Beck (2006) untuk tingkat item ($\ge 0{,}78$), namun **sedikit di bawah ambang penerimaan skala** yang umum dianut ($\ge 0{,}80$), sehingga sistem dinilai **cukup valid** sebagai pendamping konseling untuk masalah ringan hingga sedang dengan catatan perlu penyempurnaan lanjutan.

Secara keseluruhan, ketiga jalur evaluasi (RAGAS otomatis, Cohen's Kappa antar-responden, dan CVI pakar) saling melengkapi dan mengonfirmasi bahwa sistem chatbot telah berfungsi sesuai rancangan, dengan area perbaikan yang teridentifikasi jelas pada relevansi retrieval ayat dan validitas isi konseling.

---

## 5.2 Saran

Berdasarkan temuan dan keterbatasan yang teridentifikasi, penelitian ini merekomendasikan beberapa arah pengembangan lanjutan sebagai berikut:

1. **Penanganan limitasi model dan sistem.** Beberapa intent masih memiliki F1 relatif lebih rendah (mis. "Menyatakan Perasaan Sebelum Menghadapi Kejadian" dan "Menyatakan Reaksi Terkejut dan Tidak Terduga"). Disarankan penambahan volume data pelatihan pada label minoritas, eksperimen ambang (*threshold*) adaptif per-intent, serta penanganan bahasa kolokial/*slang* yang lebih baik agar sensitivitas deteksi meningkat tanpa menaikkan *false positive*.

2. **Peningkatan penyelarasan semantik pada *retrieval* ayat Alkitab (FAISS/AVI).** Skor **Cohen's Kappa 0,5346** (persetujuan moderat) mengindikasikan relevansi ayat hasil retrieval belum konsisten di mata seluruh responden dan masih di bawah ambang reliabilitas kuat ($\kappa \ge 0{,}60$). Disarankan penyempurnaan pipeline *Augmented Vector Indexing* — peninjauan dan perluasan kamus `BIBLICAL_SYNONYMS`, penyetelan strategi *chunking* dan pengayaan konteks pasal, penyetelan *oversample factor* serta filter diversitas, dan eksplorasi model *embedding* multibahasa yang lebih mutakhir — guna meningkatkan keselarasan semantik antara intent dan ayat sehingga *inter-rater reliability* dan relevansi retrieval meningkat. Perluasan jumlah responden dan kasus uji juga disarankan agar estimasi Kappa lebih stabil.

3. **Penyempurnaan iteratif fungsi konseling untuk melampaui ambang validitas isi.** Skor **CVI 0,7777777778** hasil verifikasi pakar telah menyentuh ambang validitas isi yang baik pada tingkat item ($\ge 0{,}78$), namun masih **sedikit di bawah ambang penerimaan skala** ($\ge 0{,}80$). Disarankan penyetelan iteratif bersama psikolog/konselor — perbaikan `prompt_template`, penajaman instruksi per tahap `SessionManager`, dan penanganan konteks sensitif (mis. indikasi risiko menyakiti diri/orang lain yang menjadi satu titik kegagalan pada evaluasi RAGAS tahap `pembahasan`) — hingga skor CVI melampaui **0,80**. Untuk memperkuat validitas, evaluasi selanjutnya sebaiknya melibatkan **lebih dari satu pakar** agar S-CVI dapat dihitung secara agregat (S-CVI/Ave maupun S-CVI/UA), disertai *inter-rater reliability* antar-pakar.

4. **Pemantapan evaluasi RAGAS sebagai *regression check*.** Meski skor `LABAN_Counseling_Standard` mencapai 0,9889, ukuran sampel pada beberapa tahap masih kecil (mis. `intervensi` n=4, `pembukaan` n=5). Disarankan pembesaran sampel per tahap, penambahan kriteria eksplisit terkait deteksi risiko/eskalasi, penggunaan *judge* alternatif (mis. Gemini 2.5 Pro) untuk menguji *inter-rater reliability* antar-*judge*, serta pengulangan evaluasi setelah setiap perubahan signifikan pada komponen sistem agar skor tinggi terjaga secara konsisten, bukan hanya pada satu titik waktu.

---

*Referensi Hasil: BAB 4.2 (dataset), BAB 4.4 (arsitektur LABAN), BAB 4.5 (evaluasi model), BAB 4.6 (integrasi sistem), BAB 4.7 (pembahasan), BAB 4.7.2 (evaluasi RAGAS).*
*Rujukan interpretasi metrik: Landis & Koch (1977) untuk Cohen's Kappa; Polit & Beck (2006) untuk Content Validity Index.*
