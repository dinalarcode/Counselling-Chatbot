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

Integrasi ini membuktikan bahwa intent hasil klasifikasi LABAN dapat dijadikan penentu konteks retrieval dan penyusun prompt LLM, sehingga chatbot menghasilkan respons yang dinamis, berpijak pada basis pengetahuan (faktual), dan relevan secara kontekstual. Adapun evaluasi kuantitatif kualitas respons chatbot secara menyeluruh menggunakan kerangka RAGAS (*faithfulness*, *context recall*, *answer relevancy*) direncanakan sebagai tahap validasi lanjutan dan diuraikan pada Sub-bab 5.2.

---

## 5.2 Saran

Berdasarkan temuan dan keterbatasan yang teridentifikasi, penelitian ini merekomendasikan beberapa arah pengembangan lanjutan sebagai berikut:

1. **Penanganan limitasi model dan sistem.** Beberapa intent masih memiliki F1 relatif lebih rendah (mis. "Menyatakan Perasaan Sebelum Menghadapi Kejadian" dan "Menyatakan Reaksi Terkejut dan Tidak Terduga"). Disarankan penambahan volume data pelatihan pada label minoritas, eksperimen ambang (*threshold*) adaptif per-intent, serta penanganan bahasa kolokial/*slang* yang lebih baik agar sensitivitas deteksi meningkat tanpa menaikkan *false positive*.

2. **Pengembangan dan optimasi sistem *retrieval* ayat (FAISS/AVI).** Disarankan penyempurnaan pipeline *Augmented Vector Indexing* — misalnya penyetelan strategi *chunking* dan pengayaan konteks pasal, peninjauan ulang kamus `BIBLICAL_SYNONYMS`, penyetelan *oversample factor* dan filter diversitas, serta eksplorasi *embedding* multibahasa yang lebih mutakhir — guna meningkatkan presisi dan keragaman ayat yang di-*retrieve* lintas sesi.

3. **Pelaksanaan evaluasi terukur secara medis/psikologis dengan pakar menggunakan S-CVI dan I-CVI.** Validitas chatbot sebagai pendamping konseling perlu diukur secara formal oleh pakar (konselor/psikolog) melalui:
   - **I-CVI (*Item-level Content Validity Index*)** — proporsi pakar yang menilai satu item/aspek respons chatbot sebagai relevan, dihitung per item.
   - **S-CVI (*Scale-level Content Validity Index*)** — rata-rata seluruh nilai I-CVI, mencerminkan validitas isi sistem secara keseluruhan.

   Evaluasi berbasis S-CVI dan I-CVI ini menilai kesesuaian respons dengan standar konseling untuk masalah ringan hingga sedang, dan disarankan dilengkapi dengan evaluasi otomatis berbasis **RAGAS** (*faithfulness*, *context recall*, *answer relevancy*) sebagai metrik pendukung terhadap kesetiaan dan relevansi respons berbasis dokumen.

---

*Referensi Hasil: BAB 4.2 (dataset), BAB 4.4 (arsitektur LABAN), BAB 4.5 (evaluasi model), BAB 4.6 (integrasi sistem), BAB 4.7 (pembahasan).*
