# BAB 4.7 — Pembahasan

Sub-bab ini mensintesis apa yang telah **dikerjakan dan dicapai** pada Sub-bab 4.1 (Lingkungan) hingga 4.6 (Integrasi Sistem), lalu memvalidasi setiap keputusan teknis dengan menautkannya pada temuan, formula, atau saran penelitian terdahulu. Fokus pembahasan adalah pembenaran (*justifikasi*) atas modifikasi dan hasil nyata sistem — bukan pengulangan definisi teori yang sudah dibahas pada Bab 2 dan Bab 3.

---

## 4.7.1 Fondasi Teknis dan Pemilihan Komponen (Sub-bab 4.1)

Sistem dibangun di atas tumpukan (*stack*) yang memisahkan tiga fungsi: model *embedding* (representasi semantik), basis data vektor FAISS (*similarity search*), dan LLM (pembangkit respons). Pemisahan ini bukan pilihan sembarang: struktur tripartit retrieval–generation–augmentation persis merupakan kerangka yang direkomendasikan oleh Gao et al. [2024] sebagai fondasi sistem RAG modern. Keputusan memakai `MiniLM-multilingual` (embedding lintas-bahasa 384 dimensi) untuk retrieval dan LLM terpisah untuk generasi memiliki preseden empiris langsung pada Ponmagal et al. [2025], yang membangun chatbot dukungan mental menggunakan *Sentence Transformer embeddings* untuk deteksi relevansi dan Google Gemini API untuk respons empatik — kombinasi arsitektural yang sama dengan sistem ini.

Pilihan mengoperasikan chatbot berbasis web dengan orkestrasi LangChain (`prompt_template | llm`) juga tervalidasi oleh Mahardika et al. [2025], yang menerapkan RAG berbasis LangChain pada antarmuka web dan mencapai skor *User Acceptance Test* 96,4% — bukti bahwa pola integrasi ini layak-produksi (*production-ready*).

---

## 4.7.2 Strategi Data: Augmentasi LLM dan Dimensi Spiritual (Sub-bab 4.2–4.3)

Dataset intent awal mengalami ketidakseimbangan distribusi, sehingga diperluas melalui augmentasi berbasis LLM (Groq `llama-3.1-8b-instant`, *few-shot prompting*, dedup Jaccard $\ge 0{,}75$). Strategi "memakai LLM untuk mensintesis ucapan pelatihan bagi *intent classifier*" ini divalidasi kuat oleh Liu et al. [2025], yang lewat kerangka *Chain-of-Intent* membuktikan bahwa dialog sintetis berbasis LLM efektif melatih model klasifikasi intent multi-giliran secara *domain-specific* dan multibahasa. Dengan kata lain, pipeline augmentasi yang dieksekusi di sini menempuh jalur yang secara ilmiah sudah terbukti, bukan eksperimen liar.

Pelabelan multi-label manual atas ucapan konseling nyata (10 intent, pemisah `;`) sejajar dengan metodologi Malhotra et al. [2021], yang menganotasi ~12,9K ucapan konseling ke label *dialogue-act* domain-spesifik (dataset HOPE). Keduanya berangkat dari premis yang sama: percakapan konseling bersifat *implisit* sehingga label harus dirancang khusus domain, tidak bisa memakai taksonomi intent generik.

Untuk lapisan spiritual (*web scraping* Alkitab TB, 31.102 ayat), relevansinya bagi konteks Indonesia didukung tiga temuan. Hamka et al. [2022] menunjukkan bahwa *spiritual well-being* secara signifikan menurunkan kecemasan, depresi, dan stres pada komunitas Indonesia. Subu et al. [2022] menegaskan masyarakat Indonesia kerap memaknai penyakit mental lewat kerangka religius/kultural dan condong mencari penanganan berbasis keyakinan. Nganyu [2025] menyimpulkan integrasi prinsip teologis dengan intervensi psikologis dapat meningkatkan hasil terapi — sekaligus mengingatkan perlunya pendekatan yang sensitif secara budaya, sebuah peringatan yang direspons langsung oleh mekanisme *consent* pada Sub-bab 4.6.

---

## 4.7.3 Arsitektur Klasifikasi: Adaptasi Dual-Encoder untuk ZSL (Sub-bab 4.4)

Inti klasifikasi adalah adaptasi LABAN (Wu et al., EMNLP 2021) ke domain Indonesia: backbone diganti ke IndoBERT, arsitektur dipangkas (~227 → ~45 baris), dan — kontribusi teknis paling penting — inversi matriks Gram distabilkan dengan **regularisasi Tikhonov** ($G_\varepsilon = G + \varepsilon I,\ \varepsilon = 10^{-4}$). Modifikasi ini menjawab kelemahan konkret implementasi asli: ketika dua label berkorelasi tinggi (mis. "Sedih dan Kehilangan" vs. "Takut dan Kecemasan"), `torch.inverse()` langsung rawan *near-singular* dan menghasilkan `NaN`. Regularisasi Tikhonov adalah teknik standar untuk *ill-conditioned inverse*, sehingga penambahan ini memperkuat — bukan mengubah — logika proyeksi $\mathrm{logits} = W \cdot G_\varepsilon^{-1} \cdot \sqrt{d}$.

Keputusan memakai **Sigmoid** (bukan Softmax) agar beberapa intent aktif serentak selaras dengan sifat ucapan konseling yang majemuk secara emosi — karakteristik "implisit namun berlapis" yang justru menjadi motivasi Malhotra et al. [2021] merancang klasifikasi khusus konseling. Sifat *dual-encoder* (encoder ucapan dan encoder label terpisah) yang memungkinkan *zero-shot* dengan hanya menambah teks deskripsi label saat inferensi juga konsisten dengan gagasan Liu et al. [2025] bahwa representasi berbasis teks intent lebih *generalizable* dibanding indeks kelas beku.

---

## 4.7.4 Hasil Evaluasi Model Klasifikasi (Sub-bab 4.5)

Evaluasi tiga lapisan menghasilkan bukti kuantitatif yang saling menguatkan, dan setiap lapisan memiliki dasar validasi di literatur.

Pertama, **komparasi backbone** (IndoBERT, IndoBERTweet, mBERT, MiniLM-multi) di bawah *controlled experiment* identik (50 epoch, split 80/10/10, seed=42) menempatkan IndoBERT sebagai backbone produksi dengan **Test F1-Micro 0,9243** (Precision-Micro 0,9468; Recall-Micro 0,9029). Praktik memilih model lewat perbandingan empiris metrik F1/Precision/Recall — bukan asumsi — persis rekomendasi Assayed et al. [2022], yang menunjukkan akurasi *intent classifier* sangat bergantung pada pemilihan algoritma dan ekstraksi fitur yang tepat, dan karenanya harus ditetapkan secara eksperimental. Temuan menarik penelitian ini — IndoBERT (korpus formal) mengungguli IndoBERTweet (korpus informal) meski input pengguna kolokial — memberi kontribusi empiris baru: pada arsitektur *dual-encoder*, kesesuaian register di sisi **label** lebih determinan daripada di sisi input.

Kedua, **kemampuan ZSL** terbukti kontras tajam: pada 3 intent *unseen* yang sama, Model A (produksi, *closed-set*) memperoleh F1-Macro **0,0** secara struktural, sedangkan Model B (LABAN dual-encoder) mencapai F1-Macro **0,8913** tanpa pelatihan ulang. Ini adalah realisasi empiris atas janji arsitektural LABAN dan sejalan dengan arah kerja Liu et al. [2025] mengenai klasifikasi intent yang adaptif terhadap kategori baru.

Ketiga, **analisis geometri** (*cosine heatmap*) memverifikasi bahwa F1 tinggi bukan artefak statistik: *Label Separation Gap* 1,054 (off-diagonal rata-rata negatif $-0{,}054$), *Centroid Alignment Gap* 0,601, dan *Sample Alignment Gap* 0,419. Verifikasi konsistensi proyeksi semantik lewat *cosine similarity* ini merupakan metrik yang memang direkomendasikan untuk arsitektur *label-aware* dan menegaskan model memahami semantik secara geometris, bukan sekadar menghafal pola.

---

## 4.7.5 Integrasi Sistem: RAG, State Machine, dan Mitigasi Risiko (Sub-bab 4.6)

Integrasi menyatukan klasifikasi, retrieval, dan generasi dalam satu *turn*. Beberapa keputusan rancangan spesifik memerlukan validasi.

**State machine enam tahap** (`pembukaan → pembahasan → intervensi → solusi → relaksasi → penutupan`, plus cabang `bantuan_profesional`) mengoperasionalkan alur konseling terstruktur yang paralel dengan kerangka *Psychological First Aid* "Look–Listen–Link" pada Indrayanti et al. [2025]: deteksi *distress* (Look) ≈ akumulasi intent, respons empatik (Listen) ≈ tahap `pembahasan`/`intervensi`, dan penautan ke bantuan (Link) ≈ jalur darurat `bantuan_profesional`. Rancangan bertahap ini juga selaras dengan Malhotra et al. [2021] yang memodelkan percakapan konseling sebagai urutan *dialogue-act* yang sadar-konteks dan sadar-waktu.

**Jalur darurat dua-lapis** (classifier LABAN + 17 pola *regex* untuk ekspresi bunuh diri/menyerah) yang selalu menautkan pengguna ke konselor profesional divalidasi oleh Barnett et al. [2021]: temuan mereka menegaskan chatbot paling tepat untuk tugas sempit — *screening*, *triage*, dan *referral* — sementara elemen "manusiawi" tetap tak tergantikan. Sistem ini menempatkan diri persis pada peran tersebut, bukan sebagai pengganti konselor.

**Retrieval ayat berlapis dengan LLM reranker** — konstruksi kueri via sinonim Alkitabiah, *FAISS oversample* $3\times$ + filter diversitas, lalu LLM memilih satu ayat terbaik — merupakan penerapan konkret *Advanced RAG* (optimasi pasca-retrieval / *re-ranking*) yang dipetakan Gao et al. [2024]. Pengayaan indeks dengan konteks pasal (*Augmented Vector Indexing*) sejalan dengan rekomendasi Gao et al. [2024] tentang pengayaan metadata untuk presisi retrieval. Pendekatan retrieval berlapis yang meningkatkan relevansi jawaban ini juga konsisten dengan hasil Nayinzira et al. [2024], yang menemukan varian *Multi-query RAG* lebih relevan dan empatik dibanding RAG naif pada domain kesehatan mental.

**Mitigasi halusinasi** dilakukan dengan membatasi keluaran reranker hanya pada referensi yang benar-benar ada di daftar kandidat (validasi `c["reference"] == ref`). Ini adalah bentuk *grounding* yang menjadi alasan utama RAG dianjurkan: Gao et al. [2024] menempatkan halusinasi sebagai masalah inti LLM yang justru diredam oleh RAG, dan Mahardika et al. [2025] mengonfirmasi chatbot berbasis RAG terhindar dari halusinasi karena jawaban berpijak pada basis pengetahuan.

**Gerbang persetujuan spiritual** (*tri-state consent*: `None`/`True`/`False`) memastikan ayat tidak diinjeksikan tanpa izin — sistem memberi konseling psikologis murni bila pengguna menolak. Keputusan etis ini adalah jawaban langsung atas peringatan Nganyu [2025] soal perlunya pendekatan sensitif-budaya dan penghormatan atas keyakinan pengguna. Sementara itu, *system prompt* yang secara eksplisit memosisikan agen sebagai "psikolog dan konselor Kristen" mendukung transparansi sumber respons — faktor yang terbukti memengaruhi kepercayaan dan persepsi pengguna terhadap dukungan AI menurut Shen et al. [2024] dan Jain et al. [2024].

---

## 4.7.6 Validasi Ilmiah Metrik Evaluasi Sistem (Sub-bab 4.7.2 dan Evaluasi Eksternal)

Selain metrik klasifikasi pada Sub-bab 4.5, kualitas keluaran sistem diukur melalui tiga jalur evaluasi yang saling melengkapi. Ketiganya memerlukan justifikasi metodologis, baik dari sisi *mengapa* metriknya dipilih maupun *apa* makna angka yang diperoleh.

### Evaluasi RAGAS: Otomasi Penilaian Teks Generatif

Kualitas respons LLM dinilai memakai kerangka RAGAS dengan metrik kustom `LABAN_Counseling_Standard` (berbasis `AspectCritic`), menghasilkan skor rata-rata **0,9889** atas 90 sampel (89 dari 90 memenuhi kriteria). Pemilihan pendekatan otomatis berbasis *LLM-as-a-Judge* ini — bukan penilaian manual penuh — divalidasi langsung oleh Es et al. [2024], pencipta RAGAS, yang menegaskan bahwa evaluasi RAG bersifat multidimensi (kualitas retrieval, kesetiaan generasi, dan mutu jawaban) dan dapat dinilai secara *reference-free* tanpa anotasi manusia, sehingga mempercepat siklus evaluasi arsitektur RAG. Justifikasi arsitektural yang lebih luas datang dari Gao et al. [2024], yang menempatkan kerangka evaluasi otomatis sebagai komponen matang dalam ekosistem RAG modern.

Namun, temuan penelitian ini juga mengonfirmasi peringatan metodologis penting: metrik generik RAGAS (*Faithfulness* 0,245; *Answer Relevancy* 0,015 pada iterasi awal) terbukti tidak sesuai untuk dialog konseling yang bersifat empatik dan reflektif, karena metrik tersebut dirancang untuk *question-answering* faktual. Penggantian ke `AspectCritic` — yang menilai empati, keselarasan alkitabiah, kepatuhan tahap, dan non-redundansi dalam satu kriteria biner — merupakan adaptasi domain yang justru sejalan dengan sifat modular RAGAS sebagaimana dijelaskan Es et al. [2024]. Skor tinggi yang konsisten, termasuk kelulusan seluruh kasus *adversarial*, menunjukkan bahwa metrik yang selaras-domain mampu menilai kualitas respons secara lebih sahih dibanding metrik generik.

### Evaluasi Retrieval Ayat: Cohen's Kappa dan Subjektivitas Interpretasi

Relevansi ayat hasil retrieval dievaluasi oleh **3 responden** atas **30 kasus uji** (pasangan input pengguna–ayat) dengan penilaian biner, menghasilkan rata-rata **Cohen's Kappa 0,5346443353**. Merujuk pada kriteria baku Landis & Koch [1977] — yang membagi kekuatan kesepakatan menjadi *slight* (0,00–0,20), *fair* (0,21–0,40), *moderate* (0,41–0,60), *substantial* (0,61–0,80), dan *almost perfect* (0,81–1,00) — nilai ini tergolong **persetujuan moderat** (*moderate agreement*). Artinya, sistem retrieval telah berfungsi jauh di atas kesepakatan acak, namun belum mencapai ambang reliabilitas kuat ($\kappa \ge 0{,}60$).

Nilai moderat ini dapat dijelaskan secara substantif, bukan sekadar sebagai kelemahan teknis. Penilaian relevansi ayat Alkitab terhadap keluhan konseling melibatkan **subjektivitas interpretasi teologis**: satu ayat yang dianggap sangat relevan oleh seorang responden dapat dinilai kurang tepat oleh responden lain karena perbedaan penafsiran dan pemaknaan spiritual. Hal ini tercermin pada variasi penilaian antar-responden (17, 22, dan 17 dari 30 kasus dinilai relevan), yang menandakan sumber varians utama adalah subjektivitas manusia dalam menafsirkan makna ayat, bukan semata kegagalan mesin retrieval. Temuan ini konsisten dengan sifat Kappa yang dijelaskan Landis & Koch [1977] sebagai ukuran kesepakatan terkoreksi-kebetulan yang sensitif terhadap perbedaan penilaian antar-pengamat.

### Evaluasi Validitas Konseling: Content Validity Index (CVI)

Validitas fungsi konseling secara medis/psikologis diukur oleh **1 pakar (psikolog)** yang menelaah satu sesi utuh berisi **18 pasangan** pertukaran pengguna–chatbot lintas seluruh tahap, memakai skala relevansi 1–4. Pakar menyatakan setuju (skor 3 atau 4) pada **14 dari 18 item**, menghasilkan **CVI 0,7777777778**. Menurut Polit & Beck [2006], nilai *Item-level Content Validity Index* (I-CVI) $\ge 0{,}78$ mengindikasikan validitas isi yang baik, sehingga skor **0,78** ini berada tepat pada ambang **validitas yang dapat diterima** (*acceptable validity*).

Meskipun demikian, angka ini juga menyingkap ruang perbaikan yang konkret. Empat item yang tidak disetujui pakar menandakan adanya aspek respons yang belum sepenuhnya selaras dengan standar praktik klinis. Polit & Beck [2006] menekankan bahwa untuk instrumen baru, ambang penerimaan skala (S-CVI) yang umum dianut adalah $\ge 0{,}80$ — sehingga skor 0,78 memposisikan sistem pada status "layak namun perlu penyempurnaan iteratif" sebelum siap untuk penggunaan klinis penuh. Penilaian oleh satu pakar juga membatasi kekuatan generalisasi; sebagaimana disarankan Polit & Beck [2006], pelibatan lebih banyak pakar akan memperkuat estimasi validitas isi. Dengan demikian, CVI 0,78 sebaiknya dibaca sebagai konfirmasi bahwa sistem sudah berada pada arah yang benar, sekaligus penanda area spesifik yang menuntut iterasi lanjutan bersama ahli.

---

## 4.7.7 Posisi terhadap Kesenjangan Penelitian

Secara ringkas, yang telah dikerjakan mengisi celah yang diidentifikasi di Bab 2: menggabungkan **klasifikasi multi-intent** (dengan kemampuan ZSL nyata, F1-Macro *unseen* 0,8913) dan **RAG spiritual** (retrieval ayat berbasis LLM reranker dengan *grounding*) dalam satu sistem konseling — sesuatu yang belum ditempuh gabungan penelitian terdahulu (Barnett et al. [2021]; Malhotra et al. [2021]; Nayinzira et al. [2024]). Setiap komponen — dari pemilihan backbone hingga *consent gate* — bukan keputusan intuitif, melainkan bersandar pada temuan empiris yang dapat ditelusuri.

---

## Daftar Pustaka

Assayed, S. K., Shaalan, K., & Alkhatib, M. (2022). A Chatbot Intent Classifier for Supporting High School Students. *ICST Transactions on Scalable Information Systems*, e1. https://doi.org/10.4108/eetsis.v10i2.2948

Barnett, A., Savic, M., Pienaar, K., Carter, A., Warren, N., Sandral, E., Manning, V., & Lubman, D. I. (2021). Enacting 'more-than-human' care: Clients' and counsellors' views on the multiple affordances of chatbots in alcohol and other drug counselling. *International Journal of Drug Policy*, 94, 102910. https://doi.org/10.1016/j.drugpo.2020.102910

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAs: Automated Evaluation of Retrieval Augmented Generation. Dalam N. Aletras & O. De Clercq (Ed.), *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations* (hlm. 150–158). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.eacl-demo.16

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey* (arXiv:2312.10997). arXiv. https://doi.org/10.48550/arXiv.2312.10997

Hamka, Suen, M.-W., Ramadhan, Y. A., Yusuf, M., & Wang, J.-H. (2022). Spiritual Well-Being, Depression, Anxiety, and Stress in Indonesian Muslim Communities During COVID-19. *Psychology Research and Behavior Management*, 15, 3013–3025. https://doi.org/10.2147/PRBM.S381926

Indrayanti, I., Salsabila, A. K., Amrita, V., Alhaddad, M. M., Saskia, A. B., & Ramadhani, D. P. (2025). PsyBot: A randomized controlled trial of WhatsApp-based psychological first aid to reduce loneliness among 18–22-year-old students in Yogyakarta, Indonesia. *SSM - Mental Health*, 8, 100504. https://doi.org/10.1016/j.ssmmh.2025.100504

Jain, G., Pareek, S., & Carlbring, P. (2024). Revealing the source: How awareness alters perceptions of AI and human-generated mental health responses. *Internet Interventions*, 36, 100745. https://doi.org/10.1016/j.invent.2024.100745

Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159–174. https://doi.org/10.2307/2529310

Liu, J., Tan, Y. K., Fu, B., & Lim, K. H. (2025). *From Intents to Conversations: Generating Intent-Driven Dialogues with Contrastive Learning for Multi-Turn Classification* (arXiv:2411.14252). arXiv. https://doi.org/10.48550/arXiv.2411.14252

Mahardika, B. T., & Hasan, A. M. (2025). Application of GPT in Chatbots to Facilitate Knowledge Management System Interaction Using LangChain (Case Study: PT Softbless Solutions). Vol. 5(6).

Malhotra, G., Waheed, A., Srivastava, A., Akhtar, M. S., & Chakraborty, T. (2021). *Speaker and Time-aware Joint Contextual Learning for Dialogue-act Classification in Counselling Conversations* (arXiv:2111.06647). arXiv. https://doi.org/10.48550/arXiv.2111.06647

Nayinzira, J. P., & Adda, M. (2024). SentimentCareBot: Retrieval-Augmented Generation Chatbot for Mental Health Support with Sentiment Analysis. *Procedia Computer Science*, 251, 334–341. https://doi.org/10.1016/j.procs.2024.11.118

Nganyu, G. N. (2025). Theological and Psychological Integration in Christian Psychotherapy: A Critical Review of the Literature and Implications for Church-Based Practice. *Greener Journal of Social Sciences*, 15(1), 75–82. https://doi.org/10.15580/gjss.2025.1.022525031

Polit, D. F., & Beck, C. T. (2006). The content validity index: Are you sure you know what's being reported? Critique and recommendations. *Research in Nursing & Health*, 29(5), 489–497. https://doi.org/10.1002/nur.20147

Ponmagal, R. S., Deep, H., & Yadav, D. (2025). Mental Health Support Using Gen-AI Shot Prompting Technique and Vector Embeddings. Dalam *2025 International Conference on Data Science and Business Systems* (hlm. 1–5). IEEE. https://doi.org/10.1109/ICDSBS63635.2025.11031881

Shen, J., DiPaola, D., Ali, S., Sap, M., Park, H. W., & Breazeal, C. (2024). Empathy Toward Artificial Intelligence Versus Human Experiences and the Role of Transparency in Mental Health and Social Support Chatbot Design: Comparative Study. *JMIR Mental Health*, 11, e62679. https://doi.org/10.2196/62679

Subu, M. A., Holmes, D., Arumugam, A., Al-Yateem, N., Maria Dias, J., Rahman, S. A., Waluyo, I., Ahmed, F. R., & Abraham, M. S. (2022). Traditional, religious, and cultural perspectives on mental illness: a qualitative study on causal beliefs and treatment use. *International Journal of Qualitative Studies on Health and Well-Being*, 17(1), 2123090. https://doi.org/10.1080/17482631.2022.2123090
