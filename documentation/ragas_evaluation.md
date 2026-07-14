# BAB 4.7.2 — Evaluasi RAGAS (Retrieval-Augmented Generation Assessment)

---

## 4.7.2.1 Latar Belakang dan Tujuan Evaluasi

Evaluasi kualitas sistem *Retrieval-Augmented Generation* (RAG) pada chatbot konseling Alkitab tidak dapat diukur secara memadai hanya melalui metrik klasifikasi intent (lihat Sub-bab 4.5). Diperlukan pengukuran terpisah terhadap kualitas keluaran generator: apakah respons LLM secara konsisten empatik, tidak menghakimi, selaras dengan prinsip konseling alkitabiah, dan patuh terhadap aturan tahap `SessionManager` (lihat Sub-bab 4.6.2). Untuk kebutuhan ini, penelitian menggunakan **RAGAS** (Es et al., 2024), sebuah kerangka evaluasi *open-source* yang menilai kualitas respons menggunakan pendekatan **LLM-as-a-Judge**.

RAGAS dipilih karena tidak bergantung pada metrik kesamaan teks permukaan seperti BLEU atau ROUGE, melainkan menilai kualitas semantik melalui model bahasa independen yang membaca input pengguna, konteks yang diambil, dan respons yang dihasilkan.

---

## 4.7.2.2 Metrik yang Dievaluasi: LABAN Counseling Standard (AspectCritic)

Iterasi awal evaluasi (lihat riwayat pengembangan) menggunakan empat metrik generik bawaan RAGAS — *Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall*. Metrik tersebut dirancang untuk pipeline *question-answering* faktual dan terbukti tidak cocok untuk domain dialog konseling: *Faithfulness* mengekstraksi klaim faktual diskrit dari respons empatik yang secara struktural bukan pernyataan faktual, sementara *Answer Relevancy* membandingkan makna semantik respons terhadap pertanyaan buatan dari kolom `reference`, yang tidak merepresentasikan gaya jawaban konseling yang panjang dan reflektif. Kedua metrik ini secara sistematis menghasilkan skor rendah tanpa mencerminkan kualitas respons yang sebenarnya.

Sebagai gantinya, digunakan metrik kustom tunggal berbasis **`AspectCritic`** RAGAS bernama **`LABAN_Counseling_Standard`**, didefinisikan di `evaluation/ragas_manager/ragas_engine.py`:

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

Dataset evaluasi dibangun melalui dua fase pada `evaluation/ragas_manager/ragas_engine.py`:

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

Berbeda dari pendekatan sebelumnya yang memerlukan agregasi `ragas_per_stage.csv` terpisah untuk *menyimpulkan* pelanggaran aturan tahap secara tidak langsung, definisi `LABAN_CRITERIA_DEFINITION` kini menyertakan kepatuhan tahap sebagai bagian eksplisit dari kriteria penilaian per sampel. `ragas_manager/ragas_engine.py` tetap menyisipkan prefiks tahap (`[Tahap Konseling: {stage}] ...`) ke `user_input` yang dikirim ke *judge*, sehingga *judge* memiliki konteks tahap saat menilai. Agregasi `ragas_per_stage.csv` tetap dihasilkan untuk melihat distribusi skor per tahap.

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
python evaluation/ragas_manager/ragas_engine.py --generate

# (manual) isi kolom `reference` pada evaluation/data/ragas_testset.csv

# Fase 2 — jalankan evaluasi penuh
python evaluation/ragas_manager/ragas_engine.py --evaluate
```

Fase 2 memvalidasi bahwa seluruh kolom `reference` telah terisi, menjalankan pipeline RAG penuh (klasifikasi → retrieval dual-FAISS → generasi LLM) untuk setiap sampel, menyusun `EvaluationDataset` RAGAS, lalu menjalankan metrik `LABAN_Counseling_Standard` (`AspectCritic`) dengan *judge* `gpt-4o`. Tiga berkas keluaran disimpan ke `evaluation/results/`:

| Berkas | Isi | Jumlah Baris |
|---|---|---|
| `ragas_per_sample.csv` | Skor per sampel untuk seluruh kasus uji | 90 |
| `ragas_summary.csv` | Statistik agregat (mean, median, std, min, max) | 1 (satu metrik) |
| `ragas_per_stage.csv` | Rata-rata skor per tahap konseling | 6 (satu per tahap) |

---

## 4.7.2.5 Mekanisme Eksekusi Evaluasi

Subbab ini mendokumentasikan logika kode yang mendasari mekanisme eksekusi evaluasi melalui antarmuka baris perintah (*Command Line Interface*/CLI) pada skrip `evaluation/ragas_manager/ragas_engine.py`. Pemahaman terhadap mekanisme ini penting agar pembaca dapat mereplikasi, memverifikasi, maupun memodifikasi alur evaluasi secara mandiri.

### a. Definisi Argumen CLI (`argparse`)

Titik masuk utama skrip dikendalikan oleh blok `if __name__ == "__main__"` berikut, yang mendefinisikan dua argumen eksklusif (`--generate` dan `--evaluate`) menggunakan modul `argparse` standar Python:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for Biblical Counseling Chatbot"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate",
        action="store_true",
        help="Phase 1: Generate test template CSV (opt.RAGAS_TESTSET_SIZE samples)",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Phase 2: Run RAG pipeline + RAGAS evaluation",
    )

    args = parser.parse_args()

    if args.generate:
        generate_testset()
    elif args.evaluate:
        run_evaluation()
```

Kedua argumen ditempatkan dalam `add_mutually_exclusive_group(required=True)`, yang berarti pengguna **wajib** memberikan tepat satu dari dua argumen tersebut setiap kali skrip dijalankan — tidak boleh keduanya sekaligus, dan tidak boleh tanpa argumen apapun. Kondisi `if args.generate` memanggil fungsi `generate_testset()` (Fase 1), sedangkan `elif args.evaluate` memanggil `run_evaluation()` (Fase 2).

---

### b. Fase 1: Flag `--generate` dan Fungsi `generate_testset()`

Saat flag `--generate` diumpankan ke skrip, fungsi `generate_testset()` dieksekusi. Fungsi ini bertanggung jawab membangun berkas templat CSV yang akan digunakan sebagai testset pada Fase 2. Inti alur kerjanya adalah sebagai berikut:

```python
def generate_testset():
    """Phase 1: Sample QnA pairs and create a template CSV."""

    # 1. Muat dataset QnA mentah dari data/dataset_qna.csv
    df = pd.read_csv(QNA_CSV)
    df = df.dropna(subset=["question"])
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].str.len() > 0]

    random.seed(RANDOM_SEED)       # seed=42 untuk reproduksibilitas
    np.random.seed(RANDOM_SEED)

    # 2. Klasifikasi seluruh korpus ke tahap konseling via heuristik regex
    df["stage"] = df["question"].apply(classify_stage)

    # 3. Sampling seimbang lintas enam tahap sesuai STAGE_QUOTA
    sample_df = balance_by_stage(
        df,
        total=SAMPLE_SIZE,
        quota_override=STAGE_QUOTA,
        max_caps=STAGE_MAX_CAPS,
    )

    # 4. Sisipkan kasus adversarial ke tahap pembahasan/intervensi
    sample_df = inject_adversarial_cases(sample_df)

    # 5. Bangun DataFrame keluaran dengan kolom reference kosong
    out_df = pd.DataFrame({
        "question":        sample_df["question"].values,
        "stage":           sample_df["stage"].values,
        "reference":       "",   # diisi manual oleh peneliti
        "intent_override": "",   # opsional, untuk tahap solusi/relaksasi
        "adversarial":     sample_df["adversarial"].values,
    })

    # 6. Simpan ke evaluation/data/ragas_testset.csv
    os.makedirs(TESTSET_DIR, exist_ok=True)
    out_df.to_csv(TESTSET_CSV, index=False, encoding="utf-8-sig")
```

**Penjelasan alur `--generate`:**

Fungsi ini **tidak** berinteraksi dengan LLM maupun memanggil API eksternal apapun — seluruh proses berjalan secara lokal dan tidak memerlukan koneksi jaringan. Alur dimulai dengan memuat dataset QnA mentah (`data/dataset_qna.csv`), membersihkan baris kosong, lalu mengklasifikasikan setiap pertanyaan ke salah satu dari enam tahap konseling (`pembukaan`, `pembahasan`, `intervensi`, `solusi`, `relaksasi`, `penutupan`) menggunakan fungsi `classify_stage()` berbasis pencocokan pola regex.

Setelah seluruh korpus diklasifikasikan, fungsi `balance_by_stage()` mengambil sampel sesuai kuota per tahap yang ditetapkan pada konstanta `STAGE_QUOTA` (mis. `pembahasan`: 40, `solusi`: 20) dengan `random_state=42` demi reproduksibilitas. Fungsi `inject_adversarial_cases()` kemudian menggantikan sejumlah baris sampel pada tahap `pembahasan` dan `intervensi` dengan input buatan yang secara eksplisit memancing respons solusi prematur, guna menguji ketahanan sistem terhadap tekanan pengguna.

Keluaran akhirnya adalah berkas CSV (`evaluation/data/ragas_testset.csv`) dengan kolom `reference` yang **masih kosong** — kolom ini harus diisi secara manual oleh peneliti dengan jawaban ideal per pertanyaan sebelum Fase 2 dijalankan.

---

### c. Fase 2: Flag `--evaluate` dan Fungsi `run_evaluation()`

Saat flag `--evaluate` diumpankan, fungsi `run_evaluation()` dieksekusi. Fungsi ini merupakan inti evaluasi yang melibatkan LLM generator (chatbot) dan LLM *judge* (RAGAS). Alurnya dibagi ke dalam sepuluh langkah terstruktur:

**Langkah 1–3: Validasi testset dan eksekusi pipeline RAG per sampel**

```python
def run_evaluation():
    """Phase 2: Run RAG pipeline on each sample and evaluate with RAGAS."""
    from dotenv import load_dotenv
    load_dotenv()

    # Langkah 1: Validasi berkas testset
    df = pd.read_csv(TESTSET_CSV)
    df = balance_by_stage(df, total=len(df))   # safety net untuk CSV yang diedit manual

    # Pastikan seluruh kolom reference telah terisi
    empty_refs = df["reference"].isna() | (df["reference"].astype(str).str.strip() == "")
    if empty_refs.any():
        sys.exit(1)   # hentikan proses bila ada reference yang kosong

    # Langkah 2: Inisialisasi RAG Engine
    from core.rag_engine import RAGEngine
    rag_engine = RAGEngine()

    # Langkah 3: Jalankan pipeline RAG untuk setiap sampel
    results = []
    for i, row in df.iterrows():
        question        = str(row["question"]).strip()
        stage           = str(row["stage"]).strip()
        reference       = str(row["reference"]).strip()
        intent_override_raw = str(row.get("intent_override", "")).strip()
        adversarial     = bool(row.get("adversarial", False))

        # Parse intent override (digunakan pada tahap solusi/relaksasi)
        override_intents = None
        if intent_override_raw and intent_override_raw != "nan":
            override_intents = [s.strip() for s in intent_override_raw.split(",") if s.strip()]

        # Panggil RAG Engine — spiritual_consent=True memaksa retrieval ayat aktif
        rag_result = rag_engine.generate_response(
            user_input=question,
            current_stage=stage,
            override_intents=override_intents,
            spiritual_consent=True,
        )

        response = rag_result["response"]
        ctx      = rag_result["context_used"]

        # Kumpulkan konteks yang diambil dari indeks FAISS
        retrieved_contexts = []
        if ctx.get("example_answer"):
            retrieved_contexts.append(ctx["example_answer"])
        if ctx.get("bible_verses"):
            retrieved_contexts.append(ctx["bible_verses"])
        if not retrieved_contexts:
            retrieved_contexts = ["(no context retrieved)"]

        results.append({
            "question":           question,
            "stage":              stage,
            "reference":          reference,
            "response":           response,
            "retrieved_contexts": retrieved_contexts,
            "adversarial":        adversarial,
        })
```

**Penjelasan Langkah 1–3:**

Sebelum eksekusi dimulai, skrip memuat ulang testset dari berkas CSV dan memverifikasi bahwa seluruh kolom `reference` telah terisi — proses akan dihentikan (`sys.exit(1)`) bila ada baris yang masih kosong, mencegah evaluasi berjalan di atas data yang tidak lengkap.

Setelah validasi, `RAGEngine` diinisialisasi dan digunakan untuk menghasilkan respons terhadap setiap pertanyaan dalam testset. Parameter `spiritual_consent=True` dipaksakan secara eksplisit agar retrieval ayat Alkitab dari indeks FAISS selalu aktif pada tahap yang relevan (`solusi`, `relaksasi`), terlepas dari nilai variabel *consent* pada sesi interaktif nyata. Konteks yang berhasil diambil (`example_answer` dari indeks QnA dan `bible_verses` dari indeks Alkitab) dikumpulkan sebagai `retrieved_contexts` — kolom ini merupakan input wajib bagi metrik RAGAS.

---

**Langkah 4–8: Pembangunan dataset RAGAS, konfigurasi LLM *judge*, dan eksekusi evaluasi**

```python
    # Langkah 4: Bangun EvaluationDataset RAGAS
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import AspectCritic

    samples = []
    for r in results:
        sample = SingleTurnSample(
            # Prefiks tahap membantu judge menilai kepatuhan stage-appropriateness
            user_input=f"[Tahap Konseling: {r['stage']}] {r['question']}",
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)

    # Langkah 5: Konfigurasi LLM judge via langchain_openai
    from langchain_openai import ChatOpenAI

    openai_api_key = os.getenv(opt.RAGAS_JUDGE_API_KEY_ENV)
    ragas_judge_llm = ChatOpenAI(
        model=opt.RAGAS_JUDGE_MODEL,          # "gpt-4o"
        api_key=openai_api_key,
        temperature=opt.RAGAS_JUDGE_TEMPERATURE,
    )

    # Langkah 6: Definisikan metrik AspectCritic kustom LABAN
    laban_metric = AspectCritic(
        name="LABAN_Counseling_Standard",
        definition=LABAN_CRITERIA_DEFINITION,
    )
    metrics = [laban_metric]

    # Langkah 7: Preflight check (tanpa memanggil API)
    from ragas.validation import validate_required_columns, validate_supported_metrics
    validate_required_columns(eval_dataset, metrics)
    validate_supported_metrics(eval_dataset, metrics)

    # Langkah 8: Jalankan evaluasi RAGAS
    ragas_results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_judge_llm,
    )
```

**Penjelasan Langkah 4–8:**

Pada langkah 4, setiap hasil respons dari pipeline RAG dikemas ke dalam objek `SingleTurnSample` milik RAGAS. Kolom `user_input` secara sengaja diberi prefiks `[Tahap Konseling: {stage}]` sebelum pertanyaan asli pengguna — mekanisme ini memberikan konteks tahap konseling secara eksplisit kepada LLM *judge*, sehingga *judge* dapat menilai kepatuhan tahap (misalnya, apakah chatbot memberikan solusi prematur pada tahap `pembahasan`) tanpa perlu memodifikasi definisi metrik.

Langkah 5 menginisialisasi LLM *judge* menggunakan `ChatOpenAI` dari pustaka `langchain_openai` dengan model `gpt-4o` (dikonfigurasi melalui `opt.RAGAS_JUDGE_MODEL`). LLM *judge* ini bersifat **sepenuhnya terpisah** dari LLM generator chatbot (`opt.CHATBOT_LLM_PROVIDER = "gemini"`) — pemisahan ini merupakan desain yang disengaja untuk menghindari *self-evaluation bias*, yakni kecenderungan model menilai keluarannya sendiri secara tidak objektif.

Langkah 6 mendefinisikan metrik `AspectCritic` dengan nama `LABAN_Counseling_Standard` dan kriteria yang tercantum pada Sub-bab 4.7.2.2. Fungsi `evaluate()` dari RAGAS pada langkah 8 secara otomatis menyuntikkan `ragas_judge_llm` ke dalam metrik (via parameter `llm=`) dan menjalankan penilaian biner (0 atau 1) terhadap seluruh sampel dalam `eval_dataset`. Langkah 7 (preflight check) memvalidasi kesesuaian kolom dan metrik secara lokal tanpa menghabiskan panggilan API, sehingga kesalahan konfigurasi dapat terdeteksi sebelum biaya API dikeluarkan.

---

**Langkah 9–10: Penyimpanan hasil dan ringkasan konsol**

```python
    # Langkah 9: Konversi dan simpan hasil ke tiga berkas CSV
    results_df = ragas_results.to_pandas()

    # Tambahkan kolom metadata (stage, question, adversarial)
    results_df.insert(0, "adversarial", [r["adversarial"] for r in results])
    results_df.insert(0, "stage",       [r["stage"]       for r in results])
    results_df.insert(0, "question",    [r["question"]    for r in results])

    # ragas_per_sample.csv — skor per baris sampel
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "ragas_per_sample.csv"),
        index=False, encoding="utf-8-sig",
    )

    # ragas_summary.csv — statistik agregat (mean, median, std, min, max)
    summary_data = {}
    for col in metric_names:
        values = results_df[col].dropna()
        summary_data[col] = {
            "mean": values.mean(), "median": values.median(),
            "std":  values.std(),  "min":    values.min(),
            "max":  values.max(),  "count":  len(values),
        }
    pd.DataFrame(summary_data).T.to_csv(
        os.path.join(RESULTS_DIR, "ragas_summary.csv"),
        encoding="utf-8-sig",
    )

    # ragas_per_stage.csv — rata-rata skor per tahap konseling
    stage_metrics = []
    for stage in sorted(results_df["stage"].unique()):
        stage_mask = results_df["stage"] == stage
        stage_row  = {"stage": stage, "sample_count": stage_mask.sum()}
        for col in metric_names:
            values = results_df.loc[stage_mask, col].dropna()
            stage_row[f"{col}_mean"] = values.mean() if len(values) > 0 else None
            stage_row[f"{col}_std"]  = values.std()  if len(values) > 0 else None
        stage_metrics.append(stage_row)
    pd.DataFrame(stage_metrics).to_csv(
        os.path.join(RESULTS_DIR, "ragas_per_stage.csv"),
        index=False, encoding="utf-8-sig",
    )
```

**Penjelasan Langkah 9–10:**

Setelah eksekusi `evaluate()` selesai, hasilnya dikonversi ke DataFrame pandas menggunakan metode `ragas_results.to_pandas()`. Tiga kolom metadata (`question`, `stage`, `adversarial`) disisipkan kembali ke dalam DataFrame hasil — kolom-kolom ini tidak disertakan secara otomatis oleh RAGAS karena bukan bagian dari spesifikasi `SingleTurnSample` yang relevan untuk penilaian metrik, sehingga harus ditambahkan ulang dari daftar `results` yang dibangun pada Langkah 3.

Tiga berkas CSV kemudian disimpan ke direktori `evaluation/results/ragas_aspect_critic/`:
- `ragas_per_sample.csv`: skor biner per sampel beserta metadata (tahap, pertanyaan, status adversarial);
- `ragas_summary.csv`: statistik agregat global (mean, median, std, min, max, jumlah sampel valid);
- `ragas_per_stage.csv`: rata-rata dan standar deviasi skor per tahap konseling, berguna untuk mengidentifikasi tahap yang paling banyak mengalami kegagalan.

Keseluruhan berkas keluaran ini kemudian digunakan sebagai basis analisis kuantitatif dan kualitatif yang dipaparkan pada Sub-bab 4.7.2.6 dan 4.7.2.7.

---

## 4.7.2.6 Hasil Evaluasi

Evaluasi dijalankan terhadap **90 sampel** (`n=90`, `random_state=42`), diseimbangkan lintas enam tahap konseling dan mencakup sejumlah kasus **adversarial** (input pengguna yang secara eksplisit meminta solusi/keputusan instan di tahap eksplorasi, dirancang untuk menguji kepatuhan tahap secara lebih ketat — lihat Sub-bab 4.7.2.8 poin 2 versi sebelumnya). Hasil agregat global tercantum pada Tabel 4.7.2.1.

**Tabel 4.7.2.1 — Ringkasan Skor RAGAS Global (`ragas_summary.csv`)**

| Metrik | Mean | Median | Std | Min | Max | n |
|---|---:|---:|---:|---:|---:|---:|
| LABAN Counseling Standard | 0.9889 | 1.0000 | 0.1054 | 0.0000 | 1.0000 | 90 |

**Tabel 4.7.2.2 — Skor RAGAS per Tahap Konseling (`ragas_per_stage.csv`)**

| Tahap | n | LABAN Counseling Standard (Mean) | Std |
|---|---:|---:|---:|
| `pembukaan` | 5 | 1.0000 | 0.0000 |
| `pembahasan` | 44 | 0.9773 | 0.1508 |
| `intervensi` | 4 | 1.0000 | 0.0000 |
| `solusi` | 18 | 1.0000 | 0.0000 |
| `relaksasi` | 11 | 1.0000 | 0.0000 |
| `penutupan` | 8 | 1.0000 | 0.0000 |

### Interpretasi

**89 dari 90 sampel** (98,9%) dinilai memenuhi kriteria `LABAN_Counseling_Standard` oleh *judge* `gpt-4o`. Skor rata-rata global turun tipis dari 1,0000 (evaluasi n=30 sebelumnya) menjadi 0,9889 pada testset yang diperbesar dan mencakup kasus adversarial — konsisten dengan saran pengembangan lanjutan versi sebelumnya (poin 1 dan 2) yang meminta pembesaran testset dan pengujian adversarial untuk memvalidasi apakah skor sempurna bertahan di luar testset awal yang lebih kecil dan lebih mudah. Satu-satunya kegagalan berasal dari tahap `pembahasan` (44 dari 90 sampel di tahap ini, mean turun ke 0,9773).

Ketiga sampel adversarial yang diuji (permintaan solusi instan pada tahap `pembahasan`/`intervensi`, mis. "Apa yang harus aku lakukan besok pagi? Resign atau melawan?" dan "Tolong beri tahu saya teknik atau solusi apa yang paling ampuh untuk ini sekarang juga!") seluruhnya **lulus** (skor 1) — chatbot berhasil menahan diri dari solusi prematur dan tetap merespons dengan validasi/eksplorasi sesuai aturan tahap, bahkan di bawah tekanan permintaan eksplisit pengguna.

Kualitatif atas `ragas_per_sample.csv` mendukung pola ini. Beberapa observasi:

1. **Tahap `pembukaan` (n=5, seluruhnya sapaan "Halo")** menghasilkan respons singkat dan konsisten yang selalu menegaskan ruang aman ("ini adalah ruang yang aman untukmu") tanpa menyisipkan solusi atau ayat Alkitab — sesuai aturan `BIBLE_VERSE_STAGES` yang mengecualikan `pembukaan`.
2. **Tahap `pembahasan` dan `intervensi`** secara umum merespons dengan pertanyaan eksploratif reflektif (mis. "Apa yang membuatmu...", "Bagaimana rasanya...") tanpa memberikan solusi prematur, selaras dengan aturan tahap yang menyatakan kedua tahap ini bersifat eksplorasi/validasi saja. Satu kegagalan ditemukan pada sampel `pembahasan`: input pengguna menyatakan kemarahan disertai keinginan menyakiti orang lain ("Rasanya ingin membalas atau bahkan melukai mereka"), namun respons chatbot hanya melanjutkan eksplorasi biasa tanpa memberi bobot tambahan pada indikasi risiko tersebut — *judge* kemungkinan menilai ini sebagai respons yang kurang tepat untuk konteks yang lebih sensitif dari input pembahasan pada umumnya.
3. **Tahap `solusi` dan `relaksasi`** menyisipkan ayat Alkitab yang relevan dengan konteks masalah pengguna (mis. Mazmur 126:5 untuk proses relaksasi, 1 Petrus 5:7 dan Roma 12:2 untuk kecemasan/perubahan pola pikir) — retrieval ayat berfungsi sesuai desain hanya pada tahap yang diizinkan.
4. **Tahap `penutupan`** secara konsisten merangkum sesi dan menawarkan rujukan profesional di akhir respons, sesuai desain tahap ini pada Sub-bab 4.6.2.

Ukuran sampel per tahap kini jauh lebih memadai dibanding evaluasi n=30 sebelumnya (`pembahasan` n=44, `solusi` n=18, `relaksasi` n=11), meski tahap `intervensi` (n=4) dan `pembukaan` (n=5) tetap tergolong kecil dan perlu dibaca sebagai indikasi kualitatif, bukan klaim generalisasi statistik kuat.

Dibandingkan hasil evaluasi generik sebelumnya (Faithfulness 0.245, Answer Relevancy 0.015 — lihat riwayat pengembangan), skor tinggi konsisten pada `AspectCritic` bukan indikasi bahwa kualitas generator berubah drastis, melainkan konfirmasi bahwa **metrik generik RAGAS tidak cocok untuk domain dialog konseling**, sedangkan metrik `AspectCritic` yang dirancang khusus untuk kriteria domain (empati, keselarasan alkitabiah, kepatuhan tahap, non-redundansi) mampu menilai kualitas respons secara akurat — termasuk mendeteksi satu kegagalan nyata pada testset yang lebih besar dan lebih menantang.

---

## 4.7.2.7 Perbandingan Model Generator: Gemini 2.5-Flash vs Groq

> **Catatan istilah:** *Judge* RAGAS (`ChatOpenAI`, `opt.RAGAS_JUDGE_MODEL = "gpt-4o"`, konstruksi di `evaluation/ragas_manager/ragas_engine.py`) **tidak berubah** antara kedua *run* yang dibandingkan pada sub-bab ini — keduanya dinilai oleh OpenAI `gpt-4o` yang sama. Yang berbeda adalah `opt.CHATBOT_LLM_PROVIDER`, yaitu LLM **generator** chatbot: hasil pada Sub-bab 4.7.2.6 memakai generator Google `gemini-2.5-flash` (`ragas_summary.csv`, `ragas_per_stage.csv`), sedangkan berkas berakhiran `_groq` (`ragas_summary_groq.csv`, `ragas_per_stage_groq.csv`, `ragas_per_sample_groq.csv`) dihasilkan saat `opt.CHATBOT_LLM_PROVIDER = "groq"` (`llama-3.3-70b-versatile`). Sub-bab ini karenanya membandingkan **kualitas dua model generator** di bawah *judge* yang identik, bukan dua model *judge* yang berbeda. Kedua *run* memakai testset berukuran sama (`n=90`, distribusi tahap identik: `pembukaan` 5, `pembahasan` 44, `intervensi` 4, `solusi` 18, `relaksasi` 11, `penutupan` 8) sehingga perbandingan berikut bersifat *apple-to-apple* pada level ukuran dan komposisi sampel.

**Tabel 4.7.2.3 — Perbandingan Skor Global per Generator (`ragas_summary.csv` vs `ragas_summary_groq.csv`)**

| Metrik | Gemini 2.5-Flash (n=90) | Groq `llama-3.3-70b` (n=90) |
|---|---:|---:|
| Mean | 0.9889 | 0.9556 |
| Median | 1.0000 | 1.0000 |
| Std | 0.1054 | 0.2072 |
| Min | 0.0000 | 0.0000 |
| Max | 1.0000 | 1.0000 |

**Tabel 4.7.2.4 — Perbandingan Skor per Tahap Konseling (`ragas_per_stage.csv` vs `ragas_per_stage_groq.csv`)**

| Tahap | n | Mean (Gemini) | Mean (Groq) |
|---|---:|---:|---:|
| `pembukaan` | 5 | 1.0000 | 1.0000 |
| `pembahasan` | 44 | 0.9773 | 0.9773 |
| `intervensi` | 4 | 1.0000 | 0.5000 |
| `solusi` | 18 | 1.0000 | 0.9444 |
| `relaksasi` | 11 | 1.0000 | 1.0000 |
| `penutupan` | 8 | 1.0000 | 1.0000 |

Pembacaan langsung `ragas_per_sample_groq.csv` menunjukkan 4 dari 90 sampel Groq mendapat skor 0 (`solusi` ×1, `intervensi` ×2, `pembahasan` ×1) — dibanding hanya 1 kegagalan pada generator Gemini (`pembahasan`, kasus indikasi risiko pada Sub-bab 4.7.2.6). Seluruh 4 kegagalan Groq terjadi pada kasus **non-adversarial** (skor rata-rata kasus non-adversarial Groq 0.9540, n=87; kasus adversarial tetap sempurna, mean 1.0000, n=3).

### Interpretasi

Dengan *judge* `gpt-4o` yang identik dan ukuran/komposisi testset yang sama persis (n=90), generator Gemini 2.5-flash mencetak skor rata-rata lebih tinggi (0.9889) dan lebih stabil (std 0.1054) dibanding generator Groq `llama-3.3-70b-versatile` (mean 0.9556, std 0.2072). Karena *judge* tidak berubah, perbedaan ini murni mencerminkan perbedaan kualitas keluaran generator, bukan perbedaan kelonggaran/keketatan penilai. Penurunan skor Groq terkonsentrasi tajam pada tahap `intervensi` (mean turun dari 1.0000 menjadi 0.5000, n=4) dan sedikit pada `solusi` (1.0000 → 0.9444, n=18) — dua tahap yang menuntut respons eksploratif/validasi tanpa solusi prematur (lihat Sub-bab 4.7.2.2). Tahap `pembahasan` justru identik pada kedua generator (0.9773, kegagalan yang sama pada kasus indikasi risiko), menandakan kegagalan tersebut kemungkinan bersifat sistemik pada level *prompt*/kriteria, bukan spesifik generator. Bahwa seluruh kegagalan tambahan pada Groq muncul di kasus non-adversarial (bukan kasus yang sengaja dirancang menantang) mengindikasikan penyebabnya adalah karakteristik gaya jawab generator Groq itu sendiri pada tahap `intervensi`/`solusi`, bukan tingkat kesulitan kasus uji.

Temuan ini menunjukkan bahwa pemilihan generator memengaruhi kepatuhan terhadap `LABAN_Counseling_Standard` secara independen dari pemilihan *judge*, dan mendukung keputusan konfigurasi saat ini (`opt.CHATBOT_LLM_PROVIDER = "gemini"`) sebagai generator produksi dibanding Groq, setidaknya pada tahap `intervensi` dan `solusi`.

---

## 4.7.2.8 Saran Pengembangan Lanjutan

1. **Perbesar lebih lanjut n untuk tahap yang masih kecil** (`intervensi` n=4, `pembukaan` n=5), agar skor per tahap tersebut dapat divalidasi pada populasi sampel yang lebih besar dan lebih beragam sebelum diklaim sebagai kesimpulan final.
2. **Investigasi lebih lanjut kegagalan pada tahap `pembahasan`**: sampel yang gagal melibatkan input pengguna dengan indikasi risiko (keinginan menyakiti orang lain) dan gagal identik pada kedua generator (Gemini dan Groq) — pertimbangkan menambah kriteria eksplisit terkait deteksi risiko/eskalasi ke `LABAN_CRITERIA_DEFINITION`, atau menambah kasus serupa pada testset untuk memastikan ini bukan kegagalan acak.
3. **Perluas cakupan kasus adversarial/edge-case** di luar tiga sampel saat ini (mis. input pengguna yang ambigu terhadap tahap, permintaan solusi berulang dalam satu sesi) untuk memperkuat keyakinan bahwa kepatuhan tahap tahan terhadap tekanan pengguna yang lebih beragam.
4. **Selidiki kegagalan generator Groq pada tahap `intervensi`** (Sub-bab 4.7.2.7): tinjau ulang `prompt_template` khusus tahap ini bila Groq dipertimbangkan sebagai generator produksi di masa depan, karena Gemini 2.5-flash saat ini menunjukkan kepatuhan tahap yang lebih konsisten pada ukuran sampel yang diuji.
5. **Pertimbangkan *inter-rater reliability* yang sesungguhnya**: jalankan `LABAN_Counseling_Standard` dengan *judge* alternatif (mis. Gemini 2.5 Pro atau Groq sebagai `RAGAS_JUDGE_PROVIDER`, bukan hanya sebagai generator) pada subset sampel yang sama untuk memvalidasi bahwa skor saat ini konsisten lintas model *judge* — perbandingan pada Sub-bab 4.7.2.7 menguji generator, belum menguji *judge*.
6. **Ulangi evaluasi setelah setiap perubahan signifikan** pada `prompt_template`, `BIBLICAL_SYNONYMS`, atau `MODEL_NAME` embedding, untuk memastikan skor tinggi tetap terjaga sebagai *regression check* — bukan hanya divalidasi sekali di titik waktu ini.

---

*Skrip evaluasi: `evaluation/ragas_manager/ragas_engine.py` | Hasil: `evaluation/results/ragas_summary.csv`, `ragas_per_stage.csv`, `ragas_per_sample.csv`*
*Generator: Google `gemini-2.5-flash` | Judge: OpenAI `gpt-4o` | Metrik: `LABAN_Counseling_Standard` (AspectCritic, biner) | Ukuran sampel: n=90, seed=42*
