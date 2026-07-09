# Skenario Klasifikasi Multi-Label (LABAN)

Dokumen ini menunjukkan apa yang sebenarnya terjadi di dalam classifier LABAN (`models/multilabel/predict.py` + `models/multilabel/bert_model.py`), dari kalimat mentah pengguna sampai intent yang terdeteksi. Tidak melibatkan RAG maupun `SessionManager` — murni pipeline klasifikasi.

Model ini pakai **dual encoder**: satu BERT (`bertlabelencoder`) mengubah 10 nama intent tetap menjadi vektor, satu BERT lagi (`bert`) mengubah kalimat pengguna menjadi vektor. Skor tiap intent dihitung dari seberapa "mirip" arah vektor kalimat pengguna dengan arah vektor tiap intent.

Ada 5 tahap yang dicatat tiap kali `predict()` dipanggil. Berikut jalannya untuk 2 contoh kalimat nyata.

---

## Input 1

> "Saya cukup senang karena nilai saya meningkat. IPK saya sekarang 3.06, dan beasiswa Teknik Nusantara saya masih berlanjut."

### Stage 1 — Label Encoding

10 nama intent (bukan kalimat pengguna) dilewatkan ke `bertlabelencoder`. Hasilnya, tiap intent punya satu vektor 768-dimensi (`pooler_output`) yang jadi "acuan makna" untuk intent tersebut. Tahap ini sama untuk semua kalimat pengguna — tidak berubah per input, karena label-nya tetap.

| Label | Norm | Mean |
|---|---:|---:|
| Mengisyaratkan Butuh Bantuan Profesional | 21.3768 | 0.0487 |
| Mengisyaratkan Gejala Fisik | 25.2119 | -0.0517 |
| Menyatakan Perasaan Benci dan Jijik | 17.6441 | 0.0344 |
| Menyatakan Perasaan Marah dan Frustasi | 20.7480 | 0.0579 |
| Menyatakan Perasaan Percaya | 17.6137 | 0.0289 |
| Menyatakan Perasaan Sebelum Menghadapi Kejadian | 19.3302 | 0.0482 |
| Menyatakan Perasaan Sedih dan Kehilangan | 14.9686 | 0.0216 |
| Menyatakan Perasaan Takut dan Kecemasan | 17.2459 | 0.0010 |
| Menyatakan Rasa Syukur dan Apresiasi | 20.9806 | -0.0395 |
| Menyatakan Reaksi Terkejut dan Tidak Terduga | 19.3656 | -0.0013 |

**Kenapa Norm yang dipakai sebagai wakil, bukan Mean:** Mean itu rata-rata dari 768 angka di vektor tersebut — karena output BERT sudah ter-normalisasi (lewat `tanh` di pooler), rata-ratanya selalu mendekati 0 untuk semua label, jadi Mean nyaris tidak membedakan satu intent dari intent lain (lihat kolom Mean di atas — semuanya kecil). Norm (panjang/magnitude vektor) itu beda — dia langsung ikut menentukan hasil perkalian dot product di tahap logits (`gram = clusters @ clustersᵀ`), jadi Norm punya kaitan langsung ke perhitungan sebenarnya, sementara Mean cuma statistik kosmetik.

### Stage 2 — Input Encoding (Tokenisasi)

Kalimat pengguna dipecah jadi token yang dikenali BERT.

- Jumlah token terpakai: **25** dari `max_len=50`
- Token: `['[CLS]', 'saya', 'cukup', 'senang', 'karena', 'nilai', 'saya', 'meningkat', '.', 'ipk', 'saya', 'sekarang', '3', '.', '06', ',', 'dan', 'beasiswa', 'teknik', 'nusantara', 'saya', 'masih', 'berlanjut', '.', '[SEP]']`

`[CLS]` dan `[SEP]` adalah token penanda awal/akhir kalimat yang wajib ada di BERT. Sisa slot sampai 50 token diisi padding kosong supaya semua input punya ukuran seragam.

### Stage 3 — Semantic Embedding (Vektor Kalimat)

Kalimat yang sudah ditokenisasi dimasukkan ke `bert` (encoder kedua, khusus utterance) dan menghasilkan satu vektor 768-dimensi yang merepresentasikan makna keseluruhan kalimat.

- Dimensi vektor: `(1, 768)`
- 5 nilai pertama: `[-0.3110, -0.2275, -0.9991, 0.2652, 0.3569]`
- **Norm: 18.7162**

**5 nilai pertama** cuma cuplikan dimensi ke-0 sampai ke-4 dari 768 dimensi total. Dimensi-dimensi BERT tidak punya makna individual (bukan "dimensi 0 = kebahagiaan", dst) — jadi 5 angka ini sekadar bukti vektornya terisi (bukan kosong/NaN), bukan representasi makna.

Sama seperti Stage 1: **Norm** dipilih sebagai wakil vektor semantik ini, karena dialah yang benar-benar dipakai dalam perkalian `weight = pooled_output @ clustersᵀ` yang menentukan skor akhir — cuplikan 5 nilai pertama tidak dipakai dalam perhitungan apapun.

### Stage 4 — Logits Mentah

Vektor kalimat (Stage 3) dan vektor tiap label (Stage 1) dibandingkan lewat proyeksi gram-inverse: `logits = weight @ inverse(gram) × sqrt(hidden_size)`. Secara sederhana: ini mengukur seberapa searah vektor kalimat dengan vektor tiap intent, lalu hasilnya "dikoreksi" (lewat `inverse(gram)`, dibantu regularisasi Tikhonov supaya matriksnya selalu bisa dibalik) agar label-label yang miripsatu sama lain tidak saling menumpuk skornya secara tidak adil.

| Label | Logit |
|---|---:|
| Mengisyaratkan Butuh Bantuan Profesional | -9.9317 |
| Mengisyaratkan Gejala Fisik | -31.1091 |
| Menyatakan Perasaan Benci dan Jijik | -7.2308 |
| Menyatakan Perasaan Marah dan Frustasi | -6.5104 |
| Menyatakan Perasaan Percaya | -8.7002 |
| Menyatakan Perasaan Sebelum Menghadapi Kejadian | -6.7233 |
| Menyatakan Perasaan Sedih dan Kehilangan | -9.0328 |
| Menyatakan Perasaan Takut dan Kecemasan | -8.6660 |
| **Menyatakan Rasa Syukur dan Apresiasi** | **8.4079** |
| Menyatakan Reaksi Terkejut dan Tidak Terduga | -8.0214 |

Logit belum berupa persentase — masih angka mentah, bisa negatif atau sangat besar. Yang penting dilihat di sini: satu label ("Rasa Syukur dan Apresiasi") jauh lebih positif dibanding yang lain, artinya arah vektor kalimat paling dekat dengan label itu.

### Stage 5 — Sigmoid + Threshold

Tiap logit diubah jadi probabilitas 0–1 lewat fungsi sigmoid, lalu dibandingkan ke ambang batas (`thresold = 0.5`). Kalau lolos ambang, label dianggap terdeteksi.

| Label | Sigmoid | Lolos? |
|---|---:|:---:|
| Mengisyaratkan Butuh Bantuan Profesional | 0.0000 | - |
| Mengisyaratkan Gejala Fisik | 0.0000 | - |
| Menyatakan Perasaan Benci dan Jijik | 0.0007 | - |
| Menyatakan Perasaan Marah dan Frustasi | 0.0015 | - |
| Menyatakan Perasaan Percaya | 0.0002 | - |
| Menyatakan Perasaan Sebelum Menghadapi Kejadian | 0.0012 | - |
| Menyatakan Perasaan Sedih dan Kehilangan | 0.0001 | - |
| Menyatakan Perasaan Takut dan Kecemasan | 0.0002 | - |
| **Menyatakan Rasa Syukur dan Apresiasi** | **0.9998** | **LOLOS** |
| Menyatakan Reaksi Terkejut dan Tidak Terduga | 0.0003 | - |

### Hasil Akhir Input 1

- **Intent terdeteksi:** `Menyatakan Rasa Syukur dan Apresiasi`
- **Multilabel?** Tidak (hanya 1 intent lolos threshold)

Masuk akal — kalimatnya murni bersyukur atas nilai dan beasiswa yang lancar, tidak ada indikasi emosi negatif lain.

---

## Input 2

> "Aku merasa apapun yang aku lakukan nggak pernah cukup. Aku juara kelas, tapi dibandingin sama anak orang lain. Aku mau masuk arsitektur, tapi dilarang. Aku ngerasa nggak punya kendali atas hidupku sendiri."

### Stage 1 — Label Encoding

Sama seperti Input 1 — tabel vektor label tidak berubah karena 10 nama intent tetap sama, tidak tergantung kalimat pengguna. (Lihat tabel Stage 1 di Input 1 di atas.)

### Stage 2 — Input Encoding (Tokenisasi)

- Jumlah token terpakai: **41** dari `max_len=50`
- Token: `['[CLS]', 'aku', 'merasa', 'apapun', 'yang', 'aku', 'lakukan', 'nggak', 'pernah', 'cukup', '.', 'aku', 'juara', 'kelas', ',', 'tapi', 'dibanding', '##in', 'sama', 'anak', 'orang', 'lain', '.', 'aku', 'mau', 'masuk', 'arsitektur', ',', 'tapi', 'dilarang', '.', 'aku', 'ngerasa', 'nggak', 'punya', 'kendali', 'atas', 'hidupku', 'sendiri', '.', '[SEP]']`

Catatan menarik: `dibandingin` dipecah jadi `dibanding` + `##in`. Ini normal — tokenizer BERT (subword/WordPiece) memecah kata yang tidak ada di kamusnya menjadi potongan yang lebih umum dikenali. Tanda `##` berarti "sambungan dari token sebelumnya", bukan kata baru.

### Stage 3 — Semantic Embedding (Vektor Kalimat)

- Dimensi vektor: `(1, 768)`
- 5 nilai pertama: `[0.9791, 0.4655, -0.6991, 0.5053, -0.7242]`
- **Norm: 17.2097**

Norm-nya sedikit lebih kecil dari Input 1 (17.21 vs 18.72) — bukan berarti "lebih lemah maknanya", cuma menunjukkan arah dan besar vektor makna yang berbeda karena isi kalimatnya berbeda (kalimat ini lebih panjang dan menggabungkan banyak nuansa: capek, dibandingkan, dilarang, kehilangan kendali).

### Stage 4 — Logits Mentah

| Label | Logit |
|---|---:|
| Mengisyaratkan Butuh Bantuan Profesional | -10.7322 |
| Mengisyaratkan Gejala Fisik | -12.7131 |
| Menyatakan Perasaan Benci dan Jijik | -7.2241 |
| **Menyatakan Perasaan Marah dan Frustasi** | **9.7519** |
| Menyatakan Perasaan Percaya | -8.6329 |
| Menyatakan Perasaan Sebelum Menghadapi Kejadian | -6.5919 |
| **Menyatakan Perasaan Sedih dan Kehilangan** | **9.6443** |
| Menyatakan Perasaan Takut dan Kecemasan | -7.8484 |
| Menyatakan Rasa Syukur dan Apresiasi | -9.7224 |
| Menyatakan Reaksi Terkejut dan Tidak Terduga | -7.4766 |

Kali ini ada **dua** label yang logit-nya jauh lebih positif dibanding sisanya — pertanda kalimat ini memang mengandung dua nuansa emosi sekaligus (frustrasi karena dibanding-bandingkan/dilarang, dan sedih/kehilangan kendali atas hidup sendiri). Ini contoh nyata kenapa classifier ini didesain **multi-label**, bukan cuma pilih satu label tertinggi.

### Stage 5 — Sigmoid + Threshold

| Label | Sigmoid | Lolos? |
|---|---:|:---:|
| Mengisyaratkan Butuh Bantuan Profesional | 0.0000 | - |
| Mengisyaratkan Gejala Fisik | 0.0000 | - |
| Menyatakan Perasaan Benci dan Jijik | 0.0007 | - |
| **Menyatakan Perasaan Marah dan Frustasi** | **0.9999** | **LOLOS** |
| Menyatakan Perasaan Percaya | 0.0002 | - |
| Menyatakan Perasaan Sebelum Menghadapi Kejadian | 0.0014 | - |
| **Menyatakan Perasaan Sedih dan Kehilangan** | **0.9999** | **LOLOS** |
| Menyatakan Perasaan Takut dan Kecemasan | 0.0004 | - |
| Menyatakan Rasa Syukur dan Apresiasi | 0.0001 | - |
| Menyatakan Reaksi Terkejut dan Tidak Terduga | 0.0006 | - |

### Hasil Akhir Input 2

- **Intent terdeteksi:** `Menyatakan Perasaan Marah dan Frustasi`, `Menyatakan Perasaan Sedih dan Kehilangan`
- **Multilabel?** Ya (2 intent lolos threshold sekaligus)

---

## Ringkasan Alur (Ringkas)

```
Kalimat pengguna
      │
      ▼
[Stage 2] Tokenisasi → token BERT (+ padding ke max_len)
      │
      ▼
[Stage 3] Encoder utterance (BERT) → vektor makna kalimat (768-dim, diwakili oleh Norm)
      │                                         │
      │            [Stage 1] Encoder label (BERT) → vektor makna tiap 10 intent (768-dim, diwakili oleh Norm)
      │                                         │
      └───────────────┬─────────────────────────┘
                       ▼
      [Stage 4] Proyeksi gram-inverse → logit mentah per intent
                       │
                       ▼
      [Stage 5] Sigmoid → probabilitas 0-1 → bandingkan ke threshold 0.5
                       │
                       ▼
            Intent terdeteksi (bisa lebih dari satu)
```
