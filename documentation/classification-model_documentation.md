# BAB 4.4 — Klasifikasi Multi-Intent: Arsitektur LABAN dan Alur Inferensi

---

## 4.4.1 Adaptasi Arsitektur LABAN untuk Domain Bahasa Indonesia

Sistem klasifikasi intent yang diimplementasikan dalam penelitian ini dibangun di atas arsitektur **LABAN (Label-Aware BERT Attention Network)** yang dipublikasikan oleh Wu et al. pada EMNLP 2021. Arsitektur LABAN dipilih karena kemampuannya memproses label intent sebagai teks deskriptif — bukan sebagai indeks numerik — sehingga model dapat secara teori mengenali intent yang tidak pernah dilihat selama training (*zero-shot generalization*). Namun, implementasi asli LABAN dirancang untuk dataset bahasa Inggris dengan skenario benchmark yang berbeda secara signifikan dari konteks chatbot konseling berbahasa Indonesia ini. Oleh karena itu, sejumlah adaptasi teknis dilakukan sebelum arsitektur tersebut dapat dioperasikan.

Tabel berikut merangkum perbandingan antara LABAN asli (repositori GitHub Wu et al.) dan implementasi yang digunakan dalam penelitian ini:

| Aspek | LABAN Asli (GitHub) | Implementasi Penelitian Ini |
|:---|:---|:---|
| **Backbone** | Hardcoded: `bert-base-uncased` | Configurable via `AutoModel` + `config.py` |
| **Bahasa** | Inggris | Indonesia (`indobenchmark/indobert-base-p1`) |
| **Mode Arsitektur** | 6 mode *surface encoder*, 6 mode *label-aware layer* | Hanya mode `normal` (pooler) + `zero-shot` (Gram invers) |
| **Layer Tambahan** | `Dropout`, `classifier`, `mapping`, `relations1`, `relations2` | Tidak ada — proyeksi murni tanpa komponen supervised tambahan |
| **Stabilitas Numerik** | Inversi Gram tanpa regularisasi (`torch.inverse()` langsung) | Regularisasi Tikhonov: $\varepsilon = 10^{-4}$ sebelum inversi |
| **Format Data** | File Pickle (`.pkl`) — dataset benchmark MixATIS, MixSNIPS | CSV multi-label — pemisah titik koma (`;`) |
| **Validasi Training** | Tidak ada validasi per-epoch | Split 80/10/10 (train/val/test) + *best-checkpoint saving* |
| **Ukuran Kode Model** | ~227 baris | ~45 baris |

### Adaptasi 1 — Penggantian Backbone ke IndoBERT

Implementasi asli LABAN menggunakan `bert-base-uncased` yang dilatih pada korpus bahasa Inggris. Karena seluruh data dalam penelitian ini — teks input pengguna, label intent, dan dataset training — berbahasa Indonesia, backbone diganti ke **IndoBERT** (`indobenchmark/indobert-base-p1`). Untuk memfasilitasi komparasi backbone secara empiris (lihat Sub-bab 4.5.2), kode model menggunakan `AutoModel` dan `AutoTokenizer` dari HuggingFace, sehingga backbone dapat diubah hanya dengan memodifikasi satu parameter di `config.py` tanpa perubahan kode arsitektur.

### Adaptasi 2 — Pemangkasan Arsitektur

Repositori asli LABAN menyediakan 36 kombinasi mode (6 × 6) yang dirancang untuk eksperimen ablasi akademik komprehensif. Untuk sistem produksi ini, hanya dua mode yang relevan dipertahankan: *surface encoder* `normal` (menggunakan `pooler_output` dari token `[CLS]`) dan *label-aware layer* `zero-shot` (proyeksi melalui invers matriks Gram). Pemangkasan ini mereduksi ukuran kelas model dari ~227 baris menjadi ~45 baris, menghilangkan semua layer `self.dropout`, `self.classifier`, `self.mapping`, `self.relations1`, `self.relations2`, dan semua switch `self.mode`/`self.mode2`.

### Adaptasi 3 — Regularisasi Tikhonov

LABAN asli melakukan inversi matriks Gram secara langsung dengan `torch.inverse()` tanpa pengaman numerik. Apabila dua label intent memiliki embedding yang berkorelasi tinggi — yang mungkin terjadi pada pasang intent seperti "Sedih dan Kehilangan" dan "Takut dan Kecemasan" — matriks Gram dapat menjadi *near-singular*, menyebabkan `torch.inverse()` menghasilkan nilai `NaN` atau divergensi training.

Penelitian ini menambahkan **regularisasi Tikhonov** sebelum inversi:

**File: `models/multilabel/bert_model.py`**
```python
# Implementasi asli LABAN — rentan terhadap singularitas:
logits = torch.mm(weight, torch.inverse(gram)) * np.sqrt(768)

# Implementasi penelitian ini — dengan regularisasi Tikhonov:
gram = gram + torch.eye(gram.size(0), device=gram.device) * 1e-4
logits = torch.mm(weight, torch.inverse(gram)) * np.sqrt(opt.hidden_size)
```

Penambahan $\varepsilon \cdot I$ (dengan $\varepsilon = 10^{-4}$) pada diagonal matriks Gram menjamin bahwa matriks tersebut selalu *invertible*, tanpa mempengaruhi arah proyeksi secara signifikan.

### Adaptasi 4 — Format Data dan Pipeline Training

LABAN asli bergantung pada file `.pkl` dengan struktur khusus untuk dataset benchmark berbahasa Inggris. Penelitian ini membangun pipeline data dari awal menggunakan CSV multi-label dengan label dipisahkan tanda titik koma (`;`), sesuai dengan format dataset konseling Alkitab yang dikompilasi secara independen.

Pipeline training juga diperkuat dengan validasi F1 per-epoch, penyimpanan *best checkpoint* berdasarkan F1 validasi tertinggi, dan evaluasi akhir pada test set terpisah yang tidak pernah digunakan selama training. Implementasi asli LABAN tidak menyertakan mekanisme ini.

---

## 4.4.2 Alur Inferensi: Dari Input Pengguna ke Prediksi Multi-Label

Alur inferensi LABAN terdiri dari delapan tahap berurutan yang mengubah teks input pengguna menjadi daftar intent yang terdeteksi. Seluruh alur ini dieksekusi dalam satu panggilan `predictor.predict()` pada setiap giliran percakapan.

```
Teks Input Pengguna (raw string)
    ↓ [4.4.2.1] Penerimaan Input
Teks bersih — siap diteruskan ke tokenizer
    ↓ [4.4.2.2] Enkode Input (Tokenisasi)
utterance_ids (1, 50),  utterance_mask (1, 50)
    ├── [4.4.2.3] Utterance Encoder (IndoBERT)  →  pooled_output (1, 768)
    │
    ↓ [4.4.2.4] Persiapan Label
10 teks label intent → tokenisasi
    └── [4.4.2.5] Label Encoder (IndoBERT)  →  clusters (10, 768)
                                                      ↓
    [4.4.2.6] Proyeksi Gram Invers + Scaling  →  logits (1, 10)
                                                      ↓
    [4.4.2.7] Fungsi Aktivasi Sigmoid         →  probabilities (10,) ∈ (0, 1)
                                                      ↓
    [4.4.2.8] Threshold & Output Multi-Label  →  detected_intents (list)
```

### 4.4.2.1 Penerimaan Input Pengguna

Teks yang diketik pengguna melalui antarmuka browser diterima oleh Flask sebagai `request.json['message']` dan diteruskan ke `predictor.predict()` sebagai *raw string* Python. Tidak ada langkah preprocessing agresif (stemming, stopword removal, atau normalisasi tanda baca) yang diterapkan, karena model BERT-based dirancang untuk menerima teks sealami mungkin guna menangkap nuansa emosional yang implisit dalam gaya bahasa informal pengguna.

**File: `app.py` → `core/session_manager.py`**
```python
# Menerima input dari Flask dan meneruskannya tanpa modifikasi besar
user_message = request.json.get('message', '').strip()
result = predictor.predict(user_message)
```

### 4.4.2.2 Enkode Input — Tokenisasi Utterance

Teks input pengguna dikonversi menjadi representasi numerik melalui `AutoTokenizer` yang dilatih bersama model IndoBERT. Tokenizer menggunakan algoritma **WordPiece** yang memecah kata-kata tak dikenal menjadi sub-kata untuk menghindari masalah *out-of-vocabulary*. Panjang input dibatasi pada **50 token** (`max_len` di `config.py`) — sebuah pilihan yang memadai untuk kueri chatbot yang umumnya pendek, sekaligus menghemat memori GPU dan mempercepat inferensi.

**File: `models/multilabel/predict.py`**
```python
encoding = self.tokenizer(
    text,
    add_special_tokens=True,
    max_length=opt.max_len,      # Batas: 50 token
    padding='max_length',         # Isi sisa dengan [PAD]
    truncation=True,              # Potong bila lebih panjang
    return_attention_mask=True,
    return_tensors='pt'
)

utterance_ids  = encoding['input_ids']       # Shape: (1, 50)
utterance_mask = encoding['attention_mask']  # Shape: (1, 50)
```

`attention_mask` bernilai 1 untuk setiap token konten dan 0 untuk token padding, sehingga mekanisme self-attention BERT tidak memperhitungkan padding dalam komputasinya.

### 4.4.2.3 Utterance Encoder — Self-Attentive Representation

Pasangan `utterance_ids` dan `utterance_mask` diumpankan ke **encoder pertama** (`self.bert`) yang merupakan backbone IndoBERT. Melalui 12 lapisan *multi-head self-attention*, setiap token dalam kalimat membangun representasi kontekstual dengan mempertimbangkan seluruh token lain dalam urutan yang sama. Dari seluruh output hidden state, hanya **`pooler_output`** yang diambil — yaitu representasi token `[CLS]` yang telah diproses melalui lapisan linier dan tanh, berukuran 768 dimensi.

**File: `models/multilabel/bert_model.py`**
```python
output_utterance = self.bert(
    input_ids=utterance_ids,
    attention_mask=utterance_mask,
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True
)

# Representasi ringkasan seluruh kalimat
pooled_output = output_utterance.pooler_output  # Shape: (batch_size, 768)
```

Mekanisme self-attention memastikan bahwa representasi `pooled_output` bersifat kontekstual: kata "sedih" dalam kalimat *"Aku sangat sedih setelah kehilangan pekerjaan"* akan mendapat bobot perhatian lebih tinggi terhadap "kehilangan" dibandingkan kata-kata penghubung, menghasilkan representasi yang kaya secara semantik.

### 4.4.2.4 Persiapan Label Intent

Bersamaan dengan pemrosesan kalimat pengguna, sistem mempersiapkan **10 teks label intent** yang merupakan inti dari filosofi LABAN. Label-label ini bukan representasi numerik anonimus, melainkan teks deskriptif seperti *"Menyatakan Perasaan Sedih dan Kehilangan"*, *"Mengisyaratkan Gejala Fisik"*, dan *"Menyatakan Perasaan Takut dan Kecemasan"*. Dengan merepresentasikan label sebagai teks yang diproses BERT, model dapat memahami hubungan semantik antar label dan memungkinkan generalisasi ke label baru tanpa training ulang.

**File: `models/multilabel/predict.py` dan `data/data_preparation.py`**
```python
# Daftar 10 label intent sebagai string deskriptif
listof_intent = list(mlb.classes_)  # 10 label sebagai list string

# Tokenisasi semua label sekaligus — dilakukan sekali saat inisialisasi
tokenized_intent = self.tokenizer(
    listof_intent,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

intent_ids  = tokenized_intent['input_ids']       # Shape: (10, max_label_len)
intent_mask = tokenized_intent['attention_mask']  # Shape: (10, max_label_len)
```

### 4.4.2.5 Label Encoder — Representasi Semantik Label

Sepuluh teks label yang telah ditokenisasi diumpankan ke **encoder kedua** (`self.bertlabelencoder`) — sebuah instance IndoBERT yang terpisah dari encoder utterance, meskipun berbagi arsitektur yang identik. Penggunaan encoder terpisah memungkinkan masing-masing encoder mengembangkan spesialisasi melalui training: encoder utterance mempelajari representasi gaya bahasa informal pengguna, sementara encoder label mempelajari representasi karakteristik semantik dari kategori-kategori konseling yang formal. Output encoder label adalah matriks `clusters` berukuran (10, 768) — sepuluh vektor 768-dimensi yang masing-masing merepresentasikan satu label intent.

**File: `models/multilabel/bert_model.py`**
```python
output_label = self.bertlabelencoder(
    input_ids=label_ids,
    attention_mask=Label_mask,
    return_dict=True
)

# Representasi [CLS] untuk setiap label intent
clusters = output_label.pooler_output  # Shape: (10, 768)
# clusters[0] → vektor label "Menyatakan Perasaan Sedih dan Kehilangan"
# clusters[1] → vektor label "Menyatakan Perasaan Takut dan Kecemasan"
# ... dan seterusnya untuk 8 label lainnya
```

### 4.4.2.6 Proyeksi Gram Invers dan Scaling — Komputasi Logit

Tahap ini merupakan inti matematis dari arsitektur LABAN. Dua vektor dari kedua encoder — `pooled_output` (1, 768) dan `clusters` (10, 768) — digabungkan melalui proyeksi matriks Gram invers untuk menghasilkan logit yang independen antar label.

Proses komputasinya terdiri dari empat langkah:

1. **Gram Matrix** — menangkap hubungan geometris antar semua pasang label:
$$G = C \cdot C^T \quad \text{di mana } C \in \mathbb{R}^{10 \times 768}$$

2. **Regularisasi Tikhonov** — menjamin invertibilitas:
$$G_\varepsilon = G + \varepsilon I, \quad \varepsilon = 10^{-4}$$

3. **Weight Matrix** — mengukur kedekatan kalimat dengan setiap label:
$$W = u \cdot C^T \quad \text{di mana } u \in \mathbb{R}^{1 \times 768}$$

4. **Proyeksi Invers Gram + Scaling** — menghasilkan logit yang ortogonalisasi kontribusi antar label:
$$\text{logits} = W \cdot G_\varepsilon^{-1} \cdot \sqrt{d} \quad \text{di mana } d = 768$$

Proyeksi invers Gram secara matematis "meng-ortogonalisasikan" skor setiap label terhadap label-label lainnya — memastikan bahwa skor tinggi untuk "Sedih dan Kehilangan" tidak secara artifisial mengangkat skor "Takut dan Kecemasan" hanya karena keduanya berkorelasi tinggi di ruang embedding.

**File: `models/multilabel/bert_model.py`**
```python
import numpy as np

# Langkah 1: Gram matrix — relasi geometris antar label
gram = torch.mm(clusters, clusters.permute(1, 0))  # Shape: (10, 10)

# Langkah 2: Regularisasi Tikhonov
gram = gram + torch.eye(gram.size(0), device=gram.device) * 1e-4

# Langkah 3: Weight matrix — kedekatan utterance vs. label
weight = torch.mm(pooled_output, clusters.permute(1, 0))  # Shape: (1, 10)

# Langkah 4: Proyeksi invers Gram + scaling dimensi
logits = torch.mm(weight, torch.inverse(gram)) * np.sqrt(opt.hidden_size)
# opt.hidden_size = 768 untuk semua backbone BERT-base
# logits.shape: (1, 10) — satu skor mentah per label intent

return logits
```

### 4.4.2.7 Fungsi Aktivasi Sigmoid — Probabilitas Multi-Intent

Logit mentah dari tahap sebelumnya dikonversi menjadi probabilitas melalui fungsi **Sigmoid** yang diterapkan secara independen pada setiap elemen:

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

Pemilihan Sigmoid — bukan Softmax — adalah keputusan arsitektural yang fundamental. Softmax memberlakukan normalisasi global sehingga probabilitas seluruh kelas berjumlah 1.0, yang secara implisit mengasumsikan bahwa hanya satu kelas yang aktif pada satu waktu. Sigmoid, sebaliknya, menghitung setiap probabilitas secara independen, memungkinkan beberapa intent aktif secara bersamaan — sebuah kondisi yang lazim dalam percakapan konseling di mana pengguna dapat mengekspresikan beberapa emosi dalam satu kalimat.

**File: `models/multilabel/predict.py`**
```python
# Inferensi — tidak ada gradient computation diperlukan
with torch.no_grad():
    logits = self.model(
        utterance_ids=utterance_ids,
        utterance_mask=utterance_mask,
        label_ids=self.intent_ids,
        Label_mask=self.intent_mask
    )

# Sigmoid per elemen — setiap intent dihitung independen
probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
# .squeeze() → (1, 10) → (10,)
# .cpu()     → transfer dari GPU ke RAM
# .numpy()   → konversi ke NumPy array untuk logika threshold
# probabilities.shape: (10,) — nilai ∈ (0, 1) untuk setiap intent

# Catatan training: BCEWithLogitsLoss sudah menyertakan Sigmoid secara internal
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
loss = criterion(logits, labels)  # labels: multi-hot vector shape (batch, 10)
```

### 4.4.2.8 Penentuan Kelas Aktif dan Output Multi-Label

Tahap terakhir menerapkan **threshold** pada vektor probabilitas untuk menghasilkan keputusan biner per intent. Threshold ditetapkan pada nilai **0.5** — dikonfigurasi di `config.py` dengan nama atribut `thresold` (ejaan ini merupakan konvensi kanonik proyek yang tidak diubah). Setiap intent dengan probabilitas ≥ 0.5 dinyatakan aktif dan dimasukkan ke dalam daftar `detected_intents`.

**File: `models/multilabel/predict.py`**
```python
detected_intents = []
scores = {}

for i, label_name in enumerate(self.label_names):
    score = float(probabilities[i])
    scores[label_name] = round(score, 4)
    if score >= opt.thresold:     # 'thresold' — ejaan kanonik, tidak diubah
        detected_intents.append(label_name)

return {
    "pertanyaan": text,
    "inten_terdeteksi": detected_intents,
    "apakah_multilabel": len(detected_intents) > 1,
    "skore": scores
}
```

Nilai `0.5` dipilih sebagai *titik tengah* yang netral: model dinyatakan "lebih yakin label aktif daripada tidak aktif" ketika probabilitasnya melampaui nilai ini. Nilai threshold dapat disesuaikan sesuai kebutuhan — menurunkan threshold meningkatkan sensitivitas (lebih banyak intent terdeteksi, risiko *false positive* meningkat), menaikkannya meningkatkan spesifisitas (lebih sedikit intent, risiko *miss detection* meningkat). Untuk dataset konseling ini, nilai 0.5 memberikan keseimbangan yang memadai berdasarkan hasil evaluasi kuantitatif (lihat Sub-bab 4.5).

---

## Ringkasan Alur Inferensi LABAN

Tabel berikut merangkum setiap tahap dalam alur inferensi LABAN, beserta komponen, transformasi tensor, dan output yang dihasilkan:

| # | Tahap | Komponen | Transformasi | Output |
|:---:|:---|:---|:---|:---|
| 1 | Penerimaan Input | `app.py` → `predictor.predict()` | `str` → `str.strip()` | Teks bersih *(raw string)* |
| 2 | Enkode Input | `AutoTokenizer` (IndoBERT WordPiece) | `str` → `Tensor` | `utterance_ids` (1, 50), `utterance_mask` (1, 50) |
| 3 | Utterance Encoder | `self.bert` — 12 lapisan self-attention | `(1, 50)` → BERT → `(1, 768)` | `pooled_output` (1, 768) |
| 4 | Persiapan Label | `AutoTokenizer` — 10 teks label intent | `list[str]` → `Tensor` | `intent_ids` (10, L), `intent_mask` (10, L) |
| 5 | Label Encoder | `self.bertlabelencoder` — 12 lapisan self-attention | `(10, L)` → BERT → `(10, 768)` | `clusters` (10, 768) |
| 6 | Proyeksi Gram Invers | Operasi matriks (Gram + Tikhonov + Invers + Scaling) | `(1,768)` × `(10,768)` → `(1,10)` | `logits` (1, 10) — skor mentah per intent |
| 7 | Sigmoid Activation | `torch.sigmoid()` per elemen | `ℝ` → `(0, 1)` | `probabilities` (10,) |
| 8 | Threshold & Output | Perbandingan dengan `opt.thresold = 0.5` | `float[]` → `bool[]` → `list[str]` | `detected_intents` (list), `scores` (dict) |

---

*Script Utama: `models/multilabel/bert_model.py`, `models/multilabel/predict.py`, `data/data_preparation.py`*
*Backbone Aktif: IndoBERT (`indobenchmark/indobert-base-p1`, 768-dim)*
*Framework: PyTorch + HuggingFace Transformers*
*Arsitektur: LABAN (Label-Aware BERT Attention Network) — adaptasi dari Wu et al. (EMNLP 2021)*
*Referensi kode konfigurasi: `config.py` (`opt.MODEL_NAME`, `opt.max_len`, `opt.hidden_size`, `opt.thresold`)*
