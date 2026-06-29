### 3.5.1 Input Kueri Pengguna
Tahap pertama ini ibarat kita sedang mendengarkan curhatan klien. Di sini, sistem kita menerima teks mentah (*raw text*) langsung dari input pengguna di antarmuka chat. Mengapa teks mentah? Karena di sinilah semua nuansa emosi, keluhan, dan konteks asli pengguna berada. Sebelum model AI kita bisa "membaca" data ini, kita tangkap dulu string-nya secara utuh. Bentuk datanya sangat sederhana pada tahap ini: hanya sebuah string teks biasa yang siap untuk masuk ke tahap tokenisasi.

```python
# Menerima input teks mentah dari pengguna
user_query = "Saya merasa sangat cemas dan takut dengan hari esok, rasanya dada saya sesak."
```

### 3.5.2 EnkodeInput
Di tahap ini, kita mengubah teks biasa menjadi format yang bisa dimengerti komputer. Kita pakai *tokenizer* (di project ini pakai WordPiece untuk IndoBERT) untuk memecah kalimat pengguna tadi menjadi potongan kata atau sub-kata (token). Kenapa harus dipecah? Karena model jaringan saraf kita cuma paham angka, bukan huruf. Proses ini sangat penting agar kita bisa membatasi panjang kalimat (misalnya maksimal 50 token) dan memberikan setiap token sebuah ID angka yang unik. Nantinya, string teks berubah menjadi tensor, biasanya dengan dimensi `[batch_size, max_seq_len]`, lengkap dengan *attention mask* untuk membedakan mana teks asli dan mana *padding* (isian kosong).

```python
# Mengubah teks menjadi urutan token ID
inputs = tokenizer(
    user_query, 
    max_length=50, 
    padding='max_length', 
    truncation=True, 
    return_tensors='pt'
)
# Shape inputs['input_ids']: [1, 50]
```

### 3.5.3 Self-AttentiveInput
Sekarang kueri pengguna yang sudah jadi angka kita masukkan ke dalam *encoder* pertama (model IndoBERT khusus kueri). Di dalam model ini, ada mekanisme yang namanya *Self-Attention*. Fungsinya apa? Biar model bisa tahu kata mana yang paling penting dalam kalimat tersebut. Misalnya, kata "cemas" akan diberi bobot perhatian yang lebih besar dibanding kata "saya". Dari matriks angka berukuran `[1, 50]`, *encoder* akan merangkum seluruh makna kalimat tersebut ke dalam satu vektor representasi yang padat (biasanya diambil dari *pooler output* dari token `[CLS]`). Ukuran atau dimensi hasil akhirnya adalah vektor sepanjang 768 angka, yaitu `[1, 768]`.

```python
# Memasukkan token kueri ke dalam BERT Encoder pertama
query_outputs = query_encoder(**inputs)

# Mengambil vektor rangkuman kalimat (biasanya dari token [CLS])
query_vector = query_outputs.pooler_output 
# Shape: [1, 768]
```

### 3.5.4 InputLabel
Arsitektur kita menggunakan pendekatan LABAN (*Label Attention Network*), yang artinya kita nggak cuma mendaftarkan label kelas sebagai angka indeks (misal 0 sampai 9), tapi kita perlakukan namanya sebagai kalimat itu sendiri. Di tahap ini, 10 nama *intent* emosi yang kita punya (seperti "Menyatakan Perasaan Marah", "Menyatakan Perasaan Sedih") disiapkan dalam bentuk teks mandiri. Kenapa kita butuh ini? Supaya model kita nanti bisa membaca "makna" dari nama labelnya dan membandingkannya dengan curhatan pengguna secara semantik.

```python
# Menyiapkan 10 label intent sebagai teks mandiri
labels_text = [
    "Mengisyaratkan Butuh Bantuan Profesional",
    "Mengisyaratkan Gejala Fisik",
    "Menyatakan Perasaan Benci dan Jijik",
    # ... dan 7 label lainnya
]
```

### 3.5.5 EnkodeLabel
Sama seperti cara kita memproses kueri pengguna tadi, daftar teks label ini juga akan kita ubah menjadi vektor. Bedanya, kita menggunakan *encoder* IndoBERT kedua yang terpisah khusus untuk label. Kita menggunakan *encoder* terpisah supaya model punya ruang belajar sendiri untuk memahami karakteristik setiap label tanpa terganggu oleh kerumitan kalimat pengguna. Melalui tahap ini, 10 string label tersebut ditokenisasi dan diproses hingga berubah menjadi sebuah matriks yang mewakili masing-masing label. Dimensinya berubah dari 10 teks menjadi matriks berukuran `[10, 768]`.

```python
# Tokenisasi list teks label dan proses di BERT Encoder kedua
label_inputs = tokenizer(labels_text, padding=True, return_tensors='pt')
label_outputs = label_encoder(**label_inputs)

# Mengambil vektor representasi dari ke-10 label
label_vectors = label_outputs.pooler_output 
# Shape: [10, 768]
```

### 3.5.6 Membangun Vektor Embedding Semantik
Di sinilah *matching* antara keluhan user dan label terjadi. Kita akan mempertemukan vektor kueri pengguna (dari *encoder* pertama) dengan matriks vektor label (dari *encoder* kedua). Caranya adalah dengan operasi perkalian matriks (*dot product*). Kenapa harus dikalikan? Karena dalam dunia model bahasa, perkalian *dot product* antara dua vektor berguna untuk mengukur seberapa dekat atau mirip arah keduanya. Semakin cocok makna curhatan pengguna dengan suatu label, semakin besar nilai hasil perkaliannya. Dari segi ukuran, kita mengalikan vektor kueri `[1, 768]` dengan *transpose* matriks label `[768, 10]`, yang akan menghasilkan nilai korelasi mentah berukuran `[1, 10]`.

```python
import torch

# Mengalikan vektor kueri dengan transpose matriks vektor label
# query_vector shape: [1, 768], label_vectors.T shape: [768, 10]
semantic_interaction = torch.matmul(query_vector, label_vectors.transpose(0, 1))
# Shape interaksi: [1, 10]
```

### 3.5.7 VektorEmbedding Semantik
Hasil perkalian dari tahap sebelumnya terkadang angkanya bisa terlalu besar, yang berpotensi bikin perhitungan model jadi kurang stabil saat masa training. Makanya, di tahap ini matriks korelasi tadi akan kita perkecil (skalakan) dengan cara membaginya dengan akar kuadrat dari jumlah dimensinya (yaitu akar dari 768). Pendekatan ini diadaptasi dari metode *Scaled Dot-Product Attention*. Hasil akhirnya adalah matriks akhir berukuran `[1, 10]` yang sering kita sebut sebagai matriks *logits*. Matriks ini berisi skor murni untuk 10 *intent*.

```python
import math

# Melakukan scaling (pembagian dengan akar dimensi tersembunyi model)
hidden_dim = 768
logits = semantic_interaction / math.sqrt(hidden_dim)
# Shape logits: [1, 10] (masih berupa skor logit mentah)
```

### 3.5.8 ProbabilitasMulti-Intent
Skor *logits* di atas bentuknya masih angka bebas (bisa bernilai minus atau sangat besar). Kita butuh angka yang rapi supaya bisa dibaca sebagai probabilitas (0 sampai 1). Nah, karena ini adalah kasus klasifikasi *multi-intent* (satu pengguna bisa saja merasa sedih DAN marah secara bersamaan), kita menggunakan fungsi aktivasi **Sigmoid**, bukan *Softmax*. Kenapa harus Sigmoid? Kalau *Softmax*, ke-10 probabilitas itu dipaksa berjumlah 1, jadi mereka harus saling rebutan poin. Dengan *Sigmoid*, probabilitas setiap *intent* dihitung secara independen. Jadi, sistem bisa santai saja menebak bahwa ada 90% peluang Sedih dan 85% peluang Marah di saat yang bersamaan.

```python
# Mengubah skor logits mentah menjadi rentang probabilitas (0.0 sampai 1.0)
probabilities = torch.sigmoid(logits)
# Shape probabilities: [1, 10]
```

### 3.5.9 Penentuan Kelas dan Output Multi-Label
Sampai di sini kita sudah punya angka persentase, tapi di ujung sistem kita butuh keputusan bulat: mana saja *intent* yang benar-benar terdeteksi (*True*). Cara paling *standard* adalah dengan memasang nilai ambang batas (*threshold*). Sesuai konfigurasi sistem, kita mengatur *threshold* ini di angka `0.5`. Artinya, *intent* emosi apa pun yang probabilitasnya lebih besar dari 0.5 akan langsung ditetapkan sebagai kelas yang terdeteksi (`1`), sedangkan yang di bawah itu dibuang (`0`). Output akhirnya berwujud daftar *multi-label* dari sesi chat tersebut.

```python
# Menerapkan ambang batas 0.5 (berdasarkan `opt.thresold` di config)
threshold = 0.5
predictions = (probabilities > threshold).int()

# Contoh: Jika probabilities = [0.1, 0.8, 0.6, ...], maka predictions = [0, 1, 1, ...]
# Nilai 1 berarti intent tersebut positif terdeteksi.
```
