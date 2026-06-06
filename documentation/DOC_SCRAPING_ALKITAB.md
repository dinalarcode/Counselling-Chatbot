# Dokumentasi Teknis: Proses Scraping Ayat Alkitab

---

## 1. Gambaran Umum

Sistem pengumpulan data Alkitab pada proyek ini dilakukan melalui proses *web scraping* terhadap situs **alkitab.mobi**, sebuah platform digital yang menyediakan teks Alkitab Terjemahan Baru (TB) dalam bahasa Indonesia secara daring. Seluruh proses scraping diimplementasikan dalam satu berkas skrip Python, yaitu `data/scrape_alkitab.py`, yang dirancang untuk berjalan satu kali dan menghasilkan berkas basis data ayat Alkitab lengkap dalam format CSV.

Dokumentasi ini menjelaskan secara teknis bagaimana sistem scraping tersebut bekerja, mulai dari pustaka yang digunakan, sumber data yang dituju, mekanisme pengambilan dan pemrosesan konten HTML, struktur data keluaran, hingga pemanfaatan data hasil scraping dalam pipeline konseling berbasis Alkitab.

---

## 2. Perangkat dan Pustaka yang Digunakan

Seluruh dependensi teknis yang digunakan dalam skrip scraping ini merupakan pustaka Python standar yang banyak digunakan dalam ekosistem *data engineering* dan *web scraping*. Tidak ada ketergantungan terhadap layanan berbayar maupun API eksternal — skrip ini sepenuhnya berjalan secara lokal.

Pustaka inti yang digunakan adalah **`requests`**, yaitu sebuah HTTP client untuk Python yang digunakan untuk mengirimkan permintaan GET ke setiap halaman pasal Alkitab pada situs alkitab.mobi. Setiap permintaan dilengkapi dengan header `User-Agent` yang mengidentifikasi bot sebagai proyek akademik, serta batas waktu (*timeout*) 15 detik untuk mencegah proses tergantung akibat koneksi yang tidak responsif.

```python
response = requests.get(url, timeout=15, headers={
    "User-Agent": "Mozilla/5.0 (Bible Research Bot - Academic Project)"
})
```

Untuk mem-*parsing* konten HTML yang diterima dari server, digunakan **`BeautifulSoup`** dari pustaka `bs4`. BeautifulSoup memungkinkan penelusuran struktur DOM halaman web secara terprogram untuk mengekstrak elemen-elemen yang relevan — dalam hal ini, tag `<p>` yang merepresentasikan setiap ayat Alkitab beserta nomor ayatnya.

Dua pustaka pendukung lainnya juga digunakan: **`tqdm`** untuk menampilkan progress bar pada terminal selama proses scraping berlangsung (mengingat total pasal yang harus diproses mencapai 1.189 pasal), serta **`re`** (*regular expressions*) untuk membersihkan karakter *whitespace* berlebihan yang mungkin muncul setelah proses ekstraksi teks dari HTML. Penyimpanan hasil akhir dilakukan menggunakan modul **`csv`** bawaan Python melalui antarmuka `csv.DictWriter`, yang secara otomatis mengelola penulisan *header* dan baris-baris data ke berkas CSV dengan encoding UTF-8.

---

## 3. Sumber Data

Data Alkitab diambil dari situs **[alkitab.mobi](http://alkitab.mobi/)**, sebuah platform web yang menyediakan teks Alkitab Terjemahan Baru (TB) versi digital. Terjemahan Baru (TB) dipilih karena merupakan terjemahan Alkitab bahasa Indonesia yang paling banyak digunakan oleh gereja-gereja Protestan di Indonesia, sehingga relevan dengan konteks pengguna target sistem konseling ini.

Pola URL yang digunakan untuk mengakses setiap pasal mengikuti format berikut:

```
http://alkitab.mobi/tb/{singkatan_buku}/{nomor_pasal}/
```

Sebagai contoh, untuk mengakses Mazmur pasal 23, URL yang digunakan adalah `http://alkitab.mobi/tb/Mzm/23/`. Setiap halaman URL merepresentasikan satu pasal penuh, dan seluruh ayat dalam pasal tersebut termuat dalam satu halaman HTML yang sama.

Cakupan scraping meliputi **66 buku** Alkitab — 39 buku Perjanjian Lama dan 27 buku Perjanjian Baru — sesuai dengan kanon Protestan. Setiap buku didefinisikan dalam konstanta `BOOKS` pada skrip, yang menyimpan tiga informasi per buku: singkatan yang digunakan oleh alkitab.mobi (misalnya `"Mzm"` untuk Mazmur), nama lengkap buku (misalnya `"Mazmur"`), dan total jumlah pasal dalam buku tersebut. Dengan demikian, skrip dapat secara otomatis mengiterasi seluruh pasal dari setiap buku tanpa memerlukan konfigurasi manual.

---

## 4. Cara Kerja Proses Scraping

Proses scraping dijalankan melalui fungsi utama `run_scraper()`, yang mengorkestrasi keseluruhan alur pengambilan data dari awal hingga penyimpanan ke CSV. Secara garis besar, proses ini terbagi dalam tiga tahap: iterasi atas seluruh buku dan pasal, pengambilan serta parsing konten per halaman, dan penyimpanan hasil akhir.

### Iterasi Buku dan Pasal

Fungsi `run_scraper()` melakukan iterasi bersarang (*nested iteration*) — untuk setiap buku dalam daftar `BOOKS`, skrip akan mengiterasi setiap nomor pasal dari 1 hingga jumlah maksimum pasal buku tersebut. Setelah setiap pasal selesai diproses, skrip menunggu selama **0,5 detik** (`DELAY_SECONDS = 0.5`) sebelum memproses pasal berikutnya. Mekanisme *rate limiting* ini penting untuk menghindari pemblokiran oleh server sumber akibat frekuensi permintaan yang terlalu tinggi, sekaligus menjaga kelancaran server sumber yang bukan milik peneliti.

```python
for book_abbr, book_name, num_chapters in BOOKS:
    for chapter in range(1, num_chapters + 1):
        verses = scrape_chapter(book_abbr, chapter)
        ...
        time.sleep(DELAY_SECONDS)
```

### Pengambilan dan Parsing Konten Per Halaman

Setiap halaman pasal diproses oleh fungsi `scrape_chapter(book_abbr, chapter)`. Fungsi ini pertama-tama mengirimkan permintaan HTTP GET ke URL yang sesuai. Jika permintaan gagal — baik karena *timeout*, kesalahan jaringan, maupun respons HTTP non-2xx — fungsi mengembalikan daftar kosong dan mencatat pasal yang gagal untuk dilaporkan di akhir proses.

Setelah respons HTML diterima, BeautifulSoup digunakan untuk menelusuri seluruh tag `<p>` dalam halaman. Situs alkitab.mobi merepresentasikan setiap ayat sebagai sebuah paragraf `<p>` yang di dalamnya terdapat dua elemen kunci: elemen `<span class="reftext">` yang memuat nomor ayat (dalam bentuk tautan `<a>`), dan konten teks ayat itu sendiri di luar span tersebut. Beberapa paragraf yang bukan merupakan ayat — seperti paragraf tersembunyi (*hidden*), paragraf *loading/error*, dan judul seksi (`<span class="paragraphtitle">`) — dilewati secara eksplisit.

```python
for p_tag in soup.find_all("p"):
    if p_tag.get("hidden") == "hidden":
        continue
    if p_tag.get("class") and ("loading" in p_tag.get("class") or "error" in p_tag.get("class")):
        continue

    title_span = p_tag.find("span", class_="paragraphtitle")
    if title_span:
        continue

    ref_span = p_tag.find("span", class_="reftext")
    if not ref_span:
        continue
```

Setelah nomor ayat berhasil diekstrak, teks ayat diperoleh dengan cara menyalin (*clone*) elemen `<p>` tersebut, menghapus elemen `reftext` dan `paragraphtitle` dari salinannya, kemudian mengambil teks bersih menggunakan metode `get_text()`. Langkah penghapusan dilakukan pada salinan DOM — bukan pada objek asli — untuk memastikan proses parsing pasal lainnya tidak terganggu. Terakhir, ekspresi reguler digunakan untuk menormalkan semua *whitespace* menjadi satu spasi tunggal.

```python
p_copy = BeautifulSoup(str(p_tag), "html.parser").find("p")
for ref in p_copy.find_all("span", class_="reftext"):
    ref.decompose()
for title in p_copy.find_all("span", class_="paragraphtitle"):
    title.decompose()

text = p_copy.get_text(strip=True)
text = re.sub(r'\s+', ' ', text).strip()
```

### Penanganan Kegagalan

Skrip ini dirancang untuk bersifat *fault-tolerant* — apabila sebuah pasal gagal diambil, proses tidak berhenti (*crash*) melainkan mencatat nama pasal tersebut dalam daftar `failed_chapters` dan melanjutkan ke pasal berikutnya. Di akhir proses, seluruh pasal yang gagal dilaporkan kepada pengguna, sehingga dapat dilakukan pengulangan manual apabila diperlukan.

### Penyimpanan ke CSV

Setelah seluruh iterasi selesai, semua ayat yang terkumpul dalam variabel `all_verses` ditulis sekaligus ke berkas CSV menggunakan `csv.DictWriter`. Pendekatan *write-all-at-once* ini dipilih untuk meminimalkan operasi I/O disk selama proses scraping berlangsung.

---

## 5. Struktur Data Keluaran

Proses scraping menghasilkan satu berkas CSV bernama `alkitab_tb.csv` yang disimpan di direktori `data/`. Berkas ini memuat **31.102 ayat** yang mencakup seluruh teks Alkitab Terjemahan Baru, dengan ukuran berkas sekitar 5,7 MB dan encoding UTF-8.

Setiap baris dalam CSV merepresentasikan satu ayat, dengan enam kolom sebagai berikut:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `book_abbr` | string | Singkatan buku sesuai konvensi alkitab.mobi (mis. `"Mzm"`, `"Yoh"`) |
| `book_name` | string | Nama buku lengkap dalam bahasa Indonesia (mis. `"Mazmur"`, `"Yohanes"`) |
| `chapter` | integer | Nomor pasal |
| `verse` | integer | Nomor ayat dalam pasal tersebut |
| `text` | string | Teks ayat dalam bahasa Indonesia (Terjemahan Baru) |
| `reference` | string | Referensi ayat yang telah diformat (mis. `"Mazmur 23:1"`) |

Sebagai ilustrasi, satu baris data yang khas akan terlihat sebagai berikut:

```
book_abbr,book_name,chapter,verse,text,reference
Mzm,Mazmur,23,1,"TUHAN adalah gembalaku, takkan kekurangan aku.",Mazmur 23:1
```

Kolom `reference` merupakan kolom komposit yang dibangun secara programatik selama proses scraping dengan menggabungkan `book_name`, `chapter`, dan `verse` ke dalam format yang mudah dibaca manusia. Kolom ini berfungsi sebagai *human-readable identifier* untuk setiap ayat dan digunakan secara langsung dalam respons chatbot kepada pengguna.

---

## 6. Pemanfaatan Data dalam Sistem

Data `alkitab_tb.csv` yang dihasilkan oleh proses scraping ini menjadi fondasi dari komponen *Bible verse retrieval* dalam sistem konseling berbasis Alkitab. Setelah scraping selesai, data tidak langsung digunakan dalam format CSV — melainkan diproses lebih lanjut menjadi indeks vektor FAISS yang disimpan di direktori `data/faiss_bible_index/`.

Proses pembangunan indeks ini dilakukan secara terpisah, di mana setiap teks ayat diubah menjadi vektor numerik (*embedding*) menggunakan model bahasa berbasis transformer (MiniLM). Indeks FAISS yang dihasilkan memungkinkan pencarian semantik berdimensi tinggi secara efisien — yaitu pencarian ayat yang maknanya paling dekat dengan suatu kueri, bukan hanya yang mengandung kata-kata yang sama secara harfiah.

Dalam alur konseling, basis data ayat ini diaktifkan secara selektif. Ketika sesi konseling memasuki tahap **solusi** atau **relaksasi** — yaitu tahap di mana konselor (sistem chatbot) memberikan panduan atau ayat penguat — mesin RAG (*Retrieval-Augmented Generation*) menggunakan indeks FAISS untuk mengambil sejumlah ayat yang paling relevan dengan kondisi emosional pengguna yang telah diidentifikasi. Ayat-ayat kandidat tersebut kemudian diteruskan ke LLM sebagai konteks tambahan untuk menghasilkan respons konseling yang mengandung referensi Alkitab yang akurat dan kontekstual.

Dengan kata lain, `alkitab_tb.csv` adalah sumber kebenaran tunggal (*single source of truth*) untuk seluruh konten Alkitab dalam sistem ini. Integritas teks yang dihasilkan oleh proses scraping — termasuk ketepatan nomor ayat dan kelengkapan referensi — secara langsung menentukan kualitas respons konseling yang diberikan oleh chatbot kepada pengguna.

---

*Dokumentasi ini dibuat untuk proyek Chatbot Konseling Berbasis Alkitab (Tugas Akhir) — 2026-06-07*
