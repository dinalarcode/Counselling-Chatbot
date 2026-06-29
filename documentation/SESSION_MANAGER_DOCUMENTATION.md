# Bagian 1: Kacamata Pengenalan Tahapan (Untuk Sub-bab "Pengumpulan dan Pelabelan Data")

Fokus analisis pada bagian ini adalah konseptualisasi alur konseling secara non-teknis tanpa mengacu pada struktur data atau variabel pemrograman. Tahapan konseling didefinisikan berdasarkan tujuan interaksi psikologis-spiritual dan batasan logis perpindahan antartahap.

### Sesi 1: Tahap Pembukaan
* **Nama Sesi:** Tahap Pembukaan
* **Pengertian & Fungsi:** Sesi awal yang bertujuan untuk menyapa konseli, membangun hubungan awal yang aman dan hangat (*rapport*), serta mempersiapkan mental konseli agar merasa nyaman sebelum menceritakan pergumulannya dalam suasana konseling berbasis teologi Alkitab.
* **Batasan Singkat:** Tahap ini selesai dan segera berpindah ke tahap berikutnya setelah pesan sapaan dari chatbot terkirim dan konseli memberikan tanggapan pertamanya.

### Sesi 2: Tahap Pembahasan
* **Nama Sesi:** Tahap Pembahasan
* **Pengertian & Fungsi:** Sesi eksplorasi masalah di mana konseli dipersilakan menceritakan secara luas dan mendalam segala bentuk keluhan, pergumulan, emosi negatif, serta situasi sulit yang sedang dihadapinya. Chatbot mendengarkan secara aktif untuk memetakan jenis keluhan dan mengidentifikasi intensi inti dari konseli.
* **Batasan Singkat:** Tahap ini selesai ketika konseli menyatakan bahwa penuturan keluhannya sudah cukup (tidak ada lagi yang ingin diceritakan), atau ketika percakapan mencapai batas maksimum giliran yang telah ditentukan.

### Sesi 3: Tahap Intervensi
* **Nama Sesi:** Tahap Intervensi
* **Pengertian & Fungsi:** Sesi pembingkaian ulang masalah (*reframing*) menggunakan perspektif kebenaran Alkitab. Chatbot memberikan wawasan teologis, ayat-ayat Alkitab yang relevan, serta teguran kasih atau penghiburan spiritual agar konseli dapat melihat permasalahannya sesuai sudut pandang firman Tuhan.
* **Batasan Singkat:** Tahap ini berakhir ketika konseli merespons dengan indikasi bahwa ia memahami, setuju, atau menerima pembingkaian teologis yang disampaikan oleh chatbot.

### Sesi 4: Tahap Solusi
* **Nama Sesi:** Tahap Solusi
* **Pengertian & Fungsi:** Sesi penyusunan rencana aksi praktis dan aplikatif. Tujuannya adalah membekali konseli dengan langkah-langkah konkret yang didasarkan pada prinsip-prinsip kristiani untuk diterapkan dalam kehidupan sehari-hari guna mengatasi permasalahannya.
* **Batasan Singkat:** Tahap ini selesai setelah konseli merespons dengan kesediaan untuk mencoba menerapkan solusi atau langkah praktis yang ditawarkan.

### Sesi 5: Tahap Relaksasi
* **Nama Sesi:** Tahap Relaksasi
* **Pengertian & Fungsi:** Sesi penenangan batin konseli setelah melewati proses analisis masalah dan perumusan solusi. Sesi ini diisi dengan doa penyerta, refleksi teologis yang menenangkan, atau kata-kata penguatan iman untuk memberikan damai sejahtera dan meneguhkan keyakinan konseli kepada pertolongan Tuhan.
* **Batasan Singkat:** Sesi ini berakhir ketika konseli merespons dengan rasa syukur, mengucapkan "amin", atau menyatakan bahwa jiwanya sudah merasa lebih tenang.

### Sesi 6: Tahap Penutupan
* **Nama Sesi:** Tahap Penutupan
* **Pengertian & Fungsi:** Sesi akhir untuk mengakhiri sesi konseling secara formal dengan memberikan berkat dan salam perpisahan Kristen. Pada tahap ini, jika konseli sebelumnya terdeteksi mengalami gejala fisik akibat beban mentalnya, chatbot akan menawarkan rujukan bantuan profesional Kristen (seperti psikolog Kristen atau konselor berlisensi).
* **Batasan Singkat:** Sesi ini berakhir sepenuhnya setelah salam penutup selesai disampaikan atau jika konseli memberikan tanggapan terkait tawaran bantuan profesional.

### Sesi 7: Tahap Rujukan Bantuan Profesional (Jalur Khusus/Darurat)
* **Nama Sesi:** Tahap Rujukan Bantuan Profesional
* **Pengertian & Fungsi:** Sesi khusus non-linear yang berfungsi sebagai penanganan krisis darurat medis/kejiwaan. Tujuannya adalah mendeteksi indikasi membahayakan diri sendiri (seperti keinginan bunuh diri atau keputusasaan ekstrem) atau menindaklanjuti persetujuan tawaran rujukan di akhir sesi. Chatbot akan memberikan informasi kontak bantuan profesional terakreditasi dan menyertakan ayat penguatan teologis bahwa mencari bantuan adalah tindakan bijaksana yang didukung firman Tuhan.
* **Batasan Singkat:** Tahap ini langsung menutup sesi percakapan secara permanen dan mengakhiri seluruh interaksi konseling.

---

# Bagian 2: Kacamata Sistem (Untuk Sub-bab "Hasil Integrasi Model ke Chatbot Konseling")

Fokus analisis pada bagian ini adalah implementasi teknis, arsitektur *state management*, serta logika pemrosesan data di balik kelas `SessionManager` dalam berkas `core/session_manager.py`.

### Arsitektur Umum Manajemen Sesi
Kelas `SessionManager` bertindak sebagai pengendali *state* dan berinteraksi secara langsung dengan `RAGEngine` (`core/rag_engine.py`) untuk menghasilkan respon chatbot yang dinamis. Beberapa mekanisme utama yang dikelola oleh sistem ini meliputi:
1. **Urutan Tahap Linier:** Didefinisikan melalui list `STAGE_ORDER` yang bernilai `['pembukaan', 'pembahasan', 'intervensi', 'solusi', 'relaksasi', 'penutupan']`.
2. **Akumulasi Intensi (Intent Accumulator):** Diimplementasikan menggunakan `collections.Counter` pada variabel `self.accumulated_intents` untuk menghitung frekuensi intensi yang terklasifikasi oleh model pengklasifikasi intensi (model LABAN) pada setiap giliran percakapan.
3. **Snapshot Intensi Utama (Primary Intents):** Saat bertransisi keluar dari tahap `pembahasan` ke `intervensi`, sistem mengambil snapshot 3 intensi teratas dengan frekuensi tertinggi melalui fungsi `_snapshot_primary_intents()`. Snapshot ini disimpan pada list `self.primary_intents` untuk digunakan sebagai `override_intents` pada tahap `relaksasi`. Hal ini menjamin pencarian ayat Alkitab oleh sistem FAISS tetap relevan dengan masalah utama konseli, meskipun konseli tidak lagi menyebutkan keluhannya di akhir sesi.
4. **Pelacakan Gejala Fisik:** Variabel boolean `self.has_physical_symptoms` bernilai `True` jika intensi `Mengisyaratkan Gejala Fisik` terdeteksi pada giliran mana pun dalam sesi. Variabel ini dipertahankan hingga akhir percakapan untuk memicu logika rujukan pada tahap penutupan.
5. **Jaring Pengaman Krisis (Emergency Safety Net):** Menggunakan kumpulan ekspresi reguler (`re.compile`) pada `PROFESSIONAL_KEYWORDS` untuk mendeteksi indikasi bunuh diri atau melukai diri sendiri secara langsung dari pesan mentah pengguna. Jika terdeteksi melalui `_detect_professional_keywords()`, sistem langsung melompati alur linier dan beralih ke state `bantuan_profesional`.
6. **Spiritual Consent Tracker:** Mengelola status `self.spiritual_consent` (None, True, False) dari konseli dan menanyakannya secara eksplisit di akhir tahap `solusi` guna mengontrol pemberian perspektif teologis selanjutnya.
7. **Pencegahan Pengulangan Ayat:** Melacak kitab (`self.used_verse_books`) dan referensi ayat (`self.used_verses`) yang telah diberikan untuk memastikan RAG engine tidak memberikan kutipan yang sama berulang kali.
8. **Peringkasan Konteks & Ekstraksi Teknik:** Menghasilkan rangkuman singkat permasalahan (`self.complaint_summary`) saat transisi ke `intervensi` serta mengekstraksi jenis teknik relaksasi (`self.chosen_technique`) saat transisi ke `relaksasi`, sehingga konteks inti tetap terbawa tanpa memberatkan *state memory*.

---

### Detail Konfigurasi State dan Transisi

#### 1. State: `pembukaan`
* **Konfigurasi & Logika Teknis:** Sistem mengirim sapaan awal dan memanggil `RAGEngine.generate_response()` dengan parameter `current_stage='pembukaan'`. Setiap tanggapan pengguna pada state ini dihitung ke dalam `self.turn_count`.
* **Batasan & Transisi Detail:**
  * Batasan giliran minimum (`MIN_TURNS`): 1
  * Batasan giliran maksimum (`MAX_TURNS`): 2
  * Transisi terjadi secara otomatis segera setelah `self.turn_count >= 1` dari masukan pertama pengguna. Sistem memanggil `_advance_stage()` untuk menaikkan `stage_index` dan mengubah `current_stage` menjadi `pembahasan`.

#### 2. State: `pembahasan`
* **Konfigurasi & Logika Teknis:** Chatbot berinteraksi untuk menggali detail masalah. Setiap intensi yang dihasilkan oleh model LABAN pada turn tersebut diakumulasikan ke dalam `self.accumulated_intents` dan dianalisis untuk memperbarui daftar `self.primary_intents` secara *real-time*. Histori percakapan pada tahap ini juga direkam secara sementara dalam `self.temp_pembahasan_history`.
* **Batasan & Transisi Detail:**
  * Batasan giliran minimum (`MIN_TURNS`): 3 (syarat minimum eksplorasi klinis)
  * Batasan giliran maksimum (`MAX_TURNS`): 5 (memberikan kesempatan eksplorasi yang lebih luas)
  * Transisi linier ke `intervensi` dipicu jika `self.turn_count >= 3` dan masukan pengguna cocok dengan salah satu pola regex dalam konstanta `TRANSITION_SIGNALS['pembahasan']` (seperti *cukup*, *sudah cukup*, *itu saja*, *lanjut*, dsb.).
  * **Early-exit:** Transisi ke `intervensi` dapat dipicu seketika meskipun minimum 3 giliran belum tercapai jika pengguna secara eksplisit memberikan kalimat yang cocok dengan `PEMBAHASAN_EARLY_EXIT_SIGNALS` (seperti *sudah tidak ada*, *cuma itu saja*).
  * Jika percakapan mencapai `self.turn_count == 5` tanpa adanya kata kunci transisi, sistem secara paksa (*force transition*) memindahkan state ke `intervensi`.
  * Saat keluar dari tahap ini, sistem menjalankan `_snapshot_primary_intents()` untuk menyimpan profil masalah, kemudian menggunakan `temp_pembahasan_history` guna men-generate `complaint_summary`, dan mengosongkan buffer histori tersebut.

#### 3. State: `intervensi`
* **Konfigurasi & Logika Teknis:** Sistem menghasilkan tanggapan teologis. Ayat Alkitab dicari berdasarkan intensi giliran tersebut.
* **Batasan & Transisi Detail:**
  * Batasan giliran minimum (`MIN_TURNS`): 1
  * Batasan giliran maksimum (`MAX_TURNS`): 3
  * Transisi linier ke `solusi` terjadi jika `self.turn_count >= 1` dan masukan pengguna cocok dengan regex dalam `TRANSITION_SIGNALS['intervensi']` (seperti *saya mengerti*, *benar juga*, *masuk akal*, *iya*, *baik*, dsb.).
  * Transisi paksa dilakukan jika giliran percakapan mencapai batas maksimal yaitu 3.

#### 4. State: `solusi`
* **Konfigurasi & Logika Teknis:** Memberikan saran dan solusi praktis untuk konseli. Tahap ini juga mengumpulkan buffer histori pada `self.temp_solusi_history`. Pada giliran terakhir tahap ini (berdasarkan perhitungan `MAX_TURNS`), sistem akan menyisipkan pertanyaan persetujuan rohani (`ask_spiritual_consent`) yang memengaruhi respons berbasis Alkitab selanjutnya.
* **Batasan & Transisi Detail:**
  * Batasan giliran minimum (`MIN_TURNS`): 1
  * Batasan giliran maksimum (`MAX_TURNS`): 3
  * Transisi linier ke `relaksasi` dipicu jika `self.turn_count >= 1` dan masukan pengguna cocok dengan pola dalam `TRANSITION_SIGNALS['solusi']` (seperti *akan saya coba*, *terima kasih*, *oke*, *baik*, dsb.).
  * Transisi paksa dilakukan jika giliran percakapan mencapai batas maksimal yaitu 3.
  * Saat berpindah ke `relaksasi`, sistem akan mengekstraksi jenis teknik relaksasi yang disetujui konseli dari `temp_solusi_history` ke variabel `self.chosen_technique` dan mengosongkan buffer tersebut.

#### 5. State: `relaksasi`
* **Konfigurasi & Logika Teknis:** RAG engine dipanggil dengan mengirimkan `override_intents=self.primary_intents` untuk memformulasikan respon damai sejahtera atau doa penutup rohani yang relevan.
* **Batasan & Transisi Detail:**
  * Batasan giliran minimum (`MIN_TURNS`): 1
  * Batasan giliran maksimum (`MAX_TURNS`): 3
  * Transisi linier ke `penutupan` dipicu jika `self.turn_count >= 1` dan masukan pengguna cocok dengan pola dalam `TRANSITION_SIGNALS['relaksasi']` (seperti *lebih tenang*, *lebih baik*, *lega*, *amin*, *terima kasih*, dsb.).
  * Transisi paksa dilakukan jika giliran percakapan mencapai batas maksimal yaitu 3.

#### 6. State: `penutupan`
* **Konfigurasi & Logika Teknis:** Chatbot mengembalikan pesan penutup rohani. Jika `self.has_physical_symptoms` bernilai `True`, sistem mengaktifkan logika bersyarat untuk menawarkan opsi rujukan bantuan profesional Kristen.
* **Batasan & Transisi Detail:**
  * Batasan giliran minimum (`MIN_TURNS`): 1
  * Batasan giliran maksimum (`MAX_TURNS`): 99 (tidak ada transisi paksa otomatis).
  * Jika pengguna merespons tawaran rujukan secara positif (cocok dengan regex `TRANSITION_SIGNALS['penutupan']` seperti *iya*, *mau*, *boleh*, *oke*, *siap*), sistem berpindah ke state `bantuan_profesional`.
  * Jika pengguna merespons negatif terhadap tawaran (cocok dengan `DECLINE_SIGNALS` seperti *tidak perlu*, *sudah cukup*), sistem secara halus mengakhiri sesi (menggunakan state internal `penutupan_selesai`) dan mengubah properti `self.session_ended` menjadi `True`.

#### 7. State: `bantuan_profesional` (Branch State)
* **Konfigurasi & Logika Teknis:** State non-linear yang dipicu baik lewat persetujuan rujukan di tahap `penutupan` maupun bypass darurat. Ketika state aktif, sistem melakukan hal berikut:
  1. Mengubah `self.current_stage` menjadi `'bantuan_profesional'`, me-reset `turn_count = 0`, dan menyetel `self.in_professional_stage = True`.
  2. Memanggil `RAGEngine.generate_response()` dengan `override_intents=['pertolongan harapan kekuatan Tuhan membantu tidak sendirian']`. Hal ini memaksa FAISS mengembalikan ayat Alkitab tentang pengharapan dan kebijaksanaan mencari pertolongan medis/konseling.
  3. Mengubah properti `self.session_ended` menjadi `True`.
  4. Mengembalikan nilai boolean `show_professional_button = True` pada kamus respon untuk menginstruksikan antarmuka pengguna (frontend) agar menampilkan tombol tautan kontak bantuan profesional (misalnya hotline konseling Kristen atau layanan kesehatan jiwa).
* **Batasan & Transisi Detail:** State ini langsung mengakhiri sesi percakapan secara permanen (`self.session_ended = True`), sehingga tidak memiliki transisi lanjutan ke state lain.

---

### Kode Sumber Pendukung State Management

Berikut adalah cuplikan kode utama yang mengimplementasikan logika transisi sesi, bypass darurat, dan snapshot intensi utama dalam berkas `core/session_manager.py`:

#### Cuplikan 1: Logika Keputusan Transisi Tahapan (`_should_transition`)
```python
    def _should_transition(self, user_input):
        # Sudah berada di tahap terakhir — tidak ada transisi lanjutan
        if self.stage_index >= len(self.STAGE_ORDER) - 1:
            return False

        stage = self.current_stage

        # Pembukaan: selalu bertransisi setelah respons pertama dari konseli
        if stage == 'pembukaan' and self.turn_count >= self.MIN_TURNS[stage]:
            return True

        # Batas giliran maksimum terlampaui — memicu transisi paksa
        if self.turn_count >= self.MAX_TURNS.get(stage, 99):
            return True

        # Pembahasan early-exit: bypass batasan turn_count minimum jika konseli eksplisit
        if stage == 'pembahasan' and self._detect_pembahasan_early_exit(user_input):
            return True

        # Belum memenuhi batasan giliran minimum — tidak bertransisi
        if self.turn_count < self.MIN_TURNS.get(stage, 1):
            return False

        # Memeriksa keberadaan sinyal transisi verbal pada pesan pengguna
        return self._detect_transition_signal(user_input, stage)
```

#### Cuplikan 2: Mekanisme Bypass Darurat / Jaring Pengaman Krisis (`chat`)
```python
        # --- Emergency skip: keyword-based safety net ---
        # Memeriksa kata kunci bunuh diri/melukai diri secara mentah sebelum klasifikasi intent LLM
        keyword_triggered = self._detect_professional_keywords(user_input)

        # --- Emergency skip: berdasarkan model klasifikasi LABAN ATAU kecocokan kata kunci ---
        if (self.PROFESSIONAL_INTENT in turn_intents or keyword_triggered) and not self.in_professional_stage:
            return self._enter_professional_stage(user_input)
```

#### Cuplikan 3: Snapshot Intensi Utama dan Mekanisme *Override* untuk FAISS
```python
    def _advance_stage(self):
        """Memindahkan sesi ke tahap berikutnya dan me-reset jumlah giliran."""
        if self.stage_index < len(self.STAGE_ORDER) - 1:
            self.stage_index += 1
            self.current_stage = self.STAGE_ORDER[self.stage_index]
            self.turn_count = 0
            self.turn_in_stage = 0  # reset per-stage counter on transition

            # Ketika beranjak dari tahap pembahasan, ambil snapshot dari intensi utama
            # dan men-generate satu kalimat latar belakang masalah.
            if self.current_stage == 'intervensi':
                self._snapshot_primary_intents()
                if self.temp_pembahasan_history:
                    self.complaint_summary = self.rag_engine.generate_background_summary(
                        self.temp_pembahasan_history
                    )
                    self.temp_pembahasan_history = []  # segera kosongkan

            # Ketika beranjak ke relaksasi, ekstraksi teknik relaksasi yang terpilih
            if self.current_stage == 'relaksasi':
                if self.temp_solusi_history:
                    self.chosen_technique = self.rag_engine.generate_technique_extraction(
                        self.temp_solusi_history
                    )
                    self.temp_solusi_history = []
```
```python
        # Menentukan intensi yang dikirimkan untuk pencarian ayat Alkitab
        # Hanya pada tahap relaksasi, gunakan intensi utama hasil snapshot akumulasi pembahasan
        override_intents = None
        if self.current_stage == 'relaksasi' and self.primary_intents:
            override_intents = self.primary_intents
```
```python
    def _snapshot_primary_intents(self):
        """Mengambil snapshot intensi utama saat bertransisi keluar dari pembahasan."""
        self._update_primary_intents()

    def _update_primary_intents(self):
        """Memperbarui daftar intensi utama berdasarkan frekuensi akumulasi tertinggi (top 3)."""
        if self.accumulated_intents:
            sorted_intents = self.accumulated_intents.most_common(3)
            self.primary_intents = [intent for intent, _ in sorted_intents]
```
