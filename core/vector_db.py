import os
import random
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import opt

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Indonesian stopwords: Sastrawi's comprehensive list + colloquial extras
_sastrawi_factory = StopWordRemoverFactory()
STOPWORDS_ID = set(_sastrawi_factory.get_stop_words())
# Tambahkan kata-kata informal/colloquial yang tidak ada di Sastrawi
STOPWORDS_ID.update({
    'kak', 'nggak', 'gak', 'dong', 'sih', 'nih', 'deh', 'lho', 'kan',
    'kok', 'banget', 'kayak', 'gimana', 'gitu', 'udah', 'terus',
    'aja', 'emang', 'doang', 'cuma', 'tuh', 'yah', 'wah',
    'gatau', 'gapaham', 'gajelas', 'gamau', 'gaada', 'gabisa',
})

# ── Biblical Synonyms for Query Expansion ──────────────────────────────────
# Maps each of the 10 LABAN intents to formal vocabulary found in the
# Alkitab Terjemahan Baru (TB). These terms are injected into the FAISS
# query so that informal user slang ("kesel", "capek") gets expanded with
# formal biblical words ("murka", "lelah") that actually appear in the Bible.
BIBLICAL_SYNONYMS = {
    "Mengisyaratkan Butuh Bantuan Profesional": (
        "pertolongan penolong selamatkan tolong lindungi "
        "perlindungan harapan kekuatan penghiburan pemulihan"
        "nasihat didikan hikmat teguran petunjuk jalan keluar terang kekuatan"
    ),
    "Mengisyaratkan Gejala Fisik": (
        "sakit penyakit lemah lelah tubuh daging "
        "luka menderita penderitaan kesembuhan"
        "lesu penat letih sembuh menyegarkan membalut kekuatan baru memulihkan"
    ),
    "Menyatakan Perasaan Benci dan Jijik": (
        "benci kebencian murka jijik kejijikan hina "
        "menghina najis kekejian muak"
        "kekejian muak mengampuni kasih damai memberkati pengampunan saudara"
    ),
    "Menyatakan Perasaan Marah dan Frustasi": (
        "murka amarah marah gusar geram berang "
        "kemarahan emosi mengeluh keluh kesah"
        "dendam sabar kasih karunia menahan diri damai sejahtera lemah lembut"
    ),
    "Menyatakan Perasaan Percaya": (
        "percaya iman beriman setia kesetiaan "
        "pengharapan berharap yakin teguh penyertaan"
        "berserah percaya setia dijagai janji aman tempat perlindungan"
    ),
    "Menyatakan Perasaan Sebelum Menghadapi Kejadian": (
        "kuatir khawatir gelisah waswas gentar "
        "bimbang ragu cemas menanti menunggu"
        "berjaga-jaga pencobaan ujian waspada penyertaan jangan takut berani teguh melangkah"

    ),
    "Menyatakan Perasaan Sedih dan Kehilangan": (
        "dukacita ratap tangis meratap bersedih perkabungan "
        "berkabung kehilangan air mata patah hati remuk jiwa"
        "duka ratapan hancur hati penghiburan dihibur sukacita memulihkan"
    ),
    "Menyatakan Perasaan Takut dan Kecemasan": (
        "takut ketakutan gentar ngeri kecemasan gemetar "
        "cemas waswas kuatir gelisah"
        "damai berani aman perlindungan tempat perlindungan penolong"
    ),
    "Menyatakan Rasa Syukur dan Apresiasi": (
        "syukur bersyukur puji pujian memuji ucapan syukur "
        "berkat diberkati terima kasih sukacita"
        "sukacita sorak karunia kebaikan melimpah"
    ),
    "Menyatakan Reaksi Terkejut dan Tidak Terduga": (
        "heran takjub tercengang dahsyat ajaib mujizat "
        "keajaiban terkejut terperanjat"
        "gempar kedaulatan rencana damai sejahtera pegangan pertolongan ajaib"
    ),
}


class VectorDBManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=opt.EMBED_MODEL)
        self.bible_db = None
        self.qna_db = None

    # ── Bible Index (Full TB Bible) ────────────────────────────────

    def build_bible_index(self, csv_path='data/alkitab_tb.csv', index_dir='data/faiss_bible_index'):
        """
        Membangun atau memuat FAISS index untuk seluruh Alkitab TB.
        
        Jika index sudah ada di disk, langsung dimuat (cepat).
        Jika belum, membangun dari CSV (lambat, hanya sekali).
        """
        # Coba muat dari disk terlebih dahulu
        if os.path.exists(index_dir):
            try:
                print("Memuat Bible FAISS index dari disk...")
                self.bible_db = FAISS.load_local(
                    index_dir, self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Bible index dimuat. Total dokumen: {self.bible_db.index.ntotal}")
                return
            except Exception as e:
                print(f"Gagal memuat index dari disk: {e}")
                print("Membangun ulang index...")

        # Bangun dari CSV
        if not os.path.exists(csv_path):
            print(f"[ERROR] File {csv_path} tidak ditemukan!")
            print("Jalankan 'python data/scrape_alkitab.py' terlebih dahulu.")
            return

        print(f"Membangun Bible FAISS index dari {csv_path}...")
        print("Ini akan memakan waktu ~10-30 menit (hanya sekali)...")
        
        df = pd.read_csv(csv_path)
        
        documents = []
        for _, row in df.iterrows():
            text = str(row['text']).strip()
            reference = str(row['reference']).strip()
            book_abbr = str(row['book_abbr']).strip()
            book_name = str(row['book_name']).strip()
            chapter = int(row['chapter'])
            verse = int(row['verse'])
            
            if not text or text == 'nan':
                continue
            
            doc = Document(
                page_content=text,
                metadata={
                    "book_abbr": book_abbr,
                    "book_name": book_name,
                    "chapter": chapter,
                    "verse": verse,
                    "text": text,
                    "reference": reference,
                }
            )
            documents.append(doc)
        
        if not documents:
            print("[ERROR] Tidak ada dokumen valid ditemukan dalam CSV.")
            return
        
        print(f"Mengindeks {len(documents)} ayat... (sabar ya)")
        self.bible_db = FAISS.from_documents(documents, self.embeddings)
        
        # Simpan ke disk
        self.bible_db.save_local(index_dir)
        print(f"Bible index selesai dibangun dan disimpan ke {index_dir}")
        print(f"Total dokumen: {self.bible_db.index.ntotal}")

    def retrieve_verse(self, intents, user_input, k=1):
        """
        Ambil ayat Alkitab yang paling relevan berdasarkan semua intent + input user.
        
        Langkah:
        1. Gabungkan semua intent + keyword user menjadi satu query
        2. Cari di FAISS (semantic search) → ambil kandidat
        3. Re-rank berdasarkan keyword match dari user_input
        4. Jika ada seri, random pick
        
        Args:
            intents: list of detected intent strings (e.g., ["Perasaan Sedih", "Perasaan Takut"])
            user_input: teks input pengguna
            k: jumlah ayat yang ingin dikembalikan
            
        Returns:
            list of dict: [{reference: str, text: str}, ...]
        """
        if self.bible_db is None:
            print("[WARNING] Bible index belum dibangun!")
            return []

        if not intents and not user_input:
            return []

        # --- Step 1: Bangun combined query ---
        # Gabungkan semua intent + kata kunci user + sinonim Alkitab
        keywords = self._get_keyword_list(user_input) if user_input else []
        intent_part = " ".join(intents) if intents else ""
        keyword_part = " ".join(keywords)

        # Inject biblical synonyms for each detected intent
        synonym_parts = []
        for intent in (intents or []):
            synonyms = BIBLICAL_SYNONYMS.get(intent, "")
            if synonyms:
                synonym_parts.append(synonyms)
        synonym_part = " ".join(synonym_parts)

        combined_query = f"{intent_part} {keyword_part} {synonym_part}".strip()

        if not combined_query:
            return []

        # --- Step 2: Semantic search di FAISS ---
        # Ambil lebih banyak kandidat untuk re-ranking
        candidate_count = max(k * 10, 20)

        try:
            candidates_with_scores = self.bible_db.similarity_search_with_score(
                combined_query, k=candidate_count
            )
        except Exception as e:
            print(f"[ERROR] FAISS search gagal: {e}")
            return []

        if not candidates_with_scores:
            return []

        # --- Step 3: Keyword refinement ---
        
        scored_candidates = []
        for doc, faiss_score in candidates_with_scores:
            verse_text_lower = doc.metadata.get("text", "").lower()
            
            # Hitung berapa keyword yang muncul di ayat ini
            keyword_hits = 0
            if keywords:
                for kw in keywords:
                    if kw.lower() in verse_text_lower:
                        keyword_hits += 1
            
            scored_candidates.append({
                "doc": doc,
                "faiss_score": faiss_score,  # Lower = better in FAISS L2
                "keyword_hits": keyword_hits,
                "reference": doc.metadata.get("reference", ""),
                "text": doc.metadata.get("text", ""),
            })
        
        # Sort: keyword_hits DESC, faiss_score ASC (lower = closer)
        scored_candidates.sort(key=lambda x: (-x["keyword_hits"], x["faiss_score"]))
        
        # --- Step 4: Random tiebreaker ---
        # Jika beberapa kandidat punya skor sama di top, random pick
        if len(scored_candidates) > k:
            # Ambil grup teratas (skor keyword yang sama)
            top = scored_candidates[0]
            top_group = [
                c for c in scored_candidates
                if c["keyword_hits"] == top["keyword_hits"]
                and c["faiss_score"] == top["faiss_score"]
            ]
            
            if len(top_group) > k:
                # Random dari top group
                selected = random.sample(top_group, k)
            else:
                selected = scored_candidates[:k]
        else:
            selected = scored_candidates[:k]
        
        return [{"reference": s["reference"], "text": s["text"]} for s in selected]

    def _extract_keywords(self, text):
        """Ekstrak kata kunci dari teks (hapus stopwords)."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in STOPWORDS_ID]
        return " ".join(keywords)

    def _get_keyword_list(self, text):
        """Kembalikan list kata kunci dari teks."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in STOPWORDS_ID]

    # ── QnA Index (unchanged) ─────────────────────────────────────

    def build_qna_index(self, qna_csv_path='data/dataset_qna.csv', index_dir='data/faiss_qna_index'):
        if os.path.exists(index_dir):
            try:
                print("Memuat QnA FAISS index dari disk...")
                self.qna_db = FAISS.load_local(
                    index_dir, self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"QnA index dimuat. Total dokumen: {self.qna_db.index.ntotal}")
                return
            except Exception as e:
                print(f"Gagal memuat QnA index dari disk: {e}")
                print("Membangun ulang QnA index...")

        df_qna = pd.read_csv(qna_csv_path)

        documents = []
        for index, row in df_qna.iterrows():
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            if pd.isna(question) or question == 'nan':
                continue

            page_content = f"Pertanyaan: {question}"
            doc = Document(page_content=page_content, metadata={"answer": answer})
            documents.append(doc)

        if documents:
            self.qna_db = FAISS.from_documents(documents, self.embeddings)
            self.qna_db.save_local(index_dir)
            print(f"QnA index built and saved to {index_dir}.")
        else:
            print("No valid QnA documents found.")

    def search_qna(self, query, k=1):
        if self.qna_db is None:
            self.build_qna_index()
        results = self.qna_db.similarity_search(query, k=k)
        return results


if __name__ == "__main__":
    db_manager = VectorDBManager()
    
    # Build Bible index    db_manager.build_bible_index('data/alkitab_tb.csv')
    
    # Build QnA index
    db_manager.build_qna_index('data/dataset_qna.csv')
    
    # Test retrieval
    print("\n=== Test Retrieve Verse ===")
    intents = ["Perasaan Sedih dan Kehilangan", "Perasaan Takut dan Kecemasan"]
    user_input = "saya merasa sendirian dan takut tidak ada yang peduli"
    
    results = db_manager.retrieve_verse(intents, user_input, k=1)
    for r in results:
        print(f"  {r['reference']}: {r['text']}")
