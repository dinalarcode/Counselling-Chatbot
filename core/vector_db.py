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

    # ── FAISS Semantic Search (Layer 1 + 2) ──────────────────────────────

    def _faiss_search(self, intents, user_input, k=5):
        """
        Layer 1: Build combined query (intent + keywords + biblical synonyms)
        Layer 2: FAISS semantic search → top-k candidate verses
        
        Returns:
            list of dict: [{reference: str, text: str}, ...] sorted by L2 distance
        """
        if self.bible_db is None:
            print("[WARNING] Bible index belum dibangun!")
            return []

        if not intents and not user_input:
            return []

        # --- Layer 1: Bangun combined query ---
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

        # --- Layer 2: Semantic search di FAISS ---
        try:
            candidates_with_scores = self.bible_db.similarity_search_with_score(
                combined_query, k=k
            )
        except Exception as e:
            print(f"[ERROR] FAISS search gagal: {e}")
            return []

        if not candidates_with_scores:
            return []

        # Return sorted by L2 distance (lowest = best)
        results = []
        for doc, score in candidates_with_scores:
            results.append({
                "reference": doc.metadata.get("reference", ""),
                "text": doc.metadata.get("text", ""),
                "faiss_score": score,
            })
        results.sort(key=lambda x: x["faiss_score"])
        return results

    # ── LLM Reranker ──────────────────────────────────────────────────

    def retrieve_verse_with_llm(self, intents, user_input, llm, k=10):
        """
        Retrieve the most contextually relevant Bible verse using:
        Layer 1 + 2: FAISS semantic search → top-k candidates
        LLM Reranker: Gemini picks the best verse based on user context
        
        Args:
            intents: list of detected intent strings
            user_input: raw user utterance (conversational context)
            llm: LangChain LLM instance (reuse from RAGEngine)
            k: number of FAISS candidates to present to the LLM
            
        Returns:
            list of dict: [{reference: str, text: str}] — always 1 result
        """
        # Step 1+2: Get FAISS candidates
        candidates = self._faiss_search(intents, user_input, k=k)

        if not candidates:
            return []

        # If only 1 candidate, skip LLM — return directly
        if len(candidates) == 1:
            return [{"reference": candidates[0]["reference"], "text": candidates[0]["text"]}]

        # Build numbered candidate list for the LLM
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
            candidate_lines.append(f"{i}. {c['reference']} — \"{c['text']}\"")
        candidate_list_str = "\n".join(candidate_lines)

        # Build the reranker prompt
        prompt = (
            "Kamu adalah asisten pemilih ayat Alkitab Terjemahan Baru (TB).\n\n"
            "Konteks percakapan konseling dari pengguna:\n"
            f'"{user_input}"\n\n'
            "Berikut adalah daftar ayat kandidat:\n"
            f"{candidate_list_str}\n\n"
            "Instruksi:\n"
            "1. Pilih SATU ayat yang paling sesuai dengan konteks percakapan di atas.\n"
            "2. Utamakan pilih ayat yang bersifat menguatkan dan memberi penghiburan.\n"
            "3. ⚠️ PERINGATAN: Kamu sedang berurusan dengan Firman Tuhan. "
            "JANGAN mengubah, menambahkan, atau menghilangkan satu kata pun dari teks ayat.\n"
            "4. Format output: REFERENSI|TEKS AYAT\n"
            "   Contoh: Mazmur 34:18|TUHAN itu dekat kepada orang-orang yang patah hati...\n"
            "5. Output HANYA satu baris. Tidak ada penjelasan, komentar, atau teks tambahan.\n"
        )

        try:
            response = llm.invoke(prompt)
            raw_output = response.content.strip()

            # Parse pipe-delimited output: REFERENSI|TEKS
            if "|" in raw_output:
                parts = raw_output.split("|", 1)
                ref = parts[0].strip()
                text = parts[1].strip().strip('"').strip("'")

                # Validate: reference must exist in our candidates
                for c in candidates:
                    if c["reference"] == ref:
                        print(f"[LLM Reranker] Selected: {ref}")
                        return [{"reference": ref, "text": c["text"]}]

                # LLM returned a valid format but reference not in candidates
                # Use the text from LLM but log a warning
                print(f"[LLM Reranker] WARNING: reference '{ref}' not in candidates, using LLM text")
                return [{"reference": ref, "text": text}]

            # Fallback: try to match LLM output to a candidate reference
            for c in candidates:
                if c["reference"] in raw_output:
                    print(f"[LLM Reranker] Parsed reference from raw output: {c['reference']}")
                    return [{"reference": c["reference"], "text": c["text"]}]

            # Complete fallback: LLM output unparseable → return top FAISS result
            print(f"[LLM Reranker] FALLBACK: Could not parse LLM output, using top FAISS result")
            print(f"[LLM Reranker] Raw output was: {raw_output[:200]}")

        except Exception as e:
            print(f"[LLM Reranker] ERROR: {e} — falling back to top FAISS result")

        # Fallback: return the top FAISS candidate
        return [{"reference": candidates[0]["reference"], "text": candidates[0]["text"]}]

    # ── Legacy retrieve_verse (FAISS-only, no LLM) ───────────────────

    def retrieve_verse(self, intents, user_input, k=1):
        """
        Simple FAISS-only retrieval (no LLM reranking).
        Returns the top-k results by L2 distance.
        Kept for backward compatibility and non-LLM contexts.
        """
        candidates = self._faiss_search(intents, user_input, k=k)
        return [{"reference": c["reference"], "text": c["text"]} for c in candidates]

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
