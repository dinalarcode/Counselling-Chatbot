"""FAISS vector database manager handling Bible and QnA index building, loading, and semantic search."""

import os
import random
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import opt

# Static stopwords, biblical synonyms, and diversity constants.
from core.vectordb_manager.listof_vdblist import (
    STOPWORDS_ID, BIBLICAL_SYNONYMS, OVERSAMPLE_FACTOR, MAX_PER_CHAPTER,
)


class VectorDBManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=opt.EMBED_MODEL)
        self.bible_db = None
        self.qna_db = None

    # Bible index over the full TB Bible.

    def build_bible_index(self, csv_path=opt.BIBLE_ENRICHED_CSV, index_dir=opt.FAISS_BIBLE_INDEX):
        """Build or load the FAISS index for the entire TB Bible using the enriched CSV, loading from disk when present."""
        # Try loading from disk first.
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

        # Build from the CSV.
        if not os.path.exists(csv_path):
            print(f"[ERROR] File {csv_path} tidak ditemukan!")
            print("Jalankan pipeline AVI terlebih dahulu:")
            print("  1. python data/verse_data/avi_system/generate_chapter_summaries.py")
            print("  2. python data/verse_data/avi_system/prepare_enriched_bible.py")
            return

        print(f"Membangun Bible FAISS index dari {csv_path}...")
        print("Ini akan memakan waktu ~15-45 menit (hanya sekali)...")

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

            # Use enriched_text for embedding and fall back to raw text when it is missing.
            enriched_text = str(row.get('enriched_text', '')).strip()
            page_content = enriched_text if enriched_text and enriched_text != 'nan' else text

            doc = Document(
                page_content=page_content,
                metadata={
                    "book_abbr": book_abbr,
                    "book_name": book_name,
                    "chapter": chapter,
                    "verse": verse,
                    "text": text,       # Raw verse displayed to the user.
                    "reference": reference,
                }
            )
            documents.append(doc)

        if not documents:
            print("[ERROR] Tidak ada dokumen valid ditemukan dalam CSV.")
            return

        print(f"Mengindeks {len(documents)} ayat (enriched)... (sabar ya)")
        self.bible_db = FAISS.from_documents(documents, self.embeddings)

        # Save to disk.
        self.bible_db.save_local(index_dir)
        print(f"Bible index selesai dibangun dan disimpan ke {index_dir}")
        print(f"Total dokumen: {self.bible_db.index.ntotal}")

    # FAISS semantic search covering layers 1 and 2.

    def _faiss_search(self, intents, user_input, k=10, excluded_books=None, excluded_verses=None):
        """Build the combined query, run FAISS semantic search with oversampling, and apply a diversity filter to return top-k diverse verses."""
        if self.bible_db is None:
            print("[WARNING] Bible index belum dibangun!")
            return []

        if not intents:
            return []

        if excluded_books is None:
            excluded_books = set()
        if excluded_verses is None:
            excluded_verses = set()

        # Layer 1 builds the combined query from intents and biblical synonyms only.
        intent_part = " ".join(intents)

        # Inject biblical synonyms for each detected intent.
        synonym_parts = []
        for intent in intents:
            synonyms = BIBLICAL_SYNONYMS.get(intent, "")
            if synonyms:
                synonym_parts.append(synonyms)
        synonym_part = " ".join(synonym_parts)

        combined_query = f"{intent_part} {synonym_part}".strip()

        if not combined_query:
            return []

        # Layer 2 runs the semantic search in FAISS with oversampling.
        oversample_k = k * OVERSAMPLE_FACTOR
        try:
            candidates_with_scores = self.bible_db.similarity_search_with_score(
                combined_query, k=oversample_k
            )
        except Exception as e:
            print(f"[ERROR] FAISS search gagal: {e}")
            return []

        if not candidates_with_scores:
            return []

        # Sort by L2 distance where lowest is best.
        candidates_with_scores.sort(key=lambda x: x[1])

        # Layer 2.5 applies the diversity filter over exclusions and per-chapter caps.
        results = []
        chapter_count = {}  # Maps book_name chapter key to its count in results so far.

        for doc, score in candidates_with_scores:
            reference = doc.metadata.get("reference", "")
            book_abbr = doc.metadata.get("book_abbr", "")

            # Session-level verse exclusion skips exact references already chosen.
            if reference in excluded_verses:
                continue

            # Session-level book exclusion skips all candidates from excluded books.
            if book_abbr in excluded_books:
                continue

            # Derive the chapter key from metadata when available and fall back to parsing the reference string.
            book_name = doc.metadata.get("book_name", "")
            chapter   = doc.metadata.get("chapter", None)
            if book_name and chapter is not None:
                chapter_key = f"{book_name} {chapter}"
            else:
                chapter_key = self._extract_chapter_key(reference)

            # Call-level diversity caps the per-chapter count.
            current_count = chapter_count.get(chapter_key, 0)
            if current_count >= MAX_PER_CHAPTER:
                continue

            chapter_count[chapter_key] = current_count + 1
            results.append({
                "reference": reference,
                "text": doc.metadata.get("text", ""),
                "book_abbr": book_abbr,
                "faiss_score": score,
            })

            if len(results) >= k:
                break

        print(f"[Diversity Filter] Oversample: {len(candidates_with_scores)}, "
              f"excluded books: {excluded_books or '{}'}, "
              f"excluded verses: {len(excluded_verses)}, "
              f"after filter: {len(results)} (max {MAX_PER_CHAPTER}/chapter)")

        return results

    # LLM reranker.

    def retrieve_verse_with_llm(self, intents, user_input, llm, k=10, excluded_books=None, excluded_verses=None):
        """Retrieve the most contextually relevant verse by getting FAISS diverse candidates and letting the LLM pick the best one."""
        # Get FAISS candidates with the diversity filters applied.
        candidates = self._faiss_search(intents, user_input, k=k, excluded_books=excluded_books, excluded_verses=excluded_verses)

        if not candidates:
            return []

        # If only one candidate exists, skip the LLM and return it directly.
        if len(candidates) == 1:
            return [{"reference": candidates[0]["reference"], "text": candidates[0]["text"], "book_abbr": candidates[0].get("book_abbr", "")}]

        # Print all FAISS candidate references to the terminal.
        candidate_refs = ", ".join(c["reference"] for c in candidates)
        print(f"[LLM Reranker] FAISS Candidates ({len(candidates)}): {candidate_refs}")

        # Build a numbered candidate list for the LLM.
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
            candidate_lines.append(f"{i}. {c['reference']} — \"{c['text']}\"")
        candidate_list_str = "\n".join(candidate_lines)

        # Build the reranker prompt.
        prompt = (
            "You are the Bible verse selector assistant for the Bible Terjemahan Baru (TB).\n\n"
            "The context of the user's counseling conversation:\n"
            f'"{user_input}"\n\n'
            "Here is the list of candidate verses:\n"
            f"{candidate_list_str}\n\n"
            "Instructions:\n"
            "1. Choose ONE verse that best suits the context of the conversation above.\n"
            "2. DO NOT make up book names and verses.\n"
            "3. Prioritize choosing verses that are strengthening and comforting.\n"
            "4. ⚠️ WARNING: You are dealing with the Word of God. "
            "DO NOT change, add, or remove any words from the verse text.\n"
            "5. Output format: REFERENCE|VERSE TEXT\n"
            "   Example: Mazmur 34:18|TUHAN itu dekat kepada orang-orang yang patah hati...\n"
            "6. Output ONLY one line and DO NOT translate it to english. No explanations, comments, or additional text.\n"
        )

        try:
            response = llm.invoke(prompt)
            raw_output = response.content.strip()

            # Parse the pipe-delimited output of reference and text.
            if "|" in raw_output:
                parts = raw_output.split("|", 1)
                ref = parts[0].strip()
                text = parts[1].strip().strip('"').strip("'")

                # Validate that the reference exists in our candidates.
                for c in candidates:
                    if c["reference"] == ref:
                        print(f"[LLM Reranker] Selected: {ref}")
                        return [{"reference": ref, "text": c["text"], "book_abbr": c.get("book_abbr", "")}]

                # The LLM returned a valid format but the reference is not in candidates, so use the LLM text with a warning.
                print(f"[LLM Reranker] WARNING: reference '{ref}' not in candidates, using LLM text")
                return [{"reference": ref, "text": text, "book_abbr": ""}]

            # Fallback tries to match the LLM output to a candidate reference.
            for c in candidates:
                if c["reference"] in raw_output:
                    print(f"[LLM Reranker] Parsed reference from raw output: {c['reference']}")
                    return [{"reference": c["reference"], "text": c["text"], "book_abbr": c.get("book_abbr", "")}]

            # Complete fallback when the LLM output is unparseable returns the top FAISS result.
            print(f"[LLM Reranker] FALLBACK: Could not parse LLM output, using top FAISS result")
            print(f"[LLM Reranker] Raw output was: {raw_output[:200]}")

        except Exception as e:
            print(f"[LLM Reranker] ERROR: {e} — falling back to top FAISS result")

        # Fallback returns the top FAISS candidate.
        return [{"reference": candidates[0]["reference"], "text": candidates[0]["text"], "book_abbr": candidates[0].get("book_abbr", "")}]

    # Legacy retrieve_verse using FAISS only without the LLM.

    def retrieve_verse(self, intents, user_input, k=1):
        """Return the top-k FAISS results by L2 distance without LLM reranking, kept for backward compatibility."""
        candidates = self._faiss_search(intents, user_input, k=k)
        return [{"reference": c["reference"], "text": c["text"]} for c in candidates]

    @staticmethod
    def _extract_chapter_key(reference: str) -> str:
        """Parse a Bible reference into a unique book and chapter key by dropping the verse number after the colon."""
        if ":" in reference:
            return reference.rsplit(":", 1)[0].strip()
        return reference.strip()

    def _extract_keywords(self, text):
        """Extract keywords from text by removing stopwords and return them as a joined string."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in STOPWORDS_ID]
        return " ".join(keywords)

    def _get_keyword_list(self, text):
        """Return the list of keywords from text after removing stopwords."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in STOPWORDS_ID]

    # QnA index unchanged.

    def build_qna_index(self, qna_csv_path=opt.QNA_CSV, index_dir=opt.FAISS_QNA_INDEX):
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

    # Build the Bible index.
    db_manager.build_bible_index()

    # Build the QnA index.
    db_manager.build_qna_index()

    # Test retrieval.
    print("\n=== Test Retrieve Verse ===")
    intents = ["Perasaan Sedih dan Kehilangan", "Perasaan Takut dan Kecemasan"]
    user_input = "saya merasa sendirian dan takut tidak ada yang peduli"

    results = db_manager.retrieve_verse(intents, user_input, k=1)
    for r in results:
        print(f"  {r['reference']}: {r['text']}")
