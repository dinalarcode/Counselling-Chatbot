"""
Step 1: Generate Chapter Summaries for Augmented Vector Indexing.

Reads alkitab_tb.csv, groups by (book_name, chapter), and uses Groq
(llama-3.1-8b-instant) to summarize each chapter in 2-3 sentences.

Features:
  - Checkpoint/resume: saves progress after every chapter
  - Rate limiting: 2.1s sleep between API calls (~28 RPM)
  - Language guard: re-prompts once if response is not Indonesian
  - Error handling: logs failed chapters, does not crash

Usage:
  cbenv\\Scripts\\python.exe data/verse_retrieval/generate_chapter_summaries.py

Output:
  data/verse_retrieval/chapter_summaries.csv
"""

import os
import sys
import time
import csv

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIBLE_CSV = os.path.join(PROJECT_ROOT, "alkitab_tb.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "verse_retrieval")
# ---------------------------------------------------------------------------
# LLM Provider Toggle: "groq" or "ollama"
# ---------------------------------------------------------------------------
LLM_PROVIDER = "ollama"  # ← switch to "ollama" to use local Ollama

GROQ_MODEL = "llama-3.1-8b-instant"
SLEEP_SECONDS = 2.1  # ~28 RPM, under Groq free-tier 30 RPM limit

# OLLAMA_MODEL = "gemma:2b" #gemma
# OLLAMA_MODEL = "qwen2.5:3b" #qwen laptop
OLLAMA_MODEL = "qwen3:8b" #qwen pc
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama endpoint

# Output paths: suffix based on active provider
_SUFFIX = f"_{LLM_PROVIDER}" if LLM_PROVIDER == "ollama" else ""
CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, f"chapter_summaries_checkpoint{_SUFFIX}.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"chapter_summaries{_SUFFIX}.csv")
FAILED_LOG = os.path.join(OUTPUT_DIR, f"failed_chapters{_SUFFIX}.txt")

# Indonesian stopwords for language guard
INDO_STOPWORDS = {"pasal", "ini", "yang", "dan", "adalah", "dalam",
                  "dengan", "untuk", "pada", "dari", "tentang", "kepada"}

SUMMARIZE_PROMPT = (
    "Berikan ringkasan tema utama dan konteks dari pasal Alkitab berikut "
    "dalam 2-3 kalimat yang padat. Jawab dalam Bahasa Indonesia.\n\n"
    "{chapter_text}"
)

RETRY_PROMPT_SUFFIX = "\n\nPERINGATAN: Jawab HANYA dalam Bahasa Indonesia."


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    """Load already-processed (book_name, chapter) pairs from checkpoint."""
    done = set()
    if not os.path.exists(CHECKPOINT_CSV):
        return done
    try:
        df = pd.read_csv(CHECKPOINT_CSV)
        for _, row in df.iterrows():
            done.add((str(row["book_name"]), int(row["chapter"])))
    except Exception as e:
        print(f"  [WARN] Could not read checkpoint: {e}")
    return done


def group_chapters(df: pd.DataFrame) -> dict:
    """
    Group all verses by (book_name, chapter) and concatenate verse texts.
    Returns dict: {(book_name, chapter): full_chapter_text}
    """
    chapters = {}
    grouped = df.groupby(["book_name", "chapter"])
    for (book_name, chapter), group in grouped:
        verses = group.sort_values("verse")
        full_text = "\n".join(
            f"{int(row['verse'])}. {str(row['text']).strip()}"
            for _, row in verses.iterrows()
            if str(row["text"]).strip() and str(row["text"]).strip() != "nan"
        )
        if full_text:
            chapters[(str(book_name), int(chapter))] = full_text
    return chapters


def is_indonesian(text: str) -> bool:
    """Check if text contains at least 2 Indonesian stopwords."""
    words = set(text.lower().split())
    matches = words & INDO_STOPWORDS
    return len(matches) >= 2


def summarize_chapter(llm, chapter_text: str) -> str:
    """
    Summarize a chapter using Groq LLM.
    Includes language guard: retries once if response is not Indonesian.
    """
    prompt = SUMMARIZE_PROMPT.format(chapter_text=chapter_text)

    # First attempt
    response = llm.invoke(prompt)
    summary = response.content.strip()

    # Language guard: check if response is Indonesian
    if not is_indonesian(summary):
        # Retry with explicit Indonesian instruction
        retry_prompt = prompt + RETRY_PROMPT_SUFFIX
        response = llm.invoke(retry_prompt)
        retry_summary = response.content.strip()

        if is_indonesian(retry_summary):
            return retry_summary
        else:
            # Log warning but use it anyway (don't block progress)
            print(f"  [LANG WARN] Summary may not be Indonesian, using anyway")
            return retry_summary

    return summary


def append_checkpoint(book_name: str, chapter: int, summary: str):
    """Append one row to the checkpoint CSV."""
    file_exists = os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["book_name", "chapter", "chapter_summary"])
        writer.writerow([book_name, chapter, summary])


def log_failure(book_name: str, chapter: int, error: str):
    """Log a failed chapter to the failure log."""
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{book_name} {chapter}: {error}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Chapter Summary Generator (Augmented Vector Indexing — Step 1)")
    print("=" * 65)

    # Validate inputs
    if not os.path.exists(BIBLE_CSV):
        print(f"  [ERROR] Bible CSV not found: {BIBLE_CSV}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Bible data
    print(f"\n  Loading {BIBLE_CSV}...")
    df = pd.read_csv(BIBLE_CSV)
    print(f"  Total verses: {len(df)}")

    # Group into chapters
    chapters = group_chapters(df)
    total_chapters = len(chapters)
    print(f"  Total chapters: {total_chapters}")

    # Load checkpoint (already-processed chapters)
    done = load_checkpoint()
    remaining = {k: v for k, v in chapters.items() if k not in done}
    print(f"  Already processed (checkpoint): {len(done)}")
    print(f"  Remaining to process: {len(remaining)}")

    if not remaining:
        print("\n  All chapters already processed! Consolidating output...")
    else:
        # Initialize LLM based on provider
        if LLM_PROVIDER == "groq":
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                print("  [ERROR] GROQ_API_KEY not found in .env")
                sys.exit(1)
            print(f"\n  Initializing Groq LLM ({GROQ_MODEL})...")
            llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=groq_key,
                temperature=0.3,
            )
        elif LLM_PROVIDER == "ollama":
            print(f"\n  Initializing Ollama LLM ({OLLAMA_MODEL})...")
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
            )
        else:
            print(f"  [ERROR] Unknown LLM_PROVIDER: {LLM_PROVIDER}")
            sys.exit(1)
        print("  [OK] LLM ready")

        # Process remaining chapters
        estimated_time = len(remaining) * SLEEP_SECONDS / 60
        print(f"\n  Estimated time: ~{estimated_time:.0f} minutes")
        print(f"  Rate limit: {SLEEP_SECONDS}s sleep between calls")
        print()

        failed_count = 0
        sorted_remaining = sorted(remaining.keys())

        for book_name, chapter in tqdm(sorted_remaining, desc="  Summarizing", unit="ch"):
            chapter_text = remaining[(book_name, chapter)]

            try:
                summary = summarize_chapter(llm, chapter_text)
                append_checkpoint(book_name, chapter, summary)
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                log_failure(book_name, chapter, error_msg)
                tqdm.write(f"  [FAIL] {book_name} {chapter}: {error_msg[:80]}")

            # Rate limiting (always, even on failure)
            if LLM_PROVIDER == "groq":
                time.sleep(SLEEP_SECONDS)

        if failed_count > 0:
            print(f"\n  [WARN] {failed_count} chapters failed. See: {FAILED_LOG}")

    # Consolidate checkpoint into final output
    print(f"\n  Consolidating checkpoint → {OUTPUT_CSV}...")
    if os.path.exists(CHECKPOINT_CSV):
        final_df = pd.read_csv(CHECKPOINT_CSV)
        # Remove duplicates (keep last in case of re-runs)
        final_df = final_df.drop_duplicates(
            subset=["book_name", "chapter"], keep="last"
        )
        final_df = final_df.sort_values(["book_name", "chapter"]).reset_index(drop=True)
        final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"  [OK] Saved: {OUTPUT_CSV} ({len(final_df)} chapters)")

        # Validate
        nan_count = final_df["chapter_summary"].isna().sum()
        if nan_count > 0:
            print(f"  [WARN] {nan_count} chapters have empty summaries")
        if len(final_df) < total_chapters:
            missing = total_chapters - len(final_df)
            print(f"  [WARN] {missing} chapters missing (may need re-run)")
        if len(final_df) == total_chapters and nan_count == 0:
            print(f"  [OK] All {total_chapters} chapters summarized successfully!")
    else:
        print("  [ERROR] No checkpoint file found — nothing to consolidate")
        sys.exit(1)

    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
