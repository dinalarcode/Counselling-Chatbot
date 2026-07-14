"""
Step 2: Prepare Enriched Bible Dataset for Augmented Vector Indexing.

Merges the original alkitab_tb.csv with chapter_summaries.csv and creates
an enriched_text column that prepends the chapter summary to each verse.

Usage:
  cbenv\\Scripts\\python.exe data/verse_data/avi_system/prepare_enriched_bible.py

Output:
  data/verse_data/alkitab_tb_enriched_groq.csv
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import opt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIBLE_CSV = opt.BIBLE_RAW_CSV
SUMMARIES_CSV = opt.CHAPTER_SUMMARIES_CSV
OUTPUT_CSV = opt.BIBLE_ENRICHED_CSV


def main():
    print("=" * 65)
    print("  Enriched Bible Dataset Builder (AVI — Step 2)")
    print("=" * 65)

    # --- 1. Load original Bible CSV ---
    if not os.path.exists(BIBLE_CSV):
        print(f"  [ERROR] Bible CSV not found: {BIBLE_CSV}")
        sys.exit(1)

    print(f"\n  Loading {BIBLE_CSV}...")
    df_bible = pd.read_csv(BIBLE_CSV)
    print(f"  Total verses: {len(df_bible)}")

    # --- 2. Load chapter summaries ---
    if not os.path.exists(SUMMARIES_CSV):
        print(f"  [ERROR] Chapter summaries not found: {SUMMARIES_CSV}")
        print("  Run generate_chapter_summaries.py first (Step 1).")
        sys.exit(1)

    print(f"  Loading {SUMMARIES_CSV}...")
    df_summaries = pd.read_csv(SUMMARIES_CSV)
    print(f"  Total chapters with summaries: {len(df_summaries)}")

    # --- 3. Merge on (book_name, chapter) ---
    print(f"\n  Merging datasets (LEFT JOIN on book_name + chapter)...")

    # Ensure matching dtypes
    df_bible["chapter"] = df_bible["chapter"].astype(int)
    df_summaries["chapter"] = df_summaries["chapter"].astype(int)

    df_merged = df_bible.merge(
        df_summaries[["book_name", "chapter", "chapter_summary"]],
        on=["book_name", "chapter"],
        how="left",
    )

    print(f"  Merged rows: {len(df_merged)}")

    # Check for missing summaries
    missing_mask = df_merged["chapter_summary"].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        # Show which chapters are missing
        missing_chapters = (
            df_merged[missing_mask][["book_name", "chapter"]]
            .drop_duplicates()
            .values.tolist()
        )
        print(f"  [WARN] {n_missing} verses have no chapter summary "
              f"({len(missing_chapters)} chapters missing)")
        for book, ch in missing_chapters[:10]:
            print(f"    - {book} {ch}")
        if len(missing_chapters) > 10:
            print(f"    ... and {len(missing_chapters) - 10} more")

    # --- 4. Create enriched_text column ---
    print(f"\n  Creating enriched_text column...")

    def build_enriched_text(row):
        text = str(row["text"]).strip()
        summary = row.get("chapter_summary", "")

        if pd.isna(summary) or str(summary).strip() in ("", "nan"):
            # Fallback: no summary available → use raw text only
            return text

        summary = str(summary).strip()
        return f"[Konteks Pasal: {summary}] {text}"

    df_merged["enriched_text"] = df_merged.apply(build_enriched_text, axis=1)

    # --- 5. Validate ---
    print(f"\n  Validating...")

    # Row count
    assert len(df_merged) == len(df_bible), (
        f"Row count mismatch: {len(df_merged)} vs {len(df_bible)}"
    )
    print(f"  ✓ Row count: {len(df_merged)} (matches original)")

    # NaN check in enriched_text
    nan_enriched = df_merged["enriched_text"].isna().sum()
    assert nan_enriched == 0, f"{nan_enriched} NaN values in enriched_text"
    print(f"  ✓ No NaN in enriched_text")

    # Check enriched_text format (for rows with summaries)
    has_summary = ~missing_mask
    if has_summary.any():
        enriched_with_summary = df_merged.loc[has_summary, "enriched_text"]
        starts_correct = enriched_with_summary.str.startswith("[Konteks Pasal:")
        n_correct = starts_correct.sum()
        n_total = len(enriched_with_summary)
        print(f"  ✓ {n_correct}/{n_total} enriched rows start with [Konteks Pasal:]")

    # Sample output
    print(f"\n  Sample enriched_text (first verse):")
    sample = df_merged.iloc[0]["enriched_text"]
    print(f"  {sample[:150]}...")

    # --- 6. Save ---
    df_merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Saved: {OUTPUT_CSV}")
    print(f"  Columns: {df_merged.columns.tolist()}")
    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
