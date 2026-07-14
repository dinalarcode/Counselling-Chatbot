"""
Phase 4: Write augmented data to intent_content_augmented.csv.

Reads the original intent_content.csv, appends new augmented rows (single-label),
and writes the combined output in the same wide-column format.

Usage:
    from data.augmentation.write_output import write_augmented_csv
    write_augmented_csv(original_csv, generated_dict, output_csv)
"""

import csv
import os
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# Column order must match the original intent_content.csv
INTENT_COLUMNS = [
    "Perasaan Benci dan Jijik",
    "Perasaan Percaya",
    "Rasa Syukur dan Apresiasi",
    "Perasaan Sedih dan Kehilangan",
    "Reaksi Terkejut dan Tidak Terduga",
    "Perasaan Takut dan Kecemasan",
    "Perasaan Marah dan Frustasi",
    "Perasaan Sebelum Menghadapi Kejadian",
]


def count_per_intent(csv_path: str) -> Dict[str, int]:
    """Count non-empty cells per intent column in the wide-format CSV."""
    counts = {col: 0 for col in INTENT_COLUMNS}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_indices = {}
        for i, col in enumerate(header):
            col_stripped = col.strip()
            if col_stripped in counts:
                col_indices[col_stripped] = i

        for row in reader:
            for intent, idx in col_indices.items():
                if idx < len(row) and row[idx].strip():
                    counts[intent] += 1

    return counts


def write_augmented_csv(
    original_csv: str,
    generated: Dict[str, List[str]],
    output_csv: str,
) -> None:
    """
    Write intent_content_augmented.csv by copying original rows and
    appending new augmented rows.

    Args:
        original_csv: Path to the original intent_content.csv
        generated: Dict mapping intent_name -> list of generated sentences
        output_csv: Path for the augmented output CSV
    """
    # Read all original rows
    original_rows = []
    with open(original_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # Skip fully empty rows
            if any(cell.strip() for cell in row):
                original_rows.append(row)

    # Build a column index map
    header_stripped = [col.strip() for col in header]

    # Build augmented rows — each new sentence → one row, one filled column
    augmented_rows = []
    for intent, sentences in generated.items():
        if intent not in header_stripped:
            print(f"  [WARNING] Unknown intent '{intent}', skipping.")
            continue
        col_idx = header_stripped.index(intent)
        for sentence in sentences:
            row = [''] * len(header)
            row[col_idx] = sentence
            augmented_rows.append(row)

    # Write combined output
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_stripped)
        for row in original_rows:
            # Pad if shorter than header
            padded = row + [''] * (len(header_stripped) - len(row))
            writer.writerow(padded[:len(header_stripped)])
        for row in augmented_rows:
            writer.writerow(row)

    # Print summary
    print("\n=== Augmentation Summary ===")
    before_counts = count_per_intent(original_csv)
    after_counts = count_per_intent(output_csv)

    print(f"  {'Intent':<45} {'Before':>8} {'Added':>8} {'After':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    for intent in INTENT_COLUMNS:
        before = before_counts.get(intent, 0)
        after = after_counts.get(intent, 0)
        added = after - before
        print(f"  {intent:<45} {before:>8} {added:>8} {after:>8}")

    print(f"\n  Output saved to: {output_csv}")
