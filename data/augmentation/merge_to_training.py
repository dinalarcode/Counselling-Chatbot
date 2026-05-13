"""
Phase 5: Merge validated intent_content_augmented.csv into the training format.

Reads the wide-format CSV, reconstructs full utterances per row,
detects multi-label intents, and outputs dataset_multiintent_augmented.csv
in the same (question, Intent) format as dataset_multiintent.csv.

Run this AFTER you have reviewed and edited intent_content_augmented.csv.

Usage:
    python data/augmentation/merge_to_training.py
"""

import csv
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'intent_content_augmented.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'dataset_multiintent_augmented.csv')


def merge_to_training(input_csv: str, output_csv: str) -> None:
    """
    Convert wide-format intent_content CSV to long-format training CSV.

    Each row in the input has 8 columns (one per intent). Non-empty cells
    indicate which intents are present. The full question is reconstructed
    by concatenating all non-empty cells (in column order, space-separated).

    Output format:
        question,Intent
        "full utterance text","Intent1; Intent2"
    """
    rows_out = []
    intent_counter = Counter()

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        intent_names = [col.strip() for col in header]

        for row_num, row in enumerate(reader, start=2):
            # Collect non-empty cells with their intent labels
            parts = []
            labels = []
            for i, cell in enumerate(row):
                if i >= len(intent_names):
                    break
                text = cell.strip()
                if text:
                    parts.append(text)
                    labels.append(intent_names[i])

            # Skip fully empty rows
            if not parts:
                continue

            # Reconstruct full question
            question = ' '.join(parts)

            # Build label string (semicolon-separated)
            label_str = '; '.join(labels)

            rows_out.append((question, label_str))

            # Count per intent
            for label in labels:
                intent_counter[label] += 1

    # Write output
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'Intent'])
        for question, label_str in rows_out:
            writer.writerow([question, label_str])

    # Print summary
    print("=" * 70)
    print("Merge to Training Format — Summary")
    print("=" * 70)
    print(f"\nTotal rows written: {len(rows_out)}")
    print(f"\nPer-intent distribution (a row can have multiple intents):")
    print(f"  {'Intent':<45} {'Count':>8}")
    print(f"  {'-'*45} {'-'*8}")
    for intent in intent_names:
        count = intent_counter.get(intent, 0)
        print(f"  {intent:<45} {count:>8}")

    print(f"\nOutput saved to: {output_csv}")


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Run the augmentation pipeline first, then review the output.")
        print("Expected file: data/intent_content_augmented.csv")
        return

    merge_to_training(INPUT_CSV, OUTPUT_CSV)


if __name__ == '__main__':
    main()
