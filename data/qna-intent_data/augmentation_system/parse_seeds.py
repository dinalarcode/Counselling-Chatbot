"""
Phase 1: Extract seed utterances per intent from intent_content.csv.

Reads the wide-format CSV (8 intent columns) and collects all non-empty cells
per column into a JSON dict: { intent_name: [seed1, seed2, ...] }.

Usage:
    python data/qna-intent_data/augmentation_system/parse_seeds.py
"""

import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from config import opt

INPUT_CSV = opt.INTENT_CONTENT_CSV
OUTPUT_JSON = os.path.join(SCRIPT_DIR, 'seeds.json')


def parse_intent_content(csv_path: str) -> dict:
    """
    Parse intent_content.csv and return a dict mapping
    intent_name -> list of non-empty seed sentences.
    """
    seeds = {}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Strip whitespace from header names
        intent_names = [col.strip() for col in header]

        # Initialize empty lists
        for name in intent_names:
            seeds[name] = []

        for row in reader:
            for i, cell in enumerate(row):
                if i >= len(intent_names):
                    break
                text = cell.strip()
                if text:
                    seeds[intent_names[i]].append(text)

    return seeds


def main():
    print(f"Reading: {INPUT_CSV}")
    seeds = parse_intent_content(INPUT_CSV)

    # Print summary
    print("\n=== Seed Count Per Intent ===")
    total = 0
    for intent, sentences in seeds.items():
        count = len(sentences)
        total += count
        print(f"  {intent}: {count} seeds")
    print(f"  TOTAL: {total}")

    # Save to JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
