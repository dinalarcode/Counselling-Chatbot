"""
Phase 3: Main CLI runner for LLM-based data augmentation.

Orchestrates the full pipeline:
  1. Load seeds from seeds.json
  2. Count existing samples per intent from intent_content.csv
  3. Generate new sentences via Groq (batched)
  4. Save raw output to raw_generated/*.txt
  5. Write intent_content_augmented.csv

Usage:
    python data/augmentation/run_augmentation.py
    python data/augmentation/run_augmentation.py --intents "Perasaan Benci dan Jijik"
    python data/augmentation/run_augmentation.py --batch 15 --dry_run
"""

import argparse
import json
import os
import time
from typing import Dict, List

import sys

# Load .env from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import importlib.util

def _import_from_file(module_name, file_path):
    """Import a module directly from a file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_engine_mod = _import_from_file(
    "augment_engine",
    os.path.join(SCRIPT_DIR, "augment_engine.py")
)
_output_mod = _import_from_file(
    "write_output",
    os.path.join(SCRIPT_DIR, "write_output.py")
)

AugmentEngine = _engine_mod.AugmentEngine
write_augmented_csv = _output_mod.write_augmented_csv
count_per_intent = _output_mod.count_per_intent

# =====================================================================
# CONFIGURATION — adjust targets here
# =====================================================================
TARGETS = {
    "Perasaan Sedih dan Kehilangan": 300,
    "Perasaan Takut dan Kecemasan": 300,
    "Perasaan Sebelum Menghadapi Kejadian": 300,
    "Perasaan Percaya": 350,
    "Perasaan Marah dan Frustasi": 350,
    "Rasa Syukur dan Apresiasi": 350,
    "Perasaan Benci dan Jijik": 400,
    "Reaksi Terkejut dan Tidak Terduga": 400,
}

# Groq model to use (free tier)
GROQ_MODEL = "llama-3.1-8b-instant"

# Delay between batches (seconds) — helps avoid rate limits on free tier
BATCH_DELAY = 10.0

# =====================================================================
# PATHS
# =====================================================================
SEEDS_JSON = os.path.join(SCRIPT_DIR, 'seeds.json')
ORIGINAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'intent_content.csv')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'intent_content_augmented.csv')
RAW_DIR = os.path.join(SCRIPT_DIR, 'raw_generated')


def load_seeds(path: str) -> Dict[str, List[str]]:
    """Load seed sentences from seeds.json."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_raw_output(intent: str, sentences: List[str], raw_dir: str) -> str:
    """Save generated sentences to a .txt file for inspection."""
    os.makedirs(raw_dir, exist_ok=True)
    filename = intent.replace(' ', '_') + '.txt'
    filepath = os.path.join(raw_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
    return filepath


def run_augmentation(
    intents_filter: str = "all",
    batch_size: int = 20,
    dry_run: bool = False,
) -> None:
    """
    Main augmentation loop.

    Args:
        intents_filter: "all" or a specific intent name to augment
        batch_size: Number of sentences to request per API call
        dry_run: If True, only prints the plan without calling the API
    """
    # Load API key
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")

    # Load seeds
    if not os.path.exists(SEEDS_JSON):
        raise FileNotFoundError(
            f"seeds.json not found at {SEEDS_JSON}. Run parse_seeds.py first."
        )
    seeds = load_seeds(SEEDS_JSON)

    # Use the augmented CSV as base if it already exists (preserves previous runs)
    base_csv = OUTPUT_CSV if os.path.exists(OUTPUT_CSV) else ORIGINAL_CSV
    existing_counts = count_per_intent(base_csv)
    print(f"\nBase CSV: {os.path.basename(base_csv)}")

    # Determine which intents to process
    if intents_filter == "all":
        intents_to_process = list(TARGETS.keys())
    else:
        if intents_filter not in TARGETS:
            raise ValueError(f"Unknown intent: '{intents_filter}'. Available: {list(TARGETS.keys())}")
        intents_to_process = [intents_filter]

    # Print plan
    print("=" * 70)
    print("LLM-Based Data Augmentation Plan")
    print(f"Model: {GROQ_MODEL}")
    print(f"Batch size: {batch_size}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)
    print(f"\n{'Intent':<45} {'Current':>8} {'Target':>8} {'Gap':>8}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*8}")

    for intent in intents_to_process:
        current = existing_counts.get(intent, 0)
        target = TARGETS[intent]
        gap = max(0, target - current)
        print(f"{intent:<45} {current:>8} {target:>8} {gap:>8}")

    if dry_run:
        print("\n[DRY RUN] No API calls made.")
        return

    # Initialize engine
    engine = AugmentEngine(api_key=api_key, model=GROQ_MODEL)

    # Generate for each intent
    all_generated: Dict[str, List[str]] = {}

    for intent in intents_to_process:
        current = existing_counts.get(intent, 0)
        target = TARGETS[intent]
        gap = max(0, target - current)

        if gap == 0:
            print(f"\n[SKIP] {intent}: already at {current} (target {target})")
            all_generated[intent] = []
            continue

        intent_seeds = seeds.get(intent, [])
        if len(intent_seeds) < 3:
            print(f"\n[WARNING] {intent}: only {len(intent_seeds)} seeds, need at least 3. Skipping.")
            all_generated[intent] = []
            continue

        print(f"\n{'='*70}")
        print(f"Generating for: {intent}")
        print(f"  Seeds available: {len(intent_seeds)}")
        print(f"  Need to generate: {gap}")

        collected: List[str] = []
        # Track all known sentences for deduplication
        all_known = list(intent_seeds)

        n_batches = (gap + batch_size - 1) // batch_size  # ceiling division

        for batch_num in range(1, n_batches + 1):
            remaining = gap - len(collected)
            if remaining <= 0:
                break
            n_request = min(batch_size, remaining + 5)  # request slightly more to compensate for dedup losses

            print(f"  Batch {batch_num}/{n_batches}: requesting {n_request} sentences...", end=" ", flush=True)

            new_sentences = engine.generate(
                intent=intent,
                seeds=intent_seeds,
                n_samples=n_request,
                existing_sentences=all_known + collected,
            )

            collected.extend(new_sentences)
            all_known.extend(new_sentences)
            print(f"got {len(new_sentences)}, total: {len(collected)}/{gap}")

            # Respect rate limits
            if batch_num < n_batches:
                time.sleep(BATCH_DELAY)

        # Trim to exact gap if we overshot
        collected = collected[:gap]
        all_generated[intent] = collected

        # Save raw output
        raw_path = save_raw_output(intent, collected, RAW_DIR)
        print(f"  Raw output saved: {raw_path}")
        print(f"  Final count: {len(collected)}/{gap}")

    # Write combined output CSV
    print(f"\n{'='*70}")
    print("Writing augmented CSV...")
    write_augmented_csv(base_csv, all_generated, OUTPUT_CSV)
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based data augmentation for counseling intent classification"
    )
    parser.add_argument(
        '--intents',
        type=str,
        default='all',
        help='Which intents to augment. Use "all" or a specific intent name.'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=20,
        help='Number of sentences per API call (default: 20)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print the augmentation plan without making API calls'
    )

    args = parser.parse_args()
    run_augmentation(
        intents_filter=args.intents,
        batch_size=args.batch,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
