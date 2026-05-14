"""
Multi-Label LLM Augmentation for Counseling Intent Classification
=================================================================
Generates MULTI-LABEL counseling utterances for intent COMBINATIONS
using the Groq Python library (Llama-3).

Pipeline:
    1. Read COMBINATION_CONFIG — list of (intents..., n_samples)
    2. Prompt Groq LLM for casual Indonesian student utterances
       that express ALL listed intents in a single sentence
    3. Parse pipe-delimited '|' output → multi-hot encoded DataFrame
    4. Merge with existing dataset_multiintent.csv → augmented_multilabel.csv

Usage:
    python data/augmentation/augment_multilabel.py

Configuration:
    Edit COMBINATION_CONFIG and GROQ_API_KEY below (or set in .env).
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Optional

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ========================================================================
# CONFIGURATION — Edit this section
# ========================================================================

# API Key: set directly here, or leave None to read from .env
GROQ_API_KEY: Optional[str] = None  # e.g. "gsk_abc123..."

# Groq model
GROQ_MODEL = "llama-3.3-70b-versatile"

# Delay between API calls (seconds) to avoid free-tier rate limits
CALL_DELAY = 8.0

# File paths
EXISTING_DATASET_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_multiintent.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "augmented_multilabel.csv")

# Combination config: each tuple = (*intent_names, n_samples)
# The last element is always the number of samples to generate.
# All preceding elements are intent names that must co-occur.
COMBINATION_CONFIG: list[tuple] = [
    ("Perasaan Marah dan Frustasi",  "Perasaan Sedih dan Kehilangan",     50),
    ("Perasaan Takut dan Kecemasan", "Reaksi Terkejut dan Tidak Terduga", 30),
    ("Rasa Syukur dan Apresiasi",    "Perasaan Percaya",                  30),
    ("Perasaan Benci dan Jijik",     "Perasaan Sedih dan Kehilangan",     30),
    ("Perasaan Marah dan Frustasi",  "Perasaan Takut dan Kecemasan",      30),
    # 3-label example:
    # ("Perasaan Sedih dan Kehilangan", "Perasaan Takut dan Kecemasan",
    #  "Perasaan Sebelum Menghadapi Kejadian", 20),
]


# ========================================================================
# INTENT DEFINITIONS
# ========================================================================

ALL_INTENTS: list[str] = [
    "Perasaan Benci dan Jijik",
    "Perasaan Percaya",
    "Rasa Syukur dan Apresiasi",
    "Perasaan Sedih dan Kehilangan",
    "Reaksi Terkejut dan Tidak Terduga",
    "Perasaan Takut dan Kecemasan",
    "Perasaan Marah dan Frustasi",
    "Perasaan Sebelum Menghadapi Kejadian",
]

INTENT_DESCRIPTIONS: dict[str, str] = {
    "Perasaan Benci dan Jijik":
        "rasa benci, jijik, enggan, muak, atau menolak seseorang/situasi",
    "Perasaan Percaya":
        "rasa percaya, terbuka, mau mencoba saran, atau mengikuti proses konseling",
    "Rasa Syukur dan Apresiasi":
        "rasa syukur, lega, apresiasi, merasa lebih baik, atau berterima kasih",
    "Perasaan Sedih dan Kehilangan":
        "rasa sedih, kehilangan, duka, tidak berharga, putus asa, atau kesepian",
    "Reaksi Terkejut dan Tidak Terduga":
        "kaget, tidak percaya, bingung atas sesuatu yang mengejutkan",
    "Perasaan Takut dan Kecemasan":
        "rasa takut, khawatir, cemas, panik, gugup, atau merasa terancam",
    "Perasaan Marah dan Frustasi":
        "kemarahan, frustrasi, kesal, jengkel, atau tidak sabar",
    "Perasaan Sebelum Menghadapi Kejadian":
        "antisipasi, persiapan mental, niat/rencana sebelum menghadapi tantangan",
}


# ========================================================================
# AUGMENTOR CLASS
# ========================================================================

class MultiLabelAugmentor:
    """Generates multi-label counseling utterances via Groq LLM API."""

    # Keywords that indicate the LLM leaked its own prompt into the output
    _PROMPT_LEAK_KEYWORDS = ("Mulai", "Output", "Kalimat", "Buat")

    def __init__(
        self,
        api_key: str,
        model: str = GROQ_MODEL,
        temperature: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # --- Prompt ---

    def _build_prompt(self, intents: tuple[str, ...], n_samples: int) -> str:
        """
        Build the LLM prompt for generating casual Indonesian student
        utterances that express ALL given intents simultaneously.

        Output format requested: sentences separated by '|'.
        """
        intent_lines = "\n".join(
            f"  - {intent} → {INTENT_DESCRIPTIONS[intent]}"
            for intent in intents
        )

        return (
            "Kamu adalah asisten pembuat dataset konseling untuk mahasiswa Indonesia.\n\n"
            f"Tugasmu: Buat tepat {n_samples} kalimat pendek (1 kalimat) dalam "
            "Bahasa Indonesia GAUL / sehari-hari yang biasa diucapkan oleh "
            "mahasiswa atau pelajar kepada konselor/kakak konseling.\n\n"
            "SETIAP kalimat HARUS mencerminkan SEMUA emosi/intent berikut "
            "SECARA BERSAMAAN dalam satu kalimat:\n"
            f"{intent_lines}\n\n"
            "Aturan ketat:\n"
            '1. Gaya bahasa: kasual, santai, seperti ngobrol — boleh pakai '
            '"aku", "sih", "tuh", "banget", "nggak", "gitu", dsb.\n'
            "2. Setiap kalimat = satu utterance utuh (bukan dialog, bukan narasi panjang).\n"
            "3. Kalimat HARUS mewakili SEMUA intent di atas secara alami dalam satu kalimat.\n"
            "4. JANGAN beri penomoran, label, penjelasan, atau tanda lain — hanya kalimatnya saja.\n"
            "5. Pisahkan setiap kalimat dengan karakter pipe '|' (tanpa spasi di sekitarnya).\n"
            f"6. Output: tepat {n_samples} kalimat, dipisah '|'. "
            "Contoh format: kalimat1|kalimat2|kalimat3\n\n"
            "Mulai output sekarang:"
        )

    # --- API call with retry ---

    def _call_api(self, prompt: str) -> Optional[str]:
        """
        Send prompt to Groq chat completions with exponential backoff retry.

        Returns raw response text, or None on total failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=4096,
                )
                return response.choices[0].message.content

            except Exception as exc:
                wait = self.retry_delay * attempt
                print(f"  [API Error] Attempt {attempt}/{self.max_retries}: {exc}")
                if attempt < self.max_retries:
                    print(f"  Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    print(f"  [FAILED] Giving up after {self.max_retries} attempts.")
                    return None

        return None  # Unreachable, but satisfies type checker

    # --- Response parser ---

    def _parse_pipe_output(self, raw_text: str) -> list[str]:
        """
        Parse pipe-delimited '|' LLM output into a list of clean sentences.

        Handles: newlines mixed with pipes, numbering artifacts, short fragments,
        and prompt-leaked lines.
        """
        if not raw_text:
            return []

        # Normalize: some models mix newlines with pipes
        normalized = raw_text.replace("\n", "|")
        parts = normalized.split("|")

        cleaned: list[str] = []
        for part in parts:
            sentence = part.strip()
            if not sentence:
                continue

            # Strip residual numbering: "1. ", "1) ", "- ", "• "
            sentence = re.sub(r"^[\d]+[.)]\s*", "", sentence)
            sentence = re.sub(r"^[-•]\s*", "", sentence)
            sentence = sentence.strip()

            # Skip short fragments or leaked prompt text
            if len(sentence) < 8:
                continue
            if any(sentence.startswith(kw) for kw in self._PROMPT_LEAK_KEYWORDS):
                continue

            cleaned.append(sentence)

        return cleaned

    # --- Public method ---

    def generate(self, intents: tuple[str, ...], n_samples: int) -> list[str]:
        """
        Generate n_samples sentences for an intent combination.

        Args:
            intents:   Tuple of 2+ intent names.
            n_samples: Number of sentences to request.

        Returns:
            List of parsed sentences (may be fewer on API error).
        """
        prompt = self._build_prompt(intents, n_samples)
        print(f"  Calling Groq ({self.model})...", end=" ", flush=True)

        raw = self._call_api(prompt)
        if raw is None:
            print("FAILED.")
            return []

        sentences = self._parse_pipe_output(raw)
        print(f"got {len(sentences)} sentences.")
        return sentences


# ========================================================================
# DATA HELPERS
# ========================================================================

def build_multihot_df(
    sentences: list[str],
    active_intents: tuple[str, ...],
) -> pd.DataFrame:
    """
    Create a DataFrame with multi-hot encoded intent columns.

    Args:
        sentences:      List of utterance strings.
        active_intents: Intents to set to 1 for all rows.

    Returns:
        DataFrame: columns = [text, <all 8 intent columns>]
    """
    active_set = set(active_intents)
    encoding = {intent: (1 if intent in active_set else 0) for intent in ALL_INTENTS}

    records = [{"text": text, **encoding} for text in sentences]
    return pd.DataFrame(records)


def load_existing_as_multihot(csv_path: str) -> pd.DataFrame:
    """
    Load dataset_multiintent.csv and convert to multi-hot format.

    Input format:  question, Intent  (semicolon-separated labels)
    Output format: text, <8 binary intent columns>
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={"question": "text", "Intent": "_labels_raw"})
    df = df.dropna(subset=["text"]).copy()
    df["_labels_raw"] = df["_labels_raw"].fillna("")

    records = []
    for _, row in df.iterrows():
        labels = {lbl.strip() for lbl in str(row["_labels_raw"]).split(";") if lbl.strip()}

        rec = {"text": row["text"]}
        for intent in ALL_INTENTS:
            rec[intent] = 1 if intent in labels else 0
        records.append(rec)

    return pd.DataFrame(records)


# ========================================================================
# MAIN PIPELINE
# ========================================================================

def run_multilabel_augmentation() -> None:
    """
    Full augmentation pipeline:
        1. Resolve API key
        2. Generate sentences per combination via Groq
        3. Multi-hot encode and merge with existing dataset
        4. Save to augmented_multilabel.csv
    """
    # 1. API key
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. "
            "Set it in .env or directly in this script."
        )

    # 2. Validate config
    if not COMBINATION_CONFIG:
        raise ValueError("COMBINATION_CONFIG is empty.")

    # Print plan
    print("=" * 70)
    print("Multi-Label LLM Augmentation")
    print(f"  Model  : {GROQ_MODEL}")
    print(f"  Output : {OUTPUT_CSV}")
    print("=" * 70)
    print(f"\n  {'Combination':<55} {'N':>5}")
    print(f"  {'-'*55} {'-'*5}")
    for entry in COMBINATION_CONFIG:
        *intents, n = entry
        print(f"  {' + '.join(intents):<55} {n:>5}")

    # 3. Generate
    augmentor = MultiLabelAugmentor(api_key=api_key)
    augmented_frames: list[pd.DataFrame] = []

    for i, entry in enumerate(COMBINATION_CONFIG, start=1):
        *intent_list, n_samples = entry
        intents = tuple(intent_list)

        # Validate intent names
        unknown = [x for x in intents if x not in ALL_INTENTS]
        if unknown:
            print(f"\n[SKIP] Unknown intent(s) in combo {i}: {unknown}")
            continue

        print(f"\n[{i}/{len(COMBINATION_CONFIG)}] {' + '.join(intents)}")
        print(f"  Target: {n_samples} samples")

        try:
            sentences = augmentor.generate(intents, n_samples)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            sentences = []

        if sentences:
            df_combo = build_multihot_df(sentences, intents)
            augmented_frames.append(df_combo)
            print(f"  Collected: {len(df_combo)} rows")
        else:
            print("  No sentences collected.")

        # Rate-limit delay between API calls
        if i < len(COMBINATION_CONFIG):
            print(f"  Waiting {CALL_DELAY:.0f}s...")
            time.sleep(CALL_DELAY)

    # 4. Load existing dataset
    print(f"\n{'='*70}")
    print("Loading existing dataset...")
    try:
        df_existing = load_existing_as_multihot(EXISTING_DATASET_CSV)
        print(f"  Loaded {len(df_existing)} rows from {os.path.basename(EXISTING_DATASET_CSV)}")
    except FileNotFoundError:
        print(f"  [WARNING] {EXISTING_DATASET_CSV} not found. Using augmented data only.")
        df_existing = pd.DataFrame(columns=["text"] + ALL_INTENTS)

    # 5. Merge & save
    df_augmented = (
        pd.concat(augmented_frames, ignore_index=True)
        if augmented_frames
        else pd.DataFrame(columns=["text"] + ALL_INTENTS)
    )

    df_final = pd.concat([df_existing, df_augmented], ignore_index=True)

    # Ensure intent columns are int
    for col in ALL_INTENTS:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0).astype(int)

    # Drop empty text rows
    df_final = df_final.dropna(subset=["text"])
    df_final = df_final[df_final["text"].str.strip() != ""]
    df_final = df_final.reset_index(drop=True)

    try:
        df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    except Exception as exc:
        print(f"\n[ERROR] Failed to save CSV: {exc}")
        raise

    # Summary
    print(f"\n{'='*70}")
    print("Augmentation Complete")
    print(f"  Total rows  : {len(df_final)}")
    print(f"    Existing  : {len(df_existing)}")
    print(f"    Augmented : {len(df_augmented)}")
    print(f"  Output      : {OUTPUT_CSV}")
    print(f"\n  {'Intent':<45} {'Count':>7}")
    print(f"  {'-'*45} {'-'*7}")
    for intent in ALL_INTENTS:
        count = int(df_final[intent].sum()) if intent in df_final.columns else 0
        print(f"  {intent:<45} {count:>7}")
    print()


# ========================================================================
# ENTRY POINT
# ========================================================================

if __name__ == "__main__":
    run_multilabel_augmentation()
