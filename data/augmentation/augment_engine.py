"""
Phase 2: LLM-based augmentation engine using Groq's OpenAI-compatible API.

Generates new counseling client utterances for a given intent using few-shot
prompting, then deduplicates against existing + previously generated sentences.

Usage:
    from data.augmentation.augment_engine import AugmentEngine
    engine = AugmentEngine(api_key="...")
    new_sentences = engine.generate("Perasaan Sedih dan Kehilangan", seeds, n=20)
"""

import random
import re
import time
from typing import List, Optional

import requests


# Intent descriptions in Indonesian — used in the few-shot prompt
INTENT_DESCRIPTIONS = {
    "Perasaan Benci dan Jijik": "Ekspresi kebencian, jijik, penolakan, rasa enggan, atau perasaan muak terhadap seseorang atau situasi",
    "Perasaan Percaya": "Ekspresi kepercayaan, keterbukaan, kemauan untuk mencoba saran, atau kesediaan untuk mengikuti proses konseling",
    "Rasa Syukur dan Apresiasi": "Ekspresi rasa syukur, lega, apresiasi, perasaan membaik, atau terima kasih atas bantuan yang diterima",
    "Perasaan Sedih dan Kehilangan": "Ekspresi kesedihan, kehilangan, duka, merasa tidak berharga, putus asa, atau kesepian",
    "Reaksi Terkejut dan Tidak Terduga": "Ekspresi kaget, tidak percaya, bingung atas sesuatu yang tak terduga, atau reaksi terhadap kejadian mengejutkan",
    "Perasaan Takut dan Kecemasan": "Ekspresi rasa takut, khawatir, cemas, panik, gugup, atau perasaan terancam",
    "Perasaan Marah dan Frustasi": "Ekspresi kemarahan, frustrasi, kesal, jengkel, atau perasaan tidak sabar terhadap situasi",
    "Perasaan Sebelum Menghadapi Kejadian": "Ekspresi antisipasi, persiapan mental, niat atau rencana sebelum menghadapi suatu peristiwa atau tantangan",
    "Mengisyaratkan Gejala Fisik": "Mengungkapkan keluhan fisik seperti sakit kepala, sesak napas, susah tidur, gemetar, lemas, mual, atau kondisi fisik lainnya",
    "Mengisyaratkan Butuh Bantuan Profesional": "Permintaan bantuan konseling, rujukan, nasihat keahlian, atau pengakuan tidak mampu menangani sendirian",
}


class AugmentEngine:
    """
    Generates augmented counseling utterances via Groq API.
    Uses the OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def build_prompt(
        self,
        intent: str,
        seeds: List[str],
        n_samples: int,
        n_examples: int = 8,
    ) -> str:
        """
        Build the few-shot prompt for generating counseling utterances.

        Args:
            intent: The intent label (e.g., "Perasaan Sedih dan Kehilangan")
            seeds: All available seed sentences for this intent
            n_samples: Number of new sentences to generate
            n_examples: Number of seed examples to include in prompt (5-8)
        """
        # Sample random seed examples (up to n_examples)
        n_examples = min(n_examples, len(seeds))
        selected_seeds = random.sample(seeds, n_examples)

        numbered_list = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(selected_seeds)
        )

        description = INTENT_DESCRIPTIONS.get(intent, intent)

        prompt = f"""Kamu adalah asisten yang membantu membuat dataset pelatihan untuk chatbot konseling berbasis nilai Kristen.

Intent yang ingin dilatih: "{intent}"
Artinya: {description}

Berikut contoh kalimat klien yang mencerminkan intent tersebut:
{numbered_list}

Tugas: Buat tepat {n_samples} kalimat baru dalam Bahasa Indonesia yang:
- Terdengar seperti ucapan klien nyata dalam sesi konseling
- Secara jelas mencerminkan intent "{intent}"
- Beragam dalam gaya, panjang (1–3 kalimat), dan konteks situasi
- TIDAK mengulangi atau terlalu mirip dengan contoh di atas

Output: Satu kalimat utuh per baris. Tanpa penomoran. Tanpa penjelasan tambahan."""

        return prompt

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Groq chat completions endpoint. Returns raw response text."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 4096,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)

                if response.status_code == 429:
                    # Rate limited — wait and retry
                    wait = self.retry_delay * attempt
                    print(f"  [Rate limited] Waiting {wait:.0f}s before retry {attempt}/{self.max_retries}...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                print(f"  [API Error] Attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    print(f"  [FAILED] Giving up after {self.max_retries} attempts.")
                    return None

        return None

    def _parse_response(self, raw_text: str) -> List[str]:
        """
        Parse newline-delimited LLM output into a list of clean sentences.
        Strips numbering, empty lines, and metadata.
        """
        lines = raw_text.strip().split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove common numbering patterns: "1. ", "1) ", "- "
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            line = line.strip()
            # Skip lines that look like metadata/instructions
            if line.startswith('Berikut') or line.startswith('Tugas') or line.startswith('Output'):
                continue
            if len(line) < 5:
                continue
            cleaned.append(line)
        return cleaned

    @staticmethod
    def _tokenize(text: str) -> set:
        """Simple whitespace tokenizer for Jaccard similarity."""
        return set(text.lower().split())

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def deduplicate(
        self,
        existing_sentences: List[str],
        generated: List[str],
        threshold: float = 0.75,
    ) -> List[str]:
        """
        Remove near-duplicates from generated sentences.

        Checks against both existing (seed + original) sentences AND
        already-accepted generated sentences.

        Args:
            existing_sentences: All known sentences to compare against
            generated: Newly generated sentences to filter
            threshold: Jaccard similarity threshold (>= this = duplicate)

        Returns:
            Filtered list with near-duplicates removed
        """
        existing_tokens = [self._tokenize(s) for s in existing_sentences]
        accepted = []
        accepted_tokens = []

        for sentence in generated:
            sent_tokens = self._tokenize(sentence)

            # Check against existing
            is_dup = False
            for ext_tok in existing_tokens:
                if self._jaccard_similarity(sent_tokens, ext_tok) >= threshold:
                    is_dup = True
                    break

            # Check against already-accepted generated
            if not is_dup:
                for acc_tok in accepted_tokens:
                    if self._jaccard_similarity(sent_tokens, acc_tok) >= threshold:
                        is_dup = True
                        break

            if not is_dup:
                accepted.append(sentence)
                accepted_tokens.append(sent_tokens)

        return accepted

    def generate(
        self,
        intent: str,
        seeds: List[str],
        n_samples: int,
        existing_sentences: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate n_samples new sentences for the given intent.

        Args:
            intent: Intent label string
            seeds: Seed sentences for few-shot prompting
            n_samples: Number of sentences to generate per API call
            existing_sentences: Known sentences for deduplication

        Returns:
            List of deduplicated, clean generated sentences
        """
        prompt = self.build_prompt(intent, seeds, n_samples)
        raw_text = self._call_api(prompt)

        if raw_text is None:
            return []

        parsed = self._parse_response(raw_text)

        if existing_sentences:
            parsed = self.deduplicate(existing_sentences, parsed)

        return parsed
