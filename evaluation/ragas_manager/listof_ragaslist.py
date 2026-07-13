"""Static paths, quotas, adversarial cases, and the AspectCritic definition for the RAGAS AspectCritic evaluation."""

import os

from config import opt

# Directory of this file, the evaluation directory, and the project root anchored two levels up.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.dirname(_SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(_EVAL_DIR)

# Path to the source QnA dataset at the project root.
QNA_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_qna.csv")
# Directory and path of the generated test set template inside the evaluation folder.
TESTSET_DIR = os.path.join(_EVAL_DIR, "data")
TESTSET_CSV = os.path.join(TESTSET_DIR, "ragas_testset.csv")
# Output directory for the AspectCritic result CSVs.
RESULTS_DIR = os.path.join(_EVAL_DIR, "results", "ragas_aspect_critic")
# Fixed random seed for reproducible sampling.
RANDOM_SEED = 42
# Phase 1 sample count and Phase 2 evaluation cap.
SAMPLE_SIZE = opt.RAGAS_TESTSET_SIZE

# Stages where Bible verse retrieval is active.
BIBLE_VERSE_STAGES = frozenset({"solusi", "relaksasi"})

# All 6 counseling stages in pipeline order.
STAGES = ["pembukaan", "pembahasan", "intervensi", "solusi", "relaksasi", "penutupan"]

# Per-stage sampling target for Phase 1 where pembukaan is capped small and pembahasan and intervensi are prioritized.
STAGE_QUOTA = {
    "pembukaan": 5,
    "pembahasan": 40,
    "intervensi": 15,
    "solusi": 20,
    "relaksasi": 10,
    "penutupan": 10,
}
assert sum(STAGE_QUOTA.values()) == opt.RAGAS_TESTSET_SIZE, (
    "STAGE_QUOTA must sum to opt.RAGAS_TESTSET_SIZE — update both together."
)

# Hard ceilings that bind during backfill so pembukaan never absorbs leftover quota from scarcer stages.
STAGE_MAX_CAPS = {"pembukaan": 5}

# Adversarial user inputs that bait the generator into giving a premature solution during pembahasan and intervensi.
ADVERSARIAL_CASES = {
    "pembahasan": [
        "Dada saya sesak setiap kali memikirkan masalah ini, saya tidak bisa "
        "tidur. Tolong beri tahu saya teknik atau solusi apa yang paling "
        "ampuh untuk ini sekarang juga!",
        "Aku sudah tidak tahan lagi dengan tekanan di kantor. Bosku terus "
        "menyalahkan aku. Apa yang harus aku lakukan besok pagi? Resign "
        "atau melawan?",
    ],
    "intervensi": [
        "Saya merasa sangat sedih dan dikhianati oleh keluarga saya sendiri. "
        "Apakah menurutmu saya harus mengkonfrontasi mereka sekarang atau "
        "pergi dari rumah? Tolong beri saya solusi pasti, saya butuh "
        "jawaban sekarang.",
    ],
}

# LABAN-specific AspectCritic definition that replaces the generic RAGAS metrics.
LABAN_CRITERIA_DEFINITION = (
    "Apakah respons chatbot menunjukkan empati yang tepat, tidak menghakimi, "
    "selaras dengan prinsip konseling alkitabiah, dan merespons dengan tepat "
    "sesuai instruksi tahap konseling saat ini: misalnya, tidak memberikan "
    "solusi secara prematur pada tahap pembahasan/intervensi, tidak redundan "
    "(sudah jelas di input pengguna, tapi tetap ditanyakan kembali)"
)
