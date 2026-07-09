"""
RAGAS Evaluation for Biblical Counseling Chatbot RAG Pipeline.

Two-phase workflow:
  Phase 1 (--generate): Sample QnA pairs (config.opt.RAGAS_TESTSET_SIZE),
                        auto-assign stages per STAGE_QUOTA (pembukaan capped
                        small, pembahasan/intervensi prioritized), inject
                        adversarial cases (ADVERSARIAL_CASES) into
                        pembahasan/intervensi, save template CSV for manual
                        reference authoring.
  Phase 2 (--evaluate): Run RAG pipeline on each sample, evaluate with
                        RAGAS metrics using OpenAI as judge
                        (via langchain_openai.ChatOpenAI). Generator is
                        configured via opt.CHATBOT_LLM_PROVIDER.

Usage:
  python evaluation/eval_ragas.py --generate
  python evaluation/eval_ragas.py --evaluate

Output files (Phase 2):
  evaluation/results/ragas_per_sample.csv
  evaluation/results/ragas_summary.csv
  evaluation/results/ragas_per_stage.csv
"""

import os
import sys
import re
import random
import argparse

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import opt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QNA_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_qna.csv")
TESTSET_DIR = os.path.join(SCRIPT_DIR, "data")
TESTSET_CSV = os.path.join(TESTSET_DIR, "ragas_testset.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RANDOM_SEED = 42
SAMPLE_SIZE = opt.RAGAS_TESTSET_SIZE  # Phase 1 sample count AND Phase 2 eval cap

# Stages where Bible verse retrieval is active
BIBLE_VERSE_STAGES = frozenset({"solusi", "relaksasi"})

# All 6 counseling stages, in pipeline order
STAGES = ["pembukaan", "pembahasan", "intervensi", "solusi", "relaksasi", "penutupan"]

# Per-stage sampling target for Phase 1. `pembukaan` is capped small (mostly
# redundant greetings); `pembahasan`/`intervensi` get the largest requests
# since they carry the highest risk of premature-solution violations. Actual
# yield is capped by corpus availability per stage (see balance_by_stage) —
# `intervensi` in particular is scarce in the raw QnA corpus, so its shortfall
# is backfilled into other stages automatically.
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

# Hard ceilings that also bind during backfill (unlike STAGE_QUOTA, which is
# just a starting request that spare capacity can grow past). Only
# `pembukaan` needs this: its greetings are redundant enough that we never
# want it absorbing leftover quota from scarcer stages like `intervensi`.
STAGE_MAX_CAPS = {"pembukaan": 5}

# Adversarial user inputs that bait the generator into giving a premature
# solution during pembahasan/intervensi — the one rule in
# LABAN_CRITERIA_DEFINITION enforced only by prompt instruction, not by any
# hard-coded SessionManager guard. Injected in place of sampled rows for
# their stage (see inject_adversarial_cases()).
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

# LABAN-specific AspectCritic definition (replaces generic RAGAS metrics)
LABAN_CRITERIA_DEFINITION = (
    "Apakah respons chatbot menunjukkan empati yang tepat, tidak menghakimi, "
    "selaras dengan prinsip konseling alkitabiah, dan merespons dengan tepat "
    "sesuai instruksi tahap konseling saat ini: misalnya, tidak memberikan "
    "solusi secara prematur pada tahap pembahasan/intervensi, tidak redundan "
    "(sudah jelas di input pengguna, tapi tetap ditanyakan kembali)"
)


def balance_by_stage(
    df: pd.DataFrame,
    total: int,
    seed: int = RANDOM_SEED,
    quota_override: dict = None,
    max_caps: dict = None,
) -> pd.DataFrame:
    """
    Draw `total` rows from df spread across the 6 counseling stages.

    Without `quota_override`: split as evenly as possible (round-robin,
    scarce stages max out at their full availability, surplus stages absorb
    the remainder). With `quota_override`: start from those per-stage
    targets instead of an even split, still capped by availability, with any
    shortfall backfilled round-robin into stages under their own cap
    (`max_caps`, defaulting to full availability). Stages absent from df are
    skipped with a warning rather than crashing.
    """
    groups = {s: df[df["stage"] == s] for s in STAGES}
    present = {s: g for s, g in groups.items() if len(g) > 0}
    missing = [s for s in STAGES if s not in present]
    if missing:
        print(f"  [WARN] No samples for stage(s): {missing} - excluded from balanced set.")

    max_caps = max_caps or {}

    if quota_override is not None:
        quota = {
            s: min(quota_override.get(s, 0), len(present[s]), max_caps.get(s, len(present[s])))
            for s in present
        }
        remaining = total - sum(quota.values())
        while remaining > 0:
            spare = [
                s for s in present
                if quota[s] < min(len(present[s]), max_caps.get(s, len(present[s])))
            ]
            if not spare:
                break
            for stage in spare:
                if remaining <= 0:
                    break
                quota[stage] += 1
                remaining -= 1
    else:
        total = min(total, len(df))
        quota = dict.fromkeys(present, 0)
        while sum(quota.values()) < total:
            spare = [s for s in present if quota[s] < len(present[s])]
            if not spare:
                break
            for stage in spare:
                if sum(quota.values()) >= total:
                    break
                quota[stage] += 1

    balanced = (
        pd.concat([present[s].sample(n=n, random_state=seed) for s, n in quota.items() if n])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    print(f"\n  Balanced stage distribution:")
    for stage in STAGES:
        avail = len(present.get(stage, []))
        capped = " (all available)" if quota.get(stage, 0) == avail and avail else ""
        print(f"    {stage}: {quota.get(stage, 0)} / {avail} available{capped}")
    print(f"  Balanced total: {len(balanced)} samples")

    return balanced


def inject_adversarial_cases(sample_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Swap crafted adversarial user inputs (ADVERSARIAL_CASES) into randomly
    chosen already-sampled rows of their target stage, keeping the stage
    label so they still exercise that stage's retrieval/prompt rules.
    """
    sample_df = sample_df.copy()
    sample_df["adversarial"] = False
    rng = random.Random(seed)
    injected = 0

    for stage, cases in ADVERSARIAL_CASES.items():
        stage_idx = sample_df.index[sample_df["stage"] == stage].tolist()
        if not stage_idx:
            print(f"  [WARN] No '{stage}' rows sampled - cannot inject adversarial cases for this stage.")
            continue
        targets = rng.sample(stage_idx, k=min(len(cases), len(stage_idx)))
        for idx, case_text in zip(targets, cases):
            sample_df.at[idx, "question"] = case_text
            sample_df.at[idx, "adversarial"] = True
            injected += 1

    print(f"\n  [ADVERSARIAL] Injected {injected} adversarial case(s) into stage(s): "
          f"{list(ADVERSARIAL_CASES.keys())}")
    return sample_df


# =========================================================================
#  PHASE 1: Generate test template
# =========================================================================

def classify_stage(question: str) -> str:
    """
    Heuristic stage assignment based on question content patterns.
    The user can manually adjust in the CSV before running Phase 2.
    """
    q = question.strip().lower()

    # --- pembukaan: greetings ---
    if q in ("halo", "hai", "hi"):
        return "pembukaan"

    # --- penutupan: closure signals ---
    penutupan_patterns = [
        r"terima\s*kasih.*kak.*akan\s*(coba|mencoba)",
        r"terima\s*kasih.*akan\s*coba",
        r"aku\s*merasa\s*lebih\s*lega.*terima\s*kasih",
        r"terima\s*kasih.*sudah\s*mendengarkan",
        r"aku\s*akan\s*coba\s*semuanya",
        r"aku\s*bakal\s*coba.*terima\s*kasih",
    ]
    for pat in penutupan_patterns:
        if re.search(pat, q):
            return "penutupan"

    # --- relaksasi: relaxation/gratitude signals ---
    relaksasi_patterns = [
        r"lebih\s*tenang",
        r"lebih\s*nyaman",
        r"jadi\s*lebih\s*tenang",
        r"saya\s*merasa\s*lebih\s*baik",
        r"rasanya\s*lebih\s*enak",
        r"amin",
    ]
    for pat in relaksasi_patterns:
        if re.search(pat, q):
            return "relaksasi"

    # --- solusi: action/agreement signals ---
    solusi_patterns = [
        r"(saya|aku)\s*(akan\s*)?(coba|mencoba)",
        r"oke.*bisa\s*dicoba",
        r"bisa\s*dicoba",
        r"itu\s*masuk\s*akal",
        r"itu\s*bisa\s*dicoba",
        r"mungkin\s*bisa",
        r"baik.*akan\s*(coba|mencoba)",
        r"aku\s*mau\s*coba",
    ]
    for pat in solusi_patterns:
        if re.search(pat, q):
            return "solusi"

    # --- intervensi: acknowledgment/understanding ---
    intervensi_patterns = [
        r"(saya|aku)\s*mengerti",
        r"(saya|aku)\s*paham",
        r"benar\s*juga",
        r"masuk\s*akal",
        r"setuju",
        r"oh\s*iya",
        r"memang\s*benar",
    ]
    for pat in intervensi_patterns:
        if re.search(pat, q):
            return "intervensi"

    # --- pembahasan: everything else (problem-sharing, emotional content) ---
    return "pembahasan"


def generate_testset():
    """Phase 1: Sample QnA pairs and create a template CSV."""
    print("=" * 65)
    print("  RAGAS Test Set Generator (Phase 1)")
    print("=" * 65)

    # Load QnA dataset
    if not os.path.exists(QNA_CSV):
        print(f"  [ERROR] QnA dataset not found: {QNA_CSV}")
        sys.exit(1)

    df = pd.read_csv(QNA_CSV)
    print(f"\n  QnA dataset loaded: {len(df)} rows")

    # Clean up
    df = df.dropna(subset=["question"])
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].str.len() > 0]
    print(f"  After cleanup: {len(df)} valid rows")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Classify the WHOLE corpus before sampling. Sampling first would inherit
    # the corpus's natural skew (pembahasan dominates ~80% of the QnA rows).
    df["stage"] = df["question"].apply(classify_stage)
    print(f"\n  Corpus stage distribution (all {len(df)} rows):")
    for stage in STAGES:
        count = int((df["stage"] == stage).sum())
        marker = " (Bible verse)" if stage in BIBLE_VERSE_STAGES else ""
        print(f"    {stage}: {count}{marker}")

    print(f"\n  [INFO] Testset size is strictly set to {SAMPLE_SIZE} samples.")
    sample_df = balance_by_stage(df, total=SAMPLE_SIZE, quota_override=STAGE_QUOTA, max_caps=STAGE_MAX_CAPS)
    sample_df = inject_adversarial_cases(sample_df)

    # Build output DataFrame
    out_df = pd.DataFrame({
        "question": sample_df["question"].values,
        "stage": sample_df["stage"].values,
        "reference": "",  # Empty - user fills this
        "intent_override": "",  # Optional - user can fill for Bible verse stages
        "adversarial": sample_df["adversarial"].values,
    })

    # Pre-fill intent_override hint for Bible verse stages
    for idx in out_df.index:
        if out_df.at[idx, "stage"] in BIBLE_VERSE_STAGES:
            out_df.at[idx, "intent_override"] = (
                "Menyatakan Perasaan Sedih dan Kehilangan"  # placeholder
            )

    # Save
    os.makedirs(TESTSET_DIR, exist_ok=True)
    out_df.to_csv(TESTSET_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Saved: {TESTSET_CSV}")

    print(f"\n{'=' * 65}")
    print("  NEXT STEPS:")
    print("  1. Open the CSV file above")
    print("  2. Fill in the 'reference' column with ideal responses")
    print("     - For solusi/relaksasi stages, include Bible verses")
    print("     - For other stages, write the ideal counselor response")
    print("  3. Adjust 'stage' column if auto-assignment is incorrect")
    print("  4. Adjust 'intent_override' for Bible verse stages")
    print("  5. Run: python evaluation/eval_ragas.py --evaluate")
    print(f"{'=' * 65}")


# =========================================================================
#  PHASE 2: Run RAG + RAGAS evaluation
# =========================================================================

def run_evaluation():
    """Phase 2: Run RAG pipeline on each sample and evaluate with RAGAS."""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 65)
    print("  RAGAS Evaluation (Phase 2)")
    print("=" * 65)

    # -- 1. Validate test set -----------------------------------------------
    if not os.path.exists(TESTSET_CSV):
        print(f"  [ERROR] Test set not found: {TESTSET_CSV}")
        print("  Run 'python evaluation/eval_ragas.py --generate' first.")
        sys.exit(1)

    df = pd.read_csv(TESTSET_CSV)
    print(f"\n  Test set loaded: {len(df)} samples")
    if "adversarial" in df.columns:
        n_adv = int(df["adversarial"].fillna(False).astype(bool).sum())
        print(f"  Adversarial rows in test set: {n_adv}")

    # Phase 1 already balances the template; this is a safety net for
    # hand-edited CSVs and a no-op on an already-balanced one.
    df = balance_by_stage(df, total=len(df))

    # Check for empty references (only on the balanced subset actually evaluated)
    empty_refs = df["reference"].isna() | (df["reference"].astype(str).str.strip() == "")
    if empty_refs.any():
        n_empty = empty_refs.sum()
        print(f"  [ERROR] {n_empty} samples have empty 'reference' column.")
        print("  Please fill in all reference answers before running evaluation.")
        # Show which rows are empty
        empty_indices = df[empty_refs].index.tolist()
        print(f"  Empty rows: {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}")
        sys.exit(1)

    # -- 2. Initialize RAG Engine -------------------------------------------
    print(f"\n  Initializing RAG Engine...")
    from core.rag_engine import RAGEngine
    rag_engine = RAGEngine()
    print("  [OK] RAG Engine initialized")

    # -- 3. Run RAG pipeline on each sample ---------------------------------
    print(f"\n  Running RAG pipeline on {len(df)} samples...")
    results = []

    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        stage = str(row["stage"]).strip()
        reference = str(row["reference"]).strip()
        intent_override_raw = str(row.get("intent_override", "")).strip()
        adversarial = bool(row.get("adversarial", False))

        # Parse intent overrides
        override_intents = None
        if intent_override_raw and intent_override_raw != "nan":
            override_intents = [
                s.strip() for s in intent_override_raw.split(",") if s.strip()
            ]

        # Run RAG
        try:
            rag_result = rag_engine.generate_response(
                user_input=question,
                current_stage=stage,
                override_intents=override_intents,
                # RAGAS evaluates the verse-retrieval pipeline itself, so verses
                # must always be retrieved regardless of the runtime consent gate.
                spiritual_consent=True,
            )

            response = rag_result["response"]
            ctx = rag_result["context_used"]

            # Collect retrieved contexts
            retrieved_contexts = []
            if ctx.get("example_answer"):
                retrieved_contexts.append(ctx["example_answer"])
            if ctx.get("bible_verses"):
                retrieved_contexts.append(ctx["bible_verses"])

            # Fallback: if no context was retrieved, add a placeholder
            if not retrieved_contexts:
                retrieved_contexts = ["(no context retrieved)"]

            results.append({
                "question": question,
                "stage": stage,
                "reference": reference,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "intents": ctx.get("intents", []),
                "example_answer": ctx.get("example_answer", ""),
                "bible_verses": ctx.get("bible_verses", ""),
                "adversarial": adversarial,
            })

            adv_tag = " [ADVERSARIAL]" if adversarial else ""
            print(f"    [{i+1}/{len(df)}] stage={stage}{adv_tag} | "
                  f"ctx_count={len(retrieved_contexts)} | "
                  f"resp_len={len(response)}")

        except Exception as e:
            print(f"    [{i+1}/{len(df)}] [ERROR] {e}")
            results.append({
                "question": question,
                "stage": stage,
                "reference": reference,
                "response": f"ERROR: {e}",
                "retrieved_contexts": ["(error)"],
                "intents": [],
                "example_answer": "",
                "bible_verses": "",
                "adversarial": adversarial,
            })

    n_adversarial_run = sum(1 for r in results if r["adversarial"])
    print(f"  [OK] RAG pipeline completed for {len(results)} samples "
          f"({n_adversarial_run} adversarial)")

    # -- 4. Build RAGAS dataset ---------------------------------------------
    print(f"\n  Building RAGAS evaluation dataset...")

    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import AspectCritic

    samples = []
    for r in results:
        sample = SingleTurnSample(
            # Stage prefix lets the LABAN AspectCritic judge stage-appropriateness
            # (e.g. no premature solutions during pembahasan/intervensi).
            user_input=f"[Tahap Konseling: {r['stage']}] {r['question']}",
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    print(f"  [OK] EvaluationDataset built with {len(samples)} samples")

    # -- 5. Configure OpenAI LLM judge + embeddings (langchain_openai) ------
    print(f"\n  Configuring OpenAI LLM judge via langchain_openai...")

    from langchain_openai import ChatOpenAI

    # Guard: ensure OPENAI_API_KEY is present before constructing the client.
    # dotenv has already been loaded above.
    openai_api_key = os.getenv(opt.RAGAS_JUDGE_API_KEY_ENV)
    if not openai_api_key:
        print(f"  [ERROR] {opt.RAGAS_JUDGE_API_KEY_ENV} not found in .env")
        sys.exit(1)

    # evaluate() auto-wraps a raw langchain BaseLanguageModel into its own
    # LangchainLLMWrapper and binds it to any metric with llm=None — no
    # manual wrapping needed.
    ragas_judge_llm = ChatOpenAI(
        model=opt.RAGAS_JUDGE_MODEL,
        api_key=openai_api_key,
        temperature=opt.RAGAS_JUDGE_TEMPERATURE,
    )
    print(f"  [OK] OpenAI judge configured ({opt.RAGAS_JUDGE_MODEL})")

    # -- 6. Define metrics --------------------------------------------------
    # LABAN-specific AspectCritic replaces the generic RAGAS metric stack —
    # judges empathy, non-judgmental tone, biblical-counseling alignment, and
    # stage-appropriateness in one binary criterion. llm gets auto-injected
    # by evaluate() itself via the llm= kwarg below (same mechanism the old
    # metrics relied on).
    laban_metric = AspectCritic(
        name="LABAN_Counseling_Standard",
        definition=LABAN_CRITERIA_DEFINITION,
    )
    metrics = [laban_metric]
    metric_names = ["LABAN_Counseling_Standard"]
    print(f"  Metrics: {metric_names}")

    # -- 7. Preflight check (zero API cost) ---------------------------------
    # Validate metric/LLM/embeddings wiring locally before spending any API
    # calls — reuses ragas's own internal validators so a broken refactor
    # fails immediately instead of after N billed judge calls.
    print(f"\n  Running preflight checks (no API calls)...")

    from langchain_core.language_models import BaseLanguageModel as LangchainLLM
    from ragas.metrics.base import Metric
    from ragas.validation import validate_required_columns, validate_supported_metrics

    bad_metrics = [m for m in metrics if not isinstance(m, Metric)]
    if bad_metrics:
        print(f"  [PREFLIGHT FAIL] Not valid ragas Metric objects: {bad_metrics}")
        sys.exit(1)
    if not isinstance(ragas_judge_llm, LangchainLLM):
        print(f"  [PREFLIGHT FAIL] judge llm is not a Langchain BaseLanguageModel: {type(ragas_judge_llm)}")
        sys.exit(1)
    try:
        validate_required_columns(eval_dataset, metrics)
        validate_supported_metrics(eval_dataset, metrics)
    except ValueError as e:
        print(f"  [PREFLIGHT FAIL] {e}")
        sys.exit(1)

    print("  [OK] Preflight checks passed (no API calls made)")

    # -- 8. Run RAGAS evaluation --------------------------------------------
    print(f"\n  Running RAGAS evaluation (this may take a few minutes)...")

    import traceback

    try:
        ragas_results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=ragas_judge_llm,
        )
    except Exception as e:
        print(f"  [ERROR] RAGAS evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"  [OK] RAGAS evaluation completed")

    # -- 9. Save results ----------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert to DataFrame
    results_df = ragas_results.to_pandas()

    # Add stage and metadata columns
    stages = [r["stage"] for r in results]
    results_df.insert(0, "adversarial", [r["adversarial"] for r in results])
    results_df.insert(0, "stage", stages)
    results_df.insert(0, "question", [r["question"] for r in results])

    # Per-sample CSV
    per_sample_path = os.path.join(RESULTS_DIR, "ragas_per_sample.csv")
    results_df.to_csv(per_sample_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Saved: {per_sample_path}")

    # Summary CSV (aggregate across all samples)
    summary_data = {}
    for col in metric_names:
        if col in results_df.columns:
            values = results_df[col].dropna()
            summary_data[col] = {
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "count": len(values),
            }

    summary_df = pd.DataFrame(summary_data).T
    summary_df.index.name = "metric"
    summary_path = os.path.join(RESULTS_DIR, "ragas_summary.csv")
    summary_df.to_csv(summary_path, encoding="utf-8-sig")
    print(f"  [OK] Saved: {summary_path}")

    # Per-stage CSV
    stage_metrics = []
    for stage in sorted(results_df["stage"].unique()):
        stage_mask = results_df["stage"] == stage
        stage_row = {"stage": stage, "sample_count": stage_mask.sum()}
        for col in metric_names:
            if col in results_df.columns:
                values = results_df.loc[stage_mask, col].dropna()
                stage_row[f"{col}_mean"] = values.mean() if len(values) > 0 else None
                stage_row[f"{col}_std"] = values.std() if len(values) > 0 else None
        stage_metrics.append(stage_row)

    stage_df = pd.DataFrame(stage_metrics)
    stage_path = os.path.join(RESULTS_DIR, "ragas_per_stage.csv")
    stage_df.to_csv(stage_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Saved: {stage_path}")

    # -- 10. Console summary -------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  RAGAS EVALUATION SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Judge LLM: OpenAI {opt.RAGAS_JUDGE_MODEL} (via ragas llm_factory)")
    print(f"\n  Overall Scores:")

    for col in metric_names:
        if col in results_df.columns:
            val = results_df[col].dropna()
            print(f"    {col:30s}: {val.mean():.4f} "
                  f"(+/- {val.std():.4f})")

    adv_mask = results_df["adversarial"].astype(bool)
    if adv_mask.any():
        print(f"\n  Adversarial Cases ({int(adv_mask.sum())} injected):")
        for col in metric_names:
            if col in results_df.columns:
                adv_vals = results_df.loc[adv_mask, col].dropna()
                if len(adv_vals):
                    n_fail = int((adv_vals == 0).sum())
                    print(f"    {col:30s}: {adv_vals.mean():.4f} mean "
                          f"({n_fail}/{len(adv_vals)} failed)")

    print(f"\n  Per-Stage Breakdown:")
    for _, row in stage_df.iterrows():
        stage_name = row["stage"]
        n = int(row["sample_count"])
        print(f"\n    {stage_name} (n={n}):")
        for col in metric_names:
            mean_col = f"{col}_mean"
            if mean_col in row and pd.notna(row[mean_col]):
                print(f"      {col:28s}: {row[mean_col]:.4f}")

    print(f"\n  Output files:")
    print(f"    {per_sample_path}")
    print(f"    {summary_path}")
    print(f"    {stage_path}")
    print(f"{'=' * 65}")


# =========================================================================
#  Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for Biblical Counseling Chatbot"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate",
        action="store_true",
        help="Phase 1: Generate test template CSV (opt.RAGAS_TESTSET_SIZE samples)",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Phase 2: Run RAG pipeline + RAGAS evaluation",
    )

    args = parser.parse_args()

    if args.generate:
        generate_testset()
    elif args.evaluate:
        run_evaluation()
