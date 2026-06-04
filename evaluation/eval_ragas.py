"""
RAGAS Evaluation for Biblical Counseling Chatbot RAG Pipeline.

Two-phase workflow:
  Phase 1 (--generate): Sample 50 QnA pairs, auto-assign stages,
                        save template CSV for manual reference authoring.
  Phase 2 (--evaluate): Run RAG pipeline on each sample, evaluate with
                        RAGAS metrics using Groq LLM judge.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QNA_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_qna.csv")
TESTSET_DIR = os.path.join(SCRIPT_DIR, "data")
TESTSET_CSV = os.path.join(TESTSET_DIR, "ragas_testset.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RANDOM_SEED = 42
SAMPLE_SIZE = 50

# Stages where Bible verse retrieval is active
BIBLE_VERSE_STAGES = frozenset({"solusi", "relaksasi"})


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

    # Random sample
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if len(df) < SAMPLE_SIZE:
        print(f"  [WARN] Only {len(df)} rows available, using all.")
        sample_df = df.copy()
    else:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    print(f"  Sampled: {len(sample_df)} entries")

    # Auto-assign stages
    stages = sample_df["question"].apply(classify_stage)
    stage_counts = stages.value_counts()
    print(f"\n  Auto-assigned stage distribution:")
    for stage, count in stage_counts.items():
        marker = " (Bible verse)" if stage in BIBLE_VERSE_STAGES else ""
        print(f"    {stage}: {count}{marker}")

    # Build output DataFrame
    out_df = pd.DataFrame({
        "question": sample_df["question"].values,
        "stage": stages.values,
        "reference": "",  # Empty — user fills this
        "intent_override": "",  # Optional — user can fill for Bible verse stages
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

    # Check for empty references
    empty_refs = df["reference"].isna() | (df["reference"].astype(str).str.strip() == "")
    if empty_refs.any():
        n_empty = empty_refs.sum()
        print(f"  [ERROR] {n_empty} samples have empty 'reference' column.")
        print("  Please fill in all reference answers before running evaluation.")
        # Show which rows are empty
        empty_indices = df[empty_refs].index.tolist()
        print(f"  Empty rows: {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}")
        sys.exit(1)

    stage_counts = df["stage"].value_counts()
    print(f"  Stage distribution:")
    for stage, count in stage_counts.items():
        print(f"    {stage}: {count}")

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
            })

            print(f"    [{i+1}/{len(df)}] stage={stage} | "
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
            })

    print(f"  [OK] RAG pipeline completed for {len(results)} samples")

    # -- 4. Build RAGAS dataset ---------------------------------------------
    print(f"\n  Building RAGAS evaluation dataset...")

    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        ContextRecall,
    )

    samples = []
    for r in results:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    print(f"  [OK] EvaluationDataset built with {len(samples)} samples")

    # -- 5. Configure Gemini LLM judge --------------------------------------
    print(f"\n  Configuring Gemini LLM judge...")

    from config import opt
    from langchain_google_genai import ChatGoogleGenerativeAI

    judge_api_key = os.getenv(opt.RAGAS_JUDGE_API_KEY_ENV)
    if not judge_api_key:
        print(f"  [ERROR] {opt.RAGAS_JUDGE_API_KEY_ENV} not found in .env")
        sys.exit(1)

    judge_llm = ChatGoogleGenerativeAI(
        model=opt.RAGAS_JUDGE_MODEL,
        google_api_key=judge_api_key,
        temperature=opt.RAGAS_JUDGE_TEMPERATURE,
    )
    print(f"  [OK] Gemini judge configured ({opt.RAGAS_JUDGE_MODEL})")

    # -- 6. Define metrics --------------------------------------------------
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm),
        ContextPrecisionWithoutReference(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    print(f"  Metrics: {metric_names}")

    # -- 7. Run RAGAS evaluation --------------------------------------------
    print(f"\n  Running RAGAS evaluation (this may take a few minutes)...")

    try:
        ragas_results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
        )
    except Exception as e:
        print(f"  [ERROR] RAGAS evaluation failed: {e}")
        print("  This may be due to API rate limits or model compatibility.")
        print("  Try again or reduce sample size.")
        sys.exit(1)

    print(f"  [OK] RAGAS evaluation completed")

    # -- 8. Save results ----------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert to DataFrame
    results_df = ragas_results.to_pandas()

    # Add stage and metadata columns
    stages = [r["stage"] for r in results]
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

    # -- 9. Console summary -------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  RAGAS EVALUATION SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Judge LLM: Groq llama-3.1-8b-instant")
    print(f"\n  Overall Scores:")

    for col in metric_names:
        if col in results_df.columns:
            val = results_df[col].dropna()
            print(f"    {col:30s}: {val.mean():.4f} "
                  f"(+/- {val.std():.4f})")

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
        help="Phase 1: Generate test template CSV (50 random samples)",
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
