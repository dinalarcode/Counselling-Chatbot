"""RAGAS AspectCritic evaluation engine for the biblical counseling chatbot RAG pipeline, run via the generate and evaluate CLI phases."""

import os
import sys
import re
import random
import argparse

import pandas as pd
import numpy as np

# Bootstrap the project root onto sys.path so config, core, and the static data module resolve.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import opt
from evaluation.ragas_manager.listof_ragaslist import (
    QNA_CSV, TESTSET_DIR, TESTSET_CSV, RESULTS_DIR, RANDOM_SEED, SAMPLE_SIZE,
    BIBLE_VERSE_STAGES, STAGES, STAGE_QUOTA, STAGE_MAX_CAPS, ADVERSARIAL_CASES,
    LABAN_CRITERIA_DEFINITION,
)


def balance_by_stage(
    df: pd.DataFrame,
    total: int,
    seed: int = RANDOM_SEED,
    quota_override: dict = None,
    max_caps: dict = None,
) -> pd.DataFrame:
    """Draw total rows from df spread across the 6 stages, optionally starting from per-stage quotas and backfilling any shortfall under the caps."""
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
    """Swap crafted adversarial inputs into randomly chosen sampled rows of their target stage while keeping the stage label."""
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


# Phase 1 generates the test template.

def classify_stage(question: str) -> str:
    """Heuristically assign a stage from question content patterns, which the user can adjust in the CSV before Phase 2."""
    q = question.strip().lower()

    # Pembukaan for greetings.
    if q in ("halo", "hai", "hi"):
        return "pembukaan"

    # Penutupan for closure signals.
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

    # Relaksasi for relaxation and gratitude signals.
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

    # Solusi for action and agreement signals.
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

    # Intervensi for acknowledgment and understanding signals.
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

    # Pembahasan for everything else such as problem sharing and emotional content.
    return "pembahasan"


def generate_testset():
    """Phase 1 samples QnA pairs and creates a template CSV."""
    print("=" * 65)
    print("  RAGAS Test Set Generator (Phase 1)")
    print("=" * 65)

    # Load the QnA dataset.
    if not os.path.exists(QNA_CSV):
        print(f"  [ERROR] QnA dataset not found: {QNA_CSV}")
        sys.exit(1)

    df = pd.read_csv(QNA_CSV)
    print(f"\n  QnA dataset loaded: {len(df)} rows")

    # Clean up.
    df = df.dropna(subset=["question"])
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].str.len() > 0]
    print(f"  After cleanup: {len(df)} valid rows")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Classify the whole corpus before sampling so sampling does not inherit the corpus skew.
    df["stage"] = df["question"].apply(classify_stage)
    print(f"\n  Corpus stage distribution (all {len(df)} rows):")
    for stage in STAGES:
        count = int((df["stage"] == stage).sum())
        marker = " (Bible verse)" if stage in BIBLE_VERSE_STAGES else ""
        print(f"    {stage}: {count}{marker}")

    print(f"\n  [INFO] Testset size is strictly set to {SAMPLE_SIZE} samples.")
    sample_df = balance_by_stage(df, total=SAMPLE_SIZE, quota_override=STAGE_QUOTA, max_caps=STAGE_MAX_CAPS)
    sample_df = inject_adversarial_cases(sample_df)

    # Build the output DataFrame.
    out_df = pd.DataFrame({
        "question": sample_df["question"].values,
        "stage": sample_df["stage"].values,
        "reference": "",  # Empty for the user to fill.
        "intent_override": "",  # Optional for the user to fill on Bible verse stages.
        "adversarial": sample_df["adversarial"].values,
    })

    # Pre-fill the intent_override hint for Bible verse stages.
    for idx in out_df.index:
        if out_df.at[idx, "stage"] in BIBLE_VERSE_STAGES:
            out_df.at[idx, "intent_override"] = (
                "Menyatakan Perasaan Sedih dan Kehilangan"  # Placeholder value.
            )

    # Save the template.
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
    print("  5. Run: python evaluation/ragas_manager/ragas_engine.py --evaluate")
    print(f"{'=' * 65}")


# Phase 2 runs the RAG pipeline and the RAGAS evaluation.

def run_evaluation():
    """Phase 2 runs the RAG pipeline on each sample and evaluates it with RAGAS."""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 65)
    print("  RAGAS Evaluation (Phase 2)")
    print("=" * 65)

    # Validate the test set.
    if not os.path.exists(TESTSET_CSV):
        print(f"  [ERROR] Test set not found: {TESTSET_CSV}")
        print("  Run 'python evaluation/ragas_manager/ragas_engine.py --generate' first.")
        sys.exit(1)

    df = pd.read_csv(TESTSET_CSV)
    print(f"\n  Test set loaded: {len(df)} samples")
    if "adversarial" in df.columns:
        n_adv = int(df["adversarial"].fillna(False).astype(bool).sum())
        print(f"  Adversarial rows in test set: {n_adv}")

    # Safety net that re-balances hand-edited CSVs and is a no-op on an already-balanced one.
    df = balance_by_stage(df, total=len(df))

    # Check for empty references on the balanced subset actually evaluated.
    empty_refs = df["reference"].isna() | (df["reference"].astype(str).str.strip() == "")
    if empty_refs.any():
        n_empty = empty_refs.sum()
        print(f"  [ERROR] {n_empty} samples have empty 'reference' column.")
        print("  Please fill in all reference answers before running evaluation.")
        # Show which rows are empty.
        empty_indices = df[empty_refs].index.tolist()
        print(f"  Empty rows: {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}")
        sys.exit(1)

    # Initialize the RAG engine.
    print(f"\n  Initializing RAG Engine...")
    from core.rag_engine.rag_engine import RAGEngine
    rag_engine = RAGEngine()
    print("  [OK] RAG Engine initialized")

    # Run the RAG pipeline on each sample.
    print(f"\n  Running RAG pipeline on {len(df)} samples...")
    results = []

    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        stage = str(row["stage"]).strip()
        reference = str(row["reference"]).strip()
        intent_override_raw = str(row.get("intent_override", "")).strip()
        adversarial = bool(row.get("adversarial", False))

        # Parse the intent overrides.
        override_intents = None
        if intent_override_raw and intent_override_raw != "nan":
            override_intents = [
                s.strip() for s in intent_override_raw.split(",") if s.strip()
            ]

        # Run the RAG pipeline.
        try:
            rag_result = rag_engine.generate_response(
                user_input=question,
                current_stage=stage,
                override_intents=override_intents,
                # RAGAS evaluates the verse-retrieval pipeline so verses must always be retrieved regardless of the runtime consent gate.
                spiritual_consent=True,
            )

            response = rag_result["response"]
            ctx = rag_result["context_used"]

            # Collect the retrieved contexts.
            retrieved_contexts = []
            if ctx.get("example_answer"):
                retrieved_contexts.append(ctx["example_answer"])
            if ctx.get("bible_verses"):
                retrieved_contexts.append(ctx["bible_verses"])

            # Add a placeholder when no context was retrieved.
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

    # Build the RAGAS dataset.
    print(f"\n  Building RAGAS evaluation dataset...")

    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import AspectCritic

    samples = []
    for r in results:
        sample = SingleTurnSample(
            # The stage prefix lets the LABAN AspectCritic judge stage appropriateness such as no premature solutions.
            user_input=f"[Tahap Konseling: {r['stage']}] {r['question']}",
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    print(f"  [OK] EvaluationDataset built with {len(samples)} samples")

    # Configure the OpenAI LLM judge and embeddings via langchain_openai.
    print(f"\n  Configuring OpenAI LLM judge via langchain_openai...")

    from langchain_openai import ChatOpenAI

    # Ensure the OpenAI API key is present before constructing the client since dotenv is already loaded.
    openai_api_key = os.getenv(opt.RAGAS_JUDGE_API_KEY_ENV)
    if not openai_api_key:
        print(f"  [ERROR] {opt.RAGAS_JUDGE_API_KEY_ENV} not found in .env")
        sys.exit(1)

    # evaluate() auto-wraps a raw langchain model into its own wrapper and binds it to any metric with llm None.
    ragas_judge_llm = ChatOpenAI(
        model=opt.RAGAS_JUDGE_MODEL,
        api_key=openai_api_key,
        temperature=opt.RAGAS_JUDGE_TEMPERATURE,
    )
    print(f"  [OK] OpenAI judge configured ({opt.RAGAS_JUDGE_MODEL})")

    # Define the metrics where the LABAN AspectCritic replaces the generic RAGAS metric stack in one binary criterion.
    laban_metric = AspectCritic(
        name="LABAN_Counseling_Standard",
        definition=LABAN_CRITERIA_DEFINITION,
    )
    metrics = [laban_metric]
    metric_names = ["LABAN_Counseling_Standard"]
    print(f"  Metrics: {metric_names}")

    # Preflight check validates the metric, LLM, and embeddings wiring locally before spending any API calls.
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

    # Run the RAGAS evaluation.
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

    # Save the results.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert to a DataFrame.
    results_df = ragas_results.to_pandas()

    # Add the stage and metadata columns.
    stages = [r["stage"] for r in results]
    results_df.insert(0, "adversarial", [r["adversarial"] for r in results])
    results_df.insert(0, "stage", stages)
    results_df.insert(0, "question", [r["question"] for r in results])

    # Per-sample CSV.
    per_sample_path = os.path.join(RESULTS_DIR, "ragas_per_sample.csv")
    results_df.to_csv(per_sample_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Saved: {per_sample_path}")

    # Summary CSV aggregated across all samples.
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

    # Per-stage CSV.
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

    # Console summary.
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


# Main entry point.

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
